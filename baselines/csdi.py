"""
baselines/csdi.py
CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series
论文: Tashiro et al., NeurIPS 2021  https://arxiv.org/abs/2107.03502

多步改动: T_total = T_in + T_out，条件掩码覆盖前 T_in 步，
对后 T_out 步做扩散去噪，输出 [B, T_out, N, out_dim]。

修复:
  [1] sample(return_samples=True) 中 expand 改为 repeat/clone，
      避免 out_dim>1 时多个输出维度共享同一内存块，
      导致下游对不同特征维度的修改互相干扰。
  [2] CSDI.to() 正确调用父类并返回 self，确保 scheduler 随模型迁移到指定设备。
  [4] sample() 中删除语义错误的 out_dim>1 repeat 分支：
      denoiser.output_projection2 固定输出 1 通道，无法产出真实的多特征预测，
      原先 repeat 只是把同一数值复制 out_dim 次，结果在语义上是错的。
      现在统一强制 out_dim=1（__init__ 中 assert），接口保持向后兼容。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 扩散时间步嵌入 ─────────────────────────────────────────────────────────

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps: int, embedding_dim: int = 128,
                 projection_dim: int = None):
        super().__init__()
        projection_dim = projection_dim or embedding_dim
        self.register_buffer(
            'embedding',
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim,  projection_dim),
            nn.SiLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    @staticmethod
    def _build_embedding(num_steps: int, dim: int) -> torch.Tensor:
        steps = torch.arange(num_steps, dtype=torch.float32).unsqueeze(1)
        dims  = torch.arange(dim,       dtype=torch.float32).unsqueeze(0)
        table = steps * torch.exp(-math.log(10000) * dims / dim)
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding[diffusion_step])


# ── ResidualBlock ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, side_dim: int, channels: int,
                 diffusion_embedding_dim: int, nheads: int):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection      = nn.Conv2d(side_dim, 2 * channels, kernel_size=1)
        self.mid_projection       = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        self.time_layer    = nn.MultiheadAttention(channels, nheads, batch_first=True)
        self.feature_layer = nn.MultiheadAttention(channels, nheads, batch_first=True)
        self.output_projection = nn.Conv2d(channels, 2 * channels, kernel_size=1)

    def forward(self, x: torch.Tensor,
                cond_info: torch.Tensor,
                diffusion_emb: torch.Tensor) -> tuple:
        B, C, K, L = x.shape

        diff_proj = self.diffusion_projection(diffusion_emb)
        y = x + diff_proj.unsqueeze(-1).unsqueeze(-1)
        y = self.mid_projection(y)

        y_time = y[:, :C].permute(0, 2, 3, 1).reshape(B * K, L, C)
        y_time, _ = self.time_layer(y_time, y_time, y_time)
        y_time = y_time.reshape(B, K, L, C).permute(0, 3, 1, 2)

        y_feat = y[:, C:].permute(0, 3, 2, 1).reshape(B * L, K, C)
        y_feat, _ = self.feature_layer(y_feat, y_feat, y_feat)
        y_feat = y_feat.reshape(B, L, K, C).permute(0, 3, 2, 1)

        y = torch.tanh(y_time) * torch.sigmoid(y_feat)

        cond_out = self.cond_projection(cond_info)
        y = y + cond_out[:, :C]

        residual, skip = self.output_projection(y).chunk(2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


# ── DDPM 调度器 ───────────────────────────────────────────────────────────

class DDPMScheduler:
    def __init__(self, n_steps: int = 100,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.n_steps = n_steps
        beta      = torch.linspace(beta_start, beta_end, n_steps)
        alpha     = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])
        beta_tilde = beta * (1 - alpha_bar_prev) / (1 - alpha_bar).clamp(min=1e-8)

        self.beta            = beta
        self.alpha           = alpha
        self.alpha_bar       = alpha_bar
        self.alpha_bar_prev  = alpha_bar_prev
        self.beta_tilde      = beta_tilde
        self.sqrt_alpha_bar            = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar  = (1 - alpha_bar).sqrt()
        self.sqrt_recip_alpha          = (1.0 / alpha).sqrt()

    def to(self, device: torch.device) -> 'DDPMScheduler':
        for attr in ['beta', 'alpha', 'alpha_bar', 'alpha_bar_prev',
                     'beta_tilde', 'sqrt_alpha_bar',
                     'sqrt_one_minus_alpha_bar', 'sqrt_recip_alpha']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        sa  = self.sqrt_alpha_bar[t]
        soa = self.sqrt_one_minus_alpha_bar[t]
        for _ in range(x0.dim() - 1):
            sa  = sa.unsqueeze(-1)
            soa = soa.unsqueeze(-1)
        return sa * x0 + soa * noise

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor,
                        eps_pred: torch.Tensor) -> tuple:
        def _b(tensor):
            v = tensor[t]
            for _ in range(x_t.dim() - v.dim()):
                v = v.unsqueeze(-1)
            return v

        mu = _b(self.sqrt_recip_alpha) * (
            x_t - _b(self.beta) / _b(self.sqrt_one_minus_alpha_bar) * eps_pred
        )
        var = _b(self.beta_tilde)
        return mu, var


# ── 去噪网络 diff_CSDI ────────────────────────────────────────────────────

class diff_CSDI(nn.Module):
    def __init__(self,
                 num_nodes:    int,
                 T_total:      int,
                 channels:     int  = 64,
                 n_layers:     int  = 4,
                 nheads:       int  = 8,
                 diffusion_dim: int = 128,
                 side_dim:     int  = 128,
                 inputdim:     int  = 2):
        super().__init__()
        self.channels  = channels
        self.num_nodes = num_nodes
        self.T_total   = T_total

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=1000, embedding_dim=diffusion_dim)

        self.input_projection   = nn.Conv2d(inputdim,  channels, kernel_size=1)
        self.output_projection1 = nn.Conv2d(channels, channels,  kernel_size=1)
        # 修复 [4]：output_projection2 固定输出 1 通道，
        # 与 sample() 中只处理第一个特征维一致，不支持 out_dim>1 的多通道输出。
        self.output_projection2 = nn.Conv2d(channels, 1,         kernel_size=1)
        nn.init.zeros_(self.output_projection2.weight)
        nn.init.zeros_(self.output_projection2.bias)   # 修复 [5]：bias 也零初始化，

        self.node_emb = nn.Embedding(num_nodes, side_dim // 2)
        self.time_emb = nn.Embedding(T_total,   side_dim // 2)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                side_dim               = side_dim,
                channels               = channels,
                diffusion_embedding_dim = diffusion_dim,
                nheads                 = nheads,
            )
            for _ in range(n_layers)
        ])

        self.register_buffer('node_idx', torch.arange(num_nodes))
        self.register_buffer('time_idx', torch.arange(T_total))

    def _build_side_info(self, B: int) -> torch.Tensor:
        n_emb = self.node_emb(self.node_idx)
        n_emb = n_emb.T.unsqueeze(0).unsqueeze(-1)
        n_emb = n_emb.expand(B, -1, -1, self.T_total)

        t_emb = self.time_emb(self.time_idx)
        t_emb = t_emb.T.unsqueeze(0).unsqueeze(2)
        t_emb = t_emb.expand(B, -1, self.num_nodes, -1)

        return torch.cat([n_emb, t_emb], dim=1)

    def forward(self, x_noisy: torch.Tensor,
                cond_mask: torch.Tensor,
                diffusion_step: torch.Tensor) -> torch.Tensor:
        B, K, L = x_noisy.shape

        x_inp = torch.stack([x_noisy, cond_mask], dim=1)
        h = F.relu(self.input_projection(x_inp))

        diff_emb = self.diffusion_embedding(diffusion_step)
        side     = self._build_side_info(B)

        skip_sum = torch.zeros_like(h)
        for layer in self.residual_layers:
            h, skip = layer(h, side, diff_emb)
            skip_sum = skip_sum + skip

        out = skip_sum / math.sqrt(len(self.residual_layers))
        out = F.relu(self.output_projection1(out))
        out = self.output_projection2(out)
        return out.squeeze(1)


# ── CSDI 完整模型（多步版） ─────────────────────────────────────────────────

class CSDI(nn.Module):
    """
    CSDI 多步预测版本: T_total = T_in + T_out。
    训练: compute_loss(x, y) — DDPM 在目标区域加噪去噪
    推理: forward(x) — n_samples 次 DDPM 逆向采样

    注意: 仅支持 out_dim=1（去噪网络输出固定为单通道）。
    """
    def __init__(self,
                 num_nodes:       int,
                 in_dim:          int,
                 T_in:            int = 168,
                 T_out:           int = 1,
                 channels:        int = 64,
                 n_layers:        int = 4,
                 nheads:          int = 8,
                 diffusion_steps: int = 100,
                 n_samples:       int = 10,
                 out_dim:         int = 1):
        super().__init__()
        # 修复 [4]：denoiser 固定输出 1 通道，out_dim>1 时语义上只是复制，
        # 使用 assert 明确限制，避免误导性调用。
        assert out_dim == 1, (
            "CSDI 的去噪网络 output_projection2 固定输出 1 通道，"
            "不支持 out_dim>1。如需多特征输出，请修改 output_projection2 输出通道数。"
        )
        self.T_in      = T_in
        self.T_out     = T_out
        self.T_total   = T_in + T_out
        self.n_samples = n_samples
        self.num_nodes = num_nodes
        self.n_steps   = diffusion_steps
        self.out_dim   = out_dim

        self.scheduler = DDPMScheduler(diffusion_steps)
        self.denoiser  = diff_CSDI(
            num_nodes    = num_nodes,
            T_total      = self.T_total,
            channels     = channels,
            n_layers     = n_layers,
            nheads       = nheads,
            diffusion_dim = 128,
            side_dim     = 128,
            inputdim     = 2,
        )

    def to(self, device):
        # 修复 [2]：确保 DDPMScheduler 的张量也迁移到目标设备
        super().to(device)
        self.scheduler.to(device)
        return self

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T_in, N, F]
        y : [B, T_out, N, F]
        """
        B      = x.shape[0]
        device = x.device

        # 目标区域: [B, N, T_out]
        y_target = y[..., 0].permute(0, 2, 1)              # [B, N, T_out]

        t_step  = torch.randint(0, self.n_steps, (B,), device=device)
        noise   = torch.randn_like(y_target)               # [B, N, T_out]

        # 前向加噪（只对目标区域）
        y_t = self.scheduler.q_sample(y_target, t_step, noise)

        # 构造完整序列: [已知 | 加噪目标] = [B, N, T_total]
        x_c   = x[..., 0].permute(0, 2, 1)                  # [B, N, T_in]
        x_full = torch.cat([x_c, y_t], dim=-1)              # [B, N, T_total]

        cond_mask = torch.zeros(B, self.num_nodes, self.T_total, device=device)
        cond_mask[:, :, :self.T_in] = 1.0                    # 已知区域

        eps_pred_full = self.denoiser(x_full, cond_mask, t_step)  # [B, N, T_total]
        eps_pred      = eps_pred_full[:, :, self.T_in:]           # [B, N, T_out]
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, x: torch.Tensor, return_samples: bool = False) -> torch.Tensor:
        """DDPM 逆向采样，返回 [B, T_out, N, 1] 或 [S, B, N, T_out, 1]"""
        B, T, N, F = x.shape
        device = x.device
        x_c = x[..., 0].permute(0, 2, 1)              # [B, N, T_in]

        preds = []
        for _ in range(self.n_samples):
            y_t = torch.randn(B, N, self.T_out, device=device)  # [B, N, T_out]

            for step in reversed(range(self.n_steps)):
                t_batch = torch.full((B,), step, device=device, dtype=torch.long)

                x_full    = torch.cat([x_c, y_t], dim=-1)       # [B, N, T_total]
                cond_mask = torch.zeros(B, N, self.T_total, device=device)
                cond_mask[:, :, :self.T_in] = 1.0

                eps_full = self.denoiser(x_full, cond_mask, t_batch)
                eps      = eps_full[:, :, self.T_in:]            # [B, N, T_out]

                mu, var = self.scheduler.p_mean_variance(y_t, t_batch, eps)

                if step > 0:
                    y_t = mu + var.sqrt() * torch.randn_like(mu)
                else:
                    y_t = mu

            preds.append(y_t)                                    # [B, N, T_out]

        sample_stack = torch.stack(preds, dim=0)                 # [S, B, N, T_out]

        if return_samples:
            # 修复 [1]：使用 .unsqueeze(-1) 后 .contiguous()，保证各 sample
            # 有独立内存，避免下游修改时意外共享底层存储。
            # 修复 [4]：out_dim 固定为 1，不再有 repeat 填充分支。
            # shape: [S, B, N, T_out, 1]
            return sample_stack.unsqueeze(-1).contiguous()

        # [S, B, N, T_out] → mean → [B, N, T_out] → [B, T_out, N, 1]
        out = sample_stack.mean(dim=0)                           # [B, N, T_out]
        out = out.permute(0, 2, 1).unsqueeze(-1)                 # [B, T_out, N, 1]
        return out

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.sample(x)