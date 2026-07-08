"""
baselines/tsdiff.py
TSDiff: Predict, Refine, Repeat — Rasul et al., NeurIPS 2023
论文: https://arxiv.org/abs/2307.11494
参考: https://github.com/amazon-science/unconditional-time-series-diffusion

核心思路：
  无条件扩散模型在时序上做生成，推理时以观测历史作为"引导"
  （Observation Guidance / Replacement Method）做条件预测。
  扩散骨干网络为 WaveNet 风格的因果膨胀 1-D Conv + DDPM 调度器。

实现说明（与原仓库的对应关系）：
  - DDPMScheduler      ← tsdiff/scheduler.py  (DDPM linear schedule)
  - DiffusionEmbedding ← 正弦时间步嵌入
  - CausalResidualBlock ← tsdiff/model.py WaveNetResidualBlock
                          （用因果膨胀卷积替代 S4，轻量易跑）
  - TSDiffBackbone     ← 多层因果膨胀卷积去噪网络
  - TSDiff             ← 完整模型，训练无条件 DDPM，推理 Replacement Method

引导方式（Replacement Method，原论文 §3.3）：
  每个 DDPM 逆向步骤后，将历史区域 x[:, :, :T_in] 替换为
  q_sample(x_cond, t-1) 的加噪版本，目标区域 x[:, :, T_in:] 自由演化，
  最终收敛即为条件预测。

多步预测版:
  T_total = T_in + T_out，预测结果取后 T_out 步，
  输出 [B, T_out, N, out_dim]。

修复:
  [A] CausalResidualBlock.diff_proj 输出 2*channels（对齐 causal_conv 的 2C 输出）。
  [B] TSDiffBackbone 每节点独立处理：[B,N,T]→[B*N,1,T]，保留 per-node 条件能力。
  [C] DiffusionEmbedding embedding_dim 强制为偶数，防止 sin/cos 拼接后维度不符。
  [D] CausalResidualBlock 因果卷积右侧裁剪保持时间维长度不变。
  [4] TSDiffBackbone.forward 中 n_emb.expand 后加 .contiguous()，
      避免非连续内存张量传入 Conv1d 时引发隐式 copy 或警告。
  [5] sample() 和 compute_loss() 中仅使用第一个特征维（in_dim>1 时静默丢弃），
      现在在构造函数中加警告日志，防止用户误以为多特征全部被利用。
  [7] run_tsdiff 中 channels 加上限 min(..., 128)：cfm_hidden 为 GridCFN CFM 设计
      （256~384），直接用于 TSDiff WaveNet 骨干会在大图（weather 1866节点）OOM。
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.beta                     = beta
        self.alpha                    = alpha
        self.alpha_bar                = alpha_bar
        self.alpha_bar_prev           = alpha_bar_prev
        self.beta_tilde               = beta_tilde
        self.sqrt_alpha_bar           = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt()
        self.sqrt_recip_alpha         = (1.0 / alpha).sqrt()

    def to(self, device: torch.device) -> "DDPMScheduler":
        for attr in ["beta", "alpha", "alpha_bar", "alpha_bar_prev",
                     "beta_tilde", "sqrt_alpha_bar",
                     "sqrt_one_minus_alpha_bar", "sqrt_recip_alpha"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """前向加噪：x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        sa  = self.sqrt_alpha_bar[t]
        soa = self.sqrt_one_minus_alpha_bar[t]
        for _ in range(x0.dim() - sa.dim()):
            sa  = sa.unsqueeze(-1)
            soa = soa.unsqueeze(-1)
        return sa * x0 + soa * noise

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor,
                        eps_pred: torch.Tensor) -> tuple:
        """DDPM 逆向均值和方差"""
        def _b(tensor):
            v = tensor[t]
            for _ in range(x_t.dim() - v.dim()):
                v = v.unsqueeze(-1)
            return v
        mu  = _b(self.sqrt_recip_alpha) * (
            x_t - _b(self.beta) / _b(self.sqrt_one_minus_alpha_bar) * eps_pred
        )
        var = _b(self.beta_tilde)
        return mu, var


# ── 扩散时间步嵌入 ─────────────────────────────────────────────────────────

class DiffusionEmbedding(nn.Module):
    """正弦时间步嵌入。

    Bug 修复 [C]：embedding_dim 必须为偶数，否则 sin/cos 拼接后长度
    2*(dim//2) < dim，proj Linear 维度不符，forward CRASH。
    构造函数内强制对齐为偶数，并断言保护。
    """
    def __init__(self, num_steps: int, embedding_dim: int = 128,
                 projection_dim: int = None):
        # Bug 修复 [C]：强制 embedding_dim 为偶数
        embedding_dim = max(2, (embedding_dim // 2) * 2)
        super().__init__()
        projection_dim = projection_dim or embedding_dim
        half = embedding_dim // 2
        steps = torch.arange(num_steps, dtype=torch.float32).unsqueeze(1)
        dims  = torch.arange(half,      dtype=torch.float32).unsqueeze(0)
        table = steps * torch.exp(-math.log(10000) * dims / max(half - 1, 1))
        emb   = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        assert emb.shape[1] == embedding_dim, \
            f"DiffusionEmbedding: emb dim {emb.shape[1]} != embedding_dim {embedding_dim}"
        self.register_buffer("emb", emb, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim), nn.SiLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.proj(self.emb[t])


# ── 因果膨胀卷积残差块 ─────────────────────────────────────────────────────

class CausalResidualBlock(nn.Module):
    """
    WaveNet 风格因果膨胀卷积残差块。
    输入/输出均为 [B, channels, T]（每节点独立流经此块）。

    Bug 修复 [A]：
      diff_proj 改为输出 2*channels（与 causal_conv 的 2C 输出维对齐），
      注入后再做 gated 激活，保证通道数一致，不再 CRASH。
    """
    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1, diffusion_dim: int = 128,
                 node_emb_dim: int = 0):
        super().__init__()
        # 因果卷积：左侧 padding 保持时间维不变（见修复 [D]）
        pad = (kernel_size - 1) * dilation
        self.causal_conv = nn.Conv1d(
            channels, 2 * channels, kernel_size=kernel_size,
            dilation=dilation, padding=pad)
        self.pad_size = pad

        # Bug 修复 [A]：输出 2*channels，与 h=[B,2C,T] 通道数对齐
        self.diff_proj  = nn.Linear(diffusion_dim, 2 * channels)

        self.node_emb_dim = node_emb_dim
        if node_emb_dim > 0:
            # side_proj 也输出 2*channels，与 h 对齐
            self.side_proj = nn.Conv1d(node_emb_dim, 2 * channels, kernel_size=1)

        self.res_proj  = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, diff_emb: torch.Tensor,
                side: torch.Tensor = None) -> tuple:
        """
        x        : [B, channels, T]
        diff_emb : [B, diffusion_dim]
        side     : [B, node_emb_dim, T] (可选)
        returns  : (residual [B, channels, T], skip [B, channels, T])
        """
        B, C, T = x.shape

        # 因果卷积，裁剪右侧多余时间步保持长度 T（修复 [D]）
        h = self.causal_conv(x)[..., :T]   # [B, 2C, T]

        # Bug 修复 [A]：diff_proj 输出 [B, 2C]，与 h=[B,2C,T] 通道对齐
        d = self.diff_proj(diff_emb)        # [B, 2C]
        h = h + d.unsqueeze(-1)             # [B, 2C, T]

        # 节点 side 信息注入
        if self.node_emb_dim > 0 and side is not None:
            h = h + self.side_proj(side)    # [B, 2C, T]

        # Gated 激活（WaveNet 风格）
        h_tanh    = torch.tanh(h[:, :C])    # [B, C, T]
        h_sigmoid = torch.sigmoid(h[:, C:]) # [B, C, T]
        h = h_tanh * h_sigmoid              # [B, C, T]

        residual = (x + self.res_proj(h)) / math.sqrt(2.0)
        skip     = self.skip_proj(h)
        return residual, skip


# ── TSDiff 骨干网络 ────────────────────────────────────────────────────────

class TSDiffBackbone(nn.Module):
    """
    多层因果膨胀卷积去噪网络。

    Bug 修复 [B]：
      原实现将 x_noisy [B, N, T] 以 N 为通道维整体处理，节点嵌入被
      mean(dim=0) 压缩为单向量，丧失 per-node 条件能力。
      修复：将 x_noisy 拆分为每节点独立流：
        [B, N, T] → reshape → [B*N, 1, T]
      input_proj 从 Conv1d(1, channels) 开始；
      节点嵌入保留 [N, node_emb_dim]，按节点展开为 [B*N, node_emb_dim, T]
      作为 side_info 注入各残差块；
      输出 [B*N, 1, T] → reshape → [B, N, T]。
    """
    def __init__(self,
                 num_nodes:     int,
                 T_total:       int,
                 channels:      int  = 64,
                 n_layers:      int  = 8,
                 kernel_size:   int  = 3,
                 diffusion_dim: int  = 128,
                 node_emb_dim:  int  = 16):
        super().__init__()
        self.T_total      = T_total
        self.num_nodes    = num_nodes
        self.channels     = channels
        self.node_emb_dim = node_emb_dim

        self.diff_emb   = DiffusionEmbedding(1000, diffusion_dim)
        # Bug 修复 [B]：每节点 1 个通道输入
        self.input_proj = nn.Conv1d(1, channels, kernel_size=1)
        self.node_emb   = nn.Embedding(num_nodes, node_emb_dim)
        self.register_buffer("node_idx", torch.arange(num_nodes))

        self.blocks = nn.ModuleList()
        log2T = int(math.log2(max(T_total, 2))) + 1
        for i in range(n_layers):
            dil = 2 ** (i % log2T)
            self.blocks.append(
                CausalResidualBlock(
                    channels      = channels,
                    kernel_size   = kernel_size,
                    dilation      = dil,
                    diffusion_dim = diffusion_dim,
                    node_emb_dim  = node_emb_dim,
                )
            )

        self.output_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1), nn.ReLU(),
            # 修复 [6]：固定输出 1 通道，与 out_dim=1 的约束一致
            nn.Conv1d(channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, x_noisy: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        x_noisy : [B, N, T_total]
        t       : [B]  扩散步索引
        returns : [B, N, T_total]
        """
        B, N, T = x_noisy.shape

        # Bug 修复 [B]：每节点独立处理
        # [B, N, T] → [B*N, 1, T]
        h = x_noisy.reshape(B * N, 1, T)
        h = self.input_proj(h)            # [B*N, channels, T]

        # 扩散时间步嵌入：[B, diffusion_dim] → repeat N 次 → [B*N, diffusion_dim]
        diff_emb = self.diff_emb(t)       # [B, diffusion_dim]
        diff_emb = diff_emb.repeat_interleave(N, dim=0)  # [B*N, diffusion_dim]

        # 节点嵌入：[N, node_emb_dim] → [B*N, node_emb_dim, T]
        n_emb = self.node_emb(self.node_idx)          # [N, node_emb_dim]
        n_emb = n_emb.unsqueeze(0).expand(B, -1, -1)  # [B, N, node_emb_dim]
        # 修复 [4]：expand 返回非连续张量（stride=0），传入 Conv1d 前必须
        # contiguous()，否则某些 PyTorch 版本会发出警告或触发隐式 copy。
        n_emb = (n_emb.reshape(B * N, self.node_emb_dim, 1)
                      .expand(-1, -1, T)
                      .contiguous())                   # [B*N, node_emb_dim, T]

        skip_sum = torch.zeros_like(h)
        for block in self.blocks:
            h, skip = block(h, diff_emb, n_emb)
            skip_sum = skip_sum + skip

        out = skip_sum / math.sqrt(len(self.blocks))
        out = self.output_proj(out)       # [B*N, 1, T]
        return out.reshape(B, N, T)       # [B, N, T]


# ── TSDiff 完整模型 ────────────────────────────────────────────────────────

class TSDiff(nn.Module):
    """
    TSDiff 多步预测版本。
      - 训练: 无条件 DDPM（对整段 T_total 序列加噪去噪）
      - 推理: Replacement Method 条件采样（历史区域锚定）

    T_total = T_in + T_out；
    训练时拼接 (x, y) 为无条件扩散目标；
    推理时历史区域每步替换回对应噪声水平的真实值，目标区域自由演化。

    注意:
      - 仅支持 out_dim=1（骨干网络输出固定为单通道）。
      - in_dim>1 时只使用第一个特征维，其余特征被忽略（构造时会打印警告）。
    """
    def __init__(self,
                 num_nodes:       int,
                 in_dim:          int   = 1,
                 T_in:            int   = 168,
                 T_out:           int   = 12,
                 channels:        int   = 64,
                 n_layers:        int   = 8,
                 kernel_size:     int   = 3,
                 diffusion_steps: int   = 100,
                 n_samples:       int   = 10,
                 out_dim:         int   = 1,
                 node_emb_dim:    int   = 16):
        super().__init__()
        # 修复 [6]：骨干网络输出固定 1 通道，out_dim>1 无意义
        assert out_dim == 1, (
            "TSDiff 的骨干网络 output_proj 固定输出 1 通道，"
            "不支持 out_dim>1。如需多特征输出，请修改 output_proj 最后一层输出通道数。"
        )
        # 修复 [5]：in_dim>1 时静默丢弃多余特征，打印明确警告
        if in_dim > 1:
            import warnings
            warnings.warn(
                f"[TSDiff] in_dim={in_dim} > 1，但模型仅使用第一个特征维 (index=0)，"
                "其余特征在 compute_loss 和 sample 中均被忽略。",
                UserWarning, stacklevel=2,
            )
        self.T_in      = T_in
        self.T_out     = T_out
        self.T_total   = T_in + T_out
        self.n_samples = n_samples
        self.num_nodes = num_nodes
        self.out_dim   = out_dim
        self.n_steps   = diffusion_steps

        self.scheduler = DDPMScheduler(diffusion_steps)
        self.backbone  = TSDiffBackbone(
            num_nodes     = num_nodes,
            T_total       = self.T_total,
            channels      = channels,
            n_layers      = n_layers,
            kernel_size   = kernel_size,
            diffusion_dim = 128,
            node_emb_dim  = node_emb_dim,
        )

    def to(self, device):
        super().to(device)
        self.scheduler.to(device)
        return self

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        无条件 DDPM 训练损失（对完整 T_total 序列加噪去噪）。
        x : [B, T_in,  N, F]
        y : [B, T_out, N, F]
        注: in_dim>1 时只使用 F 的第 0 维（见修复 [5]）。
        """
        B      = x.shape[0]
        device = x.device

        # 取第一个特征维度拼接成 [B, N, T_total]
        x_c = x[..., 0].permute(0, 2, 1)          # [B, N, T_in]
        y_c = y[..., 0].permute(0, 2, 1)          # [B, N, T_out]
        x0  = torch.cat([x_c, y_c], dim=-1)       # [B, N, T_total]

        t     = torch.randint(0, self.n_steps, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t   = self.scheduler.q_sample(x0, t, noise)

        eps_pred = self.backbone(x_t, t)           # [B, N, T_total]
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, x: torch.Tensor,
               return_samples: bool = False) -> torch.Tensor:
        """
        Replacement Method 条件采样（原论文 Algorithm 2）。
        每步逆向后将历史区域替换为对应噪声水平 t-1 的加噪真实值。
        返回 [B, T_out, N, 1] 或 [S, B, N, T_out, 1]。
        注: in_dim>1 时只使用 F 的第 0 维（见修复 [5]）。
        """
        B, T, N, F_dim = x.shape  # 用 F_dim 避免遮蔽模块级 F = nn.functional
        device = x.device

        # 历史条件：[B, N, T_in]
        x_cond = x[..., 0].permute(0, 2, 1)

        preds = []
        for _ in range(self.n_samples):
            x_t = torch.randn(B, N, self.T_total, device=device)

            for step in reversed(range(self.n_steps)):
                t_batch = torch.full((B,), step, device=device, dtype=torch.long)

                # DDPM 逆向一步
                eps_pred = self.backbone(x_t, t_batch)
                mu, var  = self.scheduler.p_mean_variance(x_t, t_batch, eps_pred)
                x_t = mu + var.sqrt() * torch.randn_like(mu) if step > 0 else mu

                # Replacement：历史区域替换为 q(x_cond | t-1) 的加噪值
                # t_prev = step - 1，对应原论文 Algorithm 2 第 6 行
                if step > 0:
                    t_prev = t_batch - 1                    # [B]，值域 [0, n_steps-2]
                    noise_cond      = torch.randn_like(x_cond)
                    x_cond_noisy    = self.scheduler.q_sample(x_cond, t_prev, noise_cond)
                    x_t[:, :, :self.T_in] = x_cond_noisy

            preds.append(x_t[:, :, self.T_in:])   # [B, N, T_out]

        sample_stack = torch.stack(preds, dim=0)   # [S, B, N, T_out]

        if return_samples:
            # 修复 [6]：out_dim 固定为 1，不再有 repeat 填充分支
            # shape: [S, B, N, T_out, 1]
            return sample_stack.unsqueeze(-1).contiguous()

        # 修复 [6]：删除语义错误的 out_dim>1 repeat 分支
        out = sample_stack.mean(dim=0)             # [B, N, T_out]
        out = out.permute(0, 2, 1).unsqueeze(-1)   # [B, T_out, N, 1]
        return out

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.sample(x)


# ── 内部评估（概率指标）──────────────────────────────────────────────────

def _evaluate_tsdiff(model: "TSDiff", loader, device, scaler,
                     null_val: float = None, logger=None,
                     max_batches: int = None) -> dict:
    """
    评估概率预测指标。
    model.n_samples 已由调用方在调用前设置好，此处不重复覆盖。
    max_batches: 若不为 None，只跑前 max_batches 个 batch（用于 val 快速估计）。
    """
    from baselines.utils import compute_prob_metrics

    model.eval()
    samples_list, y_list = [], []
    total = len(loader)

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            t0 = time.time()
            raw = model.sample(x, return_samples=True)   # [S, B, N, T_out, 1]
            samples_list.append(raw.cpu().numpy())
            y_list.append(y.permute(0, 2, 1, 3).numpy())  # [B, N, T_out, F]
            if logger and (i == 0 or (i + 1) % max(1, total // 5) == 0):
                logger.info(f"  [TSDiff eval] batch {i+1:4d}/{total} | "
                            f"{time.time() - t0:.1f}s/batch")

    samples_all = np.concatenate(samples_list, axis=1)  # [S, total, N, T_out, 1]
    y_all       = np.concatenate(y_list,       axis=0)  # [total, N, T_out, F]

    metrics_norm = compute_prob_metrics(samples_all, y_all)

    if scaler is not None:
        shape = samples_all.shape
        s_mean = scaler.mean[..., :1] if scaler.mean.shape[-1] > 1 else scaler.mean
        s_std  = scaler.std[..., :1]  if scaler.std.shape[-1] > 1  else scaler.std
        samples_all = (samples_all.reshape(-1) * s_std + s_mean).reshape(shape)
        y_all = (y_all.reshape(-1) * s_std + s_mean).reshape(y_all.shape)

    metrics = compute_prob_metrics(samples_all, y_all)
    for k, v in metrics_norm.items():
        metrics[f"{k}_norm"] = v
    return metrics


# ── 训练入口（baselines 统一签名）─────────────────────────────────────────

def run_tsdiff(loaders, adj, cfg, device, save_dir, logger,
               in_dim=None, num_nodes=None, scaler=None, null_val=None):
    """
    TSDiff 训练 + 测试（Replacement Method 条件采样）。
    loaders = (train_loader, val_loader, test_loader)
    """
    train_loader, val_loader, test_loader = loaders
    d, m, t_cfg = cfg.data, cfg.model, cfg.train

    # cfm_hidden 是为 GridCFN 的 CFM 设计的（256~384），直接用于 TSDiff WaveNet 骨干
    # 会在大图（weather 1866节点）上 OOM。上限 128 可覆盖所有数据集。
    channels        = min(getattr(m, "cfm_hidden",     64), 128)
    n_layers        = getattr(m, "tcn_layers",       8)
    kernel_size     = getattr(m, "kernel_size",      3)
    diffusion_steps = getattr(m, "diffusion_steps", 100)
    n_samples_val   = getattr(t_cfg, "cfm_n_samples",      10)
    n_samples_test  = getattr(t_cfg, "tsdiff_n_samples_test",
                     getattr(t_cfg, "cfm_n_samples_test", 50))
    node_emb_dim    = getattr(m, "env_dim",         16)
    # 验证时最多用这么多 batch（防止大图上卡住）。默认 50 通常足够反映趋势。
    # 可在 cfg.train 中设置 tsdiff_val_max_batches=N 覆盖。
    val_max_batches = getattr(t_cfg, "tsdiff_val_max_batches", 50)
    test_max_batches = getattr(t_cfg, "tsdiff_test_max_batches", 50)

    model = TSDiff(
        num_nodes       = num_nodes,
        in_dim          = in_dim if in_dim is not None else 1,
        T_in            = d.T_in,
        T_out           = d.T_out,
        channels        = channels,
        n_layers        = n_layers,
        kernel_size     = kernel_size,
        diffusion_steps = diffusion_steps,
        n_samples       = n_samples_val,
        out_dim         = 1,
        node_emb_dim    = node_emb_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[TSDiff] Parameters: {n_params:,}")
    logger.info(f"[TSDiff] T_total={model.T_total}, diffusion_steps={diffusion_steps}, "
                f"n_layers={n_layers}, channels={channels}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=t_cfg.lr_decay_factor,
        patience=t_cfg.lr_decay_patience)

    save_path         = os.path.join(save_dir, "tsdiff_best.pt")
    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_crps": [], "val_mae": [], "val_rmse": []}

    logger.info("[TSDiff] 开始训练 (无条件 DDPM + Replacement Method 条件采样)")
    logger.info(f"{'Epoch':>6} | {'Loss':>8} | {'Val MAE':>8} | "
                f"{'Val RMSE':>9} | {'Val CRPS':>9} | {'LR':>8} | {'Time':>6}")

    for epoch in range(1, t_cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 10)

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            loss = model.compute_loss(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

            if (i + 1) % log_every == 0 or i == 0:
                logger.info(f"  [TSDiff] Epoch {epoch:3d} | batch {i+1:4d}/{total_batches} | "
                            f"loss={loss.item():.4f} | {time.time() - t0:.1f}s")

        avg_loss = epoch_loss / max(n_batches, 1)

        # 验证：n_samples=1 大幅加速（100步×1样本 vs 100步×10样本）。
        # 对大图（如 weather 1866节点）仍很慢：每 batch 跑 n_steps 次 backbone forward，
        # 输入维度 [B*N, 1, T]，val 全集可能数千 batch。
        # 修复：最多只跑 val_max_batches 个 batch 做快速监控，足以反映训练趋势。
        model.n_samples = 1
        val_m = _evaluate_tsdiff(model, val_loader, device, scaler,
                                 null_val=null_val, logger=logger,
                                 max_batches=val_max_batches)

        scheduler.step(val_m["CRPS"])
        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(avg_loss)
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])
        history["val_rmse"].append(val_m["RMSE"])

        logger.info(
            f"{epoch:>6} | {avg_loss:>8.4f} | {val_m['MAE']:>8.4f} | "
            f"{val_m['RMSE']:>9.4f} | {val_m['CRPS']:>9.4f} | "
            f"{cur_lr:>8.2e} | {elapsed:>5.1f}s")

        if val_m["CRPS"] < best_val_crps:
            best_val_crps     = val_m["CRPS"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= t_cfg.patience:
                logger.info(f"[TSDiff] 早停于 epoch {epoch}")
                break

    # ── 测试 ──────────────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True))
    model.n_samples = n_samples_test
    test_m = _evaluate_tsdiff(model, test_loader, device, scaler,
                              null_val=null_val, logger=logger,
                              max_batches=test_max_batches
                              if test_max_batches > 0 else None)

    sep = "=" * 55
    logger.info(f"\n{sep}")
    logger.info(f"[TSDiff] TEST SET RESULTS (归一化域, T_out={d.T_out})")
    logger.info(sep)
    for k in ["MAE", "RMSE", "CRPS", "PICP", "PINAW"]:
        logger.info(f"  {k:<8}: {test_m[f'{k}_norm']:.4f}")
    logger.info(sep)
    logger.info(f"[TSDiff] TEST SET RESULTS (反归一化域, T_out={d.T_out})")
    logger.info(sep)
    for k in ["MAE", "RMSE", "CRPS", "PICP", "PINAW"]:
        logger.info(f"  {k:<8}: {test_m[k]:.4f}")
    logger.info(f"\n  {'Step':<6}  {'MAE':>8}  {'RMSE':>8}  {'CRPS':>8}")
    for h in range(d.T_out):
        mae_h  = test_m.get(f"MAE_h{h+1}",  float("nan"))
        rmse_h = test_m.get(f"RMSE_h{h+1}", float("nan"))
        crps_h = test_m.get(f"CRPS_h{h+1}", float("nan"))
        logger.info(f"  h={h+1:<4}  {mae_h:>8.4f}  {rmse_h:>8.4f}  {crps_h:>8.4f}")
    logger.info(sep)

    history["test_metrics"] = test_m
    history["test_mae"]  = test_m["MAE"]
    history["test_rmse"] = test_m["RMSE"]
    history["test_mape"] = test_m["MAPE"]
    return history
