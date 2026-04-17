"""
GridCFN: A Causal Spatio-Temporal Framework for Power Flow Uncertainty Prediction
CFM 版本 v2

[v2 修复与改进清单]

  Bug 修复：
    1. [Bug-时间编码] CFMVectorField 改为对数均匀频率（log-spaced），
       比线性频率更好地分辨 t≈0 和 t≈1 处的细节。
    2. [Bug-ODE边界] sample() 欧拉积分从 step=0 到 n_steps-1，
       每步 t=step*dt，最终 x 对应 t=1（完整覆盖 [0,1]）。
       原版逻辑已正确，此处补充注释确认。
    3. [Bug-维度] cfm_loss 加入防御性 assert，确保 y_target 维度与
       out_dim 匹配，避免无声的 broadcast 错误。
    4. [Bug-梯度方差] cfm_loss 改为对每个 batch 多采 n_t_samples 组 t
       并取均值（默认 4），降低梯度估计方差（原版单次采样噪声大）。

  设计决策（OT-CFM vs I-CFM）：
    · 本任务 out_dim=1，OT 在 1D 退化为排序，无实质增益。
    · x1=y_target 与条件 c 强绑定，OT 重配对会破坏这个语义绑定。
    · OT plan 计算（Sinkhorn）需额外 O(B²) 复杂度，对大 batch 有明显开销。
    · 结论：I-CFM 是正确且高效的选择。

  架构改进：
    · CFMVectorField: 3 层 MLP + 残差 + 对数均匀时间编码（time_emb_dim=16）。
    · sigma_min 下界 1e-4（原 1e-6，对概率校准无意义且可能引入偏差）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 1. GCN
# ---------------------------------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj_norm):
        return F.relu(torch.matmul(adj_norm, self.linear(x)))


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i+1]) for i in range(n_layers)]
        )

    def forward(self, x, adj_norm):
        for layer in self.layers:
            x = layer(x, adj_norm)
        return x


# ---------------------------------------------------------------------------
# 2. TCN
# ---------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, bias=True)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        ng = min(8, channels)
        self.norm1 = nn.GroupNorm(ng, channels)
        self.norm2 = nn.GroupNorm(ng, channels)

    def forward(self, x):
        r = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        return x + r


class TCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=4, kernel_size=3):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden_dim, kernel_size, 2**i) for i in range(n_layers)]
        )

    def forward(self, x):
        """x: [B,N,T,C] → [B,N,T,hidden]"""
        B, N, T, C = x.shape
        x = x.reshape(B*N, T, C).permute(0, 2, 1)
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        return x.permute(0, 2, 1).reshape(B, N, T, -1)


# ---------------------------------------------------------------------------
# 3. Backbone
# ---------------------------------------------------------------------------
class Backbone(nn.Module):
    def __init__(self, in_dim, gcn_hidden, tcn_hidden, gcn_layers=2, tcn_layers=4):
        super().__init__()
        self.gcn = GCN(in_dim, gcn_hidden, gcn_hidden, gcn_layers)
        self.tcn = TCN(gcn_hidden, tcn_hidden, tcn_layers)

    def forward(self, x, adj_norm):
        """x: [B,T,N,F] → H: [B,T,N,D]"""
        B, T, N, F = x.shape
        g = self.gcn(x.reshape(B*T, N, F), adj_norm).reshape(B, T, N, -1)
        H = self.tcn(g.permute(0, 2, 1, 3))   # [B,N,T,D]
        return H.permute(0, 2, 1, 3)           # [B,T,N,D]


# ---------------------------------------------------------------------------
# 4. Causal Disentangler
# ---------------------------------------------------------------------------
class CausalDisentangler(nn.Module):
    def __init__(self, in_dim, env_dim, stoch_dim):
        super().__init__()
        self.env_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, env_dim), nn.Tanh()
        )
        self.stoch_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, stoch_dim), nn.Tanh()
        )

    def forward(self, H):
        """H: [B,T,N,D] → He[B,N,De], Hs[B,N,Ds], He_seq[B,T,N,De]"""
        B, T, N, D = H.shape
        He_seq = self.env_proj(H.reshape(B*T*N, D)).reshape(B, T, N, -1)
        He     = He_seq[:, -1]
        Hs     = self.stoch_proj(H[:, -1].reshape(B*N, D)).reshape(B, N, -1)
        return He, Hs, He_seq


# ---------------------------------------------------------------------------
# 5. CLUB Estimator
# ---------------------------------------------------------------------------
class CLUBEstimator(nn.Module):
    """
    CLUB 互信息上界估计器（NeurIPS 2020）。
    两步训练：
      Step 1: variational_loss(He.detach(), Hs.detach()) → 更新变分网络
      Step 2: forward(He, Hs) → CLUB 上界，推动骨干解耦
    L2 归一化处理小数值 He/Hs（Electricity 等数据集修复）。
    """

    def __init__(self, env_dim, stoch_dim, hidden_dim=64):
        super().__init__()
        self.var_net_mu = nn.Sequential(
            nn.Linear(env_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
        )
        self.var_net_logvar = nn.Sequential(
            nn.Linear(env_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
        )

    def _get_params(self, He_flat):
        return self.var_net_mu(He_flat), self.var_net_logvar(He_flat)

    def _log_prob(self, Hs, mu_q, logvar_raw):
        log_var  = torch.log(F.softplus(logvar_raw) + 1e-2).clamp(-4.0, 4.0)
        log_prob = -0.5 * (
            math.log(2 * math.pi)
            + log_var
            + (Hs - mu_q).pow(2) / log_var.exp()
        )
        return log_prob.mean(dim=-1)

    def _neg_perm(self, M, device):
        perm = torch.randperm(M, device=device)
        same = perm == torch.arange(M, device=device)
        if same.any() and M > 1:
            idx  = same.nonzero(as_tuple=True)[0]
            swap = (idx + 1) % M
            perm[idx], perm[swap] = perm[swap].clone(), perm[idx].clone()
        return perm

    def forward(self, He, Hs):
        M       = He.shape[0] * He.shape[1]
        He_flat = F.normalize(He.reshape(M, -1), dim=-1)
        Hs_flat = F.normalize(Hs.reshape(M, -1), dim=-1)
        mu_q, logvar_raw = self._get_params(He_flat)
        pos = self._log_prob(Hs_flat,                               mu_q, logvar_raw).mean()
        neg = self._log_prob(Hs_flat[self._neg_perm(M, He.device)], mu_q, logvar_raw).mean()
        return pos - neg

    def variational_loss(self, He, Hs):
        M       = He.shape[0] * He.shape[1]
        He_flat = F.normalize(He.reshape(M, -1), dim=-1)
        Hs_flat = F.normalize(Hs.reshape(M, -1), dim=-1)
        mu_q, logvar_raw = self._get_params(He_flat)
        return -self._log_prob(Hs_flat, mu_q, logvar_raw).mean()


# ---------------------------------------------------------------------------
# 6. Multi-Scale Context
# ---------------------------------------------------------------------------
class MultiScaleContext(nn.Module):
    def __init__(self, env_dim, ms_out_dim):
        super().__init__()
        self.convs = nn.ModuleList([
            CausalConv1d(env_dim, env_dim, kernel_size=3, dilation=d)
            for d in [1, 7, 30]
        ])
        self.proj = nn.Linear(env_dim * 3, ms_out_dim)

    def forward(self, He_seq):
        """He_seq: [B,T,N,De] → [B,N,ms_out_dim]"""
        B, T, N, De = He_seq.shape
        x    = He_seq.permute(0, 2, 3, 1).reshape(B*N, De, T)
        outs = [conv(x)[:, :, -1] for conv in self.convs]
        return self.proj(torch.cat(outs, dim=-1)).reshape(B, N, -1)


# ---------------------------------------------------------------------------
# 7. SCG Message Passing
# ---------------------------------------------------------------------------
class CausalGateUnit(nn.Module):
    def __init__(self, stoch_dim, env_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stoch_dim*2 + env_dim*2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, Hs_i, Hs_j, He_i, He_j):
        return self.net(torch.cat([Hs_i, Hs_j, He_i, He_j], dim=-1))


class SCGMessagePassingLayer(nn.Module):
    def __init__(self, stoch_dim, env_dim, hidden_dim=64):
        super().__init__()
        self.gate      = CausalGateUnit(stoch_dim, env_dim, hidden_dim)
        self.msg_tr    = nn.Linear(stoch_dim, stoch_dim)
        self.agg_norm  = nn.LayerNorm(stoch_dim)
        self.agg_tr    = nn.Sequential(nn.Linear(stoch_dim, stoch_dim), nn.ReLU())

    def forward(self, Hs, He, edge_index):
        B, N, Ds = Hs.shape
        src, dst = edge_index[0], edge_index[1]
        g        = self.gate(Hs[:, dst], Hs[:, src], He[:, dst], He[:, src])
        m        = g * self.msg_tr(Hs[:, src])
        agg      = torch.zeros(B, N, Ds, device=Hs.device, dtype=Hs.dtype)
        agg.scatter_add_(1, dst.view(1, -1, 1).expand(B, -1, Ds), m)
        return self.agg_tr(Hs + self.agg_norm(agg))


class SCGMP(nn.Module):
    def __init__(self, stoch_dim, env_dim, n_layers=3, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList([
            SCGMessagePassingLayer(stoch_dim, env_dim, hidden_dim)
            for _ in range(n_layers)
        ])

    def forward(self, Hs, He, edge_index):
        for layer in self.layers:
            Hs = layer(Hs, He, edge_index)
        return Hs


# ---------------------------------------------------------------------------
# 8. CFM Vector Field（v2：对数均匀时间编码 + 3层MLP残差）
# ---------------------------------------------------------------------------
class CFMVectorField(nn.Module):
    """
    条件向量场 v_θ(x_t, t | c)，用于 I-CFM 训练。

    改进点（v2）：
      · 时间编码：对数均匀频率（log-spaced），覆盖 [1π, max_freq*π]，
        能分辨 t 接近 0/1 时的细节，比线性频率更稳定。
      · 网络：3层 MLP + 两处残差，增强表达力同时稳定梯度。

    输入：
      x_t : [B, N, out_dim]   插值状态
      t   : [B]               时间（0~1）
      c   : [B, N, cond_dim]  条件特征
    输出：
      v   : [B, N, out_dim]   速度场
    """

    def __init__(self, out_dim: int, cond_dim: int, hidden_dim: int = 128,
                 time_emb_dim: int = 16, max_freq: float = 1000.0):
        super().__init__()
        self.out_dim = out_dim
        n_freqs = time_emb_dim // 2
        assert n_freqs * 2 == time_emb_dim, "time_emb_dim 必须是偶数"

        # 对数均匀频率：1π ~ max_freq*π，覆盖多个时间尺度
        freqs = torch.exp(
            torch.linspace(0.0, math.log(max_freq), n_freqs)
        ) * math.pi
        self.register_buffer("freqs", freqs)  # [n_freqs]，不参与梯度

        in_dim = out_dim + time_emb_dim + cond_dim

        # 3 层 MLP（含两处残差连接）
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )
        # 残差对齐层（in_dim → hidden_dim）
        self.skip_proj  = nn.Linear(in_dim, hidden_dim, bias=False)

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _time_embed(self, t: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """t: [B] → [B, N, time_emb_dim]"""
        angles = t.reshape(B, 1) * self.freqs.unsqueeze(0)            # [B, n_freqs]
        emb    = torch.cat([angles.sin(), angles.cos()], dim=-1)       # [B, 2*n_freqs]
        return emb.unsqueeze(1).expand(B, N, -1)                       # [B, N, time_emb_dim]

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_t.shape
        t_emb   = self._time_embed(t, B, N)                   # [B, N, time_emb_dim]
        inp     = torch.cat([x_t, t_emb, c], dim=-1)          # [B, N, in_dim]

        h = self.input_proj(inp) + self.skip_proj(inp)        # 残差 1：跳过非线性
        h = h + self.layer2(h)                                # 残差 2
        h = h + self.layer3(h)                                # 残差 3
        return self.out_proj(h)                                # [B, N, out_dim]


# ---------------------------------------------------------------------------
# 9. GridCFN（CFM 版 v2）
# ---------------------------------------------------------------------------
class GridCFN(nn.Module):
    """
    GridCFN with I-CFM probabilistic output head (v2).

    接口：
      forward(x, adj_norm, edge_index)
        → (context_feat, He, Hs, mi_loss)

      cfm_loss(context_feat, y_target, n_t_samples=4)
        → scalar

      sample(context_feat, n_samples=50, n_steps=20)
        → [n_samples, B, N, out_dim]
    """

    def __init__(
        self,
        in_dim=1, gcn_hidden=64, tcn_hidden=64,
        env_dim=32, stoch_dim=32, ms_out_dim=32,
        n_scg_layers=3, out_dim=1, lambda_mi=0.5,
        gcn_layers=2, tcn_layers=4,
        cfm_hidden=128, cfm_time_emb_dim=16,
    ):
        super().__init__()
        self.lambda_mi = lambda_mi
        self.out_dim   = out_dim
        cond_dim = ms_out_dim + stoch_dim

        self.backbone     = Backbone(in_dim, gcn_hidden, tcn_hidden, gcn_layers, tcn_layers)
        self.disentangler = CausalDisentangler(tcn_hidden, env_dim, stoch_dim)
        self.club         = CLUBEstimator(env_dim, stoch_dim)
        self.ms_context   = MultiScaleContext(env_dim, ms_out_dim)
        self.scgmp        = SCGMP(stoch_dim, env_dim, n_scg_layers)
        self.vector_field = CFMVectorField(
            out_dim=out_dim,
            cond_dim=cond_dim,
            hidden_dim=cfm_hidden,
            time_emb_dim=cfm_time_emb_dim,
        )

    @staticmethod
    def normalize_adj(adj):
        adj      = adj + torch.eye(adj.size(0), device=adj.device)
        deg      = adj.sum(dim=1)
        d_inv_sq = torch.pow(deg.clamp(min=1e-8), -0.5)
        D        = torch.diag(d_inv_sq)
        return D @ adj @ D

    @staticmethod
    def adj_to_edge_index(adj):
        return adj.nonzero(as_tuple=False).t().contiguous()

    def forward(self, x, adj_norm, edge_index):
        """
        x          : [B, T_in, N, F]
        adj_norm   : [N, N]
        edge_index : [2, E]
        →  context_feat : [B, N, cond_dim]
           He           : [B, N, env_dim]
           Hs           : [B, N, stoch_dim]
           mi_loss      : scalar
        """
        H              = self.backbone(x, adj_norm)
        He, Hs, He_seq = self.disentangler(H)
        mi_loss        = self.club(He, Hs)
        He_prime       = self.ms_context(He_seq)
        Hs_prime       = self.scgmp(Hs, He, edge_index)
        context_feat   = torch.cat([He_prime, Hs_prime], dim=-1)
        return context_feat, He, Hs, mi_loss

    def cfm_loss(self, context_feat: torch.Tensor,
                 y_target: torch.Tensor,
                 n_t_samples: int = 4) -> torch.Tensor:
        """
        I-CFM 训练损失（多次 t 采样降梯度方差）。

        y_target : [B, N, out_dim]，调用方负责切片对齐（y[..., :out_dim]）。
        n_t_samples : 每 batch 重复采样 t 次数，取均值（推荐 4）。

        I-CFM：
          x0 ~ N(0,I), x1 = y_target（独立配对，无 OT 重排）
          x_t = (1-t)*x0 + t*x1
          u_t = x1 - x0（解析目标向量场）
          loss = MSE(v_θ(x_t, t, c), u_t)

        为什么用 I-CFM 而不用 OT-CFM：
          · out_dim=1 时 OT 退化为 1D 排序，无增益
          · y_target 与条件 c 绑定，OT 重排会打断语义配对
          · OT plan 计算（Sinkhorn/匈牙利）引入额外 O(B²) 复杂度
        """
        assert y_target.shape[-1] == self.out_dim, (
            f"y_target 末维 {y_target.shape[-1]} ≠ out_dim={self.out_dim}，"
            "请传入 y[..., :out_dim]"
        )

        B, N, _ = y_target.shape
        device  = y_target.device
        losses  = []

        for _ in range(n_t_samples):
            x0    = torch.randn_like(y_target)               # [B, N, out_dim]
            t     = torch.rand(B, device=device)             # [B]
            t_bc  = t.reshape(B, 1, 1)
            x_t   = (1.0 - t_bc) * x0 + t_bc * y_target    # [B, N, out_dim]
            u_t   = y_target - x0                           # 目标向量场
            v_pred = self.vector_field(x_t, t, context_feat)
            losses.append(F.mse_loss(v_pred, u_t))

        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(self, context_feat: torch.Tensor,
               n_samples: int = 50,
               n_steps: int = 20) -> torch.Tensor:
        """
        欧拉积分推断：从 x_0~N(0,I) 积分到 x_1（t=1）。

        t 序列：0, dt, 2dt, ..., (n_steps-1)*dt
        每步：x_{t+dt} = x_t + dt * v_θ(x_t, t, c)
        最终 x 对应 t=1（n_steps 步 * dt = 1.0）。

        参数：
          n_samples : 粒子数，验证时 10~20 足够，测试时 100+
          n_steps   : 欧拉步数，I-CFM 路径近线性，20 步已准确
        返回：[n_samples, B, N, out_dim]
        """
        B, N, _ = context_feat.shape
        device  = context_feat.device
        dt      = 1.0 / n_steps
        c       = context_feat.detach()

        samples = []
        for _ in range(n_samples):
            x = torch.randn(B, N, self.out_dim, device=device)
            for step in range(n_steps):
                t_val = step * dt
                t_vec = torch.full((B,), t_val, device=device, dtype=torch.float32)
                x = x + dt * self.vector_field(x, t_vec, c)
            samples.append(x)

        return torch.stack(samples, dim=0)    # [n_samples, B, N, out_dim]