"""
GridCFN: Causal Spatio-Temporal Framework for Power Flow Uncertainty Prediction

架构概览：
  Backbone (GCN+TCN) → CausalDisentangler → He（环境）/ Hs（随机）
  He → MultiScaleContext → He_prime（多尺度环境背景）
  Hs → SCGMP            → Hs_prime（图传播后随机表征）
  CFMVectorField(x_t, t, He_prime, Hs_prime) → 双流 AdaLN 条件向量场
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
        H = self.tcn(g.permute(0, 2, 1, 3))
        return H.permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# 4. Causal Disentangler
# ---------------------------------------------------------------------------
class CausalDisentangler(nn.Module):
    """
    将 Backbone 输出 H 分解为：
      He_seq : [B, T, N, env_dim]   全序列环境表征，供 MultiScaleContext 使用
      He     : [B, N, env_dim]      最后时间步环境表征，供 CLUB 和 SCGMP 使用
      Hs     : [B, N, stoch_dim]    最后时间步随机表征
    """
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
        B, T, N, D = H.shape
        He_seq = self.env_proj(H.reshape(B*T*N, D)).reshape(B, T, N, -1)
        He     = He_seq[:, -1]
        Hs     = self.stoch_proj(H[:, -1].reshape(B*N, D)).reshape(B, N, -1)
        return He, Hs, He_seq


# ---------------------------------------------------------------------------
# 5. CLUB Mutual Information Estimator
# ---------------------------------------------------------------------------
class CLUBEstimator(nn.Module):
    """
    CLUB 互信息上界估计器（Cheng et al., 2020）。

    变分网络 q(Hs|He) 用于近似条件分布，训练目标为最大化 log q(Hs|He)。
    MI 上界：E[log q(Hs|He)] - E[log q(Hs'|He)]，其中 Hs' 来自随机置换。

    logvar 使用 softplus + log 的双重变换，clamp(-6, 4) 支持小方差场景。
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
        log_var  = torch.log(F.softplus(logvar_raw) + 1e-2).clamp(-6.0, 4.0)
        log_prob = -0.5 * (
            math.log(2 * math.pi) + log_var
            + (Hs - mu_q).pow(2) / log_var.exp()
        )
        return log_prob.mean(dim=-1)

    def _neg_perm(self, M, device):
        """生成无固定点的随机置换（derangement 近似）。"""
        perm = torch.randperm(M, device=device)
        same = perm == torch.arange(M, device=device)
        if same.any() and M > 1:
            idx  = same.nonzero(as_tuple=True)[0]
            swap = (idx + 1) % M
            perm[idx], perm[swap] = perm[swap].clone(), perm[idx].clone()
        return perm

    def forward(self, He, Hs):
        """返回 MI 上界估计，用于主网络 loss 中的 MI 正则项。"""
        M        = He.shape[0] * He.shape[1]
        He_flat  = He.reshape(M, -1)
        Hs_flat  = Hs.reshape(M, -1)
        mu_q, logvar_raw = self._get_params(He_flat)
        pos = self._log_prob(Hs_flat,                               mu_q, logvar_raw).mean()
        neg = self._log_prob(Hs_flat[self._neg_perm(M, He.device)], mu_q, logvar_raw).mean()
        return pos - neg

    def variational_loss(self, He, Hs):
        """变分网络训练目标：最大化 log q(Hs|He)，即最小化负对数似然。"""
        M        = He.shape[0] * He.shape[1]
        He_flat  = He.reshape(M, -1)
        Hs_flat  = Hs.reshape(M, -1)
        mu_q, logvar_raw = self._get_params(He_flat)
        return -self._log_prob(Hs_flat, mu_q, logvar_raw).mean()


# ---------------------------------------------------------------------------
# 6. Multi-Scale Context
# ---------------------------------------------------------------------------
class MultiScaleContext(nn.Module):
    """
    多尺度时间上下文提取。

    对环境表征序列 He_seq 分别用不同 dilation 的因果卷积提取多时间尺度特征，
    取最后时间步拼接后投影为 ms_out_dim 维输出。

    dilations 应按数据集时间分辨率配置，捕捉实际物理时间尺度：
      Solar (10min):         (1, 24, 84)  → 10min / 4h / 14h
      Electricity/Weather (1h): (1, 12, 84) → 1h / 12h / 3.5d
    """
    def __init__(self, env_dim, ms_out_dim, dilations=(1, 7, 30)):
        super().__init__()
        self.dilations = list(dilations)
        self.convs = nn.ModuleList([
            CausalConv1d(env_dim, env_dim, kernel_size=3, dilation=d)
            for d in self.dilations
        ])
        self.proj = nn.Linear(env_dim * len(self.dilations), ms_out_dim)

    def forward(self, He_seq):
        """He_seq: [B, T, N, env_dim] → [B, N, ms_out_dim]"""
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
    """
    因果门控图消息传递层。

    对边进行分块处理（chunk_size），避免一次性创建 [B, E, dim] 张量导致 OOM。
    Weather 数据集 E~200k 时，chunk_size=4096 将单次显存占用从 ~400MB 降至 ~16MB。
    """
    def __init__(self, stoch_dim, env_dim, hidden_dim=64, chunk_size: int = 8192):
        super().__init__()
        self.gate       = CausalGateUnit(stoch_dim, env_dim, hidden_dim)
        self.msg_tr     = nn.Linear(stoch_dim, stoch_dim)
        self.agg_norm   = nn.LayerNorm(stoch_dim)
        self.agg_tr     = nn.Sequential(nn.Linear(stoch_dim, stoch_dim), nn.ReLU())
        self.chunk_size = chunk_size

    def forward(self, Hs, He, edge_index):
        B, N, Ds = Hs.shape
        src, dst = edge_index[0], edge_index[1]
        E        = src.shape[0]
        agg      = torch.zeros(B, N, Ds, device=Hs.device, dtype=Hs.dtype)

        for start in range(0, E, self.chunk_size):
            end   = min(start + self.chunk_size, E)
            s_idx = src[start:end]
            d_idx = dst[start:end]
            g = self.gate(Hs[:, d_idx], Hs[:, s_idx],
                          He[:, d_idx], He[:, s_idx])   # [B, chunk, 1]
            m = g * self.msg_tr(Hs[:, s_idx])            # [B, chunk, Ds]
            agg.scatter_add_(1, d_idx.view(1, -1, 1).expand(B, -1, Ds), m)

        return self.agg_tr(Hs + self.agg_norm(agg))


class SCGMP(nn.Module):
    def __init__(self, stoch_dim, env_dim, n_layers=3, hidden_dim=64, chunk_size=8192):
        super().__init__()
        self.layers = nn.ModuleList([
            SCGMessagePassingLayer(stoch_dim, env_dim, hidden_dim, chunk_size)
            for _ in range(n_layers)
        ])

    def forward(self, Hs, He, edge_index):
        for layer in self.layers:
            Hs = layer(Hs, He, edge_index)
        return Hs


# ---------------------------------------------------------------------------
# 8. CFM Vector Field（双流 AdaLN）
# ---------------------------------------------------------------------------
class CFMVectorField(nn.Module):
    """
    条件向量场 v_θ(x_t, t | He_prime, Hs_prime)。

    双流 AdaLN 设计：
      He_prime → env_proj   → env_shift        控制分布中心（shift）
      Hs_prime → stoch_proj → stoch_scale_extra 控制分布宽度（额外 scale）
      t_emb    → time_proj  → 基础 (scale, shift)

    每个残差块的融合方式：
      scale = 1 + time_s + stoch_scale_extra
      shift =     time_b + env_shift

    物理含义：环境因素决定"预测中心在哪"，随机因素决定"分布有多宽"，
    二者在生成过程中扮演不同角色。
    """

    def __init__(self, out_dim: int, env_dim: int, stoch_dim: int,
                 hidden_dim: int = 128, time_emb_dim: int = 16,
                 max_freq: float = 1000.0):
        super().__init__()
        self.out_dim = out_dim
        n_freqs = time_emb_dim // 2
        assert n_freqs * 2 == time_emb_dim, "time_emb_dim 必须是偶数"

        freqs = torch.exp(
            torch.linspace(0.0, math.log(max_freq), n_freqs)
        ) * math.pi
        self.register_buffer("freqs", freqs)

        # 时间嵌入 → 3个残差块各自的基础 (scale, shift)，共 hidden*6
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )

        # He 流：env_dim → 3个残差块各一个 shift，共 hidden*3
        self.env_proj = nn.Sequential(
            nn.LayerNorm(env_dim),
            nn.Linear(env_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )

        # Hs 流：stoch_dim → 3个残差块各一个额外 scale，共 hidden*3
        self.stoch_proj = nn.Sequential(
            nn.LayerNorm(stoch_dim),
            nn.Linear(stoch_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )

        self.input_proj = nn.Linear(out_dim, hidden_dim)

        self.layer1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())

        self.out_proj = nn.Linear(hidden_dim, out_dim)

        # 零初始化输出层，稳定早期训练
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _time_embed(self, t: torch.Tensor, B: int, N: int) -> torch.Tensor:
        angles = t.reshape(B, 1) * self.freqs.unsqueeze(0)
        emb    = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb.unsqueeze(1).expand(B, N, -1)

    def forward(self, x_t, t, He_prime, Hs_prime):
        """
        x_t      : [B, N, out_dim]
        t        : [B]
        He_prime : [B, N, env_dim]
        Hs_prime : [B, N, stoch_dim]
        """
        B, N, _ = x_t.shape

        t_emb = self._time_embed(t, B, N)
        t_s1, t_b1, t_s2, t_b2, t_s3, t_b3 = \
            self.time_proj(t_emb).chunk(6, dim=-1)

        b1_env, b2_env, b3_env = self.env_proj(He_prime).chunk(3, dim=-1)
        s1_st, s2_st, s3_st   = self.stoch_proj(Hs_prime).chunk(3, dim=-1)

        h = self.input_proj(x_t)

        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.layer1(h_norm * (1.0 + t_s1 + s1_st) + (t_b1 + b1_env))

        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.layer2(h_norm * (1.0 + t_s2 + s2_st) + (t_b2 + b2_env))

        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.layer3(h_norm * (1.0 + t_s3 + s3_st) + (t_b3 + b3_env))

        return self.out_proj(h)


# ---------------------------------------------------------------------------
# 9. GridCFN
# ---------------------------------------------------------------------------
class GridCFN(nn.Module):

    def __init__(
        self,
        in_dim=1, gcn_hidden=64, tcn_hidden=64,
        env_dim=32, stoch_dim=32, ms_out_dim=32,
        n_scg_layers=3, out_dim=1, lambda_mi=0.5,
        gcn_layers=2, tcn_layers=4,
        cfm_hidden=128, cfm_time_emb_dim=16,
        chunk_size=8192,
        ms_dilations=(1, 7, 30),
    ):
        super().__init__()
        self.lambda_mi = lambda_mi
        self.out_dim   = out_dim
        self.env_dim   = env_dim
        self.stoch_dim = stoch_dim

        self.backbone     = Backbone(in_dim, gcn_hidden, tcn_hidden, gcn_layers, tcn_layers)
        self.disentangler = CausalDisentangler(tcn_hidden, env_dim, stoch_dim)
        self.club         = CLUBEstimator(env_dim, stoch_dim)
        self.ms_context   = MultiScaleContext(env_dim, ms_out_dim, dilations=ms_dilations)
        self.scgmp        = SCGMP(stoch_dim, env_dim, n_scg_layers, chunk_size=chunk_size)
        self.vector_field = CFMVectorField(
            out_dim=out_dim, env_dim=ms_out_dim, stoch_dim=stoch_dim,
            hidden_dim=cfm_hidden, time_emb_dim=cfm_time_emb_dim,
        )

    @staticmethod
    def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """
        对称归一化：D^{-1/2} A_hat D^{-1/2}，A_hat = A + I

        先清零对角线再加单位矩阵，防止已有自环导致度偏差。
        用广播替代 diag(d) @ adj @ diag(d)，避免构建 N×N 的 D 矩阵（Weather 下节省显存）。
        """
        adj = adj.clone()
        adj.fill_diagonal_(0)
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg      = adj.sum(dim=1)
        d_inv_sq = deg.clamp(min=1e-8).pow(-0.5)
        return d_inv_sq.unsqueeze(1) * adj * d_inv_sq.unsqueeze(0)

    @staticmethod
    def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
        return adj.nonzero(as_tuple=False).t().contiguous()

    def forward(self, x, adj_norm, edge_index):
        H              = self.backbone(x, adj_norm)
        He, Hs, He_seq = self.disentangler(H)
        mi_loss        = self.club(He, Hs)
        He_prime       = self.ms_context(He_seq)
        Hs_prime       = self.scgmp(Hs, He, edge_index)
        return He_prime, Hs_prime, He, Hs, mi_loss

    def cfm_loss(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor,
                 y_target: torch.Tensor,
                 n_t_samples: int = 4,
                 sigma_min: float = 0.01) -> torch.Tensor:
        """
        OT-CFM 训练损失。

        使用 sigma_min > 0 的 OT-CFM 路径（Flow Matching 论文 §3.3 标准做法）：
          x_t = (1 - (1-sigma_min)*t)*x0 + t*y
          u_t = y - (1-sigma_min)*x0

        sigma_min 保证 t=1 时条件分布不退化为 delta，采样自然保留 spread，
        避免直接线性插值时 PICP 系统性偏低的问题。
        """
        assert y_target.shape[-1] == self.out_dim
        B, N, _ = y_target.shape
        device  = y_target.device
        losses  = []
        for _ in range(n_t_samples):
            x0    = torch.randn_like(y_target)
            t     = torch.rand(B, device=device)
            t_bc  = t.reshape(B, 1, 1)
            x_t   = (1.0 - (1.0 - sigma_min) * t_bc) * x0 + t_bc * y_target
            u_t   = y_target - (1.0 - sigma_min) * x0
            v_pred = self.vector_field(x_t, t, He_prime, Hs_prime)
            losses.append(F.mse_loss(v_pred, u_t))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor,
               n_samples: int = 50,
               n_steps: int = 20,
               sigma_min: float = 0.01,
               x0_scale: float = 1.0) -> torch.Tensor:
        """
        并行 Euler ODE 采样，返回 [n_samples, B, N, out_dim]。

        x0_scale 控制初始噪声幅值，PICP 偏低时可从 1.0 逐步调大至 1.5。
        """
        B, N, _ = He_prime.shape
        S       = n_samples
        device  = He_prime.device
        dt      = 1.0 / n_steps

        he = He_prime.detach().repeat_interleave(S, dim=0)
        hs = Hs_prime.detach().repeat_interleave(S, dim=0)
        x  = torch.randn(B * S, N, self.out_dim, device=device) * x0_scale

        for step in range(n_steps):
            t_val = step * dt
            t_vec = torch.full((B * S,), t_val, device=device, dtype=torch.float32)
            x = x + dt * self.vector_field(x, t_vec, he, hs)

        return x.reshape(B, S, N, self.out_dim).permute(1, 0, 2, 3).contiguous()