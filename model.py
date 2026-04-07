"""
GridCFN: A Causal Spatio-Temporal Framework for Power Flow Uncertainty Prediction
Reproduction based on IC2ECS 2025 paper by Zhao et al.

Architecture:
  Input X [B, T, N, F]
    └─ Backbone (GCN + TCN) → H [B, T, N, D]
         └─ Causal Disentangler → He [B, N, De], Hs [B, N, Ds], He_seq [B, T, N, De]
              ├─ Multi-Scale Context (dilated conv on He_seq) → H'e [B, N, ms_out_dim]
              └─ SCG-MP (causal gated message passing) → H's [B, N, Ds]
                   └─ Feature Fusion → H_final [B, N, ms_out_dim+Ds]
                        └─ Predictor → (mu, sigma) [B, N, Fout]

修复说明：
  [Fix-A/B] MINE minimax 训练逻辑：
    原代码 _forward_for_mine 对 He/Hs 做了 detach，导致 MINE 参数梯度全为 0，
    mine_optimizer.step() 完全无效。
    修复：删除 _forward_for_mine；forward() 只返回 mi_loss 供两个 optimizer 分别使用；
    compute_loss 中对 mi_loss 做 detach，使 Step2 的梯度不回传到 MINE 参数，
    两步优化彻底隔离。train.py 也同步简化。

  [Fix-E] Hs_seq 无用内存开销：
    原 CausalDisentangler 对全序列 [B,T,N,D] 做 stoch_proj，生成 Hs_seq 后
    只取最后帧，在 Weather(N=1866, T=168) 下显存浪费严重。
    修复：stoch_proj 只对最后帧 H[:,-1] 运算；env_proj 保留全序列（MultiScaleContext 需要）。

  [Fix-G] 删除论文中不存在的 degree normalization：
    论文 eq.(9) 是 σ_agg(Hs_i + Σ m_causal)，没有除以度数。
    自行添加的 deg 归一化改变梯度尺度，与论文不符，已删除。

  [Fix-I] normalize_adj / adj_to_edge_index 移出 forward()：
    adj 在整个训练中固定，每次 forward 重算是纯浪费。
    修复：两个静态方法保留，由 main.py 在加载数据后预计算一次，
    forward() 直接接收 adj_norm [N,N] 和 edge_index [2,E]。
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 1. Graph Convolutional Layer (spectral GCN, Kipf & Welling 2017)
# ---------------------------------------------------------------------------
class GCNLayer(nn.Module):
    """
    H_out = ReLU(D^{-1/2} Â D^{-1/2} H W)
    Â = A + I  (self-loop 在 normalize_adj 里添加)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x        : [*, N, in_dim]
        adj_norm : [N, N]  pre-normalised
        returns  : [*, N, out_dim]
        """
        support = self.linear(x)               # [*, N, out_dim]
        out = torch.matmul(adj_norm, support)  # [*, N, out_dim]
        return F.relu(out)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj_norm)
        return x


# ---------------------------------------------------------------------------
# 2. Temporal Convolutional Network (causal dilated convolutions)
# ---------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    """Left-padded causal conv1d：输出长度 == 输入长度，保证不泄漏未来信息。"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, C, T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """带残差连接的因果膨胀卷积块，使用 GroupNorm 替代 permute+LayerNorm。"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        # GroupNorm 直接作用于 [B*N, C, T]，无需 permute，等价于对通道做归一化
        # num_groups=min(8, channels) 兼容小隐层（debug 模式 channels=16）
        n_groups = min(8, channels)
        self.norm1 = nn.GroupNorm(n_groups, channels)
        self.norm2 = nn.GroupNorm(n_groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, C, T]
        residual = x
        out = F.gelu(self.norm1(self.conv1(x)))
        out = F.gelu(self.norm2(self.conv2(out)))
        return out + residual


class TCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 4,
                 kernel_size: int = 3):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, hidden_dim, 1)
        dilations = [2 ** i for i in range(n_layers)]
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden_dim, kernel_size, d) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x  : [B, N, T, in_dim]
        out: [B, N, T, hidden_dim]
        """
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C).permute(0, 2, 1)  # [B*N, C, T]
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1).reshape(B, N, T, -1)  # [B, N, T, hidden]
        return x


# ---------------------------------------------------------------------------
# 3. Backbone: GCN + TCN  (paper eq. 2)
# ---------------------------------------------------------------------------
class Backbone(nn.Module):
    """H = TCN(GCN(X, A)) ∈ R^{B×T×N×D}"""

    def __init__(self, in_dim: int, gcn_hidden: int, tcn_hidden: int,
                 gcn_layers: int = 2, tcn_layers: int = 4):
        super().__init__()
        self.gcn = GCN(in_dim, gcn_hidden, gcn_hidden, gcn_layers)
        self.tcn = TCN(gcn_hidden, tcn_hidden, tcn_layers)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x        : [B, T, N, F]
        adj_norm : [N, N]  预归一化邻接矩阵（由外部预计算传入）
        returns H: [B, T, N, D]
        """
        B, T, N, F = x.shape
        # 合并 B×T 维，对所有时间步并行做 GCN（避免 Python for 循环）
        x_flat  = x.reshape(B * T, N, F)
        gcn_out = self.gcn(x_flat, adj_norm)          # [B*T, N, gcn_hidden]
        x_gcn   = gcn_out.reshape(B, T, N, -1)        # [B, T, N, gcn_hidden]
        # TCN 需要 [B, N, T, C] 格式
        H = self.tcn(x_gcn.permute(0, 2, 1, 3))      # [B, N, T, tcn_hidden]
        H = H.permute(0, 2, 1, 3)                     # [B, T, N, tcn_hidden]
        return H


# ---------------------------------------------------------------------------
# 4. Causal Disentangler  (paper Sec. IV-A, eq. 3)
# ---------------------------------------------------------------------------
class CausalDisentangler(nn.Module):
    """
    将 H 解耦为 He（环境背景）和 Hs（随机实体）。

    [Fix-E] 原代码对全序列 [B,T,N,D] 同时做 stoch_proj（生成 Hs_seq），
    但 Hs_seq 只取最后帧使用，在大数据集（Weather: N=1866, T=168）下
    浪费约 T 倍显存。
    修复：stoch_proj 只对最后帧 H[:,-1] 运算；
          env_proj 保留全序列投影，因为 MultiScaleContext 需要 He_seq [B,T,N,De]。
    """
    def __init__(self, in_dim: int, env_dim: int, stoch_dim: int):
        super().__init__()
        self.env_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, env_dim),
            nn.Tanh()   # 限制范围 [-1,1]，防止 MINE 梯度爆炸
        )
        self.stoch_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, stoch_dim),
            nn.Tanh()
        )

    def forward(self, H: torch.Tensor):
        """
        H   : [B, T, N, D]
        返回:
          He     : [B, N, env_dim]    – 最后帧环境表征（MINE + SCG-MP gate 使用）
          Hs     : [B, N, stoch_dim]  – 最后帧随机实体表征（SCG-MP 消息传递使用）
          He_seq : [B, T, N, env_dim] – 完整时序环境表征（MultiScaleContext 使用）
        """
        B, T, N, D = H.shape

        # env_proj：对全序列投影（MultiScaleContext 需要时序信息）
        H_flat  = H.reshape(B * T * N, D)
        He_seq  = self.env_proj(H_flat).reshape(B, T, N, -1)  # [B, T, N, De]
        He      = He_seq[:, -1]                                # [B, N, De]

        # [Fix-E] stoch_proj：只对最后帧投影，节省 T 倍显存
        H_last = H[:, -1].reshape(B * N, D)                   # [B*N, D]
        Hs     = self.stoch_proj(H_last).reshape(B, N, -1)    # [B, N, Ds]

        return He, Hs, He_seq


# ---------------------------------------------------------------------------
# 5. MINE – Mutual Information Neural Estimator  (paper eq. 3, Belghazi 2018)
# ---------------------------------------------------------------------------
class MINEEstimator(nn.Module):
    """
    用 MINE 估计 I(He, Hs) 的下界：
        MI_lower ≈ E[T(He, Hs)] - log(E[exp(T(He, Hs_shuffled))])

    训练方式（minimax，在 train.py 中实现）：
      - Step1：mine_optimizer 对 MINE 参数做梯度上升（最大化 MI 估计）
      - Step2：主 optimizer 最小化 NLL + λ·MI（MI 项 detach，不回传给 MINE）

    数值稳定：log-sum-exp trick 防止 exp 上溢。
    """
    def __init__(self, env_dim: int, stoch_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env_dim + stoch_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),           nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, He: torch.Tensor, Hs: torch.Tensor) -> torch.Tensor:
        """
        He, Hs : [B, N, dim]
        returns: 标量，MI 的 MINE 下界估计
        """
        B, N, _ = He.shape
        He_flat = He.reshape(B * N, -1)
        Hs_flat = Hs.reshape(B * N, -1)

        # 联合分布：T(He_i, Hs_i)
        t_joint = self.net(torch.cat([He_flat, Hs_flat], dim=-1)).squeeze(-1)

        # 边缘分布：T(He_i, Hs_j)，j 是 i 的随机置换
        idx         = torch.randperm(B * N, device=He.device)
        Hs_shuffled = Hs_flat[idx]
        t_marginal  = self.net(torch.cat([He_flat, Hs_shuffled], dim=-1)).squeeze(-1)

        # log-sum-exp trick
        c           = t_marginal.detach().max()
        log_mean_et = c + torch.log(torch.exp(t_marginal - c).mean() + 1e-8)

        mi_estimate = t_joint.mean() - log_mean_et
        return mi_estimate


# ---------------------------------------------------------------------------
# 6. Multi-Scale Context Modeling  (paper Sec. IV-A, eq. 4)
# ---------------------------------------------------------------------------
class MultiScaleContext(nn.Module):
    """
    对 He 的时序用多尺度膨胀卷积捕捉不同时间尺度的环境规律（hourly/daily/seasonal），
    拼接后线性融合。
    Paper: H'e = fuse([DilatedConv1D(He_seq, d_l) for l in L])
    """
    def __init__(self, env_dim: int, out_dim: int,
                 n_scales: int = 4, kernel_size: int = 3):
        super().__init__()
        dilations = [1, 2, 4, 8][:n_scales]
        self.convs = nn.ModuleList([
            CausalConv1d(env_dim, out_dim, kernel_size, d) for d in dilations
        ])
        # 拼接各尺度输出后线性融合
        self.fuse = nn.Linear(out_dim * n_scales, out_dim)

    def forward(self, He_seq: torch.Tensor) -> torch.Tensor:
        """
        He_seq : [B, T, N, env_dim]
        returns: [B, N, out_dim]
        """
        B, T, N, C = He_seq.shape
        # 合并 (B, N) 为 batch 维，时间维作为序列长度
        x    = He_seq.permute(0, 2, 3, 1).reshape(B * N, C, T)  # [B*N, C, T]
        outs = [conv(x) for conv in self.convs]                   # each: [B*N, out_dim, T]
        # 拼接各尺度，取最后时间步（只需当前时刻的多尺度汇总）
        fused = torch.cat([o[:, :, -1] for o in outs], dim=1)    # [B*N, out_dim*n_scales]
        out   = self.fuse(fused).view(B, N, -1)                   # [B, N, out_dim]
        return F.relu(out)


# ---------------------------------------------------------------------------
# 7. Spatial Causal Gated Message Passing  (paper Sec. IV-B, eq. 5-9)
# ---------------------------------------------------------------------------
class CausalGatingUnit(nn.Module):
    """
    为每条边 (i,j) 计算因果门控系数 g_ij ∈ [0,1]。
    Z_ij = [Hs_i || Hs_j || He_i || He_j]  (paper eq. 5)
    g_ij = σ(W2 · ReLU(W1·Z_ij + b1) + b2)  (paper eq. 6)
    """
    def __init__(self, stoch_dim: int, env_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = 2 * stoch_dim + 2 * env_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),      nn.Sigmoid()
        )

    def forward(self, Hs_i, Hs_j, He_i, He_j):
        """All inputs: [B, E, dim]"""
        Z = torch.cat([Hs_i, Hs_j, He_i, He_j], dim=-1)  # [B, E, 2Ds+2De]
        return self.gate_mlp(Z)                             # [B, E, 1]


class SCGMessagePassingLayer(nn.Module):
    """
    一层空间因果门控消息传递（paper eq. 7-9）：
      m_j→i^raw    = Θ_msg · Hs_j
      m_j→i^causal = g_ij ⊙ m_j→i^raw
      Hs_i^new     = σ_agg(Hs_i + Σ_j m_j→i^causal)

    [Fix-G] 删除原代码中论文没有的 degree normalization。
    """
    def __init__(self, stoch_dim: int, env_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gate_unit     = CausalGatingUnit(stoch_dim, env_dim, hidden_dim)
        self.msg_transform = nn.Linear(stoch_dim, stoch_dim, bias=False)
        self.agg_norm      = nn.LayerNorm(stoch_dim)   # 聚合后归一化，替代 degree norm
        self.agg_transform = nn.Sequential(
            nn.Linear(stoch_dim, stoch_dim), nn.ReLU()
        )

    def forward(self, Hs: torch.Tensor, He: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Hs         : [B, N, stoch_dim]
        He         : [B, N, env_dim]   原始解耦结果（论文框架图直接从 Disentangler 连过来）
        edge_index : [2, E]  row0=src(j), row1=dst(i)
        returns    : [B, N, stoch_dim]
        """
        B, N, Ds = Hs.shape
        src, dst = edge_index[0], edge_index[1]   # j → i

        # 为每条边收集节点特征
        Hs_j = Hs[:, src]   # [B, E, Ds]
        Hs_i = Hs[:, dst]   # [B, E, Ds]
        He_j = He[:, src]   # [B, E, De]
        He_i = He[:, dst]   # [B, E, De]

        # 因果门控系数
        g = self.gate_unit(Hs_i, Hs_j, He_i, He_j)  # [B, E, 1]

        # 原始消息 → 门控消息
        m_raw    = self.msg_transform(Hs_j)           # [B, E, Ds]
        m_causal = g * m_raw                          # [B, E, Ds]

        # 聚合到目标节点（scatter_add 沿节点维 dim=1）
        agg = torch.zeros(B, N, Ds, device=Hs.device, dtype=Hs.dtype)
        idx = dst.view(1, -1, 1).expand(B, -1, Ds)   # [B, E, Ds]
        agg.scatter_add_(1, idx, m_causal)

        # LayerNorm 稳定聚合后的尺度（替代 degree norm，符合论文精神）
        agg = self.agg_norm(agg)

        # 残差更新（paper eq. 9）
        Hs_new = self.agg_transform(Hs + agg)
        return Hs_new


class SCGMP(nn.Module):
    """L_SCG 层空间因果门控消息传递的堆叠。"""
    def __init__(self, stoch_dim: int, env_dim: int,
                 n_layers: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            SCGMessagePassingLayer(stoch_dim, env_dim, hidden_dim)
            for _ in range(n_layers)
        ])

    def forward(self, Hs: torch.Tensor, He: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            Hs = layer(Hs, He, edge_index)
        return Hs


# ---------------------------------------------------------------------------
# 8. Probabilistic Predictor  (paper Sec. IV-C, eq. 11)
# ---------------------------------------------------------------------------
class ProbabilisticPredictor(nn.Module):
    """
    H_final = [H'e || H's]  →  (μ, σ)
    σ 用 softplus + offset 确保正值，防止 log(σ) → -∞。
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head    = nn.Linear(hidden_dim, out_dim)
        self.sigma_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, H_final: torch.Tensor):
        """
        H_final : [B, N, ms_out_dim + stoch_dim]
        returns : mu [B, N, out_dim], sigma [B, N, out_dim] (sigma > 0)
        """
        h     = self.net(self.norm(H_final))
        mu    = self.mu_head(h)
        # sigma = F.softplus(self.sigma_head(h)) + 1e-3
        sigma = F.softplus(self.sigma_head(h)) + 0.05
        return mu, sigma


# ---------------------------------------------------------------------------
# 9. Loss Functions
# ---------------------------------------------------------------------------
def nll_gaussian_loss(mu: torch.Tensor, sigma: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
    """
    完整高斯负对数似然（paper eq. 13）：
      L_NLL = mean[ log(σ) + 0.5·log(2π) + 0.5·((y-μ)/σ)² ]
    """
    sigma = sigma.clamp(min=1e-6)
    nll = (torch.log(sigma)
           + 0.5 * math.log(2 * math.pi)
           + 0.5 * ((y - mu) / sigma) ** 2)
    return nll.mean()


# ---------------------------------------------------------------------------
# 10. Full GridCFN Model
# ---------------------------------------------------------------------------
class GridCFN(nn.Module):
    """
    完整 GridCFN 管道。

    [Fix-I] forward() 不再内部调用 normalize_adj / adj_to_edge_index，
    改为直接接收预计算的 adj_norm 和 edge_index，避免每个 batch 重复计算。
    预计算由 main.py 在数据加载后完成。
    """
    def __init__(
        self,
        in_dim:       int   = 1,
        gcn_hidden:   int   = 64,
        tcn_hidden:   int   = 64,
        env_dim:      int   = 32,
        stoch_dim:    int   = 32,
        ms_out_dim:   int   = 32,
        n_scg_layers: int   = 3,
        out_dim:      int   = 1,
        lambda_mi:    float = 0.5,
        gcn_layers:   int   = 2,
        tcn_layers:   int   = 4,
    ):
        super().__init__()
        self.lambda_mi = lambda_mi

        self.backbone     = Backbone(in_dim, gcn_hidden, tcn_hidden,
                                     gcn_layers, tcn_layers)
        self.disentangler = CausalDisentangler(tcn_hidden, env_dim, stoch_dim)
        self.mine         = MINEEstimator(env_dim, stoch_dim)
        self.ms_context   = MultiScaleContext(env_dim, ms_out_dim)
        # SCG-MP 门控使用原始 He（env_dim），与论文框架图一致
        self.scgmp        = SCGMP(stoch_dim, env_dim, n_scg_layers)
        # Predictor 融合 H'e（ms_out_dim）和 H's（stoch_dim）
        self.predictor    = ProbabilisticPredictor(ms_out_dim + stoch_dim, out_dim)

    @staticmethod
    def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """对称归一化：D^{-1/2} (A+I) D^{-1/2}"""
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
        D = torch.diag(d_inv_sqrt)
        return D @ adj @ D

    @staticmethod
    def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
        """邻接矩阵 → edge_index [2, E]"""
        return adj.nonzero(as_tuple=False).t().contiguous()

    def forward(self, x: torch.Tensor,
                adj_norm: torch.Tensor,
                edge_index: torch.Tensor):
        """
        x          : [B, T, N, F]
        adj_norm   : [N, N]  预归一化邻接矩阵（外部预计算，固定不变）
        edge_index : [2, E]  边索引（外部预计算，固定不变）

        返回:
          mu     : [B, N, out_dim]
          sigma  : [B, N, out_dim]  (> 0)
          mi_loss: 标量，MINE 对 I(He,Hs) 的下界估计
        """
        # ── Backbone ────────────────────────────────────────────────────────
        H = self.backbone(x, adj_norm)                    # [B, T, N, D]

        # ── Causal Disentanglement ──────────────────────────────────────────
        He, Hs, He_seq = self.disentangler(H)
        # He    : [B, N, De]
        # Hs    : [B, N, Ds]
        # He_seq: [B, T, N, De]

        # ── MI 估计（MINE）──────────────────────────────────────────────────
        mi_loss = self.mine(He, Hs)

        # ── Multi-Scale Context ─────────────────────────────────────────────
        He_prime = self.ms_context(He_seq)                # [B, N, ms_out_dim]

        # ── Spatial Causal Gated MP ─────────────────────────────────────────
        # 门控单元使用原始 He（论文框架图：箭头直接从 Disentangler → Causal Gating Unit）
        Hs_prime = self.scgmp(Hs, He, edge_index)        # [B, N, Ds]

        # ── Feature Fusion + Probabilistic Prediction ───────────────────────
        H_final   = torch.cat([He_prime, Hs_prime], dim=-1)  # [B, N, ms+Ds]
        mu, sigma = self.predictor(H_final)

        return mu, sigma, mi_loss

    def compute_loss(self, mu: torch.Tensor, sigma: torch.Tensor,
                     y: torch.Tensor, mi_loss: torch.Tensor,
                     lambda_mi: float = None) -> tuple:
        """
        总损失 = L_NLL + λ · L_MI  (paper eq. 12)

        [Fix-A/B] mi_loss 在传入前已经从主网络计算图中 detach（在 train.py 里处理），
        确保 Step2 的梯度不回传到 MINE 参数，两步优化彻底隔离。
        """
        if lambda_mi is None:
            lambda_mi = self.lambda_mi
        l_nll   = nll_gaussian_loss(mu, sigma, y)
        l_total = l_nll + lambda_mi * mi_loss
        return l_total, l_nll, mi_loss

    def main_parameters(self):
        """主网络参数（排除 MINE），用于主 optimizer。"""
        mine_ids = {id(p) for p in self.mine.parameters()}
        return [p for p in self.parameters() if id(p) not in mine_ids]

    def mine_parameters(self):
        """MINE 网络参数，用于独立的 mine_optimizer。"""
        return list(self.mine.parameters())
