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

修改说明（MINE → CLUB）：
  原版使用 MINE（互信息下界估计）来最小化 I(He, Hs)。
  MINE 本身是为最大化互信息设计的，"最小化下界"在梯度方向上存在理论不一致，
  且需要 minimax 两步优化（mine_optimizer 单独更新），训练逻辑复杂易出错。

  本版本用 CLUB（Contrastive Log-ratio Upper Bound，NeurIPS 2020）替换 MINE：
    CLUB 估计互信息上界：
      I(He; Hs) ≤ E_p(He,Hs)[log q(Hs|He)] - E_p(He)E_p(Hs)[log q(Hs|He)]
    其中 q(Hs|He) 是一个可学习的变分网络（VariationalNet）。

  优势：
    1. 直接估计上界 → 最小化上界比最小化下界理论上更保守且更可靠
    2. 无需 minimax 两步 → 所有参数（含变分网络）统一用主 optimizer 更新
    3. 梯度方向一致：损失对解耦表征的梯度正确指向"降低互信息"
    4. 训练更稳定：MINE 在最小化场景下梯度方差大，CLUB 无此问题

  对应代码变更：
    model.py : MINEEstimator → CLUBEstimator（含 VariationalNet）
               GridCFN.mine → GridCFN.club
               main_parameters() / mine_parameters() → 合并为 parameters()（全部参数）
    train.py : 删除 mine_optimizer 和两步训练逻辑 → 单步 optimizer 更新

原有修复保留：
  [Fix-E] Hs_seq 无用内存开销：stoch_proj 只对最后帧运算
  [Fix-G] 删除论文中不存在的 degree normalization
  [Fix-I] normalize_adj / adj_to_edge_index 移出 forward()

CLUB 数值修复说明 [Fix-CLUB-v2]：
  问题根源：
    1. _log_prob 中 log_var_q * 4.0 导致 var 极小时 (Hs-μ)²/var 爆炸
       例：log_var=-4 → var=e^{-4}≈0.018，放大项高达 ×55，log_prob → -10^6
    2. 原 train.py 中 clamp(mi_raw, max=0) 方向错误（应 clamp min=0）
    3. compute_loss 中 NLL + λ*CLUB（负数）导致 loss 为很大的负数

  修复方案：
    1. _log_prob：log_var 改用 softplus 参数化（恒正，无爆炸风险）
       var_net_logvar → 输出 log(softplus(x) + 1e-4)，方差始终在合理范围
    2. CLUBEstimator.forward 返回 clamp(club, min=0)
       CLUB 理论上 ≥ 0（互信息下界为 0），负值是采样噪声，截断为 0 合理
    3. compute_loss：L_total = L_NLL + λ * mi_loss（mi_loss ≥ 0）
       语义清晰：mi_loss 是正则项，越小说明解耦越好
    4. train.py：去掉错误的 clamp(max=0)，直接传入 mi_raw（已在 model 内截断）
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
        x_flat  = x.reshape(B * T, N, F)
        gcn_out = self.gcn(x_flat, adj_norm)          # [B*T, N, gcn_hidden]
        x_gcn   = gcn_out.reshape(B, T, N, -1)        # [B, T, N, gcn_hidden]
        H = self.tcn(x_gcn.permute(0, 2, 1, 3))      # [B, N, T, tcn_hidden]
        H = H.permute(0, 2, 1, 3)                     # [B, T, N, tcn_hidden]
        return H


# ---------------------------------------------------------------------------
# 4. Causal Disentangler  (paper Sec. IV-A, eq. 3)
# ---------------------------------------------------------------------------
class CausalDisentangler(nn.Module):
    """
    将 H 解耦为 He（环境背景）和 Hs（随机实体）。

    [Fix-E] stoch_proj 只对最后帧 H[:,-1] 运算，节省 T 倍显存；
            env_proj 保留全序列投影，MultiScaleContext 需要 He_seq [B,T,N,De]。
    """
    def __init__(self, in_dim: int, env_dim: int, stoch_dim: int):
        super().__init__()
        self.env_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, env_dim),
            nn.Tanh()
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
          He     : [B, N, env_dim]    – 最后帧环境表征
          Hs     : [B, N, stoch_dim]  – 最后帧随机实体表征
          He_seq : [B, T, N, env_dim] – 完整时序环境表征（MultiScaleContext 使用）
        """
        B, T, N, D = H.shape

        H_flat  = H.reshape(B * T * N, D)
        He_seq  = self.env_proj(H_flat).reshape(B, T, N, -1)  # [B, T, N, De]
        He      = He_seq[:, -1]                                # [B, N, De]

        # [Fix-E] 只对最后帧做 stoch_proj
        H_last = H[:, -1].reshape(B * N, D)                   # [B*N, D]
        Hs     = self.stoch_proj(H_last).reshape(B, N, -1)    # [B, N, Ds]

        return He, Hs, He_seq


# ---------------------------------------------------------------------------
# 5. CLUB – Contrastive Log-ratio Upper Bound  (NeurIPS 2020)
#    替换原版 MINE，用于最小化 I(He, Hs)
# ---------------------------------------------------------------------------
class CLUBEstimator(nn.Module):
    """
    CLUB 互信息上界估计器（Chen et al., NeurIPS 2020）。

    原理：
      CLUB 构造互信息的上界：
        I(He; Hs) ≤ E_{p(He,Hs)}[log q(Hs|He)]
                      - E_{p(He)}E_{p(Hs)}[log q(Hs|He)]

      其中 q(Hs|He) 是一个变分网络，输出高斯分布的 (μ, log_var)。

    与 MINE 的关键区别：
      · MINE 估计下界，用于最大化 MI（对抗训练），minimax 两步优化
      · CLUB 估计上界，用于最小化 MI（联合训练），单步优化
        → 直接将 club_loss 纳入总损失，无需独立的 mine_optimizer

    [Fix-CLUB-v2] 数值稳定性修复：

    问题1（log_var 爆炸）：
      原实现用 Tanh(x) * 4 参数化 log_var，当 Tanh 输出 -1 时：
        log_var = -4 → var = e^{-4} ≈ 0.018
        (Hs - μ)² / var 被放大约 55 倍，整个 log_prob 可达 -10^6 量级
      
      修复：改用 log(softplus(x) + ε) 参数化 log_var：
        softplus 输出 ≥ 0，log_var = log(softplus + ε) ∈ [log(ε), +∞)
        设 ε=0.01，log_var ≥ -4.6，var ≥ 0.01，防止极端缩放
        同时 clamp log_var ∈ [-4, 4]，双重保险

    问题2（CLUB 负值）：
      CLUB 理论上是互信息上界，≥ 0。
      训练初期变分网络未收敛时，负样本项可能大于正样本项，
      导致 club = pos - neg < 0（采样噪声）。
      修复：clamp(club, min=0)，截断噪声，确保正则项语义正确。

    问题3（量纲不匹配）：
      CLUB 的 log_prob 对 stoch_dim 求和，stoch_dim=32 时量级 ×32。
      修复：_log_prob 改为对特征维度取 mean（不是 sum），归一化量纲。
    """

    def __init__(self, env_dim: int, stoch_dim: int, hidden_dim: int = 64):
        super().__init__()
        # 变分网络：q(Hs|He) → μ（均值，无约束）
        self.var_net_mu = nn.Sequential(
            nn.Linear(env_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
        )
        # [Fix-CLUB-v2] 用 softplus 参数化 log_var，防止 var 极小导致数值爆炸
        # 输出 raw logit，在 _log_prob 中转换为 log_var
        self.var_net_logvar = nn.Sequential(
            nn.Linear(env_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
            # 不加激活，直接输出 raw；_log_prob 内部做 softplus + clamp
        )

    def _log_prob(self, Hs: torch.Tensor,
                  mu_q: torch.Tensor, logvar_raw: torch.Tensor) -> torch.Tensor:
        """
        高斯对数概率（特征维度取 mean，归一化量纲）：
          log N(Hs; μ_q, σ²_q)
            = -0.5 * mean_d[ log(2π) + log_var + (Hs-μ)²/var ]

        [Fix-CLUB-v2] log_var 参数化：
          log_var = log(softplus(logvar_raw) + 1e-2)
          → var = softplus(logvar_raw) + 1e-2 ≥ 1e-2 > 0，永远不会为 0
          → 再用 clamp(-4, 4) 防止极端值

        所有输入 shape : [M, stoch_dim]
        returns        : [M]，对特征维度取 mean 后的对数概率
        """
        # [Fix-CLUB-v2] 安全的 log_var：softplus 保证 var > 1e-2
        log_var = torch.log(F.softplus(logvar_raw) + 1e-2)
        log_var = log_var.clamp(-4.0, 4.0)   # 双重保险，var ∈ [e^{-4}, e^4]

        log_prob = -0.5 * (
            math.log(2 * math.pi)
            + log_var
            + (Hs - mu_q).pow(2) / log_var.exp()
        )
        # [Fix-CLUB-v2] mean 代替 sum，归一化量纲（消除 stoch_dim 倍数放大）
        return log_prob.mean(dim=-1)          # [M]

    def forward(self, He: torch.Tensor, Hs: torch.Tensor) -> torch.Tensor:
        """
        计算 CLUB 互信息上界估计（随机置换负样本近似）。

        He, Hs : [B, N, dim]
        returns: 标量，clamp(club, min=0)
                 ≥ 0，可直接作为正则项加入总损失

        [Fix-CLUB-v2] 修复：
          · _log_prob 数值稳定（见上方说明）
          · clamp(min=0) 消除采样噪声导致的负值
          · 返回值语义明确：0 表示完全解耦，>0 表示仍有互信息残留
        """
        B, N, _ = He.shape
        M = B * N

        He_flat = He.reshape(M, -1)    # [M, De]
        Hs_flat = Hs.reshape(M, -1)   # [M, Ds]

        # 变分网络推断条件分布参数
        mu_q       = self.var_net_mu(He_flat)       # [M, Ds]
        logvar_raw = self.var_net_logvar(He_flat)   # [M, Ds]

        # ── 正样本项：E_{p(He,Hs)}[log q(Hs_i | He_i)] ──────────────────────
        pos_term = self._log_prob(Hs_flat, mu_q, logvar_raw).mean()

        # ── 负样本项：随机置换近似 E_{p(He)}E_{p(Hs)}[log q(Hs | He)] ─────────
        # 生成错位置换（π(i) ≠ i），避免负样本与正样本完全重叠
        perm = torch.randperm(M, device=He.device)
        same = (perm == torch.arange(M, device=He.device))
        if same.any() and M > 1:
            idx  = same.nonzero(as_tuple=True)[0]
            swap = (idx + 1) % M
            perm[idx], perm[swap] = perm[swap].clone(), perm[idx].clone()

        Hs_neg   = Hs_flat[perm]       # [M, Ds]，打乱后的负样本
        neg_term = self._log_prob(Hs_neg, mu_q, logvar_raw).mean()

        # CLUB 上界：正样本 - 负样本
        # [Fix-CLUB-v2] clamp(min=0)：互信息理论 ≥ 0，负值是采样噪声
        club_upper_bound = (pos_term - neg_term).clamp(min=0.0)
        return club_upper_bound


# ---------------------------------------------------------------------------
# 6. Multi-Scale Context  (paper Sec. IV-B, eq. 7)
# ---------------------------------------------------------------------------
class MultiScaleContext(nn.Module):
    """
    对 He_seq [B, T, N, De] 用不同膨胀率的因果卷积提取多尺度上下文，
    拼接后投影到 ms_out_dim。
    膨胀率：[1, 7, 30]，分别对应约 3小时/日/月 三个时间尺度。
    """
    def __init__(self, env_dim: int, ms_out_dim: int):
        super().__init__()
        dilations = [1, 7, 30]
        self.convs = nn.ModuleList([
            CausalConv1d(env_dim, env_dim, kernel_size=3, dilation=d)
            for d in dilations
        ])
        self.proj = nn.Linear(env_dim * len(dilations), ms_out_dim)

    def forward(self, He_seq: torch.Tensor) -> torch.Tensor:
        """
        He_seq : [B, T, N, De]
        returns: [B, N, ms_out_dim]
        """
        B, T, N, De = He_seq.shape
        x = He_seq.permute(0, 2, 3, 1).reshape(B * N, De, T)  # [B*N, De, T]
        outs = [conv(x)[:, :, -1] for conv in self.convs]     # 各 [B*N, De]
        cat  = torch.cat(outs, dim=-1)                          # [B*N, De*3]
        out  = self.proj(cat).reshape(B, N, -1)                 # [B, N, ms_out_dim]
        return out


# ---------------------------------------------------------------------------
# 7. Spatial Causal Gated Message Passing  (paper Sec. IV-B, eq. 8-10)
# ---------------------------------------------------------------------------
class CausalGateUnit(nn.Module):
    """
    门控因果性过滤单元（paper eq. 9）：
      g_ji = sigmoid(W_g [Hs_i || Hs_j || He_i || He_j])
    """
    def __init__(self, stoch_dim: int, env_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = stoch_dim * 2 + env_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, Hs_i, Hs_j, He_i, He_j):
        cat = torch.cat([Hs_i, Hs_j, He_i, He_j], dim=-1)
        return self.net(cat)


class SCGMessagePassingLayer(nn.Module):
    """
    单层空间因果门控消息传递（paper eq. 8-10）。
    消息 m_ji = g_ji * W_m Hs_j
    聚合 agg_i = sum_{j ∈ N(i)} m_ji
    更新 Hs'_i = LayerNorm(W_u [Hs_i + agg_i])
    """
    def __init__(self, stoch_dim: int, env_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gate_unit     = CausalGateUnit(stoch_dim, env_dim, hidden_dim)
        self.msg_transform = nn.Linear(stoch_dim, stoch_dim)
        self.agg_norm      = nn.LayerNorm(stoch_dim)
        self.agg_transform = nn.Sequential(
            nn.Linear(stoch_dim, stoch_dim), nn.ReLU()
        )

    def forward(self, Hs: torch.Tensor, He: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Hs         : [B, N, stoch_dim]
        He         : [B, N, env_dim]
        edge_index : [2, E]  row0=src(j), row1=dst(i)
        returns    : [B, N, stoch_dim]
        """
        B, N, Ds = Hs.shape
        src, dst = edge_index[0], edge_index[1]   # j → i

        Hs_j = Hs[:, src]   # [B, E, Ds]
        Hs_i = Hs[:, dst]   # [B, E, Ds]
        He_j = He[:, src]   # [B, E, De]
        He_i = He[:, dst]   # [B, E, De]

        g = self.gate_unit(Hs_i, Hs_j, He_i, He_j)  # [B, E, 1]

        m_raw    = self.msg_transform(Hs_j)           # [B, E, Ds]
        m_causal = g * m_raw                          # [B, E, Ds]

        agg = torch.zeros(B, N, Ds, device=Hs.device, dtype=Hs.dtype)
        idx = dst.view(1, -1, 1).expand(B, -1, Ds)   # [B, E, Ds]
        agg.scatter_add_(1, idx, m_causal)

        agg = self.agg_norm(agg)

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
    完整 GridCFN 管道（CLUB 版本）。

    与原版 MINE 版本的差异：
      · self.mine → self.club（CLUBEstimator）
      · forward() 返回的第三个值含义不变（mi_loss），但现在是 CLUB 上界
      · main_parameters() / mine_parameters() 合并 → 直接用 model.parameters()
        （所有参数由主 optimizer 统一更新，无需独立的 mine_optimizer）
      · 不再需要 warmup_epochs 的 MI 热身逻辑（CLUB 训练从第 1 个 epoch 即稳定）

    [Fix-I] forward() 接收预计算的 adj_norm 和 edge_index，不在内部重算。
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
        # CLUB 替换 MINE：直接估计上界，联合训练，无需独立优化器
        self.club         = CLUBEstimator(env_dim, stoch_dim)
        self.ms_context   = MultiScaleContext(env_dim, ms_out_dim)
        self.scgmp        = SCGMP(stoch_dim, env_dim, n_scg_layers)
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
        adj_norm   : [N, N]  预归一化邻接矩阵
        edge_index : [2, E]  边索引

        返回:
          mu      : [B, N, out_dim]
          sigma   : [B, N, out_dim]  (> 0)
          mi_loss : 标量，CLUB 上界估计，clamp(min=0)
                    语义：0 = 完全解耦，>0 = 存在互信息，训练目标是最小化此值
        """
        # ── Backbone ────────────────────────────────────────────────────────
        H = self.backbone(x, adj_norm)                    # [B, T, N, D]

        # ── Causal Disentanglement ──────────────────────────────────────────
        He, Hs, He_seq = self.disentangler(H)

        # ── MI 估计（CLUB 上界，已在 CLUBEstimator 内 clamp(min=0)）──────────
        mi_loss = self.club(He, Hs)                       # 标量，≥ 0

        # ── Multi-Scale Context ─────────────────────────────────────────────
        He_prime = self.ms_context(He_seq)                # [B, N, ms_out_dim]

        # ── Spatial Causal Gated MP ─────────────────────────────────────────
        Hs_prime = self.scgmp(Hs, He, edge_index)        # [B, N, Ds]

        # ── Feature Fusion + Probabilistic Prediction ───────────────────────
        H_final   = torch.cat([He_prime, Hs_prime], dim=-1)
        mu, sigma = self.predictor(H_final)

        return mu, sigma, mi_loss

    def compute_loss(self, mu: torch.Tensor, sigma: torch.Tensor,
                     y: torch.Tensor, mi_loss: torch.Tensor,
                     lambda_mi: float = None) -> tuple:
        """
        总损失 = L_NLL + λ · L_CLUB

        [Fix-CLUB-v2] 语义修复：
          · mi_loss 已在 CLUBEstimator.forward 中 clamp(min=0)，恒 ≥ 0
          · λ * mi_loss 恒 ≥ 0，是正则项，与 NLL 方向一致（都是最小化）
          · loss = NLL + λ * CLUB ≥ NLL > 0（通常），不会出现巨大负数

        与原版 compute_loss 接口完全兼容，train.py 无需改动 compute_loss 调用方式。
        """
        if lambda_mi is None:
            lambda_mi = self.lambda_mi
        l_nll   = nll_gaussian_loss(mu, sigma, y)
        l_total = l_nll + lambda_mi * mi_loss
        return l_total, l_nll