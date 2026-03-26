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

修复说明（对照论文逐条）：
  [Bug-1] MINE 训练方向错误：原代码用单一 optimizer 同时更新所有参数，
          导致 MINE 网络无法正确收敛（MINE 需要 minimax：网络参数最大化 MI 估计，
          主网络最小化 MI）。修复：将 MINE 网络参数暴露给 train.py 用独立 optimizer。

  [Bug-2] CausalDisentangler 只对最后一帧 H[:,-1] 投影，而 MultiScaleContext
          需要完整时序的 He_seq [B,T,N,De]。原 forward() 里在 GridCFN 外部
          重复调用 env_proj（权重共享但冗余且架构混乱）。
          修复：Disentangler 内部直接对全序列投影，返回 He_seq。

  [Bug-3] NLL 损失公式与论文不符：论文 eq.(3-9) 是标准高斯 NLL，
          包含 log(2π σ²)/2 + (Y-μ)²/(2σ²)，即 log(σ) + 常数 + MSE/σ²。
          F.gaussian_nll_loss 的 eps 参数是对方差 var 的下界裁剪，设为 1e-2
          等价于 sigma 最小 0.1，对归一化数据过大。修复为标准手写实现，eps 改为 1e-6。

  [Bug-4] train_one_epoch 中定义了 curr_lambda_mi（热身逻辑）但从未传入
          compute_loss，热身期 MI 正则照常生效。修复：compute_loss 接受
          lambda_mi 参数，train.py 动态传入。

  [Bug-5] SCG-MP 各层共享同一个固定 He（初始解耦值），而论文 eq.(3-7) 暗示
          每层的 gate 应使用当前层的上下文。修复：将 He_prime（经 MultiScaleContext
          精炼后的上下文）传入 SCGMP，而非原始 He。

  [Bug-6] nll_gaussian_loss 使用 F.gaussian_nll_loss 的 full=False 默认值，
          丢掉了 log(2πσ²)/2 中的 log(2π) 常数（训练时无影响但与论文公式不符，
          且 full=True 对校准有帮助）。修复：手写完整 NLL。
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
    H_out = σ(D^{-1/2} Â D^{-1/2} H W)
    Â = A + I  (self-loop 已在 normalize_adj 里添加)
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
    """Left-padded causal conv1d：输出长度 == 输入长度。"""
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
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, C, T]
        residual = x
        out = F.gelu(self.conv1(x))
        out = self.norm1(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = F.gelu(self.conv2(out))
        out = self.norm2(out.permute(0, 2, 1)).permute(0, 2, 1)
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
        adj_norm : [N, N]
        returns H: [B, T, N, D]
        """
        B, T, N, F = x.shape
        # 合并 B×T 一次性完成所有时间步 GCN（避免 Python for 循环）
        x_flat  = x.reshape(B * T, N, F)
        gcn_out = self.gcn(x_flat, adj_norm)          # [B*T, N, gcn_hidden]
        x_gcn   = gcn_out.reshape(B, T, N, -1)        # [B, T, N, gcn_hidden]

        # TCN 沿时间轴处理（需要 [B, N, T, C] 格式）
        H = self.tcn(x_gcn.permute(0, 2, 1, 3))      # [B, N, T, tcn_hidden]
        H = H.permute(0, 2, 1, 3)                     # [B, T, N, tcn_hidden]
        return H


# ---------------------------------------------------------------------------
# 4. Causal Disentangler  (paper Sec. IV-A / 论文图像 Sec. 3.x)
# ---------------------------------------------------------------------------
class CausalDisentangler(nn.Module):
    """
    将 H 解耦为 He（环境背景）和 Hs（随机实体）两个统计独立子表征。

    [Bug-2 修复]
    原代码只对最后一帧 H[:,-1] 投影，但 MultiScaleContext 需要完整时序的 He_seq。
    原实现在 GridCFN.forward() 里重复调用 env_proj，既冗余又架构不清晰。
    修复：forward() 内部同时对全序列 H 投影，返回 He_seq [B,T,N,De]。
    """
    def __init__(self, in_dim: int, env_dim: int, stoch_dim: int):
        super().__init__()
        # 环境背景编码器
        self.env_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, env_dim),
            nn.Tanh()   # 限制范围 [-1,1]，防止 MINE 梯度爆炸
        )
        # 随机实体编码器
        self.stoch_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, stoch_dim),
            nn.Tanh()
        )

    def forward(self, H: torch.Tensor):
        """
        H   : [B, T, N, D]
        返回:
          He     : [B, N, env_dim]    – 最后一帧的环境背景表征（用于 MINE 和 SCG-MP gate）
          Hs     : [B, N, stoch_dim]  – 最后一帧的随机实体表征（用于 SCG-MP 消息传递）
          He_seq : [B, T, N, env_dim] – 完整时序的环境表征（用于 MultiScaleContext）
        """
        B, T, N, D = H.shape

        # [Bug-2 修复] 对完整序列一次性投影，替代 GridCFN.forward() 里的冗余调用
        H_flat    = H.reshape(B * T * N, D)
        He_seq    = self.env_proj(H_flat).reshape(B, T, N, -1)   # [B, T, N, De]
        Hs_seq    = self.stoch_proj(H_flat).reshape(B, T, N, -1) # [B, T, N, Ds]

        # 取最后一帧作为当前状态（供 MINE 估计和 SCG-MP 使用）
        He = He_seq[:, -1]   # [B, N, env_dim]
        Hs = Hs_seq[:, -1]   # [B, N, stoch_dim]

        return He, Hs, He_seq


# ---------------------------------------------------------------------------
# 5. MINE – Mutual Information Neural Estimator  (paper eq. 3 / Belghazi 2018)
# ---------------------------------------------------------------------------
class MINEEstimator(nn.Module):
    """
    用 MINE 估计 I(He, Hs) 的下界：
        MI_lower ≈ E[T(He, Hs)] - log(E[exp(T(He, Hs'))])
    其中 Hs' 是 Hs 的随机打乱（边缘分布采样）。

    [Bug-1 修复说明]
    MINE 是 minimax 问题：
      - MINE 网络参数需要被「最大化」（使 T 网络更好地估计真实 MI）
      - 主网络参数需要被「最小化」MI（让 He 和 Hs 独立）
    因此 MINE 网络参数必须使用独立的 optimizer 做梯度上升。
    具体训练逻辑见 train.py 的 train_one_epoch()。

    数值稳定性：使用 log-sum-exp trick 防止 exp 上溢。
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
        returns: 标量，MI 的 MINE 下界估计（正值表示两者有依赖，目标是让主网络最小化它）
        """
        B, N, _ = He.shape
        He_flat = He.reshape(B * N, -1)
        Hs_flat = Hs.reshape(B * N, -1)

        # 联合分布项：T(He_i, Hs_i)
        t_joint = self.net(
            torch.cat([He_flat, Hs_flat], dim=-1)
        ).squeeze(-1)                         # [B*N]

        # 边缘分布项：T(He_i, Hs_j)，j 是 i 的随机置换
        idx         = torch.randperm(B * N, device=He.device)
        Hs_shuffled = Hs_flat[idx]
        t_marginal  = self.net(
            torch.cat([He_flat, Hs_shuffled], dim=-1)
        ).squeeze(-1)                         # [B*N]

        # log-sum-exp trick：log(E[exp(t)]) = c + log(mean(exp(t-c)))，避免 exp 上溢
        c           = t_marginal.detach().max()
        log_mean_et = c + torch.log(torch.exp(t_marginal - c).mean() + 1e-8)

        mi_estimate = t_joint.mean() - log_mean_et
        return mi_estimate   # 正值 ≈ 真实 MI；主网络目标：minimize 此值


# ---------------------------------------------------------------------------
# 6. Multi-Scale Context Modeling  (paper eq. 4)
# ---------------------------------------------------------------------------
class MultiScaleContext(nn.Module):
    """
    对 He 的时序用多尺度膨胀卷积捕捉不同时间尺度的环境规律，
    然后拼接融合。
    Paper: H'e = fuse([DilatedConv(He_seq, d_l) for l in L])
    """
    def __init__(self, env_dim: int, out_dim: int,
                 n_scales: int = 4, kernel_size: int = 3):
        super().__init__()
        dilations = [1, 2, 4, 8][:n_scales]
        self.convs = nn.ModuleList([
            CausalConv1d(env_dim, out_dim, kernel_size, d) for d in dilations
        ])
        self.fuse = nn.Linear(out_dim * n_scales, out_dim)

    def forward(self, He_seq: torch.Tensor) -> torch.Tensor:
        """
        He_seq : [B, T, N, env_dim]
        returns: [B, N, out_dim]
        """
        B, T, N, C = He_seq.shape
        # 将 (B, N) 合并为 batch 维，时间维作为序列长度
        x = He_seq.permute(0, 2, 3, 1).reshape(B * N, C, T)  # [B*N, C, T]
        outs = [conv(x) for conv in self.convs]                # each [B*N, out_dim, T]
        fused = torch.cat(outs, dim=1)                         # [B*N, out_dim*scales, T]
        fused = fused[:, :, -1]                                # 取最后时间步 [B*N, out_dim*scales]
        out = self.fuse(fused).view(B, N, -1)                  # [B, N, out_dim]
        return F.relu(out)


# ---------------------------------------------------------------------------
# 7. Spatial Causal Gated Message Passing (paper Sec. IV-B, eq. 5-9)
# ---------------------------------------------------------------------------
class CausalGatingUnit(nn.Module):
    """
    为每条边 (i,j) 计算因果门控系数 g_ij ∈ [0,1]。
    Z_ij = [Hs_i || Hs_j || He_i || He_j]  (paper eq. 3-3)
    g_ij = σ(W2 · ReLU(W1·Z_ij + b1) + b2)  (paper eq. 3-4, 3-5)
    """
    def __init__(self, stoch_dim: int, env_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = 2 * stoch_dim + 2 * env_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),      nn.Sigmoid()
        )

    def forward(self, Hs_i, Hs_j, He_i, He_j):
        """All inputs: [B, E, dim] where E = number of edges."""
        Z = torch.cat([Hs_i, Hs_j, He_i, He_j], dim=-1)  # [B, E, 2Ds+2De]
        return self.gate_mlp(Z)                             # [B, E, 1]


class SCGMessagePassingLayer(nn.Module):
    """
    一层空间因果门控消息传递，实现 paper eq. (3-6)(3-7)：
      m_j→i^(causal) = g_ij ⊙ Θ_msg · Hs_j
      Hs_i^(new) = σ_agg(Hs_i + Σ_j m_j→i^(causal))

    [Bug-5 修复]
    原代码 SCG-MP 各层传入固定的初始 He，而论文中 gate 的意义是
    「用当前最精炼的环境表征来判断因果强度」。
    修复：SCGMP.forward() 接收 He_prime（经 MultiScaleContext 后的环境上下文），
    而非原始 He，从第一层起即使用质量更高的上下文来门控。
    """
    def __init__(self, stoch_dim: int, env_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gate_unit     = CausalGatingUnit(stoch_dim, env_dim, hidden_dim)
        self.msg_transform = nn.Linear(stoch_dim, stoch_dim, bias=False)
        self.agg_transform = nn.Sequential(
            nn.Linear(stoch_dim, stoch_dim), nn.ReLU()
        )

    def forward(self, Hs: torch.Tensor, He: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Hs         : [B, N, stoch_dim]
        He         : [B, N, env_dim]   – 用于门控的环境表征（传入 He_prime 更优）
        edge_index : [2, E]  row0=src(j), row1=dst(i)
        returns H's: [B, N, stoch_dim]
        """
        B, N, Ds = Hs.shape
        src, dst = edge_index[0], edge_index[1]   # j → i

        # 为每条边收集节点特征
        Hs_j = Hs[:, src]   # [B, E, Ds]
        Hs_i = Hs[:, dst]   # [B, E, Ds]
        He_j = He[:, src]   # [B, E, De]
        He_i = He[:, dst]   # [B, E, De]

        # 因果门控系数  (eq. 3-4, 3-5)
        g = self.gate_unit(Hs_i, Hs_j, He_i, He_j)  # [B, E, 1]

        # 原始消息  (eq. 3-6 中的 m_raw)
        m_raw = self.msg_transform(Hs_j)              # [B, E, Ds]

        # 门控消息  (eq. 3-6)
        m_causal = g * m_raw                          # [B, E, Ds]

        # 聚合到目标节点  (eq. 3-7)
        agg = torch.zeros(B, N, Ds, device=Hs.device)
        agg.scatter_add_(
            1,
            dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, Ds),
            m_causal
        )

        # 均值归一化（degree normalization），防止高度数节点特征尺度爆炸
        deg = torch.zeros(N, device=Hs.device)
        deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=Hs.device))
        deg = deg.view(1, N, 1).clamp(min=1.0)
        agg = agg / deg

        # 残差更新  (eq. 3-7)
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
        """
        He 建议传入经 MultiScaleContext 精炼后的 He_prime，
        以提供更高质量的因果判别信号。[Bug-5 修复]
        """
        for layer in self.layers:
            Hs = layer(Hs, He, edge_index)
        return Hs


# ---------------------------------------------------------------------------
# 8. Probabilistic Predictor  (paper Sec. 3.2.4, eq. 3-8)
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
        H_final : [B, N, De'+Ds']
        returns : mu [B, N, out_dim], sigma [B, N, out_dim] (sigma > 0)
        """
        H_final = self.norm(H_final)
        h       = self.net(H_final)
        mu      = self.mu_head(h)
        # softplus 保证正值，+1e-3 防止 sigma 趋近 0 导致 log(σ) 数值不稳定
        sigma   = F.softplus(self.sigma_head(h)) + 1e-3
        return mu, sigma


# ---------------------------------------------------------------------------
# 9. Loss Functions
# ---------------------------------------------------------------------------
def nll_gaussian_loss(mu: torch.Tensor, sigma: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
    """
    完整高斯负对数似然损失（paper eq. 3-9）：
      L_NLL = (1/T'·N·F_out) Σ [ log(2π σ²)/2 + (Y - μ)²/(2σ²) ]
            = (1/T'·N·F_out) Σ [ log(σ) + 0.5·log(2π) + (Y - μ)²/(2σ²) ]

    [Bug-3 修复]
    原代码使用 F.gaussian_nll_loss(eps=1e-2)，相当于对方差的下界裁剪到 0.01，
    等价于 sigma 最小 0.1，对归一化后的数据（sigma 通常 < 0.5）过于保守，
    压制了模型的不确定性精度。
    修复：手写完整 NLL，eps 改为 1e-6（仅用于数值安全，不影响正常范围的 sigma）。
    log(2π)/2 是常数，训练时不影响优化方向，但保留以和论文公式对齐。
    """
    eps = 1e-6
    sigma = sigma.clamp(min=eps)
    nll = torch.log(sigma) + 0.5 * math.log(2 * math.pi) + \
          0.5 * ((y - mu) / sigma) ** 2
    return nll.mean()


def crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor,
                  y: torch.Tensor) -> torch.Tensor:
    """
    高斯 CRPS 闭合解（仅用于评估，非训练 loss）：
    CRPS = σ · [z·(2Φ(z)-1) + 2φ(z) - 1/√π]，z=(y-μ)/σ
    """
    from torch.distributions import Normal
    dist = Normal(mu, sigma.clamp(min=1e-6))
    z   = (y - mu) / sigma.clamp(min=1e-6)
    phi = dist.log_prob(y).exp()
    Phi = dist.cdf(y)
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / math.sqrt(math.pi))
    return crps.mean()


# ---------------------------------------------------------------------------
# 10. Full GridCFN Model
# ---------------------------------------------------------------------------
class GridCFN(nn.Module):
    """
    完整 GridCFN 管道。

    超参数（默认对齐论文：λ=0.5, L_SCG=3, d_hidden=64）：
      in_dim      : 输入特征维度 F
      gcn_hidden  : GCN 隐藏层维度
      tcn_hidden  : TCN 输出维度 D
      env_dim     : He 维度 De
      stoch_dim   : Hs 维度 Ds
      ms_out_dim  : H'e 维度（MultiScaleContext 输出）
      n_scg_layers: L_SCG 层数
      out_dim     : 输出变量数 F_out（单步单变量为 1）
      lambda_mi   : MI 正则化权重 λ
    """
    def __init__(
        self,
        in_dim:       int   = 7,
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
        # [Bug-5 修复] SCGMP 接收 ms_out_dim 作为 env_dim（传入 He_prime 而非 He）
        self.scgmp        = SCGMP(stoch_dim, ms_out_dim, n_scg_layers)
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

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        x   : [B, T, N, F]
        adj : [N, N]  原始邻接矩阵（0/1 或加权，normalize_adj 内部处理）
        返回:
          mu     : [B, N, out_dim]
          sigma  : [B, N, out_dim]  (> 0)
          mi_loss: 标量，MINE 估计的 MI 下界（主网络训练目标之一：最小化）
        """
        adj_norm   = self.normalize_adj(adj)
        edge_index = self.adj_to_edge_index(adj)

        # ── Backbone ──────────────────────────────────────────────────────
        H = self.backbone(x, adj_norm)                # [B, T, N, D]

        # ── Causal Disentanglement ────────────────────────────────────────
        # [Bug-2 修复] Disentangler 内部完成完整时序的投影，不再在 forward 里重复调用
        He, Hs, He_seq = self.disentangler(H)
        # He    : [B, N, De]      – 最后帧环境表征
        # Hs    : [B, N, Ds]      – 最后帧随机实体表征
        # He_seq: [B, T, N, De]   – 完整时序环境表征

        # ── MI loss（供 MINE optimizer 最大化，供主 optimizer 最小化）────
        mi_loss = self.mine(He, Hs)

        # ── Multi-Scale Context ──────────────────────────────────────────
        He_prime = self.ms_context(He_seq)            # [B, N, ms_out_dim]

        # ── Spatial Causal Gated MP ──────────────────────────────────────
        # [Bug-5 修复] 传入 He_prime 而非原始 He，提供更精炼的因果判别信号
        Hs_prime = self.scgmp(Hs, He_prime, edge_index)  # [B, N, Ds]

        # ── Feature Fusion + Probabilistic Prediction ────────────────────
        H_final       = torch.cat([He_prime, Hs_prime], dim=-1)  # [B, N, ms+Ds]
        mu, sigma     = self.predictor(H_final)

        return mu, sigma, mi_loss

    def compute_loss(self, mu: torch.Tensor, sigma: torch.Tensor,
                     y: torch.Tensor, mi_loss: torch.Tensor,
                     lambda_mi: float = None) -> tuple:
        """
        总损失 = L_NLL + λ · L_MI  (paper eq. 3-10)

        [Bug-4 修复]
        增加 lambda_mi 参数，允许 train.py 动态传入（热身阶段传 0.0，
        之后传 self.lambda_mi），替代原来写死的 self.lambda_mi。

        y : [B, N, out_dim]
        """
        if lambda_mi is None:
            lambda_mi = self.lambda_mi
        l_nll   = nll_gaussian_loss(mu, sigma, y)
        l_total = l_nll + lambda_mi * mi_loss
        return l_total, l_nll, mi_loss

    def main_parameters(self):
        """返回除 MINE 之外的主网络参数（用于主 optimizer）。"""
        mine_ids = {id(p) for p in self.mine.parameters()}
        return [p for p in self.parameters() if id(p) not in mine_ids]

    def mine_parameters(self):
        """返回 MINE 网络参数（用于独立的 MINE optimizer）。"""
        return list(self.mine.parameters())
