"""
GridCFN: A Causal Spatio-Temporal Framework for Power Flow Uncertainty Prediction
CFM 版本 v5（仅升级 CFMVectorField 为 AdaLN，其余完全同 v4）

修复列表：
  Fix-CLUB     : 去掉 CLUBEstimator 中的 F.normalize()，logvar clamp 放宽到 (-6, 4)
  Fix-MSC      : MultiScaleContext 改回原版 3 路卷积 dilation=[1,7,30]，proj 输入 env_dim*3
  Fix-AdjOOM   : normalize_adj 用广播替代 D@adj@D，Weather(N=1866) 下节省约 1/3 显存
  Fix-AdjSelf  : normalize_adj 加自环前先 fill_diagonal_(0)，防止孤立节点补自环后度偏差
  Fix-GateDim  : SCGMessagePassingLayer 对边分块处理（chunk_edges=8192），Weather 防 OOM
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
# 5. CLUB Estimator
# Fix-CLUB: 移除 F.normalize()；logvar clamp 放宽到 (-6, 4)
# ---------------------------------------------------------------------------
class CLUBEstimator(nn.Module):
    """
    CLUB 互信息上界估计器。

    [Fix-CLUB]
    原版在 forward/variational_loss 中对 He/Hs 做 F.normalize()，
    投影到单位球面。Electricity 经 log1p+Z-score 后 He/Hs 方差极小，
    normalize 后所有向量几乎相同，pos≈neg，CLUB 塌缩（VarLoss≈-0.95 不动）。

    修复：直接使用原始 He/Hs，不做 normalize。
    logvar clamp 从 (-4,4) 放宽到 (-6,4)，支持小方差场景。

    对 Solar：影响很小（数据分布本已较规范）。
    对 Electricity：VarLoss 应从 -0.95 恢复到接近 0 的有效范围。
    对 Weather：同 Solar，影响小。
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
        # [Fix-CLUB] clamp 放宽到 (-6, 4)
        log_var  = torch.log(F.softplus(logvar_raw) + 1e-2).clamp(-6.0, 4.0)
        log_prob = -0.5 * (
            math.log(2 * math.pi) + log_var
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
        # [Fix-CLUB] 移除 F.normalize，直接 reshape
        M        = He.shape[0] * He.shape[1]
        He_flat  = He.reshape(M, -1)
        Hs_flat  = Hs.reshape(M, -1)
        mu_q, logvar_raw = self._get_params(He_flat)
        pos = self._log_prob(Hs_flat,                               mu_q, logvar_raw).mean()
        neg = self._log_prob(Hs_flat[self._neg_perm(M, He.device)], mu_q, logvar_raw).mean()
        return pos - neg

    def variational_loss(self, He, Hs):
        # [Fix-CLUB] 移除 F.normalize
        M        = He.shape[0] * He.shape[1]
        He_flat  = He.reshape(M, -1)
        Hs_flat  = Hs.reshape(M, -1)
        mu_q, logvar_raw = self._get_params(He_flat)
        return -self._log_prob(Hs_flat, mu_q, logvar_raw).mean()


# ---------------------------------------------------------------------------
# 6. Multi-Scale Context
# Fix-MSC: 改回原版 3 路卷积 dilation=[1,7,30]，proj 输入 env_dim*3
# ---------------------------------------------------------------------------
class MultiScaleContext(nn.Module):
    """
    [Fix-MSC]
    修复版错误地使用了 4 路卷积（dilation=[1,2,4,8]）和 env_dim*4 的投影层，
    与原版（3路，dilation=[1,7,30]，env_dim*3）不一致，导致：
      1. cond_dim 变化（96→128），权重无法与旧版互相加载
      2. dilation=[1,2,4,8] 的设计对应 TCN 的指数增长，但 MultiScale 的目标
         是捕捉小时/天/周等实际时间尺度，dilation=[1,7,30] 更贴合论文意图

    恢复为原版设计：3路 dilation=[1,7,30]，proj: env_dim*3 → ms_out_dim

    [v6-Dilation] 支持可配置 dilations，不同数据集分辨率对应不同的时间尺度：
      - Solar (10min):  → 10min / 日(1440min) / 周(10080min)
      - Electricity/Weather (1h):  → 1h / 日(24h) / 周(168h)
    默认保留原版 [1, 7, 30] 以向后兼容。
    """
    def __init__(self, env_dim, ms_out_dim, dilations=(1, 7, 30)):
        super().__init__()
        self.dilations = list(dilations)
        self.convs = nn.ModuleList([
            CausalConv1d(env_dim, env_dim, kernel_size=3, dilation=d)
            for d in self.dilations
        ])
        # proj 输入维度 = env_dim * n_scales
        self.proj = nn.Linear(env_dim * len(self.dilations), ms_out_dim)

    def forward(self, He_seq):
        """He_seq: [B, T, N, env_dim] → [B, N, ms_out_dim]"""
        B, T, N, De = He_seq.shape
        x    = He_seq.permute(0, 2, 3, 1).reshape(B*N, De, T)
        outs = [conv(x)[:, :, -1] for conv in self.convs]   # 取最后时间步
        return self.proj(torch.cat(outs, dim=-1)).reshape(B, N, -1)


# ---------------------------------------------------------------------------
# 7. SCG Message Passing
# Fix-GateDim: 对边分块处理，防止 Weather(N=1866, E~200k) 下 OOM
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
    [Fix-GateDim]
    原版对所有边一次性计算 gate，创建 [B, E, dim] 张量。
    Weather: B=4, E≈200k, concat_dim=128 → 约 410MB/层，3层 SCG-MP × 梯度 ≈ 2.4GB，OOM。

    修复：对边进行分块（chunk）处理，每次只处理 chunk_size 条边。
    chunk_size=8192 时：B=4, chunk=8192, dim=128 → 约 16MB/chunk，安全。
    Solar(E=34512) 和 Electricity(E=34512) 边数适中，分块开销可忽略。
    """
    def __init__(self, stoch_dim, env_dim, hidden_dim=64, chunk_size: int = 8192):
        super().__init__()
        self.gate      = CausalGateUnit(stoch_dim, env_dim, hidden_dim)
        self.msg_tr    = nn.Linear(stoch_dim, stoch_dim)
        self.agg_norm  = nn.LayerNorm(stoch_dim)
        self.agg_tr    = nn.Sequential(nn.Linear(stoch_dim, stoch_dim), nn.ReLU())
        self.chunk_size = chunk_size

    def forward(self, Hs, He, edge_index):
        B, N, Ds = Hs.shape
        src, dst = edge_index[0], edge_index[1]
        E        = src.shape[0]
        agg      = torch.zeros(B, N, Ds, device=Hs.device, dtype=Hs.dtype)

        # [Fix-GateDim] 分块处理，每次处理 chunk_size 条边
        for start in range(0, E, self.chunk_size):
            end   = min(start + self.chunk_size, E)
            s_idx = src[start:end]
            d_idx = dst[start:end]

            g = self.gate(Hs[:, d_idx], Hs[:, s_idx],
                          He[:, d_idx], He[:, s_idx])          # [B, chunk, 1]
            m = g * self.msg_tr(Hs[:, s_idx])                  # [B, chunk, Ds]
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
# 8. CFM Vector Field（v6: 双流 AdaLN，He→shift，Hs→scale）
# ---------------------------------------------------------------------------
class CFMVectorField(nn.Module):
    """
    条件向量场 v_θ(x_t, t | He_prime, Hs_prime)。

    v5 问题：
      He_prime（环境背景）和 Hs_prime（内生随机）无差别拼接为 c，
      再共同投影为 scale/shift，CFM 无法区分"分布中心"和"分布宽度"
      两类信号，Hs 的随机性信息容易被 He 的确定性信号淹没。

    v6 改动（双流 AdaLN）：
      [DualStream]
        · He_prime → env_proj   → (env_shift)        控制分布中心（shift）
        · Hs_prime → stoch_proj → (stoch_scale_extra) 控制分布宽度（scale）
        · t_emb    → time_proj  → 基础 (scale, shift)（时间步信号）
      融合方式（每个残差块）：
        scale = 1 + time_s + stoch_scale_extra   # Hs 决定"有多宽"
        shift =     time_b + env_shift            # He 决定"中心在哪"
      物理含义：
        · 环境因素告诉模型"预测分布的中心在哪"（shift）
        · 内生随机因素告诉模型"分布有多宽"（额外 scale 扰动）
        · 二者在生成过程中扮演不同角色，对应论文核心贡献叙事

    接口变化：
      v5: forward(x_t, t, c)              c = cat([He_prime, Hs_prime])
      v6: forward(x_t, t, He_prime, Hs_prime)  两流分开传入
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

        # 时间嵌入 → 每个残差块的基础 (scale, shift)，共 hidden*6
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )

        # [DualStream] He 流：环境背景 → shift（控制分布中心）
        # 输出 hidden*3：每个残差块一个 env_shift
        self.env_proj = nn.Sequential(
            nn.LayerNorm(env_dim),
            nn.Linear(env_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),   # b1_env, b2_env, b3_env
        )

        # [DualStream] Hs 流：内生随机 → extra scale（控制分布宽度）
        # 输出 hidden*3：每个残差块一个额外 scale
        self.stoch_proj = nn.Sequential(
            nn.LayerNorm(stoch_dim),
            nn.Linear(stoch_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),   # s1_st, s2_st, s3_st
        )

        # x_t 单独投影
        self.input_proj = nn.Linear(out_dim, hidden_dim)

        # 3 个残差块
        self.layer1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())

        self.out_proj = nn.Linear(hidden_dim, out_dim)

        # [ZeroInit] 零初始化输出层，稳定早期训练
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
        He_prime : [B, N, env_dim]    （来自 MultiScaleContext，控制分布中心）
        Hs_prime : [B, N, stoch_dim]  （来自 SCGMP，控制分布宽度）
        """
        B, N, _ = x_t.shape

        # ── 时间嵌入 → 6份基础 AdaLN 参数 ─────────────────────────────────
        t_emb = self._time_embed(t, B, N)                        # [B, N, time_emb_dim]
        t_s1, t_b1, t_s2, t_b2, t_s3, t_b3 = \
            self.time_proj(t_emb).chunk(6, dim=-1)               # 各 [B, N, hidden]

        # ── He 流：环境背景 → shift（分布中心）─────────────────────────────
        b1_env, b2_env, b3_env = \
            self.env_proj(He_prime).chunk(3, dim=-1)             # 各 [B, N, hidden]

        # ── Hs 流：内生随机 → extra scale（分布宽度）──────────────────────
        s1_st, s2_st, s3_st = \
            self.stoch_proj(Hs_prime).chunk(3, dim=-1)           # 各 [B, N, hidden]

        # ── x_t 投影 ─────────────────────────────────────────────────────
        h = self.input_proj(x_t)                                 # [B, N, hidden]

        # ── 残差块 1：Hs决定宽度，He决定中心 ─────────────────────────────
        h_norm = F.layer_norm(h, [h.shape[-1]])
        scale1 = 1.0 + t_s1 + s1_st   # 时间基础scale + Hs额外scale
        shift1 = t_b1 + b1_env         # 时间基础shift + He环境shift
        h = h + self.layer1(h_norm * scale1 + shift1)

        # ── 残差块 2 ─────────────────────────────────────────────────────
        h_norm = F.layer_norm(h, [h.shape[-1]])
        scale2 = 1.0 + t_s2 + s2_st
        shift2 = t_b2 + b2_env
        h = h + self.layer2(h_norm * scale2 + shift2)

        # ── 残差块 3 ─────────────────────────────────────────────────────
        h_norm = F.layer_norm(h, [h.shape[-1]])
        scale3 = 1.0 + t_s3 + s3_st
        shift3 = t_b3 + b3_env
        h = h + self.layer3(h_norm * scale3 + shift3)

        return self.out_proj(h)                                  # [B, N, out_dim]


# ---------------------------------------------------------------------------
# 9. GridCFN（CFM 版 v6，双流条件注入 + 可配置 dilation）
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
        ms_dilations=(1, 7, 30),   # [v6-Dilation] 可按数据集分辨率配置
    ):
        super().__init__()
        self.lambda_mi = lambda_mi
        self.out_dim   = out_dim
        self.env_dim   = env_dim    # [v6-DualStream] 保留供 sample/cfm_loss 传参
        self.stoch_dim = stoch_dim

        self.backbone     = Backbone(in_dim, gcn_hidden, tcn_hidden, gcn_layers, tcn_layers)
        self.disentangler = CausalDisentangler(tcn_hidden, env_dim, stoch_dim)
        self.club         = CLUBEstimator(env_dim, stoch_dim)
        # [v6-Dilation] 透传 ms_dilations，支持各数据集的物理时间尺度
        self.ms_context   = MultiScaleContext(env_dim, ms_out_dim, dilations=ms_dilations)
        self.scgmp        = SCGMP(stoch_dim, env_dim, n_scg_layers, chunk_size=chunk_size)
        # [v6-DualStream] He_prime 和 Hs_prime 分开传入，不再拼接为 cond_dim
        self.vector_field = CFMVectorField(
            out_dim=out_dim, env_dim=ms_out_dim, stoch_dim=stoch_dim,
            hidden_dim=cfm_hidden, time_emb_dim=cfm_time_emb_dim,
        )

    @staticmethod
    def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """
        对称归一化：D^{-1/2} A_hat D^{-1/2}，A_hat = A + I

        [Fix-AdjSelf] 先 fill_diagonal_(0) 清零已有自环，再统一加 I，
        防止孤立节点补自环后 degree=2 导致归一化偏差。

        [Fix-AdjOOM] 用广播替代 torch.diag(d) @ adj @ torch.diag(d)：
          原版：两次矩阵乘，中间需构建 N×N 的 D 矩阵
            D = diag(d_inv_sq)  → N×N 稠密矩阵
            D @ adj → N×N matmul
            结果 @ D → N×N matmul
          新版：直接广播乘，无需构建 D 矩阵：
            d_inv_sq[:, None] * adj * d_inv_sq[None, :]
          Weather(N=1866): 节省 1866×1866×4B≈13MB 的 D 矩阵，以及两次 O(N²) matmul
        """
        adj = adj.clone()
        adj.fill_diagonal_(0)                                    # [Fix-AdjSelf] 清零已有自环
        adj = adj + torch.eye(adj.size(0), device=adj.device)   # 统一加 I
        deg      = adj.sum(dim=1)
        d_inv_sq = deg.clamp(min=1e-8).pow(-0.5)
        # [Fix-AdjOOM] 广播替代 diag @ @ diag
        return d_inv_sq.unsqueeze(1) * adj * d_inv_sq.unsqueeze(0)

    @staticmethod
    def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
        return adj.nonzero(as_tuple=False).t().contiguous()

    def forward(self, x, adj_norm, edge_index):
        H              = self.backbone(x, adj_norm)
        He, Hs, He_seq = self.disentangler(H)
        mi_loss        = self.club(He, Hs)
        He_prime       = self.ms_context(He_seq)           # [B, N, ms_out_dim]
        Hs_prime       = self.scgmp(Hs, He, edge_index)   # [B, N, stoch_dim]
        # [v6-DualStream] 返回分开的 He_prime 和 Hs_prime，供 cfm_loss/sample 双流传入
        return He_prime, Hs_prime, He, Hs, mi_loss

    def cfm_loss(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor,
                 y_target: torch.Tensor,
                 n_t_samples: int = 4,
                 sigma_min: float = 0.01) -> torch.Tensor:
        """
        [Fix-Sigma] 使用 sigma_min 缩放的 OT-CFM 路径，替代原始线性插值。

        原版问题：
          x_t = (1-t)*x0 + t*y，u_t = y - x0
          在 t→0 时 x_t 完全由 x0 决定，在 t→1 时完全由 y 决定。
          向量场目标幅值 = ||y - x0||，在重尾数据（Electricity）下幅值极大，
          模型学到的向量场偏向均值，采样粒子 spread 系统性偏小，PICP 严重不足。

        修复（OT-CFM with sigma_min，Flow Matching 论文标准做法）：
          x_t = (1 - (1 - sigma_min)*t)*x0 + t*y
          u_t = y - (1 - sigma_min)*x0
          sigma_min > 0 保证 t=1 时 x_1 = sigma_min*x0 + y ≠ y，
          条件分布不退化为 delta，采样自然保留 spread。
          sigma_min=0.01 是标准推荐值（Flow Matching 原论文 §3.3）。

        [v6-DualStream] He_prime 和 Hs_prime 分开传入向量场，不再拼接。
        """
        assert y_target.shape[-1] == self.out_dim, (
            f"y_target 末维 {y_target.shape[-1]} ≠ out_dim={self.out_dim}"
        )
        B, N, _ = y_target.shape
        device  = y_target.device
        losses  = []
        for _ in range(n_t_samples):
            x0    = torch.randn_like(y_target)
            t     = torch.rand(B, device=device)
            t_bc  = t.reshape(B, 1, 1)
            # [Fix-Sigma] OT-CFM 路径
            x_t   = (1.0 - (1.0 - sigma_min) * t_bc) * x0 + t_bc * y_target
            u_t   = y_target - (1.0 - sigma_min) * x0
            # [v6-DualStream] 双流传入
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
        并行 ODE 采样（v6：双流条件 + OT-CFM sigma_min 对齐）。

        [Fix-Sigma] 采样时与训练路径保持一致：
          - 初始噪声 x0 乘以 x0_scale（默认1.0，可在 config 里调大补偿 spread）
          - Euler 积分步长不变

        [v6-DualStream] He_prime 和 Hs_prime 分开 repeat_interleave 后传入向量场。

        x0_scale 调参建议：
          - 如果 PICP 仍然偏低（< 0.90），把 x0_scale 从 1.0 逐步升到 1.5

        返回：[n_samples, B, N, out_dim]
        """
        B, N, _ = He_prime.shape
        S       = n_samples
        device  = He_prime.device
        dt      = 1.0 / n_steps

        # [v6-DualStream] 两路条件分别扩展 S 倍
        he = He_prime.detach().repeat_interleave(S, dim=0)     # [B*S, N, ms_out_dim]
        hs = Hs_prime.detach().repeat_interleave(S, dim=0)     # [B*S, N, stoch_dim]

        # [Fix-Sigma] x0_scale 控制初始噪声幅值，补偿 spread 欠估计
        x = torch.randn(B * S, N, self.out_dim, device=device) * x0_scale

        for step in range(n_steps):
            t_val = step * dt
            t_vec = torch.full((B * S,), t_val, device=device, dtype=torch.float32)
            # [v6-DualStream] 双流传入
            x = x + dt * self.vector_field(x, t_vec, he, hs)

        return x.reshape(B, S, N, self.out_dim).permute(1, 0, 2, 3).contiguous()