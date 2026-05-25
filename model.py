"""
GridCFN-DMSD: Dual-track Multi-Scale Disentanglement
（在多步预测版基础上引入 DMSD 架构改进）

改进说明（相对多步版）：
  1. FrequencyDecomposer：可学习多尺度移动平均，将输入分解为趋势支路（X_low）
     和残差支路（X_high），截止频率由数据自适应决定。
  2. LowRankGCN：邻接矩阵参数化为 U@U^T（rank-r），强制只学习全局同步模式。
     配合显式秩正则 L_rank 防止退化。
  3. SparseGCN：可微稀疏化（软阈值 sigmoid）替代硬 Top-K，梯度稳定。
     SDWPF 可选叠加风向先验掩码。
  4. DualTrackBackbone：X_low → LowRankGCN+TCN_e → He_seq/He；
     X_high → SparseGCN+TCN_s → Hs。各自独立 LayerNorm，独立 TCN 参数。
  5. 移除 CausalDisentangler：解耦职责提前到前端架构，不再依赖后端线性层拆分。
  6. CLUBEstimator 改为两个独立实例：
       club_e: MI(He, X_low_pooled)  — 环境表征不应含高频残差信息
       club_s: MI(Hs, X_high_pooled) — 因果表征不应含低频趋势信息
  7. MultiScaleContext / SCGMP / CFMVectorField 接口完全不变。

架构数据流：
  X → FrequencyDecomposer → X_low, X_high
  X_low  → LowRankGCN + TCN_e → He_seq [B,T,N,env_dim]
                               → He     [B,N,env_dim]   (注意力时间池化)
  X_high → SparseGCN  + TCN_s → Hs     [B,N,stoch_dim] (注意力时间池化)
  He_seq → MultiScaleContext   → He_prime [B,N,ms_out_dim]
  Hs,He  → SCGMP               → Hs_prime [B,N,stoch_dim]
  He_prime, Hs_prime → CFMVectorField → 速度场
  CLUB: MI(He, X_low_pooled) + MI(Hs, X_high_pooled)  (一致性惩罚)
  L_rank: -log det(U^T U + εI)  (低秩正则)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 1. FrequencyDecomposer — 可学习多尺度趋势/残差分解
# ---------------------------------------------------------------------------

class FrequencyDecomposer(nn.Module):
    """
    将输入序列分解为趋势支路（X_low）和残差支路（X_high）。

    做法：
      - 提供 4 个候选移动平均窗口（由 candidates 指定）
      - 权重由输入全局均值经线性层 + softmax 动态生成
      - X_low  = sum_k(w_k * MA_k(X))
      - X_high = X - X_low

    候选窗口建议：
      Solar/SDWPF (10min 粒度): candidates=[12, 24, 48, 96]
      Electricity (1h 粒度):    candidates=[6, 12, 24, 48]

    填充方式用反射填充（reflect），避免边缘趋势估计偏低。
    """

    def __init__(self, in_dim: int, candidates=(12, 24, 48, 96)):
        super().__init__()
        self.candidates = list(candidates)
        n = len(candidates)
        # 权重生成：全局 mean → Linear → softmax
        self.weight_net = nn.Linear(in_dim, n)
        # 独立 LayerNorm，不共享参数
        self.norm_low  = nn.LayerNorm(in_dim)
        self.norm_high = nn.LayerNorm(in_dim)

    def _moving_avg(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        x: [B*N, T, F]，对时间维做 kernel=k 的均值池化（反射填充）。
        返回同形状。

        修复说明：之前 pad_l/pad_r 被截断后，k_eff = pad_l + pad_r + 1 不再等于
        原始 k，导致移动平均窗口缩水（极端情况下退化为 k_eff=1，频率分流失效）。
        正确做法：先用截断后的 pad 做填充，然后始终用原始 k 做 avg_pool1d，
        输出长度 = padded_T - k + 1，最后裁剪/复制对齐到 T_len。
        """
        if k <= 1:
            return x
        T_len = x.shape[1]   # x: [B*N, T, F]
        # 反射填充要求 padding < input_size，保护性截断
        pad_l = min((k - 1) // 2,       T_len - 1)
        pad_r = min(k - 1 - (k - 1) // 2, T_len - 1)
        x_t = x.permute(0, 2, 1)           # [B*N, F, T]
        x_t = F.pad(x_t, (pad_l, pad_r), mode="reflect")
        # 始终使用原始 k 而非截断后的 k_eff，保证每个候选窗口语义不同
        k_actual = min(k, x_t.shape[-1])
        x_t = F.avg_pool1d(x_t, kernel_size=k_actual, stride=1, padding=0)
        # 输出长度 = padded_T - k_actual + 1，对齐到原始 T_len
        if x_t.shape[-1] > T_len:
            x_t = x_t[:, :, :T_len]
        elif x_t.shape[-1] < T_len:
            x_t = F.pad(x_t, (0, T_len - x_t.shape[-1]), mode="replicate")
        return x_t.permute(0, 2, 1)        # [B*N, T, F]

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, N, F]
        返回: X_low_norm [B,T,N,F], X_high_norm [B,T,N,F]
        """
        B, T, N, Fin = x.shape
        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, Fin)  # [B*N, T, Fin]

        # 全局均值生成候选权重
        ctx = x_bn.mean(dim=1)                                # [B*N, Fin]
        w   = F.softmax(self.weight_net(ctx), dim=-1)         # [B*N, n_cand]

        # 加权融合各候选 MA
        ma_list = [self._moving_avg(x_bn, k) for k in self.candidates]
        ma_stack = torch.stack(ma_list, dim=-1)              # [B*N, T, F, n_cand]
        w_bc     = w.unsqueeze(1).unsqueeze(2)               # [B*N, 1, 1, n_cand]
        x_low_bn = (ma_stack * w_bc).sum(dim=-1)             # [B*N, T, F]

        x_high_bn = x_bn - x_low_bn                         # [B*N, T, F]

        # reshape 回 [B, T, N, Fin]
        x_low  = x_low_bn.reshape(B, N, T, Fin).permute(0, 2, 1, 3)
        x_high = x_high_bn.reshape(B, N, T, Fin).permute(0, 2, 1, 3)

        # 独立 LayerNorm（防止两路能量差异导致梯度不平衡）
        return self.norm_low(x_low), self.norm_high(x_high)


# ---------------------------------------------------------------------------
# 2. LowRankGCN — 低秩邻接矩阵图卷积（环境支路）
# ---------------------------------------------------------------------------

class LowRankGCN(nn.Module):
    """
    邻接矩阵参数化为 A = softmax(U @ U^T / sqrt(r))，rank = r。
    强制只能表达 r 个独立扩散模式，天然捕捉全局同步集群。

    秩正则项由 rank_loss() 方法返回，需在训练损失中加入：
        L_rank = -log(det(U^T @ U + ε·I))
    """

    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int, out_dim: int,
                 rank_r: int = 8, n_layers: int = 2):
        super().__init__()
        self.rank_r   = rank_r
        self.n_nodes  = n_nodes
        self.U = nn.Parameter(torch.randn(n_nodes, rank_r) * 0.1)

        # in → hidden（中间层）→ out（最后层）
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(n_layers)
        ])

    def _adj(self) -> torch.Tensor:
        """计算低秩邻接矩阵 [N, N]，行 softmax 归一化。"""
        A = self.U @ self.U.T / math.sqrt(self.rank_r)  # [N, N]
        return F.softmax(A, dim=-1)

    def rank_loss(self) -> torch.Tensor:
        """秩正则：鼓励 U 的列向量线性独立，防止低秩退化为秩-1。"""
        G   = self.U.T @ self.U                           # [r, r]
        eps = 1e-4 * torch.eye(self.rank_r, device=self.U.device)
        # log det(G + εI)，优先用 Cholesky（数值更稳定）
        try:
            L      = torch.linalg.cholesky(G + eps)
            logdet = 2.0 * L.diagonal().log().sum()
        except Exception:
            logdet = torch.logdet(G + eps)
        # 防止数值异常（nan/inf）静默传播到总 loss 导致训练崩溃；
        # 返回 0 相当于本 step 跳过秩正则，比 nan 蔓延代价小。
        if not torch.isfinite(logdet):
            return G.new_tensor(0.0)
        return -logdet

    def forward(self, x: torch.Tensor, wind_mask=None) -> torch.Tensor:
        """
        x: [B*T, N, F]
        返回: [B*T, N, out_dim]
        wind_mask 参数保留接口一致性，低秩通道不使用。
        """
        A = self._adj()
        for i, layer in enumerate(self.layers):
            x = torch.matmul(A, layer(x))
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# 3. SparseGCN — 可微稀疏邻接矩阵图卷积（因果支路）
# ---------------------------------------------------------------------------

class SparseGCN(nn.Module):
    """
    可微稀疏化替代硬 Top-K，梯度连续。

    A_sparse = A_raw * sigmoid((A_raw - threshold) / temperature)
    行归一化后做图卷积。

    threshold 和 temperature 均为可学习标量：
      - threshold 初始化为 0（中性），训练中自适应
      - temperature 初始化为 1.0，训练中自然衰减趋向稀疏

    SDWPF 可选传入 wind_mask [N, N]（上风向为 1，下风向为 0.1），
    叠加物理先验；其他数据集不传，默认 None（等价于全 1 mask）。
    """

    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, emb_dim: int = 32):
        super().__init__()
        self.n_nodes   = n_nodes
        # 节点 embedding 用独立 emb_dim，不与 out_dim 耦合
        self.emb       = nn.Embedding(n_nodes, emb_dim)
        self.threshold   = nn.Parameter(torch.zeros(1))
        self.temperature = nn.Parameter(torch.ones(1))

        # in → hidden（中间层）→ out（最后层）
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(n_layers)
        ])

    def _adj(self, device, wind_mask=None) -> torch.Tensor:
        """计算可微稀疏邻接矩阵 [N, N]。"""
        idx   = torch.arange(self.n_nodes, device=device)
        E     = self.emb(idx)                                    # [N, emb_dim]
        A_raw = E @ E.T / math.sqrt(E.shape[-1])                 # [N, N]

        temp = self.temperature.abs().clamp(min=1e-3)            # 防止除零
        A    = A_raw * torch.sigmoid((A_raw - self.threshold) / temp)

        if wind_mask is not None:
            A = A * wind_mask.to(device)

        # 行归一化
        row_sum = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return A / row_sum

    def forward(self, x: torch.Tensor, wind_mask=None) -> torch.Tensor:
        """
        x: [B*T, N, F]
        返回: [B*T, N, out_dim]
        """
        A = self._adj(x.device, wind_mask)
        for i, layer in enumerate(self.layers):
            x = torch.matmul(A, layer(x))
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# 4. TCN（保留，不变）
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
            [TCNBlock(hidden_dim, kernel_size, 2 ** i) for i in range(n_layers)]
        )

    def forward(self, x):
        """x: [B,N,T,C] → [B,N,T,hidden]"""
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C).permute(0, 2, 1)
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        return x.permute(0, 2, 1).reshape(B, N, T, -1)


# ---------------------------------------------------------------------------
# 5. AttentionPool — 可学习注意力时间池化（供双轨共用）
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """
    对 [B, T, N, D] 的时间维做注意力加权池化，输出 [B, N, D]。
    与原 CausalDisentangler 中的池化逻辑一致，抽出为独立模块供复用。
    """
    def __init__(self, dim: int):
        super().__init__()
        hidden = max(1, dim // 2)
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N, D] → [B, N, D]"""
        score  = self.attn(x)                      # [B, T, N, 1]
        weight = F.softmax(score, dim=1)
        return (weight * x).sum(dim=1)             # [B, N, D]


# ---------------------------------------------------------------------------
# 6. DualTrackBackbone — 双轨 Backbone（替换原 Backbone + CausalDisentangler）
# ---------------------------------------------------------------------------

class DualTrackBackbone(nn.Module):
    """
    完整的双轨特征提取，输出与原 Backbone+Disentangler 接口等价：
      He_seq : [B, T, N, env_dim]   — 供 MultiScaleContext（带梯度）
      He     : [B, N, env_dim]      — 供 SCGMP 和 CLUB_e（注意力池化）
      Hs     : [B, N, stoch_dim]    — 供 SCGMP 和 CLUB_s（注意力池化）

    同时返回 X_low/X_high 的时间均值池化 + 线性投影，供 CLUB 一致性约束使用：
      X_low_pooled  : [B, N, env_dim]
      X_high_pooled : [B, N, stoch_dim]
    """

    def __init__(self, n_nodes: int, in_dim: int,
                 env_dim: int, stoch_dim: int,
                 gcn_hidden: int = 64, tcn_hidden: int = 64,
                 gcn_layers: int = 2, tcn_layers: int = 4,
                 rank_r: int = 8,
                 freq_candidates: tuple = (12, 24, 48, 96),
                 wind_mask=None):
        super().__init__()

        # 频率分流
        self.decomposer = FrequencyDecomposer(in_dim, candidates=freq_candidates)

        # 环境支路：LowRankGCN 内部经过 gcn_hidden 再投影到 env_dim，TCN 在 tcn_hidden 宽度运行
        self.gcn_e   = LowRankGCN(n_nodes, in_dim, gcn_hidden, env_dim,
                                   rank_r=rank_r, n_layers=gcn_layers)
        self.tcn_e   = TCN(in_dim=env_dim, hidden_dim=tcn_hidden, n_layers=tcn_layers)
        # TCN 输出是 tcn_hidden，需要投影回 env_dim 供下游使用
        self.proj_tcn_e = nn.Linear(tcn_hidden, env_dim) if tcn_hidden != env_dim else nn.Identity()
        self.pool_e  = AttentionPool(env_dim)

        # 因果支路：SparseGCN 内部经过 gcn_hidden 再投影到 stoch_dim
        self.gcn_s   = SparseGCN(n_nodes, in_dim, gcn_hidden, stoch_dim, n_layers=gcn_layers)
        self.tcn_s   = TCN(in_dim=stoch_dim, hidden_dim=tcn_hidden, n_layers=tcn_layers)
        self.proj_tcn_s = nn.Linear(tcn_hidden, stoch_dim) if tcn_hidden != stoch_dim else nn.Identity()
        self.pool_s  = AttentionPool(stoch_dim)

        # X_low / X_high 的时间均值池化后的线性投影，供 CLUB 使用
        self.proj_low  = nn.Linear(in_dim, env_dim)
        self.proj_high = nn.Linear(in_dim, stoch_dim)

        # 风向掩码（仅 SDWPF 非 None）
        if wind_mask is not None:
            self.register_buffer("wind_mask", wind_mask)
        else:
            self.wind_mask = None

    def rank_loss(self) -> torch.Tensor:
        """代理 LowRankGCN 的秩正则，由 GridCFN.rank_loss() 调用。"""
        return self.gcn_e.rank_loss()

    def forward(self, x: torch.Tensor):
        """
        x: [B, T_in, N, F]
        返回:
          He_seq        : [B, T, N, env_dim]
          He            : [B, N, env_dim]
          Hs            : [B, N, stoch_dim]
          X_low_pooled  : [B, N, env_dim]
          X_high_pooled : [B, N, stoch_dim]
        """
        B, T, N, Fin = x.shape

        # ── 频率分流 ──────────────────────────────────────────────────────
        X_low, X_high = self.decomposer(x)       # [B,T,N,Fin] × 2，已 LayerNorm

        # ── CLUB 用的参考表征：时间均值池化 + 线性投影 ────────────────────
        X_low_pooled  = self.proj_low(X_low.mean(dim=1))    # [B, N, env_dim]
        X_high_pooled = self.proj_high(X_high.mean(dim=1))  # [B, N, stoch_dim]

        # ── 环境支路：LowRankGCN → TCN_e → proj_tcn_e → He_seq / He ──────
        e_gcn = self.gcn_e(
            X_low.reshape(B * T, N, Fin), wind_mask=None
        ).reshape(B, T, N, -1)                              # [B,T,N,env_dim]
        # TCN 输出是 tcn_hidden，proj_tcn_e 投影回 env_dim
        He_seq_raw = self.tcn_e(e_gcn.permute(0, 2, 1, 3)  # [B,N,T,env_dim]
                                ).permute(0, 2, 1, 3)       # [B,T,N,tcn_hidden]
        He_seq = self.proj_tcn_e(He_seq_raw)                # [B,T,N,env_dim]
        He = self.pool_e(He_seq)                            # [B,N,env_dim]

        # ── 因果支路：SparseGCN → TCN_s → proj_tcn_s → Hs ──────────────
        s_gcn = self.gcn_s(
            X_high.reshape(B * T, N, Fin), wind_mask=self.wind_mask
        ).reshape(B, T, N, -1)                              # [B,T,N,stoch_dim]
        Hs_seq_raw = self.tcn_s(s_gcn.permute(0, 2, 1, 3)
                                ).permute(0, 2, 1, 3)       # [B,T,N,tcn_hidden]
        Hs_seq = self.proj_tcn_s(Hs_seq_raw)                # [B,T,N,stoch_dim]
        Hs = self.pool_s(Hs_seq)                            # [B,N,stoch_dim]

        return He_seq, He, Hs, X_low_pooled, X_high_pooled


# ---------------------------------------------------------------------------
# 7. CLUBEstimator（保留，接口不变，实例化两个）
# ---------------------------------------------------------------------------

class CLUBEstimator(nn.Module):
    """
    CLUB 互信息上界估计器（Cheng et al., 2020）。
    接口不变，由 GridCFN 实例化两个：
      club_e: MI(He, X_low_pooled)   — env 表征不应含高频残差
      club_s: MI(Hs, X_high_pooled)  — stoch 表征不应含低频趋势
    输入维度分别为 (env_dim, env_dim) 和 (stoch_dim, stoch_dim)。
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.var_net_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
        )
        self.var_net_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
            nn.Tanh(),
        )

    def _get_params(self, x_flat):
        return self.var_net_mu(x_flat), self.var_net_logvar(x_flat)

    def _log_prob(self, y, mu_q, logvar_raw):
        log_var  = logvar_raw.clamp(-6.0, 4.0)
        log_prob = -0.5 * (
            math.log(2 * math.pi) + log_var
            + (y - mu_q).pow(2) / log_var.exp()
        )
        return log_prob.mean(dim=-1)

    def _neg_perm(self, M, device):
        perm = torch.randperm(M, device=device)
        same = perm == torch.arange(M, device=device)
        if same.any() and M > 1:
            idx  = same.nonzero(as_tuple=True)[0]
            swap = (idx + 1) % M
            # 使用临时变量避免批量赋值时的读写竞争：
            # 当 idx/swap 为多元素 tensor 时，perm[idx], perm[swap] = perm[swap], perm[idx]
            # 的右侧求值顺序未定义，可能导致部分元素被覆盖后再读。
            tmp          = perm[swap].clone()
            perm[swap]   = perm[idx].clone()
            perm[idx]    = tmp
        return perm

    def forward(self, x, y):
        """
        x: [B, N, x_dim]，y: [B, N, y_dim]
        返回 CLUB 互信息上界估计（标量）。
        """
        M            = x.shape[0] * x.shape[1]
        x_flat       = x.reshape(M, -1)
        y_flat       = y.reshape(M, -1)
        mu_q, lv_raw = self._get_params(x_flat)
        pos = self._log_prob(y_flat, mu_q, lv_raw).mean()
        neg = self._log_prob(y_flat[self._neg_perm(M, x.device)],
                             mu_q, lv_raw).mean()
        return pos - neg

    def variational_loss(self, x, y):
        M            = x.shape[0] * x.shape[1]
        x_flat       = x.reshape(M, -1)
        y_flat       = y.reshape(M, -1)
        mu_q, lv_raw = self._get_params(x_flat)
        return -self._log_prob(y_flat, mu_q, lv_raw).mean()


# ---------------------------------------------------------------------------
# 8. MultiScaleContext（保留，不变）
# ---------------------------------------------------------------------------

class MultiScaleContext(nn.Module):
    """多尺度时间上下文提取。"""

    def __init__(self, env_dim, ms_out_dim, dilations=(1, 7, 30), T_in: int = None):
        super().__init__()
        if T_in is not None:
            # 感受野 = (kernel_size-1)*dilation = 2*dilation（kernel=3）
            # 条件改为 <= T_in，避免感受野恰好等于 T_in 时被误过滤
            # 原条件 2*d < T_in 会错误丢弃 Solar/SDWPF 的 dilation=84（2*84=168=T_in）
            valid = [d for d in dilations if 2 * d <= T_in]
            if not valid:
                valid = [1]
            dilations = valid
        self.dilations = list(dilations)
        self.convs = nn.ModuleList([
            CausalConv1d(env_dim, env_dim, kernel_size=3, dilation=d)
            for d in self.dilations
        ])
        self.proj = nn.Linear(env_dim * len(self.dilations), ms_out_dim)

    def forward(self, He_seq):
        """He_seq: [B, T, N, env_dim] → [B, N, ms_out_dim]"""
        B, T, N, De = He_seq.shape
        x    = He_seq.permute(0, 2, 3, 1).reshape(B * N, De, T)
        outs = [conv(x)[:, :, -1] for conv in self.convs]
        return self.proj(torch.cat(outs, dim=-1)).reshape(B, N, -1)


# ---------------------------------------------------------------------------
# 9. SCG Message Passing（保留，不变）
# ---------------------------------------------------------------------------

class CausalGateUnit(nn.Module):
    def __init__(self, stoch_dim, env_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stoch_dim * 2 + env_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, Hs_i, Hs_j, He_i, He_j):
        return self.net(torch.cat([Hs_i, Hs_j, He_i, He_j], dim=-1))


class SCGMessagePassingLayer(nn.Module):
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
                          He[:, d_idx], He[:, s_idx])
            m = g * self.msg_tr(Hs[:, s_idx])
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
# 10. CFM Vector Field（保留，不变）
# ---------------------------------------------------------------------------

class CFMVectorField(nn.Module):
    """
    条件向量场 v_θ(x_t, t | He_prime, Hs_prime)。
    双流 AdaLN + 时序位置编码，接口和实现完全不变。
    """

    def __init__(self, out_dim: int, env_dim: int, stoch_dim: int,
                 hidden_dim: int = 128, time_emb_dim: int = 16,
                 max_freq: float = 1000.0,
                 T_out: int = 1, feat_dim: int = 1):
        super().__init__()
        self.out_dim  = out_dim
        self.T_out    = T_out
        self.feat_dim = feat_dim

        n_freqs = time_emb_dim // 2
        assert n_freqs * 2 == time_emb_dim, "time_emb_dim 必须是偶数"

        freqs = torch.exp(
            torch.linspace(0.0, math.log(max_freq), n_freqs)
        ) * math.pi
        self.register_buffer("freqs", freqs)

        self.temporal_pe = nn.Parameter(torch.zeros(T_out, feat_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
        self.env_proj = nn.Sequential(
            nn.LayerNorm(env_dim),
            nn.Linear(env_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )
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
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _time_embed(self, t: torch.Tensor, B: int, N: int) -> torch.Tensor:
        angles = t.reshape(B, 1) * self.freqs.unsqueeze(0)
        emb    = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb.unsqueeze(1).expand(B, N, -1)

    def _add_temporal_pe(self, x_t: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_t.shape
        x_3d = x_t.reshape(B, N, self.T_out, self.feat_dim)
        x_3d = x_3d + self.temporal_pe.unsqueeze(0).unsqueeze(0)
        return x_3d.reshape(B, N, self.out_dim)

    def forward(self, x_t, t, He_prime, Hs_prime):
        B, N, _ = x_t.shape
        x_t   = self._add_temporal_pe(x_t)
        t_emb = self._time_embed(t, B, N)
        t_s1, t_b1, t_s2, t_b2, t_s3, t_b3 = \
            self.time_proj(t_emb).chunk(6, dim=-1)
        b1_env, b2_env, b3_env = self.env_proj(He_prime).chunk(3, dim=-1)
        s1_st,  s2_st,  s3_st  = self.stoch_proj(Hs_prime).chunk(3, dim=-1)

        h = self.input_proj(x_t)
        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.layer1(h_norm * (1.0 + t_s1 + s1_st) + (t_b1 + b1_env))
        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.layer2(h_norm * (1.0 + t_s2 + s2_st) + (t_b2 + b2_env))
        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.layer3(h_norm * (1.0 + t_s3 + s3_st) + (t_b3 + b3_env))
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# 11. GridCFN（DMSD 版，集成所有改进）
# ---------------------------------------------------------------------------

class GridCFN(nn.Module):
    """
    GridCFN-DMSD 多步预测版。

    新增参数（相比原版）：
      rank_r           : 低秩 GCN 的秩（默认 8）
      lambda_rank      : 秩正则权重（默认 0.01）
      lambda_club      : CLUB 一致性惩罚权重（默认 0.05，替代原 lambda_mi）
      freq_candidates  : 频率分流候选窗口（tuple，数据集相关）
      wind_mask        : SDWPF 风向先验掩码 [N,N]，其他数据集传 None
    """

    def __init__(
        self,
        n_nodes: int,
        in_dim=1, gcn_hidden=64, tcn_hidden=64,
        env_dim=32, stoch_dim=32, ms_out_dim=32,
        n_scg_layers=3, out_dim=1,
        # 新参数
        lambda_mi=0.5,      # 保留字段名兼容 config，实际语义变为 lambda_club
        lambda_rank=0.01,
        rank_r=8,
        freq_candidates=(12, 24, 48, 96),
        wind_mask=None,
        # 以下与原版相同
        gcn_layers=2, tcn_layers=4,
        cfm_hidden=128, cfm_time_emb_dim=16,
        chunk_size=8192,
        ms_dilations=(1, 7, 30),
        T_out=1,
        T_in=None,
        adap_dim=16,        # 保留参数名兼容 main.py，DMSD 中不使用
    ):
        super().__init__()
        self.lambda_club = lambda_mi   # config 里的 lambda_mi 在此充当 lambda_club
        self.lambda_rank = lambda_rank
        self.feat_dim    = out_dim
        self.T_out       = T_out
        self.cfm_dim     = T_out * out_dim
        self.env_dim     = env_dim
        self.stoch_dim   = stoch_dim

        # ── 双轨 Backbone（替换原 Backbone + CausalDisentangler）──────────
        self.backbone = DualTrackBackbone(
            n_nodes         = n_nodes,
            in_dim          = in_dim,
            env_dim         = env_dim,
            stoch_dim       = stoch_dim,
            gcn_hidden      = gcn_hidden,
            tcn_hidden      = tcn_hidden,
            gcn_layers      = gcn_layers,
            tcn_layers      = tcn_layers,
            rank_r          = rank_r,
            freq_candidates = freq_candidates,
            wind_mask       = wind_mask,
        )

        # ── CLUB：两个独立实例 ──────────────────────────────────────────
        # club_e: MI(He, X_low_pooled)   — x_dim=env_dim,   y_dim=env_dim
        # club_s: MI(Hs, X_high_pooled)  — x_dim=stoch_dim, y_dim=stoch_dim
        self.club_e = CLUBEstimator(env_dim,   env_dim)
        self.club_s = CLUBEstimator(stoch_dim, stoch_dim)

        # ── 下游模块（接口完全不变）────────────────────────────────────
        self.ms_context = MultiScaleContext(env_dim, ms_out_dim,
                                            dilations=ms_dilations, T_in=T_in)
        self.scgmp      = SCGMP(stoch_dim, env_dim, n_scg_layers,
                                chunk_size=chunk_size)
        self.vector_field = CFMVectorField(
            out_dim       = self.cfm_dim,
            env_dim       = ms_out_dim,
            stoch_dim     = stoch_dim,
            hidden_dim    = cfm_hidden,
            time_emb_dim  = cfm_time_emb_dim,
            T_out         = T_out,
            feat_dim      = out_dim,
        )

    # ------------------------------------------------------------------
    # 图工具（不变）
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        adj = adj.clone()
        adj.fill_diagonal_(0)
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg      = adj.sum(dim=1)
        d_inv_sq = deg.clamp(min=1e-8).pow(-0.5)
        return d_inv_sq.unsqueeze(1) * adj * d_inv_sq.unsqueeze(0)

    @staticmethod
    def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
        return adj.nonzero(as_tuple=False).t().contiguous()

    # ------------------------------------------------------------------
    # 秩正则（代理 DualTrackBackbone.rank_loss）
    # ------------------------------------------------------------------

    def rank_loss(self) -> torch.Tensor:
        return self.backbone.rank_loss()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, adj_norm, edge_index):
        """
        x         : [B, T_in, N, F]
        adj_norm  : [N, N]  — 传入保持接口兼容，DMSD 内部不再使用固定图（双轨 GCN 自学习邻接）
        edge_index: [2, E]  — SCGMP 使用

        返回:
          He_prime      : [B, N, ms_out_dim]
          Hs_prime      : [B, N, stoch_dim]
          He            : [B, N, env_dim]
          Hs            : [B, N, stoch_dim]
          X_low_pooled  : [B, N, env_dim]
          X_high_pooled : [B, N, stoch_dim]
        """
        He_seq, He, Hs, X_low_pooled, X_high_pooled = self.backbone(x)
        He_prime = self.ms_context(He_seq)
        Hs_prime = self.scgmp(Hs, He, edge_index)
        return He_prime, Hs_prime, He, Hs, X_low_pooled, X_high_pooled

    # ------------------------------------------------------------------
    # CFM 训练损失（不变）
    # ------------------------------------------------------------------

    def cfm_loss(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor,
                 y_target: torch.Tensor,
                 n_t_samples: int = 4,
                 sigma_min: float = 0.01) -> torch.Tensor:
        assert y_target.shape[-1] == self.cfm_dim, (
            f"y_target.shape[-1]={y_target.shape[-1]} != cfm_dim={self.cfm_dim}"
        )
        B, N, _ = y_target.shape
        device  = y_target.device
        losses  = []
        for k in range(n_t_samples):
            x0    = torch.randn_like(y_target)
            t     = (k + torch.rand(B, device=device)) / n_t_samples
            t_bc  = t.reshape(B, 1, 1)
            x_t   = (1.0 - (1.0 - sigma_min) * t_bc) * x0 + t_bc * y_target
            u_t   = y_target - (1.0 - sigma_min) * x0
            v_pred = self.vector_field(x_t, t, He_prime, Hs_prime)
            losses.append(F.mse_loss(v_pred, u_t))
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # CFM 采样（Heun 二阶 ODE，不变）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor,
               n_samples: int = 50,
               n_steps: int = 20,
               sigma_min: float = 0.01,
               x0_scale: float = 1.0) -> torch.Tensor:
        B, N, _ = He_prime.shape
        S       = n_samples
        device  = He_prime.device
        dt      = 1.0 / n_steps

        he = He_prime.detach().repeat_interleave(S, dim=0)
        hs = Hs_prime.detach().repeat_interleave(S, dim=0)
        x  = torch.randn(B * S, N, self.cfm_dim, device=device) * x0_scale

        for step in range(n_steps):
            t_val  = step * dt
            t_next = t_val + dt
            t_vec      = torch.full((B * S,), t_val,  device=device, dtype=torch.float32)
            t_next_vec = torch.full((B * S,), t_next, device=device, dtype=torch.float32)
            v1    = self.vector_field(x, t_vec, he, hs)
            x_hat = x + dt * v1
            v2    = self.vector_field(x_hat, t_next_vec, he, hs)
            x     = x + dt * 0.5 * (v1 + v2)

        x = x.reshape(B, S, N, self.T_out, self.feat_dim)
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        return x