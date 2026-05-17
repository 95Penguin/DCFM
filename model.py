"""
GridCFN: Causal Spatio-Temporal Framework for Power Flow Uncertainty Prediction
（多步预测版，Direct Multi-Output）

改进说明（相对原版）：
  1. AdaptiveGCN：在固定皮尔逊图基础上叠加可学习自适应邻接矩阵（参考 GWN），
     图结构随训练更新，不再依赖固定阈值。
  2. CausalDisentangler：He/Hs 改为注意力时间池化，不再只取最后一个时间步，
     充分利用整个序列的上下文信息。
  3. CFMVectorField：输入 x_t 增加可学习时序位置编码，让网络感知各预测步的
     相对位置，而不是把 T_out 步完全展平处理。
  4. GridCFN.sample：Euler → Heun 二阶 ODE 求解器，推理质量更好，
     相同 n_steps 下误差更低。
  5. GridCFN.cfm_loss：t 改为分层随机采样（stratified sampling），
     覆盖更均匀，收敛更快。

架构概览：
  Backbone (AdaptiveGCN+TCN) → CausalDisentangler → He（环境）/ Hs（随机）
  He → MultiScaleContext → He_prime（多尺度环境背景）
  Hs → SCGMP            → Hs_prime（图传播后随机表征）
  CFMVectorField(x_t + temporal_pe, t, He_prime, Hs_prime) → 双流 AdaLN 条件向量场
  向量场输出维度 = T_out * feat_dim（直接多步输出）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 1. GCN → AdaptiveGCN
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj_norm):
        return F.relu(torch.matmul(adj_norm, self.linear(x)))


class AdaptiveGCN(nn.Module):
    """
    自适应图卷积网络。

    在固定的归一化邻接矩阵 adj_norm 基础上，叠加一个数据驱动的自适应邻接矩阵：
        A_adap = softmax(ReLU(E1 @ E2^T))
        A_total = α * A_adap + (1-α) * adj_norm

    E1, E2 是可学习节点嵌入，维度为 adap_dim。
    α 为可学习的混合系数（初始化为 0.5）。

    优点：
      - 不依赖预计算的皮尔逊阈值图（固定图的邻接矩阵不参与梯度）
      - 能捕捉静态图结构无法表达的隐式依赖关系
      - 参数量增加很少（2 * N * adap_dim + 1）
    """
    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int,
                 out_dim: int, n_layers: int = 2, adap_dim: int = 16):
        super().__init__()
        self.n_nodes  = n_nodes
        self.adap_dim = adap_dim

        # 可学习节点嵌入，用于生成自适应邻接矩阵
        self.E1 = nn.Embedding(n_nodes, adap_dim)
        self.E2 = nn.Embedding(n_nodes, adap_dim)

        # 混合系数：α∈(0,1)，初始 0.5
        self.alpha = nn.Parameter(torch.tensor(0.5))

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)]
        )

    def _adaptive_adj(self, device):
        """计算归一化自适应邻接矩阵 A_adap: [N, N]"""
        idx = torch.arange(self.n_nodes, device=device)
        A   = F.relu(self.E1(idx) @ self.E2(idx).T)        # [N, N]，非负
        A   = F.softmax(A, dim=-1)                          # 行归一化
        return A

    def forward(self, x, adj_norm):
        """
        x        : [B*T, N, F] 或 [B, N, F]
        adj_norm : [N, N]  固定归一化邻接矩阵（不参与梯度）
        """
        A_adap  = self._adaptive_adj(x.device)              # [N, N]
        alpha   = torch.sigmoid(self.alpha)                  # 约束到 (0,1)
        A_mix   = alpha * A_adap + (1.0 - alpha) * adj_norm # [N, N]
        for layer in self.layers:
            x = layer(x, A_mix)
        return x


# ---------------------------------------------------------------------------
# 2. TCN（不变）
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
# 3. Backbone（使用 AdaptiveGCN）
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int,
                 gcn_hidden: int, tcn_hidden: int,
                 gcn_layers: int = 2, tcn_layers: int = 4,
                 adap_dim: int = 16):
        super().__init__()
        self.gcn = AdaptiveGCN(n_nodes, in_dim, gcn_hidden, gcn_hidden,
                               gcn_layers, adap_dim)
        # tcn_input_dim = gcn_hidden：GCN 输出维即 TCN 输入维
        self.tcn = TCN(in_dim=gcn_hidden, hidden_dim=tcn_hidden, n_layers=tcn_layers)

    def forward(self, x, adj_norm):
        """x: [B,T,N,F] → H: [B,T,N,D]"""
        B, T, N, F = x.shape
        g = self.gcn(x.reshape(B * T, N, F), adj_norm).reshape(B, T, N, -1)
        H = self.tcn(g.permute(0, 2, 1, 3))
        return H.permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# 4. CausalDisentangler（改为注意力时间池化）
# ---------------------------------------------------------------------------

class CausalDisentangler(nn.Module):
    """
    将 Backbone 输出 H 分解为 He（环境）和 Hs（随机）表征。

    改进：He 和 Hs 均通过可学习注意力时间池化聚合全序列，不再只取最后一步。

    注意力池化：
        score_t = w^T * tanh(W * H_t)      (标量分数，逐节点)
        weight  = softmax(score, dim=T)
        He      = sum_t(weight_t * env_proj(H_t))

    这样网络可以根据任务自适应地关注不同时间步，
    而不是强制使用最近的时间步作为唯一代理。

    He_seq 保留用于 MultiScaleContext（全序列多尺度卷积不受影响）。
    """

    def __init__(self, in_dim: int, env_dim: int, stoch_dim: int):
        super().__init__()
        # 环境表征投影
        self.env_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, env_dim), nn.Tanh()
        )
        # 随机表征投影
        self.stoch_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, stoch_dim), nn.Tanh()
        )
        # 时间注意力评分（用于 He 的池化）
        self.attn_env = nn.Sequential(
            nn.Linear(env_dim, env_dim // 2), nn.Tanh(),
            nn.Linear(env_dim // 2, 1)        # → [B, T, N, 1]
        )
        # 时间注意力评分（用于 Hs 的池化）
        self.attn_stoch = nn.Sequential(
            nn.Linear(stoch_dim, stoch_dim // 2), nn.Tanh(),
            nn.Linear(stoch_dim // 2, 1)
        )

    def forward(self, H):
        """
        H : [B, T, N, D]
        返回:
          He     : [B, N, env_dim]    注意力加权的环境表征
          Hs     : [B, N, stoch_dim]  注意力加权的随机表征
          He_seq : [B, T, N, env_dim] 全序列环境表征（供 MultiScaleContext 使用）

        stop-gradient：两个投影头各自接收 H.detach()，使 backbone 的梯度
        不再被两个分支同时拉扯，迫使 backbone 学习对两者都有用的中性表征，
        解耦压力落在投影头而非 backbone。
        """
        B, T, N, D = H.shape
        H_flat = H.reshape(B * T * N, D)
        H_sg   = H.detach()                                  # stop-gradient
        H_sg_flat = H_sg.reshape(B * T * N, D)

        # ── 环境分支（stop-gradient 输入）────────────────────────────────
        He_seq = self.env_proj(H_sg_flat).reshape(B, T, N, -1)
        env_score  = self.attn_env(He_seq)                   # [B, T, N, 1]
        env_weight = F.softmax(env_score, dim=1)
        He = (env_weight * He_seq).sum(dim=1)                # [B, N, env_dim]

        # ── 随机分支（stop-gradient 输入）────────────────────────────────
        Hs_seq = self.stoch_proj(H_sg_flat).reshape(B, T, N, -1)
        stoch_score  = self.attn_stoch(Hs_seq)               # [B, T, N, 1]
        stoch_weight = F.softmax(stoch_score, dim=1)
        Hs = (stoch_weight * Hs_seq).sum(dim=1)              # [B, N, stoch_dim]

        # He_seq 供 MultiScaleContext 使用，保留梯度（从 H_flat 重新投影）
        He_seq_grad = self.env_proj(H_flat).reshape(B, T, N, -1)

        return He, Hs, He_seq_grad


# ---------------------------------------------------------------------------
# 5. CLUB Mutual Information Estimator（不变）
# ---------------------------------------------------------------------------

class CLUBEstimator(nn.Module):
    """CLUB 互信息上界估计器（Cheng et al., 2020）。"""

    def __init__(self, env_dim, stoch_dim, hidden_dim=64):
        super().__init__()
        self.var_net_mu = nn.Sequential(
            nn.Linear(env_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
        )
        # 末尾加 Tanh 将原始输出压到 (-1, 1)，配合 _log_prob 内的 clamp(-6, 4)，
        # 避免训练初期 log_var 跑到极端值导致数值不稳定。
        self.var_net_logvar = nn.Sequential(
            nn.Linear(env_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
            nn.Tanh(),
        )

    def _get_params(self, He_flat):
        return self.var_net_mu(He_flat), self.var_net_logvar(He_flat)

    def _log_prob(self, Hs, mu_q, logvar_raw):
        # 直接将网络输出视为 log σ²，用 clamp 限制范围。
        # 原来的 log(softplus(x)+1e-2) 是双重非线性：softplus 已保证正数，
        # 再取 log 得到 log(log(1+e^x)+1e-2)，语义不对且梯度路径混乱。
        log_var  = logvar_raw.clamp(-6.0, 4.0)
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
        M                = He.shape[0] * He.shape[1]
        He_flat          = He.reshape(M, -1)
        Hs_flat          = Hs.reshape(M, -1)
        mu_q, logvar_raw = self._get_params(He_flat)
        pos = self._log_prob(Hs_flat, mu_q, logvar_raw).mean()
        neg = self._log_prob(Hs_flat[self._neg_perm(M, He.device)],
                             mu_q, logvar_raw).mean()
        return pos - neg

    def variational_loss(self, He, Hs):
        M                = He.shape[0] * He.shape[1]
        He_flat          = He.reshape(M, -1)
        Hs_flat          = Hs.reshape(M, -1)
        mu_q, logvar_raw = self._get_params(He_flat)
        return -self._log_prob(Hs_flat, mu_q, logvar_raw).mean()


# ---------------------------------------------------------------------------
# 6. Multi-Scale Context（不变）
# ---------------------------------------------------------------------------

class MultiScaleContext(nn.Module):
    """多尺度时间上下文提取。"""

    def __init__(self, env_dim, ms_out_dim, dilations=(1, 7, 30), T_in: int = None):
        super().__init__()
        # 感受野 = (kernel_size-1)*dilation = 2*dilation，需满足 2*d < T_in
        if T_in is not None:
            valid = [d for d in dilations if 2 * d < T_in]
            if not valid:
                valid = [1]   # 至少保留 dilation=1
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
# 7. SCG Message Passing（不变）
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
    """门控图消息传递层，分块处理避免 OOM。"""

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
# 8. CFM Vector Field（双流 AdaLN + 时序位置编码）
# ---------------------------------------------------------------------------

class CFMVectorField(nn.Module):
    """
    条件向量场 v_θ(x_t, t | He_prime, Hs_prime)。

    改进：输入 x_t 在进入 input_proj 之前，加上可学习时序位置编码。
    x_t 形状为 [B, N, T_out * feat_dim]，先 reshape 为 [B, N, T_out, feat_dim]，
    对 T_out 维度加位置编码，再 reshape 回来。这让网络知道每个预测步的相对位置，
    而不是把所有步完全对称地展平处理。

    双流 AdaLN：He_prime 控制 shift，Hs_prime 控制 scale。
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

        # 可学习时序位置编码：[T_out, feat_dim]，加到 x_t 的各预测步上
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
        """
        x_t : [B, N, T_out * feat_dim]
        时序位置编码 temporal_pe : [T_out, feat_dim]
        广播加到各步，返回同形状 tensor。
        """
        B, N, _ = x_t.shape
        # reshape → 加 pe → reshape 回
        x_3d = x_t.reshape(B, N, self.T_out, self.feat_dim)
        x_3d = x_3d + self.temporal_pe.unsqueeze(0).unsqueeze(0)  # 广播 [B,N,T_out,feat_dim]
        return x_3d.reshape(B, N, self.out_dim)

    def forward(self, x_t, t, He_prime, Hs_prime):
        """
        x_t      : [B, N, out_dim]   out_dim = T_out * feat_dim
        t        : [B]
        He_prime : [B, N, env_dim]
        Hs_prime : [B, N, stoch_dim]
        """
        B, N, _ = x_t.shape

        # 加时序位置编码
        x_t = self._add_temporal_pe(x_t)

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
# 9. GridCFN（多步版，集成全部改进）
# ---------------------------------------------------------------------------

class GridCFN(nn.Module):
    """
    多步预测版 GridCFN。

    参数变化（相比原版）：
      n_nodes  : 节点数，AdaptiveGCN 需要（从 load_data 的 adj.shape[0] 传入）
      adap_dim : 自适应邻接矩阵的节点嵌入维度（默认 16）
      T_out    : 预测步长
    """

    def __init__(
        self,
        n_nodes: int,
        in_dim=1, gcn_hidden=64, tcn_hidden=64,
        env_dim=32, stoch_dim=32, ms_out_dim=32,
        n_scg_layers=3, out_dim=1, lambda_mi=0.5,
        gcn_layers=2, tcn_layers=4,
        cfm_hidden=128, cfm_time_emb_dim=16,
        chunk_size=8192,
        ms_dilations=(1, 7, 30),
        T_out=1,
        T_in=None,                  # ← 新增：传给 MultiScaleContext 做 dilation 裁剪
        adap_dim=16,
    ):
        super().__init__()
        self.lambda_mi = lambda_mi
        self.feat_dim  = out_dim
        self.T_out     = T_out
        self.cfm_dim   = T_out * out_dim
        self.env_dim   = env_dim
        self.stoch_dim = stoch_dim

        self.backbone     = Backbone(n_nodes, in_dim, gcn_hidden, tcn_hidden,
                                     gcn_layers, tcn_layers, adap_dim)
        self.disentangler = CausalDisentangler(tcn_hidden, env_dim, stoch_dim)
        self.club         = CLUBEstimator(env_dim, stoch_dim)
        self.ms_context   = MultiScaleContext(env_dim, ms_out_dim, dilations=ms_dilations,
                                              T_in=T_in)
        self.scgmp        = SCGMP(stoch_dim, env_dim, n_scg_layers, chunk_size=chunk_size)
        self.vector_field = CFMVectorField(
            out_dim=self.cfm_dim,
            env_dim=ms_out_dim,
            stoch_dim=stoch_dim,
            hidden_dim=cfm_hidden,
            time_emb_dim=cfm_time_emb_dim,
            T_out=T_out,
            feat_dim=out_dim,
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
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, adj_norm, edge_index):
        """
        x : [B, T_in, N, F]
        返回: He_prime, Hs_prime, He, Hs, mi_loss
        """
        H              = self.backbone(x, adj_norm)
        He, Hs, He_seq = self.disentangler(H)
        mi_loss        = self.club(He, Hs)
        He_prime       = self.ms_context(He_seq)
        Hs_prime       = self.scgmp(Hs, He, edge_index)
        return He_prime, Hs_prime, He, Hs, mi_loss

    # ------------------------------------------------------------------
    # CFM 训练损失（分层 t 采样）
    # ------------------------------------------------------------------

    def cfm_loss(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor,
                 y_target: torch.Tensor,
                 n_t_samples: int = 4,
                 sigma_min: float = 0.01) -> torch.Tensor:
        """
        OT-CFM 训练损失。

        改进：t 使用分层随机采样（stratified sampling），把 [0,1] 均分为
        n_t_samples 个区间，每个区间内随机取一个点。相比纯随机采样，
        t 的覆盖更均匀，训练初期收敛更快，避免 t 集中在某个区域。

        y_target : [B, N, T_out * feat_dim]
        """
        assert y_target.shape[-1] == self.cfm_dim, (
            f"y_target.shape[-1]={y_target.shape[-1]} != cfm_dim={self.cfm_dim}"
        )
        B, N, _ = y_target.shape
        device  = y_target.device
        losses  = []

        for k in range(n_t_samples):
            x0 = torch.randn_like(y_target)
            # 分层采样：第 k 个区间 [k/n, (k+1)/n) 内均匀采样
            t = (k + torch.rand(B, device=device)) / n_t_samples   # ← 改进
            t_bc  = t.reshape(B, 1, 1)
            x_t   = (1.0 - (1.0 - sigma_min) * t_bc) * x0 + t_bc * y_target
            u_t   = y_target - (1.0 - sigma_min) * x0
            v_pred = self.vector_field(x_t, t, He_prime, Hs_prime)
            losses.append(F.mse_loss(v_pred, u_t))

        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # CFM 采样（Heun 二阶 ODE 求解器）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor,
               n_samples: int = 50,
               n_steps: int = 20,
               sigma_min: float = 0.01,
               x0_scale: float = 1.0) -> torch.Tensor:
        """
        Heun 二阶 ODE 求解器（比 Euler 精度高，相同步数下轨迹误差更小）。

        Heun 方法（预测-校正）：
            v1    = f(x_t,   t)
            x_hat = x_t + dt * v1          （Euler 预测步）
            v2    = f(x_hat, t + dt)        （在预测点再算一次向量场）
            x_t+1 = x_t + dt * (v1 + v2) / 2  （梯形校正）

        每步比 Euler 多一次 vector_field forward，但可以用更少的步数
        达到同等质量，实际推理时间相近。

        返回 : [n_samples, B, N, T_out, feat_dim]
        """
        B, N, _ = He_prime.shape
        S       = n_samples
        device  = He_prime.device
        dt      = 1.0 / n_steps

        he = He_prime.detach().repeat_interleave(S, dim=0)   # [B*S, N, ms_out_dim]
        hs = Hs_prime.detach().repeat_interleave(S, dim=0)   # [B*S, N, stoch_dim]
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

        # [B*S, N, T_out*feat_dim] → [S, B, N, T_out, feat_dim]
        x = x.reshape(B, S, N, self.T_out, self.feat_dim)
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        return x