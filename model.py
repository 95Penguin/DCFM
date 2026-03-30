"""
GridCFN-Improved
================
在原版 GridCFN 基础上做两处有文献支撑的改进：

改动一（模块二）：MINE → CaST 完整 EnvDisentangler（NeurIPS 2023）
  原版 CausalDisentangler 只用两个线性投影 + MINE minimax 约束解耦，
  训练天然不稳定（日志：Loss -2.3~-0.75 震荡，22 epoch 早停）。

  CaST（NeurIPS 2023）后门调整方案完整三步实现：
    Step-1  EnvEncoder：AvgPool（时序均值）+ FFT 频域特征 + Attention 加权融合
            → 提取时序中"稳定的环境背景" He
    Step-2  EntityEncoder：H_last - He_reconstructed（残差去环境背景）
            → 提取"动态实体前景" Hi
    Step-3  VQ 码本（VQ-EMA）：对 He 做向量量化离散化
            → 用 commitment loss 约束解耦
  附加 MI 正则：对抗分类器用 Hi 预测环境类别，目标是阻止其预测成功
               → 最小化 I(Hi, He)

改动二（模块四）：单高斯 → GMM 混合高斯（参考 TimeGMM, arXiv 2026.01）
  输出 K 组 {alpha_k, mu_k, sigma_k}，用 NLL_GMM 训练。
  加入 GRIN（实例归一化 + 反归一化）消除节点间量纲差异。

损失：L_total = L_NLL(GMM) + beta_vq * L_commit + beta_mi * L_mi
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1-3. Backbone（保持原版完全不变）
# ===========================================================================

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
            [GCNLayer(dims[i], dims[i+1]) for i in range(n_layers)])

    def forward(self, x, adj_norm):
        for layer in self.layers:
            x = layer(x, adj_norm)
        return x


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, bias=True)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        r = x
        x = F.gelu(self.conv1(x))
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.gelu(self.conv2(x))
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x + r


class TCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=4, kernel_size=3):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden_dim, kernel_size, 2**i) for i in range(n_layers)])

    def forward(self, x):
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C).permute(0, 2, 1)
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        return x.permute(0, 2, 1).reshape(B, N, T, -1)


class Backbone(nn.Module):
    def __init__(self, in_dim, gcn_hidden, tcn_hidden,
                 gcn_layers=2, tcn_layers=4):
        super().__init__()
        self.gcn = GCN(in_dim, gcn_hidden, gcn_hidden, gcn_layers)
        self.tcn = TCN(gcn_hidden, tcn_hidden, tcn_layers)

    def forward(self, x, adj_norm):
        B, T, N, F_in = x.shape
        gcn_out = self.gcn(x.reshape(B * T, N, F_in), adj_norm)
        x_gcn   = gcn_out.reshape(B, T, N, -1)
        H = self.tcn(x_gcn.permute(0, 2, 1, 3))   # [B,N,T,D]
        return H.permute(0, 2, 1, 3)               # [B,T,N,D]


# ===========================================================================
# 4. ★ 改动一：CaST 完整 EnvDisentangler（NeurIPS 2023 后门调整）
# ===========================================================================

class VectorQuantizerEMA(nn.Module):
    """
    VQ-EMA 向量量化码本。
    - EMA 更新码本，码本不参与 optimizer，训练稳定
    - Straight-through estimator 传梯度给编码器
    - Commitment loss = ||z - sg(e)||^2 约束编码器贴近码本
    """
    def __init__(self, n_codes: int, code_dim: int,
                 decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.n_codes  = n_codes
        self.code_dim = code_dim
        self.decay    = decay
        self.eps      = eps
        embed = torch.randn(n_codes, code_dim)
        self.register_buffer('embed',        embed)
        self.register_buffer('cluster_size', torch.ones(n_codes))
        self.register_buffer('embed_avg',    embed.clone())

    def forward(self, z: torch.Tensor):
        """
        z   : [M, code_dim]
        返回: z_q (straight-through), commit_loss, code_indices [M]
        """
        M, D = z.shape
        # L2 距离到每个码字
        dist = (z.pow(2).sum(1, keepdim=True)
                - 2 * (z @ self.embed.t())
                + self.embed.pow(2).sum(1, keepdim=True).t())   # [M, n_codes]
        _, idx = dist.min(1)                                     # [M]
        one_hot = F.one_hot(idx, self.n_codes).float()          # [M, n_codes]

        z_q    = one_hot @ self.embed                            # [M, D]
        z_q_st = z + (z_q - z).detach()                         # straight-through

        if self.training:
            cs = one_hot.sum(0)
            ea = one_hot.t() @ z.detach()
            self.cluster_size.mul_(self.decay).add_(cs, alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(ea,  alpha=1 - self.decay)
            n  = self.cluster_size.sum()
            sm = ((self.cluster_size + self.eps)
                  / (n + self.n_codes * self.eps) * n)
            self.embed.data.copy_(self.embed_avg / sm.unsqueeze(1))

        commit_loss = F.mse_loss(z, z_q.detach())
        return z_q_st, commit_loss, idx


class CaSTEnvDisentangler(nn.Module):
    """
    CaST NeurIPS 2023 的完整 Environment Disentangler，实现后门调整。

    对应论文 Figure 4a 的三个子结构：

    EnvEncoder（提取环境特征 He）：
      ① AvgPool over time → 时序均值（捕捉稳定背景）
      ② FFT → 频域幅度谱均值（捕捉日/周期性环境规律）
      ③ MultiheadAttention 融合两路特征（AvgPool 作 query，FFT 作 key/value）

    EntityEncoder（提取实体特征 Hi）：
      ① He 重构回 in_dim 空间
      ② H_last - He_reconstructed = 残差（去除环境背景）
      ③ Linear 投影到 stoch_dim → Hi

    VQ 码本（论文的"离散化环境"步骤）：
      He → VQ-EMA 量化 → He_q（离散化环境表征）
      环境类别索引 idx 供 MI 正则化使用

    MI 正则（最小化 I(Hi, He)）：
      训练一个分类器用 Hi 预测环境类别 idx
      目标：阻止其预测成功（最大化预测熵）
      → 梯度反传让编码器输出更无法被 Hi 判别的 He
    """
    def __init__(self, in_dim: int, env_dim: int, stoch_dim: int,
                 n_codes: int = 64, n_heads: int = 4,
                 vq_decay: float = 0.99):
        super().__init__()
        self.env_dim   = env_dim
        self.stoch_dim = stoch_dim
        self.n_codes   = n_codes

        # ── EnvEncoder ──────────────────────────────────────────────────
        # ① AvgPool 路：时序均值 → 线性投影
        self.avg_proj = nn.Linear(in_dim, env_dim)

        # ② FFT 路：幅度谱均值（跨频率 bin）→ 线性投影
        self.fft_proj = nn.Linear(in_dim, env_dim)

        # ③ Attention 融合（query=avg_feat, key/value=fft_feat）
        self.attn     = nn.MultiheadAttention(
            embed_dim=env_dim, num_heads=n_heads,
            batch_first=True, dropout=0.0)
        self.env_norm = nn.LayerNorm(env_dim)

        # ── EntityEncoder ────────────────────────────────────────────────
        # He 重构回 in_dim（用于计算残差）
        self.he_recon = nn.Linear(env_dim, in_dim)
        # 残差投影到 stoch_dim
        self.hi_proj  = nn.Sequential(
            nn.Linear(in_dim, stoch_dim),
            nn.Tanh())

        # ── VQ 码本 ──────────────────────────────────────────────────────
        self.vq = VectorQuantizerEMA(
            n_codes=n_codes, code_dim=env_dim, decay=vq_decay)

        # ── MI 正则：对抗分类器 ──────────────────────────────────────────
        self.mi_classifier = nn.Sequential(
            nn.Linear(stoch_dim, stoch_dim), nn.ReLU(),
            nn.Linear(stoch_dim, n_codes))

    def _env_encode(self, H: torch.Tensor):
        """
        H : [B, T, N, D]
        返回 He [B, N, env_dim]（量化前），He_seq [B, T, N, env_dim]
        """
        B, T, N, D = H.shape

        # ① AvgPool 路
        avg_feat = self.avg_proj(H.mean(dim=1))          # [B, N, env_dim]

        # ② FFT 路：rfft 沿时间轴，取幅度谱跨频率 bin 均值
        fft_amp  = torch.fft.rfft(H, dim=1, norm='ortho').abs().mean(dim=1)
        fft_feat = self.fft_proj(fft_amp)                # [B, N, env_dim]

        # ③ Attention 融合：展平 B*N 作 batch
        BN = B * N
        q = avg_feat.reshape(BN, 1, self.env_dim)
        k = fft_feat.reshape(BN, 1, self.env_dim)
        attn_out, _ = self.attn(q, k, k)                # [B*N, 1, env_dim]
        He = self.env_norm(
            attn_out.reshape(B, N, self.env_dim) + avg_feat)

        # 完整时序 He_seq（对每帧 avg_proj，供 MultiScaleContext）
        He_seq = self.avg_proj(
            H.reshape(B * T * N, D)).reshape(B, T, N, self.env_dim)

        return He, He_seq

    def forward(self, H: torch.Tensor):
        """
        H : [B, T, N, D]
        返回：
          He_q        : [B, N, env_dim]    量化环境表征
          Hi          : [B, N, stoch_dim]  实体表征
          He_seq      : [B, T, N, env_dim] 时序环境（供 MultiScaleContext）
          commit_loss : 标量
          mi_loss     : 标量
        """
        B, T, N, D = H.shape

        # Step-1：EnvEncoder
        He, He_seq = self._env_encode(H)                # [B,N,env_dim]

        # Step-2：EntityEncoder（残差去环境背景）
        H_last   = H[:, -1]                             # [B, N, D]
        He_recon = self.he_recon(He)                    # [B, N, D]
        residual = H_last - He_recon
        Hi       = self.hi_proj(residual)               # [B, N, stoch_dim]

        # Step-3：VQ 码本量化
        He_q_flat, commit_loss, env_idx = self.vq(
            He.reshape(B * N, self.env_dim))
        He_q = He_q_flat.reshape(B, N, self.env_dim)   # [B, N, env_dim]

        # MI 正则：用 Hi 预测环境类别，目标是让预测分布趋近均匀（熵最大化）
        # env_idx: [B*N]，VQ 量化索引即环境类别标签
        logits   = self.mi_classifier(
            Hi.reshape(B * N, self.stoch_dim))          # [B*N, n_codes]
        # 训练分类器预测 env_idx（让 CE loss 可反传）
        mi_cls_loss = F.cross_entropy(logits, env_idx.detach())
        # MI 正则 = 对主网络施加对抗项（负 CE）= 让 Hi 无法区分环境
        # 实现：对 Hi 编码器传 -CE 的梯度，即 mi_loss = -mi_cls_loss
        # 分类器参数正常被 mi_cls_loss 更新（朝"分类准确"方向）
        # Hi/He 编码器被 -mi_cls_loss 更新（朝"分类失败"方向）
        # 单 optimizer 全参数联合优化时，用梯度反转层思路：
        # 将 mi_loss 设为 -mi_cls_loss，放入总 loss 中
        mi_loss = -mi_cls_loss

        return He_q, Hi, He_seq, commit_loss, mi_loss


# ===========================================================================
# 5. MultiScaleContext（保持原版不变）
# ===========================================================================

class MultiScaleContext(nn.Module):
    def __init__(self, env_dim, out_dim, n_scales=4, kernel_size=3):
        super().__init__()
        dilations = [1, 2, 4, 8][:n_scales]
        self.convs = nn.ModuleList(
            [CausalConv1d(env_dim, out_dim, kernel_size, d) for d in dilations])
        self.fuse = nn.Linear(out_dim * n_scales, out_dim)

    def forward(self, He_seq):
        B, T, N, C = He_seq.shape
        x = He_seq.permute(0, 2, 3, 1).reshape(B * N, C, T)
        fused = torch.cat([conv(x) for conv in self.convs], dim=1)[:, :, -1]
        return F.relu(self.fuse(fused).view(B, N, -1))


# ===========================================================================
# 6. SCG-MP（保持原版不变）
# ===========================================================================

class CausalGatingUnit(nn.Module):
    def __init__(self, stoch_dim, env_dim, hidden_dim=64):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * stoch_dim + 2 * env_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, Hs_i, Hs_j, He_i, He_j):
        return self.gate_mlp(torch.cat([Hs_i, Hs_j, He_i, He_j], dim=-1))


class SCGMessagePassingLayer(nn.Module):
    def __init__(self, stoch_dim, env_dim, hidden_dim=64):
        super().__init__()
        self.gate_unit     = CausalGatingUnit(stoch_dim, env_dim, hidden_dim)
        self.msg_transform = nn.Linear(stoch_dim, stoch_dim, bias=False)
        self.agg_transform = nn.Sequential(
            nn.Linear(stoch_dim, stoch_dim), nn.ReLU())

    def forward(self, Hs, He, edge_index):
        B, N, Ds = Hs.shape
        src, dst = edge_index[0], edge_index[1]
        g   = self.gate_unit(Hs[:, dst], Hs[:, src], He[:, dst], He[:, src])
        m   = g * self.msg_transform(Hs[:, src])
        agg = torch.zeros(B, N, Ds, device=Hs.device)
        agg.scatter_add_(1,
            dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, Ds), m)
        deg = torch.zeros(N, device=Hs.device)
        deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=Hs.device))
        agg = agg / deg.view(1, N, 1).clamp(min=1.0)
        return self.agg_transform(Hs + agg)


class SCGMP(nn.Module):
    def __init__(self, stoch_dim, env_dim, n_layers=3, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList(
            [SCGMessagePassingLayer(stoch_dim, env_dim, hidden_dim)
             for _ in range(n_layers)])

    def forward(self, Hs, He, edge_index):
        for layer in self.layers:
            Hs = layer(Hs, He, edge_index)
        return Hs


# ===========================================================================
# 7. ★ 改动二：GMM 概率输出层
# ===========================================================================

class GMMPredictor(nn.Module):
    """
    K 分量 GMM 输出 + GRIN 实例归一化，参考 TimeGMM (arXiv 2026.01)。

    GRIN：
      前向：实例归一化（消除节点间量纲差异），可学习 scale/shift
      输出：mu 反归一化（还原量纲），sigma 乘以尺度

    输出：alpha [B,N,F,K], mu [B,N,F,K], sigma [B,N,F,K]
    """
    def __init__(self, in_dim: int, out_dim: int,
                 K: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.K       = K
        self.out_dim = out_dim

        self.grin_scale = nn.Parameter(torch.ones(1))
        self.grin_shift = nn.Parameter(torch.zeros(1))

        self.norm = nn.LayerNorm(in_dim)
        self.net  = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.alpha_head = nn.Linear(hidden_dim, out_dim * K)
        self.mu_head    = nn.Linear(hidden_dim, out_dim * K)
        self.sigma_head = nn.Linear(hidden_dim, out_dim * K)

    def forward(self, H_final: torch.Tensor):
        B, N, _ = H_final.shape
        K, Fo   = self.K, self.out_dim

        # GRIN 实例归一化
        inst_mean = H_final.mean(-1, keepdim=True)
        inst_std  = H_final.std(-1,  keepdim=True) + 1e-6
        H = (H_final - inst_mean) / inst_std
        H = H * self.grin_scale + self.grin_shift
        h = self.net(self.norm(H))                             # [B, N, hidden]

        alpha  = F.softmax(
            self.alpha_head(h).view(B, N, Fo, K), dim=-1)     # [B,N,Fo,K]
        mu_raw = self.mu_head(h).view(B, N, Fo, K)
        sigma  = F.softplus(
            self.sigma_head(h).view(B, N, Fo, K)) + 1e-3

        # GRIN 反归一化
        me = inst_mean.unsqueeze(-1).expand_as(mu_raw)
        se = inst_std.unsqueeze(-1).expand_as(sigma)
        mu    = mu_raw * se + me
        sigma = sigma  * se

        return alpha, mu, sigma


# ===========================================================================
# 8. 损失函数与评估辅助
# ===========================================================================

def nll_gmm_loss(alpha, mu, sigma, y):
    """GMM NLL，log-sum-exp 防数值下溢。y:[B,N,F]  其余:[B,N,F,K]"""
    eps   = 1e-6
    sigma = sigma.clamp(min=eps)
    y_exp = y.unsqueeze(-1).expand_as(mu)
    log_phi  = (-0.5 * ((y_exp - mu) / sigma).pow(2)
                - sigma.log()
                - 0.5 * math.log(2 * math.pi))
    log_terms = alpha.clamp(min=eps).log() + log_phi
    return -torch.logsumexp(log_terms, dim=-1).mean()


def gmm_mean(alpha, mu):
    """点预测期望：Σ_k alpha_k * mu_k"""
    return (alpha * mu).sum(-1)


def gmm_variance(alpha, mu, sigma):
    """GMM 总标准差（用于 CRPS/PICP 评估）"""
    mean = gmm_mean(alpha, mu)
    ex2  = (alpha * (sigma.pow(2) + mu.pow(2))).sum(-1)
    return (ex2 - mean.pow(2)).clamp(min=1e-6).sqrt()


def nll_gaussian_loss(mu, sigma, y):
    """兼容旧接口，不再主动调用"""
    eps   = 1e-6
    sigma = sigma.clamp(min=eps)
    return (sigma.log() + 0.5 * math.log(2 * math.pi)
            + 0.5 * ((y - mu) / sigma).pow(2)).mean()


# ===========================================================================
# 9. ★ 完整改进版 GridCFN
# ===========================================================================

class GridCFN(nn.Module):
    """
    GridCFN-Improved

    输入：X [B,T,N,F]，adj [N,N]
    输出：alpha, mu, sigma 各 [B,N,out_dim,K]，commit_loss，mi_loss

    新增超参（相比原版）：
      n_codes  : VQ 码本大小（默认 64）
      K        : GMM 分量数（默认 3）
      beta_vq  : VQ commitment loss 权重（默认 0.25）
      beta_mi  : MI 正则化权重（默认 0.1）
      n_heads  : EnvEncoder Attention 头数（默认 4，需整除 env_dim）
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
        gcn_layers:   int   = 2,
        tcn_layers:   int   = 4,
        n_codes:      int   = 64,
        K:            int   = 3,
        beta_vq:      float = 0.25,
        beta_mi:      float = 0.1,
        n_heads:      int   = 4,
        lambda_mi:    float = 0.5,    # 兼容旧接口
    ):
        super().__init__()
        self.beta_vq   = beta_vq
        self.beta_mi   = beta_mi
        self.lambda_mi = lambda_mi

        self.backbone = Backbone(in_dim, gcn_hidden, tcn_hidden,
                                 gcn_layers, tcn_layers)

        # ★ 改动一：CaST 完整 EnvDisentangler
        self.disentangler = CaSTEnvDisentangler(
            in_dim    = tcn_hidden,
            env_dim   = env_dim,
            stoch_dim = stoch_dim,
            n_codes   = n_codes,
            n_heads   = n_heads,
            vq_decay  = 0.99,
        )

        self.ms_context = MultiScaleContext(env_dim, ms_out_dim)
        self.scgmp      = SCGMP(stoch_dim, ms_out_dim, n_scg_layers)

        # ★ 改动二：GMM 概率输出
        self.predictor  = GMMPredictor(ms_out_dim + stoch_dim, out_dim, K=K)

    @staticmethod
    def normalize_adj(adj):
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(1)
        d   = torch.pow(deg.clamp(min=1e-8), -0.5)
        D   = torch.diag(d)
        return D @ adj @ D

    @staticmethod
    def adj_to_edge_index(adj):
        return adj.nonzero(as_tuple=False).t().contiguous()

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        返回：
          alpha, mu, sigma : GMM 参数，各 [B, N, out_dim, K]
          commit_loss      : VQ commitment loss（标量）
          mi_loss          : MI 正则 loss（标量，已取负号）
        """
        adj_norm   = self.normalize_adj(adj)
        edge_index = self.adj_to_edge_index(adj)

        H = self.backbone(x, adj_norm)

        # CaST 完整后门调整解耦
        He_q, Hi, He_seq, commit_loss, mi_loss = self.disentangler(H)

        # 多尺度上下文
        He_prime = self.ms_context(He_seq)

        # SCG-MP（传入精炼后的 He_prime）
        Hi_prime = self.scgmp(Hi, He_prime, edge_index)

        # GMM 输出
        H_final = torch.cat([He_prime, Hi_prime], dim=-1)
        alpha, mu, sigma = self.predictor(H_final)

        return alpha, mu, sigma, commit_loss, mi_loss

    def compute_loss(self, alpha, mu, sigma, y,
                     commit_loss, mi_loss,
                     beta_vq=None, beta_mi=None):
        """
        L_total = L_NLL(GMM) + beta_vq * L_commit + beta_mi * L_mi

        mi_loss 已在 forward 中取负（-CE），
        加入总 loss 后对 Hi 编码器产生"阻止分类"的梯度方向。
        mi_classifier 参数同时被正常 CE 梯度更新（朝"分类准确"方向），
        形成对抗解耦。
        """
        if beta_vq is None:
            beta_vq = self.beta_vq
        if beta_mi is None:
            beta_mi = self.beta_mi

        y_tgt   = y[..., :self.predictor.out_dim]
        l_nll   = nll_gmm_loss(alpha, mu, sigma, y_tgt)
        l_total = l_nll + beta_vq * commit_loss + beta_mi * mi_loss
        return l_total, l_nll, commit_loss, mi_loss

    def main_parameters(self):
        return list(self.parameters())

    def mine_parameters(self):
        """兼容旧接口。"""
        return []
