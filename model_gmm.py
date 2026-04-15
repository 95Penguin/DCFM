"""
GridCFN + GMM 概率输出头 v2
============================

借鉴 TimeGMM 的三处改进：
  [TimeGMM-1] GRIN：GMM-adapted Reversible Instance Normalization
    在 GMMHead 输出后做 denorm，把归一化域的 mu/sigma 还原到原始数据尺度。
    对 Electricity 重尾数据效果显著：不同节点的量级差异由 per-node 参数保留。
    注意：GRIN 只在输出头做 denorm，不在 backbone 前做 norm，
    避免破坏 GCN 节点间相对尺度。

  [TimeGMM-2] GMM 复合损失
    L_total = L_NLL + lambda_mean * L_mean(Huber) + lambda_weight * L_weight
    L_mean 用 Huber loss 替换 MSE，对大误差样本线性惩罚。

  [TimeGMM-3] 重尾分量
    K 个分量中最后一个使用更大的 sigma_min_heavy，专门捕捉异常值。

自身修复：
  [Fix-CLUB-Elec] CLUBEstimator 增加 projection MLP
    在 L2 normalize 前先将 He/Hs 投影到判别空间，
    解决 Electricity 数据极小数值范围导致 MI=0 的问题。
    梯度流向正确：proj_he/proj_hs 是 club 参数，
    variational_loss 中 He.detach() 只阻断 backbone 梯度，
    proj 参数仍由 club_optimizer 更新。

  [Fix-VarLoss-Clamp] variational_loss clamp(min=-10)
    防止 log_var 崩塌时 VarLoss 无限下降。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
# 5. [Fix-CLUB-Elec] CLUBEstimator
# ---------------------------------------------------------------------------
class CLUBEstimator(nn.Module):
    """
    CLUB 互信息上界估计器（NeurIPS 2020）。

    [Fix-CLUB-Elec] 新增 proj_he / proj_hs：
      两层 MLP，在 L2 normalize 之前先将 He/Hs 投影到 proj_dim 维判别空间。

      梯度流向说明（关键）：
        forward(He, Hs)：He/Hs 保留梯度，proj + var_net 均正常反传到 backbone。
        variational_loss(He.detach(), Hs.detach())：
          He/Hs.detach() 只阻断流向 backbone 的梯度（阻断发生在 detach() 调用处，
          即进入 _project 之前）。proj_he/proj_hs 作为 club 自身参数，
          依然参与此次 backward，由 club_optimizer 正确更新。
          这与原版 L2 norm 的梯度行为完全一致，无需额外处理。
    """
    def __init__(self, env_dim, stoch_dim, hidden_dim=64, proj_dim=64):
        super().__init__()
        self.proj_he = nn.Sequential(
            nn.Linear(env_dim,  proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.proj_hs = nn.Sequential(
            nn.Linear(stoch_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim,  proj_dim),
        )
        self.var_net_mu = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.var_net_logvar = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def _project(self, He, Hs):
        M = He.shape[0] * He.shape[1]
        he_proj = F.normalize(self.proj_he(He.reshape(M, -1)), dim=-1)
        hs_proj = F.normalize(self.proj_hs(Hs.reshape(M, -1)), dim=-1)
        return he_proj, hs_proj

    def _get_params(self, he_proj):
        return self.var_net_mu(he_proj), self.var_net_logvar(he_proj)

    def _log_prob(self, hs_proj, mu_q, logvar_raw):
        log_var  = torch.log(F.softplus(logvar_raw) + 1e-2).clamp(-4.0, 4.0)
        log_prob = -0.5 * (
            math.log(2 * math.pi)
            + log_var
            + (hs_proj - mu_q).pow(2) / log_var.exp()
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
        he_proj, hs_proj = self._project(He, Hs)
        M = he_proj.shape[0]
        mu_q, logvar_raw = self._get_params(he_proj)
        pos = self._log_prob(hs_proj,                               mu_q, logvar_raw).mean()
        neg = self._log_prob(hs_proj[self._neg_perm(M, He.device)], mu_q, logvar_raw).mean()
        return pos - neg

    def variational_loss(self, He, Hs):
        he_proj, hs_proj = self._project(He, Hs)
        mu_q, logvar_raw = self._get_params(he_proj)
        loss = -self._log_prob(hs_proj, mu_q, logvar_raw).mean()
        return loss.clamp(min=-10.0)   # [Fix-VarLoss-Clamp]


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
        self.gate     = CausalGateUnit(stoch_dim, env_dim, hidden_dim)
        self.msg_tr   = nn.Linear(stoch_dim, stoch_dim)
        self.agg_norm = nn.LayerNorm(stoch_dim)
        self.agg_tr   = nn.Sequential(nn.Linear(stoch_dim, stoch_dim), nn.ReLU())

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
# 8. [TimeGMM-1] GRIN
# ---------------------------------------------------------------------------
class GRIN(nn.Module):
    """
    GMM-adapted Reversible Instance Normalization（TimeGMM Sec.2.1）。

    在 GMMHead 输出的归一化域 mu/sigma 上做 denorm，
    还原到原始数据尺度，使 GMM NLL 在原始尺度优化。

    per-node 可学习仿射参数 a [1,N,1], b [1,N,1]：
      mu_orig    = (mu_norm - b) / (a + eps) * std + mean
      sigma_orig = sigma_norm / (|a| + eps) * std

    fit(x_train) 在数据加载后调用一次，初始化 mean/std buffer。
    x_train: [T, N] 或 [T, N, F]，训练集归一化后的数据。
    """
    def __init__(self, n_nodes: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.a   = nn.Parameter(torch.ones(1, n_nodes, 1))
        self.b   = nn.Parameter(torch.zeros(1, n_nodes, 1))
        self.register_buffer("mean_x", torch.zeros(1, n_nodes, 1))
        self.register_buffer("std_x",  torch.ones(1, n_nodes, 1))

    def fit(self, x_train: torch.Tensor):
        if x_train.dim() == 2:
            x_train = x_train.unsqueeze(-1)
        mean = x_train.mean(dim=0, keepdim=True)[..., :1]
        std  = x_train.std(dim=0,  keepdim=True)[..., :1].clamp(min=self.eps)
        self.mean_x.data.copy_(mean.to(self.mean_x.device))
        self.std_x.data.copy_(std.to(self.std_x.device))

    def denorm_mu(self, mu_norm: torch.Tensor) -> torch.Tensor:
        std  = self.std_x.to(mu_norm.device)
        mean = self.mean_x.to(mu_norm.device)
        return (mu_norm - self.b) / (self.a + self.eps) * std + mean

    def denorm_sigma(self, sigma_norm: torch.Tensor) -> torch.Tensor:
        std = self.std_x.to(sigma_norm.device)
        return sigma_norm / (self.a.abs() + self.eps) * std


# ---------------------------------------------------------------------------
# 9. [TimeGMM-3] GMMHead
# ---------------------------------------------------------------------------
class GMMHead(nn.Module):
    """
    GMM 概率输出头 v2。

    [TimeGMM-3] 最后一个分量使用 sigma_min_heavy，其余使用 sigma_min。
    [TimeGMM-1] use_grin=True 时，forward 对 mu/sigma 做 GRIN denorm。
    """
    def __init__(self, in_dim: int, n_components: int = 3,
                 hidden_dim: int = 128,
                 sigma_min: float = 0.05,
                 sigma_min_heavy: float = 0.5,
                 use_grin: bool = True,
                 n_nodes: int = 1):
        super().__init__()
        self.K               = n_components
        self.sigma_min       = sigma_min
        self.sigma_min_heavy = sigma_min_heavy
        self.use_grin        = use_grin

        self.norm   = nn.LayerNorm(in_dim)
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.w_head     = nn.Linear(hidden_dim, n_components)
        self.mu_head    = nn.Linear(hidden_dim, n_components)
        self.sigma_head = nn.Linear(hidden_dim, n_components)

        if use_grin:
            self.grin = GRIN(n_nodes)

    def fit_grin(self, x_train: torch.Tensor):
        if self.use_grin:
            self.grin.fit(x_train)

    def forward(self, H_final):
        h       = self.shared(self.norm(H_final))
        w       = self.w_head(h)
        mu_norm = self.mu_head(h)
        sig_raw = F.softplus(self.sigma_head(h))

        if self.K == 1:
            sigma_norm = sig_raw + self.sigma_min
        else:
            sigma_norm = torch.cat([
                sig_raw[..., :-1] + self.sigma_min,
                sig_raw[...,  -1:] + self.sigma_min_heavy,
            ], dim=-1)

        if self.use_grin:
            mu    = self.grin.denorm_mu(mu_norm)
            sigma = self.grin.denorm_sigma(sigma_norm)
        else:
            mu, sigma = mu_norm, sigma_norm

        return w, mu, sigma


# ---------------------------------------------------------------------------
# 10. [TimeGMM-2] GMM 复合损失
# ---------------------------------------------------------------------------
def gmm_nll_loss(
    w: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    lambda_mean: float   = 0.1,
    lambda_weight: float = 0.01,
    huber_delta: float   = 1.0,
) -> tuple:
    """
    L_total = L_NLL + lambda_mean * L_mean(Huber) + lambda_weight * L_weight

    use_grin=True 时，mu/sigma/y 均在原始数据尺度（train.py 负责 y 的还原）。
    use_grin=False 时，均在归一化域，行为与 v1 相同。
    """
    sigma = sigma.clamp(min=1e-6)
    y_k   = y[..., 0:1].expand_as(mu)

    log_prob_k = (
        -0.5 * math.log(2 * math.pi)
        - torch.log(sigma)
        - 0.5 * ((y_k - mu) / sigma) ** 2
    )
    l_nll = -torch.logsumexp(F.log_softmax(w, dim=-1) + log_prob_k, dim=-1).mean()

    w_soft  = F.softmax(w, dim=-1)
    mu_mean = (w_soft * mu).sum(dim=-1)
    l_mean  = F.huber_loss(mu_mean, y[..., 0], delta=huber_delta)

    l_weight = F.mse_loss(w_soft.sum(dim=-1), torch.ones_like(w_soft.sum(dim=-1)))

    return l_nll + lambda_mean * l_mean + lambda_weight * l_weight, l_nll, l_mean, l_weight


# ---------------------------------------------------------------------------
# 11. GridCFN 主模型
# ---------------------------------------------------------------------------
class GridCFN(nn.Module):
    """
    GridCFN + GMM v2。

    forward() 返回 6 个值（与所有之前版本接口相同）：
      (w, mu, sigma, He, Hs, mi_loss)

    GRIN 初始化流程（main.py 中调用）：
      model = build_model(cfg, in_dim=in_dim, n_nodes=N).to(device)
      model.fit_grin(train_data_tensor)   # train_data: [T, N] 归一化后
    """
    def __init__(
        self,
        in_dim=1, gcn_hidden=64, tcn_hidden=64,
        env_dim=32, stoch_dim=32, ms_out_dim=32,
        n_scg_layers=3, out_dim=1, lambda_mi=0.5,
        gcn_layers=2, tcn_layers=4,
        n_components: int    = 3,
        lambda_mean: float   = 0.1,
        lambda_weight: float = 0.01,
        proj_dim: int        = 64,
        sigma_min_heavy: float = 0.5,
        huber_delta: float   = 1.0,
        use_grin: bool       = True,
        n_nodes: int         = 1,
    ):
        super().__init__()
        self.lambda_mi     = lambda_mi
        self.lambda_mean   = lambda_mean
        self.lambda_weight = lambda_weight
        self.huber_delta   = huber_delta
        self.use_grin      = use_grin

        self.backbone     = Backbone(in_dim, gcn_hidden, tcn_hidden, gcn_layers, tcn_layers)
        self.disentangler = CausalDisentangler(tcn_hidden, env_dim, stoch_dim)
        self.club         = CLUBEstimator(env_dim, stoch_dim, proj_dim=proj_dim)
        self.ms_context   = MultiScaleContext(env_dim, ms_out_dim)
        self.scgmp        = SCGMP(stoch_dim, env_dim, n_scg_layers)
        self.predictor    = GMMHead(
            ms_out_dim + stoch_dim,
            n_components=n_components,
            sigma_min_heavy=sigma_min_heavy,
            use_grin=use_grin,
            n_nodes=n_nodes,
        )

    def fit_grin(self, x_train: torch.Tensor):
        """x_train: [T, N] 或 [T, N, F]，训练集归一化后的数据。"""
        if self.use_grin:
            self.predictor.fit_grin(x_train)

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
        H              = self.backbone(x, adj_norm)
        He, Hs, He_seq = self.disentangler(H)
        mi_loss        = self.club(He, Hs)
        He_prime       = self.ms_context(He_seq)
        Hs_prime       = self.scgmp(Hs, He, edge_index)
        w, mu, sigma   = self.predictor(torch.cat([He_prime, Hs_prime], dim=-1))
        return w, mu, sigma, He, Hs, mi_loss

    def compute_loss(self, w, mu, sigma, y, mi_loss, lambda_mi=None):
        if lambda_mi is None:
            lambda_mi = self.lambda_mi
        gmm_loss, l_nll, l_mean, l_weight = gmm_nll_loss(
            w, mu, sigma, y,
            lambda_mean=self.lambda_mean,
            lambda_weight=self.lambda_weight,
            huber_delta=self.huber_delta,
        )
        return gmm_loss + lambda_mi * mi_loss, l_nll, l_mean, l_weight


def nll_gaussian_loss(mu, sigma, y):
    """已废弃，仅向后兼容。"""
    sigma = sigma.clamp(min=1e-6)
    return (torch.log(sigma)
            + 0.5 * math.log(2 * math.pi)
            + 0.5 * ((y - mu) / sigma) ** 2).mean()
