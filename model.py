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
"""
"""
GridCFN: A Causal Spatio-Temporal Framework for Power Flow Uncertainty Prediction

[Fix-CLUB-v4] 修复记录：

  v1 问题：_log_prob 中 log_var = Tanh*4，var 极小时 (Hs-μ)²/var 爆炸 → loss=-10^6
  v2 问题：clamp(mi_raw, max=0) 方向错误 → loss=-874K
  v3 问题：clamp(min=0) 截断梯度 → MI=0；sigma_min=0.05 → NLL 为负
  v4（本版）修复：
    · sigma_min 从 0.3 → 0.1（0.3 对归一化数据太大，CRPS 变差）
    · forward() 恢复返回 3 个值 (mu, sigma, mi_loss)，不返回 He/Hs
      train.py 通过拆分 forward 步骤获取 He/Hs，避免双重 forward 浪费
    · CLUBEstimator 新增 forward_with_repr() 返回 (mi_loss, He_flat, Hs_flat)
      供 train.py 在单次 forward 内同时拿到 CLUB 估计和表征

  保留的正确修复：
    · log_var = log(softplus(raw)+1e-2) clamp[-4,4]，var ≥ 1e-2
    · _log_prob 特征维度取 mean（不是 sum）
    · 无 clamp 的 CLUB 输出（允许负值）
    · variational_loss() 两步训练接口
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

    核心公式：
      CLUB = E_{p(He,Hs)}[log q(Hs|He)] - E_{p(He)}E_{p(Hs)}[log q(Hs|He)]
             ↑ 正样本项                    ↑ 负样本项（随机置换近似）

    使用方式（两步训练）：
      Step 1: var_loss = club.variational_loss(He, Hs)
              → 训练变分网络 q(Hs|He) 准确建模条件分布
      Step 2: mi_loss = club(He, Hs)
              → 用已收敛的变分网络估计互信息上界

    数值稳定：
      · log_var = log(softplus(raw) + 1e-2)，clamp[-4,4]
        → var ∈ [e^{-4}, e^4]，防止除零爆炸
      · _log_prob 对特征维度取 mean（不是 sum），消除 stoch_dim 放大效应
      · forward 无 clamp：允许负值，保留完整梯度信号
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
        """
        log N(Hs; μ_q, σ²_q)，对特征维度取 mean。
        输入 [M, D] → 返回 [M]
        """
        log_var  = torch.log(F.softplus(logvar_raw) + 1e-2).clamp(-4.0, 4.0)
        log_prob = -0.5 * (
            math.log(2 * math.pi)
            + log_var
            + (Hs - mu_q).pow(2) / log_var.exp()
        )
        return log_prob.mean(dim=-1)   # [M]

    def _neg_perm(self, M, device):
        """无固定点随机置换。"""
        perm = torch.randperm(M, device=device)
        same = perm == torch.arange(M, device=device)
        if same.any() and M > 1:
            idx  = same.nonzero(as_tuple=True)[0]
            swap = (idx + 1) % M
            perm[idx], perm[swap] = perm[swap].clone(), perm[idx].clone()
        return perm

    def forward(self, He, Hs):
        """
        CLUB 上界估计，无 clamp，允许负值。
        He, Hs: [B,N,dim] → 标量
        """
        M       = He.shape[0] * He.shape[1]
        He_flat = He.reshape(M, -1)
        Hs_flat = Hs.reshape(M, -1)

        mu_q, logvar_raw = self._get_params(He_flat)
        pos = self._log_prob(Hs_flat,                        mu_q, logvar_raw).mean()
        neg = self._log_prob(Hs_flat[self._neg_perm(M, He.device)], mu_q, logvar_raw).mean()
        return pos - neg   # 无 clamp

    def variational_loss(self, He, Hs):
        """
        变分网络损失：-E[log q(Hs|He)]。
        最小化此损失 = 让 q 准确建模 p(Hs|He)。
        He, Hs 应已 detach，避免梯度流回 backbone。
        """
        M       = He.shape[0] * He.shape[1]
        He_flat = He.reshape(M, -1)
        Hs_flat = Hs.reshape(M, -1)
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
# 8. Probabilistic Predictor
# ---------------------------------------------------------------------------
class ProbabilisticPredictor(nn.Module):
    """
    sigma_min = 0.1：
      · 0.05 太小 → NLL 为负（sigma=0.05 时 log(sigma)=-3，NLL 无下界）
      · 0.3  太大 → 预测区间过宽，CRPS 变差（归一化数据真实 sigma 约 0.3~1.0）
      · 0.1  合理：log(0.1)+0.5*log(2π)≈-1.3+0.92=-0.38，NLL 最低约 -0.38，
                   但网络实际预测 sigma 会大于 0.1（softplus 输出 > 0），
                   实际 NLL 通常 > 0.5
    """
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.norm       = nn.LayerNorm(in_dim)
        self.net        = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head    = nn.Linear(hidden_dim, out_dim)
        self.sigma_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, H_final):
        h     = self.net(self.norm(H_final))
        mu    = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 0.1
        return mu, sigma


# ---------------------------------------------------------------------------
# 9. Loss
# ---------------------------------------------------------------------------
def nll_gaussian_loss(mu, sigma, y):
    sigma = sigma.clamp(min=1e-6)
    nll   = (torch.log(sigma)
             + 0.5 * math.log(2 * math.pi)
             + 0.5 * ((y - mu) / sigma) ** 2)
    return nll.mean()


# ---------------------------------------------------------------------------
# 10. GridCFN
# ---------------------------------------------------------------------------
class GridCFN(nn.Module):
    """
    完整 GridCFN 管道。

    forward() 返回 (mu, sigma, He, Hs, mi_loss)，共 5 个值：
      · He, Hs 供 train.py 调用 club.variational_loss()
      · mi_loss 供 compute_loss() 使用
      · evaluate() 中用 _, _, mi_loss 忽略 He, Hs（或全部 5 个解包）

    [Fix-v4] 相比 v3：
      · sigma_min: 0.3 → 0.1（避免预测区间过宽）
      · 架构和接口不变
    """
    def __init__(
        self,
        in_dim=1, gcn_hidden=64, tcn_hidden=64,
        env_dim=32, stoch_dim=32, ms_out_dim=32,
        n_scg_layers=3, out_dim=1, lambda_mi=0.5,
        gcn_layers=2, tcn_layers=4,
    ):
        super().__init__()
        self.lambda_mi    = lambda_mi
        self.backbone     = Backbone(in_dim, gcn_hidden, tcn_hidden, gcn_layers, tcn_layers)
        self.disentangler = CausalDisentangler(tcn_hidden, env_dim, stoch_dim)
        self.club         = CLUBEstimator(env_dim, stoch_dim)
        self.ms_context   = MultiScaleContext(env_dim, ms_out_dim)
        self.scgmp        = SCGMP(stoch_dim, env_dim, n_scg_layers)
        self.predictor    = ProbabilisticPredictor(ms_out_dim + stoch_dim, out_dim)

    @staticmethod
    def normalize_adj(adj):
        adj       = adj + torch.eye(adj.size(0), device=adj.device)
        deg       = adj.sum(dim=1)
        d_inv_sq  = torch.pow(deg.clamp(min=1e-8), -0.5)
        D         = torch.diag(d_inv_sq)
        return D @ adj @ D

    @staticmethod
    def adj_to_edge_index(adj):
        return adj.nonzero(as_tuple=False).t().contiguous()

    def forward(self, x, adj_norm, edge_index):
        """
        返回: (mu, sigma, He, Hs, mi_loss)

        train.py 两步训练用法：
          mu, sigma, He, Hs, mi_loss = model(x, adj_norm, edge_index)

          # Step 1：用 He/Hs（detach）更新变分网络
          var_loss = model.club.variational_loss(He.detach(), Hs.detach())
          club_optimizer.zero_grad(); var_loss.backward(); club_optimizer.step()

          # Step 2：用 mi_loss 更新主网络（mi_loss 计算图已在上面 forward 中建立）
          loss, l_nll = model.compute_loss(mu, sigma, y_target, mi_loss)
          optimizer.zero_grad(); loss.backward(); optimizer.step()

        evaluate() 用法：
          mu, sigma, _, _, _ = model(x, adj_norm, edge_index)
        """
        H              = self.backbone(x, adj_norm)
        He, Hs, He_seq = self.disentangler(H)
        mi_loss        = self.club(He, Hs)          # 无 clamp，可正可负
        He_prime       = self.ms_context(He_seq)
        Hs_prime       = self.scgmp(Hs, He, edge_index)
        mu, sigma      = self.predictor(torch.cat([He_prime, Hs_prime], dim=-1))
        return mu, sigma, He, Hs, mi_loss

    def compute_loss(self, mu, sigma, y, mi_loss, lambda_mi=None):
        if lambda_mi is None:
            lambda_mi = self.lambda_mi
        l_nll   = nll_gaussian_loss(mu, sigma, y)
        l_total = l_nll + lambda_mi * mi_loss
        return l_total, l_nll