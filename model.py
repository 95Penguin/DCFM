"""
GridCFN: A Causal Spatio-Temporal Framework for Power Flow Uncertainty Prediction

[CFM 版本] 将概率输出头从参数化高斯替换为 Conditional Flow Matching（CFM）：

  原来：ProbabilisticPredictor → 直接输出 (mu, sigma) → NLL 损失
  现在：CFMVectorField → 学习向量场 u_t(x|c) → CFM MSE 损失
        推断时：ODE 积分（Euler/RK4） → 多次采样 → 用样本均值/标准差估计不确定性

架构变化说明：
  · 骨干 + 因果解耦 + CLUB + MS-Context + SCG-MP 全部保留不变
  · 最终特征 H_final = [He_prime; Hs_prime]（同原版，作为条件向量 c）
  · CFMVectorField(x_t, t, c) → v，学习从噪声到数据的速度场
  · forward() 返回 (context_feat, He, Hs, mi_loss)
      context_feat: [B,N,De'+Ds'] 条件特征，用于 CFM 损失计算
  · cfm_loss(context_feat, y_target) → 标量 CFM 训练损失
  · sample(context_feat, n_samples, n_steps) → [S,B,N,out_dim] 样本集合

CFM 训练损失：
  给定条件特征 c，目标值 x1（归一化后的 y），源 x0 ~ N(0,I)：
    插值：x_t = (1-t)*x0 + t*x1
    目标向量场：u_t = x1 - x0
    损失：MSE(CFMVectorField(x_t, t, c), u_t)

推断（ODE 积分）：
  x_0 ~ N(0,I)  [n_samples 个]
  欧拉积分：x_{t+Δt} = x_t + Δt * CFMVectorField(x_t, t, c)
  最终 x_1 即预测样本
  用 n_samples 个样本的均值/标准差作为 mu/sigma 返回（与 train.py evaluate 接口兼容）

MI 正则（CLUB）：
  与原版完全相同，不受 CFM 替换影响
  总损失 = CFM_loss + lambda_mi * mi_loss
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

    使用方式（两步训练）：
      Step 1: var_loss = club.variational_loss(He, Hs)
      Step 2: mi_loss = club(He, Hs)
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
        log_var  = torch.log(F.softplus(logvar_raw) + 1e-2).clamp(-4.0, 4.0)
        log_prob = -0.5 * (
            math.log(2 * math.pi)
            + log_var
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
        """
        CLUB 上界估计，无 clamp，允许负值。
        He, Hs: [B,N,dim] → 标量

        [Fix-Elec] L2 归一化：
          Electricity 等数据集归一化后 He/Hs 数值范围极小，
          变分网络无法区分正负样本，导致 MI≈0、VarLoss<0。
          将 He_flat/Hs_flat 投影到单位球面后，
          无论原始数值尺度多小，变分网络都能有效学习条件分布。
          eps=1e-8 防止全零向量除零。
        """
        M       = He.shape[0] * He.shape[1]
        He_flat = F.normalize(He.reshape(M, -1), dim=-1)
        Hs_flat = F.normalize(Hs.reshape(M, -1), dim=-1)
        mu_q, logvar_raw = self._get_params(He_flat)
        pos = self._log_prob(Hs_flat,                               mu_q, logvar_raw).mean()
        neg = self._log_prob(Hs_flat[self._neg_perm(M, He.device)], mu_q, logvar_raw).mean()
        return pos - neg

    def variational_loss(self, He, Hs):
        """
        变分网络损失：-E[log q(Hs|He)]。
        最小化此损失 = 让 q 准确建模 p(Hs|He)。
        He, Hs 应已 detach，避免梯度流回 backbone。

        [Fix-Elec] 同 forward，先做 L2 归一化再送入变分网络。
        """
        M       = He.shape[0] * He.shape[1]
        He_flat = F.normalize(He.reshape(M, -1), dim=-1)
        Hs_flat = F.normalize(Hs.reshape(M, -1), dim=-1)
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
# 8. CFM Vector Field Network（替换原 ProbabilisticPredictor）
# ---------------------------------------------------------------------------
class CFMVectorField(nn.Module):
    """
    条件向量场网络 v_θ(x_t, t | c)。

    输入：
      x_t : [B, N, out_dim]  当前状态（插值点）
      t   : [B, 1, 1]        当前时间（0~1），broadcast 到 [B,N,1]
      c   : [B, N, cond_dim] 条件特征（来自骨干网络的 H_final）

    输出：
      v   : [B, N, out_dim]  向量场（速度），预测 dx/dt

    网络结构：
      将 x_t + t_embed + c 拼接后通过 2 层 MLP 预测速度。
      时间编码：sin/cos 傅里叶特征（4 个频率 → 8 维），
        比直接输入标量 t 更稳定。
    """

    def __init__(self, out_dim: int, cond_dim: int, hidden_dim: int = 128,
                 time_emb_dim: int = 8):
        super().__init__()
        self.out_dim     = out_dim
        self.time_emb_dim = time_emb_dim

        # 时间傅里叶编码频率（固定，不可学习）
        n_freqs = time_emb_dim // 2
        self.register_buffer(
            "freqs",
            torch.arange(1, n_freqs + 1, dtype=torch.float32) * math.pi
        )

        in_dim = out_dim + time_emb_dim + cond_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _time_embed(self, t: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """
        t: 标量或 [B] 或 [B,1,1]  → 返回 [B, N, time_emb_dim]
        """
        t = t.reshape(-1)                   # [B]
        angles = t.unsqueeze(1) * self.freqs.unsqueeze(0)   # [B, n_freqs]
        emb    = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, time_emb_dim]
        return emb.unsqueeze(1).expand(B, N, -1)            # [B, N, time_emb_dim]

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        """
        x_t : [B, N, out_dim]
        t   : [B] 或 标量（batch 内同一时间步）
        c   : [B, N, cond_dim]
        → v : [B, N, out_dim]
        """
        B, N, _ = x_t.shape
        t_emb   = self._time_embed(t, B, N)          # [B, N, time_emb_dim]
        inp     = torch.cat([x_t, t_emb, c], dim=-1) # [B, N, in_dim]
        return self.net(inp)                          # [B, N, out_dim]


# ---------------------------------------------------------------------------
# 9. GridCFN（CFM 版）
# ---------------------------------------------------------------------------
class GridCFN(nn.Module):
    """
    GridCFN with Conditional Flow Matching output head.

    forward() 返回 (context_feat, He, Hs, mi_loss)：
      context_feat : [B, N, De'+Ds']  条件特征，用于 cfm_loss() 和 sample()
      He, Hs       : 供 CLUB 变分网络更新
      mi_loss      : CLUB 互信息上界（用于总损失）

    cfm_loss(context_feat, y_target) → 标量 CFM 训练损失
      训练：对每个 batch 随机采样 t, x0，计算插值 x_t 和目标向量场 u_t，
            最小化 MSE(v_θ(x_t,t,c), u_t)

    sample(context_feat, n_samples, n_steps) → [S, B, N, out_dim]
      推断：从 x0 ~ N(0,I) 出发，欧拉积分到 x1，重复 n_samples 次

    训练总损失：
      L = cfm_loss + lambda_mi * mi_loss
    """

    def __init__(
        self,
        in_dim=1, gcn_hidden=64, tcn_hidden=64,
        env_dim=32, stoch_dim=32, ms_out_dim=32,
        n_scg_layers=3, out_dim=1, lambda_mi=0.5,
        gcn_layers=2, tcn_layers=4,
        cfm_hidden=128, cfm_time_emb_dim=8,
    ):
        super().__init__()
        self.lambda_mi = lambda_mi
        self.out_dim   = out_dim

        cond_dim = ms_out_dim + stoch_dim   # H_final 的维度

        self.backbone     = Backbone(in_dim, gcn_hidden, tcn_hidden, gcn_layers, tcn_layers)
        self.disentangler = CausalDisentangler(tcn_hidden, env_dim, stoch_dim)
        self.club         = CLUBEstimator(env_dim, stoch_dim)
        self.ms_context   = MultiScaleContext(env_dim, ms_out_dim)
        self.scgmp        = SCGMP(stoch_dim, env_dim, n_scg_layers)

        # CFM 向量场网络（替代 ProbabilisticPredictor）
        self.vector_field = CFMVectorField(
            out_dim=out_dim,
            cond_dim=cond_dim,
            hidden_dim=cfm_hidden,
            time_emb_dim=cfm_time_emb_dim,
        )

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
        """
        返回: (context_feat, He, Hs, mi_loss)

        与 train.py 两步训练配合：
          context_feat, He, Hs, mi_loss = model(x, adj_norm, edge_index)

          # Step 1: 更新变分网络
          var_loss = model.club.variational_loss(He.detach(), Hs.detach())
          club_optimizer.zero_grad(); var_loss.backward(); club_optimizer.step()

          # Step 2: 重新估计 MI，更新主网络
          mi_loss_new = model.club(He, Hs)
          cfm_loss    = model.cfm_loss(context_feat, y_target)
          loss        = cfm_loss + model.lambda_mi * mi_loss_new
          optimizer.zero_grad(); loss.backward(); optimizer.step()
        """
        H              = self.backbone(x, adj_norm)
        He, Hs, He_seq = self.disentangler(H)
        mi_loss        = self.club(He, Hs)
        He_prime       = self.ms_context(He_seq)
        Hs_prime       = self.scgmp(Hs, He, edge_index)
        context_feat   = torch.cat([He_prime, Hs_prime], dim=-1)  # [B,N,De'+Ds']
        return context_feat, He, Hs, mi_loss

    def cfm_loss(self, context_feat: torch.Tensor,
                 y_target: torch.Tensor) -> torch.Tensor:
        """
        Conditional Flow Matching 训练损失（MSE on vector field）。

        参数：
          context_feat : [B, N, cond_dim]  条件特征（来自 forward）
          y_target     : [B, N, out_dim]   目标值（归一化后的真实 y）

        算法：
          1. x1 = y_target（目标分布样本）
          2. x0 ~ N(0, I)（源分布噪声）
          3. t ~ Uniform(0, 1)（随机时间步）
          4. x_t = (1-t) * x0 + t * x1（线性插值路径）
          5. u_t = x1 - x0（目标向量场，CFM 的 ground truth）
          6. loss = MSE(v_θ(x_t, t, c), u_t)

        注：这是最基础的 I-CFM（Independent CFM），
            t=0 对应噪声，t=1 对应真实数据。
        """
        B, N, out_dim = y_target.shape
        device = y_target.device

        x1 = y_target                                             # [B, N, out_dim]
        x0 = torch.randn_like(x1)                                # [B, N, out_dim]
        t  = torch.rand(B, device=device)                        # [B]

        # 线性插值（OT-CFM/I-CFM 的 μ_t(x0,x1) = (1-t)*x0 + t*x1）
        t_bc   = t.reshape(B, 1, 1)                              # [B, 1, 1]
        x_t    = (1.0 - t_bc) * x0 + t_bc * x1                  # [B, N, out_dim]
        u_t    = x1 - x0                                         # [B, N, out_dim] 目标向量场

        v_pred = self.vector_field(x_t, t, context_feat)         # [B, N, out_dim]
        return F.mse_loss(v_pred, u_t)

    @torch.no_grad()
    def sample(self, context_feat: torch.Tensor,
               n_samples: int = 100,
               n_steps: int = 20) -> torch.Tensor:
        """
        ODE 积分推断，返回多组样本。

        参数：
          context_feat : [B, N, cond_dim]
          n_samples    : 每个位置采样的粒子数（越多，µ/σ 估计越稳定）
          n_steps      : 欧拉积分步数（越多，ODE 积分越精确；20~50 通常足够）

        返回：
          samples : [n_samples, B, N, out_dim]

        推断流程：
          t: 0 → 1，步长 dt = 1/n_steps
          x_{t+dt} = x_t + dt * v_θ(x_t, t, c)
          最终 x_1 即为样本
        """
        B, N, cond_dim = context_feat.shape
        device = context_feat.device
        dt = 1.0 / n_steps

        all_samples = []
        for _ in range(n_samples):
            x = torch.randn(B, N, self.out_dim, device=device)   # [B, N, out_dim]
            for step in range(n_steps):
                t_val = step * dt
                t_vec = torch.full((B,), t_val, device=device)   # [B]
                v = self.vector_field(x, t_vec, context_feat)     # [B, N, out_dim]
                x = x + dt * v
            all_samples.append(x)

        return torch.stack(all_samples, dim=0)   # [n_samples, B, N, out_dim]
