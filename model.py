"""
GridCFN: A Causal Spatio-Temporal Framework for Power Flow Uncertainty Prediction
CFM 版本 v3（修复并行 sample() 的 reshape bug）

[Bug 说明]
  并行 sample() 里 repeat_interleave + reshape 的维度顺序不匹配：

  repeat_interleave(S, dim=0) 的内存排列：
    [b0s0, b0s1, ..., b0s(S-1), b1s0, ..., b(B-1)s(S-1)]
    即"每个 batch 元素连续重复 S 次"，外层是 B，内层是 S。

  错误写法：x.reshape(S, B, N, out_dim)
    PyTorch reshape 按行优先读取，把前 B 个元素放第一行：
    s=0 行变成 [b0s0, b0s1, b0s2, b1s0]（混入了不同 sample 和不同 batch）
    导致 mean/std 完全在错误的元素上计算，预测结果混乱。

  正确写法：x.reshape(B, S, N, out_dim).permute(1, 0, 2, 3).contiguous()
    先按实际内存排列解包（外层 B，内层 S），再把 S 维移到最前面。
    这样 samples[s, b, n, :] = b 号样本的第 s 个粒子，语义正确。

[效果影响]
  这个 bug 使并行采样的 mean/std 完全错误（混入了其他 batch 的预测值），
  等效于在随机数上算统计量，所以：
    · mu（均值）几乎变成噪声，MAE 大幅上升（约 2~3 倍）
    · sigma 也失去意义，PICP/CRPS 全部失真
  这解释了为什么 v3 并行版比 v2 串行版指标差那么多。

[修复后预期]
  修复后并行版与串行版数值应完全一致（仅速度不同），
  v3 的所有速度收益（3~5x）在修复后都能正常享受。
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
# ---------------------------------------------------------------------------
class CLUBEstimator(nn.Module):
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
        M       = He.shape[0] * He.shape[1]
        He_flat = F.normalize(He.reshape(M, -1), dim=-1)
        Hs_flat = F.normalize(Hs.reshape(M, -1), dim=-1)
        mu_q, logvar_raw = self._get_params(He_flat)
        pos = self._log_prob(Hs_flat,                               mu_q, logvar_raw).mean()
        neg = self._log_prob(Hs_flat[self._neg_perm(M, He.device)], mu_q, logvar_raw).mean()
        return pos - neg

    def variational_loss(self, He, Hs):
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
        g   = self.gate(Hs[:, dst], Hs[:, src], He[:, dst], He[:, src])
        m   = g * self.msg_tr(Hs[:, src])
        agg = torch.zeros(B, N, Ds, device=Hs.device, dtype=Hs.dtype)
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
# 8. CFM Vector Field
# ---------------------------------------------------------------------------
class CFMVectorField(nn.Module):
    """
    条件向量场 v_θ(x_t, t | c)。
    对数均匀时间编码 + 3 层 MLP + 残差。
    """

    def __init__(self, out_dim: int, cond_dim: int, hidden_dim: int = 128,
                 time_emb_dim: int = 16, max_freq: float = 1000.0):
        super().__init__()
        self.out_dim = out_dim
        n_freqs = time_emb_dim // 2
        assert n_freqs * 2 == time_emb_dim, "time_emb_dim 必须是偶数"

        freqs = torch.exp(
            torch.linspace(0.0, math.log(max_freq), n_freqs)
        ) * math.pi
        self.register_buffer("freqs", freqs)

        in_dim = out_dim + time_emb_dim + cond_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )
        self.skip_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer2    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.layer3    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.out_proj  = nn.Linear(hidden_dim, out_dim)

    def _time_embed(self, t: torch.Tensor, B: int, N: int) -> torch.Tensor:
        angles = t.reshape(B, 1) * self.freqs.unsqueeze(0)
        emb    = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb.unsqueeze(1).expand(B, N, -1)

    def forward(self, x_t, t, c):
        B, N, _ = x_t.shape
        t_emb = self._time_embed(t, B, N)
        inp   = torch.cat([x_t, t_emb, c], dim=-1)
        h  = self.input_proj(inp) + self.skip_proj(inp)
        h  = h + self.layer2(h)
        h  = h + self.layer3(h)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# 9. GridCFN（CFM 版 v3，修复并行 sample()）
# ---------------------------------------------------------------------------
class GridCFN(nn.Module):

    def __init__(
        self,
        in_dim=1, gcn_hidden=64, tcn_hidden=64,
        env_dim=32, stoch_dim=32, ms_out_dim=32,
        n_scg_layers=3, out_dim=1, lambda_mi=0.5,
        gcn_layers=2, tcn_layers=4,
        cfm_hidden=128, cfm_time_emb_dim=16,
    ):
        super().__init__()
        self.lambda_mi = lambda_mi
        self.out_dim   = out_dim
        cond_dim = ms_out_dim + stoch_dim

        self.backbone     = Backbone(in_dim, gcn_hidden, tcn_hidden, gcn_layers, tcn_layers)
        self.disentangler = CausalDisentangler(tcn_hidden, env_dim, stoch_dim)
        self.club         = CLUBEstimator(env_dim, stoch_dim)
        self.ms_context   = MultiScaleContext(env_dim, ms_out_dim)
        self.scgmp        = SCGMP(stoch_dim, env_dim, n_scg_layers)
        self.vector_field = CFMVectorField(
            out_dim=out_dim, cond_dim=cond_dim,
            hidden_dim=cfm_hidden, time_emb_dim=cfm_time_emb_dim,
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
        H              = self.backbone(x, adj_norm)
        He, Hs, He_seq = self.disentangler(H)
        mi_loss        = self.club(He, Hs)
        He_prime       = self.ms_context(He_seq)
        Hs_prime       = self.scgmp(Hs, He, edge_index)
        context_feat   = torch.cat([He_prime, Hs_prime], dim=-1)
        return context_feat, He, Hs, mi_loss

    def cfm_loss(self, context_feat: torch.Tensor,
                 y_target: torch.Tensor,
                 n_t_samples: int = 4) -> torch.Tensor:
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
            x_t   = (1.0 - t_bc) * x0 + t_bc * y_target
            u_t   = y_target - x0
            v_pred = self.vector_field(x_t, t, context_feat)
            losses.append(F.mse_loss(v_pred, u_t))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(self, context_feat: torch.Tensor,
               n_samples: int = 50,
               n_steps: int = 20) -> torch.Tensor:
        """
        并行 ODE 采样（修复 reshape 维度顺序 bug）。

        核心修复：
          repeat_interleave(S, dim=0) 的内存排列是"每个 batch 元素连续重复 S 次"：
            [b0s0, b0s1, ..., b0s(S-1), b1s0, ..., b(B-1)s(S-1)]
            外层循环是 B（batch），内层循环是 S（samples）。

          错误写法：x.reshape(S, B, N, out_dim)
            PyTorch 按行优先（C顺序）读取，把前 B 个连续元素当作第一行，
            即 s=0 行变成 [b0s0, b0s1, b0s2, b1s0]——把来自同一 batch 的
            不同 sample 和来自不同 batch 的 sample 混在一起，完全错误。

          正确写法：x.reshape(B, S, N, out_dim).permute(1, 0, 2, 3).contiguous()
            先按实际内存结构解包为 [B, S, N, out_dim]（外 B 内 S），
            再把 S 维换到最前面得到 [S, B, N, out_dim]。
            此时 result[s, b, n, :] = b号batch的第s个粒子，语义正确。

        参数：
          n_samples : 粒子数（验证时 50，测试时 200）
          n_steps   : 欧拉步数（I-CFM 路径近线性，20 步已足够）

        显存注意（并行版）：
          实际 forward 的 batch size = B * n_samples
          Weather: B=4, S=50, N=1866 → 4*50*1866=373200 节点同时处理
          若 OOM，在 config 里把 cfm_n_samples 降到 20

        返回：[n_samples, B, N, out_dim]
        """
        B, N, _ = context_feat.shape
        S       = n_samples
        device  = context_feat.device
        dt      = 1.0 / n_steps

        # [B, N, D] → [B*S, N, D]（每个 batch 元素连续重复 S 次）
        c = context_feat.detach().repeat_interleave(S, dim=0)   # [B*S, N, D]
        x = torch.randn(B * S, N, self.out_dim, device=device)  # [B*S, N, out_dim]

        for step in range(n_steps):
            t_val = step * dt
            t_vec = torch.full((B * S,), t_val, device=device, dtype=torch.float32)
            x = x + dt * self.vector_field(x, t_vec, c)

        # [B*S, N, out_dim]
        # 内存排列：[b0s0, b0s1, ..., b0s(S-1), b1s0, ..., b(B-1)s(S-1)]
        # reshape(B, S, N, out_dim) 正确按外 B 内 S 解包
        # permute(1, 0, 2, 3) → [S, B, N, out_dim]
        return x.reshape(B, S, N, self.out_dim).permute(1, 0, 2, 3).contiguous()