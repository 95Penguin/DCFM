# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 1. FrequencyDecomposer — 可学习多尺度趋势/残差分解
# ---------------------------------------------------------------------------

class FrequencyDecomposer(nn.Module):
    def __init__(self, in_dim: int, candidates=(12, 24, 48, 96)):
        super().__init__()
        self.candidates = list(candidates)
        n = len(candidates)
        self.weight_net = nn.Linear(in_dim, n)
        self.scale_low  = nn.Parameter(torch.ones(in_dim))
        self.scale_high = nn.Parameter(torch.ones(in_dim))

    def _moving_avg(self, x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 1:
            return x
        T_len = x.shape[1]   # x: [B*N, T, F]
        pad_l = min((k - 1) // 2,       T_len - 1)
        pad_r = min(k - 1 - (k - 1) // 2, T_len - 1)
        x_t = x.permute(0, 2, 1)           # [B*N, F, T]
        x_t = F.pad(x_t, (pad_l, pad_r), mode="reflect")
        k_actual = min(k, x_t.shape[-1])
        x_t = F.avg_pool1d(x_t, kernel_size=k_actual, stride=1, padding=0)
        if x_t.shape[-1] > T_len:
            x_t = x_t[:, :, :T_len]
        elif x_t.shape[-1] < T_len:
            x_t = F.pad(x_t, (0, T_len - x_t.shape[-1]), mode="replicate")
        return x_t.permute(0, 2, 1)

    def forward(self, x: torch.Tensor):
        B, T, N, Fin = x.shape
        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, Fin)

        ctx = x_bn.mean(dim=1)
        w   = F.softmax(self.weight_net(ctx), dim=-1)

        ma_list = [self._moving_avg(x_bn, k) for k in self.candidates]
        ma_stack = torch.stack(ma_list, dim=-1)
        w_bc     = w.unsqueeze(1).unsqueeze(2)
        x_low_bn = (ma_stack * w_bc).sum(dim=-1)

        x_high_bn = x_bn - x_low_bn

        x_low  = x_low_bn.reshape(B, N, T, Fin).permute(0, 2, 1, 3)
        x_high = x_high_bn.reshape(B, N, T, Fin).permute(0, 2, 1, 3)

        return x_low * self.scale_low, x_high * self.scale_high


# ---------------------------------------------------------------------------
# 2. LowRankGCN — 混合先验低秩图卷积
# ---------------------------------------------------------------------------

class LowRankGCN(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int, out_dim: int,
                 rank_r: int = 8, n_layers: int = 2):
        super().__init__()
        self.rank_r   = rank_r
        self.n_nodes  = n_nodes
        self.U = nn.Parameter(torch.randn(n_nodes, rank_r) * 0.1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(n_layers)
        ])

    def _adj(self, adj_norm=None) -> torch.Tensor:
        A_adaptive = self.U @ self.U.T / math.sqrt(self.rank_r)
        A_adaptive = F.softmax(A_adaptive, dim=-1)
        
        if adj_norm is not None:
            alpha = torch.sigmoid(self.alpha)
            return (1.0 - alpha) * A_adaptive + alpha * adj_norm.to(device=A_adaptive.device, dtype=A_adaptive.dtype)
        return A_adaptive

    def rank_loss(self) -> torch.Tensor:
        G   = self.U.T @ self.U
        eps = 1e-4 * torch.eye(self.rank_r, device=self.U.device)
        try:
            L      = torch.linalg.cholesky(G + eps)
            logdet = 2.0 * L.diagonal().log().sum()
        except Exception:
            logdet = torch.logdet(G + eps)
        if not torch.isfinite(logdet):
            return G.new_tensor(0.0)
        return -logdet / self.rank_r

    def forward(self, x: torch.Tensor, adj_norm=None) -> torch.Tensor:
        A = self._adj(adj_norm)
        for i, layer in enumerate(self.layers):
            x = torch.matmul(A, layer(x))
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# 3. SparseGCN — 混合先验可微稀疏图卷积
# ---------------------------------------------------------------------------

class SparseGCN(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, emb_dim: int = 32):
        super().__init__()
        self.n_nodes   = n_nodes
        self.emb       = nn.Embedding(n_nodes, emb_dim)
        self.threshold   = nn.Parameter(torch.zeros(1))
        self.temperature = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.tensor(0.0))

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(n_layers)
        ])

    def _adj(self, device, adj_norm=None, wind_mask=None) -> torch.Tensor:
        idx   = torch.arange(self.n_nodes, device=device)
        E     = self.emb(idx)
        A_raw = E @ E.T / math.sqrt(E.shape[-1])

        temp = self.temperature.abs().clamp(min=1e-3)
        A_adaptive = A_raw * torch.sigmoid((A_raw - self.threshold) / temp)

        if wind_mask is not None:
            A_adaptive = A_adaptive * wind_mask.to(device)

        row_sum = A_adaptive.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        A_adaptive = A_adaptive / row_sum
        
        if adj_norm is not None:
            alpha = torch.sigmoid(self.alpha)
            return (1.0 - alpha) * A_adaptive + alpha * adj_norm.to(device=A_adaptive.device, dtype=A_adaptive.dtype)
        return A_adaptive

    def forward(self, x: torch.Tensor, adj_norm=None, wind_mask=None) -> torch.Tensor:
        A = self._adj(x.device, adj_norm, wind_mask)
        for i, layer in enumerate(self.layers):
            x = torch.matmul(A, layer(x))
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# 4. TCN 模块
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, bias=True)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        ng = min(8, channels)
        self.norm1 = nn.GroupNorm(ng, channels)
        self.norm2 = nn.GroupNorm(ng, channels)
        self.dropout = nn.Dropout(dropout)  # ─── [调整] 引入 Dropout 结构防止过拟合 ───

    def forward(self, x):
        r = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = F.gelu(self.norm2(self.conv2(x)))
        x = self.dropout(x)
        return x + r


class TCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=4, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden_dim, kernel_size, 2 ** i, dropout) for i in range(n_layers)]
        )

    def forward(self, x):
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C).permute(0, 2, 1)
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        return x.permute(0, 2, 1).reshape(B, N, T, -1)


# ---------------------------------------------------------------------------
# 5. AttentionPool — 可学习注意力时间池化
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden = max(1, dim // 2)
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score  = self.attn(x)
        weight = F.softmax(score, dim=1)
        return (weight * x).sum(dim=1)


# ---------------------------------------------------------------------------
# 6. DualTrackBackbone — 双轨 Backbone (优化能级尺度稳定)
# ---------------------------------------------------------------------------

class DualTrackBackbone(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int,
                 env_dim: int, stoch_dim: int,
                 gcn_hidden: int = 64, tcn_hidden: int = 64,
                 gcn_layers: int = 2, tcn_layers: int = 4,
                 rank_r: int = 8,
                 freq_candidates: tuple = (12, 24, 48, 96),
                 wind_mask=None,
                 dropout: float = 0.1):
        super().__init__()

        self.decomposer = FrequencyDecomposer(in_dim, candidates=freq_candidates)

        # 环境支路
        self.gcn_e   = LowRankGCN(n_nodes, in_dim, gcn_hidden, env_dim,
                                   rank_r=rank_r, n_layers=gcn_layers)
        self.norm_gcn_e = nn.LayerNorm(env_dim)  
        self.tcn_e   = TCN(in_dim=env_dim, hidden_dim=tcn_hidden, n_layers=tcn_layers, dropout=dropout)
        self.proj_tcn_e = nn.Linear(tcn_hidden, env_dim) if tcn_hidden != env_dim else nn.Identity()
        self.pool_e  = AttentionPool(env_dim)

        # 因果支路
        self.gcn_s   = SparseGCN(n_nodes, in_dim, gcn_hidden, stoch_dim, n_layers=gcn_layers)
        self.norm_gcn_s = nn.LayerNorm(stoch_dim)  
        self.tcn_s   = TCN(in_dim=stoch_dim, hidden_dim=tcn_hidden, n_layers=tcn_layers, dropout=dropout)
        self.proj_tcn_s = nn.Linear(tcn_hidden, stoch_dim) if tcn_hidden != stoch_dim else nn.Identity()
        self.pool_s  = AttentionPool(stoch_dim)

        # ─── [优化点] 分轨设计可学习隐特征比例乘子，相比原 decomposer.scale 更加匹配特征维度 ───
        self.scale_env = nn.Parameter(torch.ones(env_dim))
        self.scale_stoch = nn.Parameter(torch.ones(stoch_dim))

        if wind_mask is not None:
            self.register_buffer("wind_mask", wind_mask)
        else:
            self.wind_mask = None

    def rank_loss(self) -> torch.Tensor:
        return self.gcn_e.rank_loss()

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor = None):
        B, T, N, Fin = x.shape

        X_low, X_high = self.decomposer(x)

        # ─── [优化点] 趋势分量使用 mean 表达基准能级；随机分量使用 abs().mean 表达波动能量，防止求均值正负抵消 ───
        X_low_pooled  = X_low.mean(dim=1)
        X_high_pooled = X_high.abs().mean(dim=1)

        # 环境支路：GCN -> LayerNorm -> 尺度还原 -> TCN
        e_gcn = self.gcn_e(
            X_low.reshape(B * T, N, Fin), adj_norm=adj_norm
        ).reshape(B, T, N, -1)
        
        e_gcn = self.norm_gcn_e(e_gcn) * self.scale_env
        
        He_seq_raw = self.tcn_e(e_gcn.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        He_seq = self.proj_tcn_e(He_seq_raw)
        He = self.pool_e(He_seq)

        # 因果支路：GCN -> LayerNorm -> 尺度还原 -> TCN
        s_gcn = self.gcn_s(
            X_high.reshape(B * T, N, Fin), adj_norm=adj_norm, wind_mask=self.wind_mask
        ).reshape(B, T, N, -1)
        
        s_gcn = self.norm_gcn_s(s_gcn) * self.scale_stoch
        
        Hs_seq_raw = self.tcn_s(s_gcn.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        Hs_seq = self.proj_tcn_s(Hs_seq_raw)
        Hs = self.pool_s(Hs_seq)

        return He_seq, He, Hs, X_low_pooled, X_high_pooled


# ---------------------------------------------------------------------------
# 7. CLUBEstimator — 互信息估计器
# ---------------------------------------------------------------------------

class CLUBEstimator(nn.Module):
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

    # def _neg_perm(self, M, device):
    #     perm = torch.randperm(M, device=device)
    #     same = perm == torch.arange(M, device=device)
    #     if same.any() and M > 1:
    #         idx  = same.nonzero(as_tuple=True)[0]
    #         swap = (idx + 1) % M
    #         tmp          = perm[swap].clone()
    #         perm[swap]   = perm[idx].clone()
    #         perm[idx]    = tmp
    #     return perm

    def _neg_perm(self, M: int, device):
        perm = torch.randperm(M, device=device)
        arange = torch.arange(M, device=device)
        clash = (perm == arange).nonzero(as_tuple=True)[0]
        if len(clash) == 0:
            return perm
        if len(clash) >= 2:
            # clash>=2：整体循环右移一位，数学保证无不动点且仍是合法置换。
            roll_idx = torch.roll(torch.arange(len(clash), device=device), 1)
            perm[clash] = perm[clash[roll_idx]].clone()
        else:
            # clash==1：torch.roll 单元素无效，与任意非clash位置交换。
            # 置换特性保证交换后两个位置均无不动点。
            i = int(clash[0])
            non_clash = (perm != arange).nonzero(as_tuple=True)[0]
            j = int(non_clash[0])
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
        return perm

    def forward(self, x, y):
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
# 8. MultiScaleContext
# ---------------------------------------------------------------------------

class MultiScaleContext(nn.Module):
    def __init__(self, env_dim, ms_out_dim, dilations=(1, 7, 30), T_in: int = None):
        super().__init__()
        if T_in is not None:
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
        B, T, N, De = He_seq.shape
        x    = He_seq.permute(0, 2, 3, 1).reshape(B * N, De, T)
        outs = [conv(x)[:, :, -1] for conv in self.convs]
        return self.proj(torch.cat(outs, dim=-1)).reshape(B, N, -1)


# ---------------------------------------------------------------------------
# 9. SCG Message Passing
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
# 10. CFM Vector Field
# ---------------------------------------------------------------------------

class CFMVectorField(nn.Module):
    def __init__(self, out_dim: int, env_dim: int, stoch_dim: int,
                 hidden_dim: int = 128, time_emb_dim: int = 16,
                 max_freq: float = 1000.0,
                 T_out: int = 1, feat_dim: int = 1,
                 dropout: float = 0.1):
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
        self.dropout = nn.Dropout(dropout)  # ─── [调整] CFM场映射回归加上高维Dropout，防范尾流波动过拟合 ───

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
        h = h + self.dropout(self.layer1(h_norm * (1.0 + t_s1 + s1_st) + (t_b1 + b1_env)))
        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.dropout(self.layer2(h_norm * (1.0 + t_s2 + s2_st) + (t_b2 + b2_env)))
        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.dropout(self.layer3(h_norm * (1.0 + t_s3 + s3_st) + (t_b3 + b3_env)))
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# 11. DCFM — 主模型集成
# ---------------------------------------------------------------------------

class DCFM(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        in_dim=1, gcn_hidden=64, tcn_hidden=64,
        env_dim=32, stoch_dim=32, ms_out_dim=32,
        n_scg_layers=3, out_dim=1,
        lambda_mi=0.5,
        lambda_rank=0.1,
        rank_r=8,
        freq_candidates=(12, 24, 48, 96),
        wind_mask=None,
        gcn_layers=2, tcn_layers=4,
        cfm_hidden=128, cfm_time_emb_dim=16,
        chunk_size=8192,
        ms_dilations=(1, 7, 30),
        T_out=1,
        T_in=None,
        dropout=0.1,
    ):
        super().__init__()
        self.lambda_club = lambda_mi
        self.lambda_rank = lambda_rank
        self.feat_dim    = out_dim
        self.T_out       = T_out
        self.cfm_dim     = T_out * out_dim
        self.env_dim     = env_dim
        self.stoch_dim   = stoch_dim

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
            dropout         = dropout,
        )

        self.club_e = CLUBEstimator(env_dim,   in_dim)
        self.club_s = CLUBEstimator(stoch_dim, in_dim)

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
            dropout       = dropout,
        )

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

    def rank_loss(self) -> torch.Tensor:
        return self.backbone.rank_loss()

    def forward(self, x, adj_norm, edge_index):
        He_seq, He, Hs, X_low_pooled, X_high_pooled = self.backbone(x, adj_norm=adj_norm)
        He_prime = self.ms_context(He_seq)
        Hs_prime = self.scgmp(Hs, He, edge_index)
        return He_prime, Hs_prime, He, Hs, X_low_pooled, X_high_pooled

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


# Backward compatibility for checkpoints and scripts created before the rename.
# New code should import and instantiate ``DCFM`` directly.
GridCFN = DCFM
