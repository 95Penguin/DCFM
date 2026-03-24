"""
GridCFN: A Causal Spatio-Temporal Framework for Power Flow Uncertainty Prediction
Reproduction based on IC2ECS 2025 paper by Zhao et al.

Architecture:
  Input X [T, N, F]
    └─ Backbone (GCN + TCN) → H [T, N, D]
         └─ Causal Disentangler → He [N, De], Hs [N, Ds]
              ├─ Multi-Scale Context (dilated conv) → H'e [N, De']
              └─ SCG-MP (causal gated message passing) → H's [N, Ds']
                   └─ Feature Fusion → H_final [N, De'+Ds']
                        └─ Predictor → (mu, sigma) [N, Fout]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Graph Convolutional Layer (simple spectral GCN)
# ---------------------------------------------------------------------------
class GCNLayer(nn.Module):
    """
    H_out = sigma(D^{-1/2} A_hat D^{-1/2} H W)
    A_hat = A + I  (self-loop added)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x        : [B, N, in_dim]  or  [N, in_dim]
        adj_norm : [N, N]  (pre-normalised adjacency, symmetric)
        returns  : [B, N, out_dim]
        """
        support = self.linear(x)                  # [..., N, out_dim]
        out = torch.matmul(adj_norm, support)     # [..., N, out_dim]
        return F.relu(out)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
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
    """Left-padded 1-D conv so output length == input length (causal)."""
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
        x = x.reshape(B * N, T, C).permute(0, 2, 1)   # [B*N, C, T]
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1).reshape(B, N, T, -1)   # [B, N, T, hidden]
        return x


# ---------------------------------------------------------------------------
# 3. Backbone: GCN + TCN
# ---------------------------------------------------------------------------
class Backbone(nn.Module):
    """
    Produces H ∈ R^{T×N×D} fusing spatial topology and temporal dynamics.
    Paper eq. (2): H = TCN(GCN(X, A))
    """
    def __init__(self, in_dim: int, gcn_hidden: int, tcn_hidden: int,
                 gcn_layers: int = 2, tcn_layers: int = 4):
        super().__init__()
        self.gcn = GCN(in_dim, gcn_hidden, gcn_hidden, gcn_layers)
        self.tcn = TCN(gcn_hidden, tcn_hidden, tcn_layers)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x        : [B, T, N, F]
        adj_norm : [N, N]
        returns H: [B, T, N, D]  (we keep T dim for disentangler)
        """
        B, T, N, F = x.shape
        # Apply GCN at each time step
        x_gcn = []
        for t in range(T):
            x_gcn.append(self.gcn(x[:, t], adj_norm))   # [B, N, gcn_hidden]
        x_gcn = torch.stack(x_gcn, dim=1)               # [B, T, N, gcn_hidden]

        # Apply TCN across time
        H = self.tcn(x_gcn.permute(0, 2, 1, 3))        # [B, N, T, tcn_hidden]
        H = H.permute(0, 2, 1, 3)                       # [B, T, N, tcn_hidden]
        return H


# ---------------------------------------------------------------------------
# 4. Causal Disentangler  (paper Sec. IV-A)
# ---------------------------------------------------------------------------
class CausalDisentangler(nn.Module):
    """
    Decomposes H → (He, Hs) via learned projections.
    MI minimisation is handled externally by the MINE estimator.

    We take the last time step of H as the 'current state', and
    project it into two independent subspaces.
    """
    def __init__(self, in_dim: int, env_dim: int, stoch_dim: int):
        super().__init__()
        self.env_proj   = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, env_dim)
        )
        self.stoch_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, stoch_dim)
        )

    def forward(self, H: torch.Tensor):
        """
        H   : [B, T, N, D]
        returns:
          He : [B, N, env_dim]    – environmental context
          Hs : [B, N, stoch_dim]  – stochastic entity repr
          H_seq: [B, T, N, D]     – full sequence kept for multi-scale context
        """
        h_last = H[:, -1]                # [B, N, D]  – use last time step
        He = self.env_proj(h_last)       # [B, N, env_dim]
        Hs = self.stoch_proj(h_last)     # [B, N, stoch_dim]
        return He, Hs, H


# ---------------------------------------------------------------------------
# 5. MINE – Mutual Information Neural Estimator  (paper eq. 3 / [11])
# ---------------------------------------------------------------------------
class MINEEstimator(nn.Module):
    """
    Estimates I(He, Hs) via the MINE lower bound:
        I >= E[T(x,y)] - log(E[e^{T(x,y')}])
    where y' is sampled from the marginal (i.e., shuffled batch).
    Returns the MI estimate as a scalar loss term (to be minimised).
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
        returns: scalar MI estimate
        """
        B, N, _ = He.shape
        He_flat = He.view(B * N, -1)
        Hs_flat = Hs.view(B * N, -1)

        # Joint score
        t_joint = self.net(torch.cat([He_flat, Hs_flat], dim=-1))

        # Marginal score: shuffle Hs along batch dimension
        idx = torch.randperm(B * N, device=He.device)
        Hs_shuffled = Hs_flat[idx]
        t_marginal = self.net(torch.cat([He_flat, Hs_shuffled], dim=-1))

        # MINE bound: E[T_joint] - log(E[exp(T_marginal)])
        mi_estimate = t_joint.mean() - torch.log(t_marginal.exp().mean() + 1e-8)
        return mi_estimate   # minimise this → minimise MI


# ---------------------------------------------------------------------------
# 6. Multi-Scale Context Modeling  (paper eq. 4)
# ---------------------------------------------------------------------------
class MultiScaleContext(nn.Module):
    """
    Processes He temporal sequence with L parallel dilated conv1d,
    then fuses them.  Paper: H'e = fuse([DilatedConv(He,T, d_l) for l in L])
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
        He_seq : [B, T, N, env_dim]  – full temporal sequence of env context
        returns: [B, N, out_dim]     – multi-scale fused context
        """
        B, T, N, C = He_seq.shape
        x = He_seq.permute(0, 2, 3, 1).reshape(B * N, C, T)   # [B*N, C, T]
        outs = [conv(x) for conv in self.convs]                 # each [B*N, out_dim, T]
        fused = torch.cat(outs, dim=1)                          # [B*N, out_dim*scales, T]
        fused = fused[:, :, -1]                                  # take last time step
        out = self.fuse(fused).view(B, N, -1)                   # [B, N, out_dim]
        return F.relu(out)


# ---------------------------------------------------------------------------
# 7. Spatial Causal Gated Message Passing  (paper Sec. IV-B, eq. 5-9)
# ---------------------------------------------------------------------------
class CausalGatingUnit(nn.Module):
    """
    Computes gate g_ij ∈ [0,1] for each edge (i,j).
    Input: Z_ij = [Hs_i || Hs_j || He_i || He_j]
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
        Z = torch.cat([Hs_i, Hs_j, He_i, He_j], dim=-1)   # [B, E, 2*ds+2*de]
        return self.gate_mlp(Z)                              # [B, E, 1]


class SCGMessagePassingLayer(nn.Module):
    """
    One layer of Spatial Causal Gated Message Passing.
    Implements eq. (7)-(9) from the paper.
    """
    def __init__(self, stoch_dim: int, env_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gate_unit = CausalGatingUnit(stoch_dim, env_dim, hidden_dim)
        self.msg_transform = nn.Linear(stoch_dim, stoch_dim, bias=False)
        self.agg_transform  = nn.Sequential(
            nn.Linear(stoch_dim, stoch_dim), nn.ReLU()
        )

    def forward(self, Hs: torch.Tensor, He: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Hs         : [B, N, stoch_dim]
        He         : [B, N, env_dim]
        edge_index : [2, E]  (source j, target i)
        returns H's: [B, N, stoch_dim]
        """
        B, N, Ds = Hs.shape
        src, dst = edge_index[0], edge_index[1]   # j, i

        # Gather node features for each edge
        Hs_j = Hs[:, src]   # [B, E, Ds]
        Hs_i = Hs[:, dst]   # [B, E, Ds]
        He_j = He[:, src]   # [B, E, De]
        He_i = He[:, dst]   # [B, E, De]

        # Compute causal gate  (eq. 6)
        g = self.gate_unit(Hs_i, Hs_j, He_i, He_j)  # [B, E, 1]

        # Raw message  (eq. 7)
        m_raw = self.msg_transform(Hs_j)              # [B, E, Ds]

        # Gated message  (eq. 8)
        m_causal = g * m_raw                          # [B, E, Ds]

        # Aggregate onto target nodes  (eq. 9)
        agg = torch.zeros(B, N, Ds, device=Hs.device)
        agg.scatter_add_(1,
                         dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, Ds),
                         m_causal)

        # Residual update
        Hs_new = self.agg_transform(Hs + agg)
        return Hs_new


class SCGMP(nn.Module):
    """Stack of L_SCG causal gated message passing layers."""
    def __init__(self, stoch_dim: int, env_dim: int,
                 n_layers: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            SCGMessagePassingLayer(stoch_dim, env_dim, hidden_dim)
            for _ in range(n_layers)
        ])

    def forward(self, Hs: torch.Tensor, He: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            Hs = layer(Hs, He, edge_index)
        return Hs


# ---------------------------------------------------------------------------
# 8. Probabilistic Predictor  (paper Sec. IV-C, eq. 10-11)
# ---------------------------------------------------------------------------
class ProbabilisticPredictor(nn.Module):
    """
    H_final = concat(H'e, H's)
    Outputs (mu, sigma) for each node and output feature.
    sigma is softplus-activated to ensure positivity.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head    = nn.Linear(hidden_dim, out_dim)
        self.sigma_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, H_final: torch.Tensor):
        """
        H_final : [B, N, De'+Ds']
        returns : mu [B, N, out_dim], sigma [B, N, out_dim]
        """
        h = self.net(H_final)
        mu    = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-6   # ensure > 0
        return mu, sigma


# ---------------------------------------------------------------------------
# 9. Loss Functions
# ---------------------------------------------------------------------------
def nll_gaussian_loss(mu: torch.Tensor, sigma: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
    """
    Negative Log-Likelihood for Gaussian  (paper eq. 13).
    L_NLL = mean[ (y - mu)^2 / (2*sigma^2) + log(sigma) ]
    """
    loss = ((y - mu) ** 2) / (2 * sigma ** 2) + torch.log(sigma)
    return loss.mean()


def crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor,
                  y: torch.Tensor) -> torch.Tensor:
    """
    CRPS for Gaussian predictive distribution (closed form).
    CRPS = sigma * [ (y-mu)/sigma * (2*Phi((y-mu)/sigma) - 1)
                     + 2*phi((y-mu)/sigma) - 1/sqrt(pi) ]
    Used as an evaluation metric, not a training loss.
    """
    from torch.distributions import Normal
    dist = Normal(mu, sigma)
    z = (y - mu) / sigma
    phi = dist.log_prob(y).exp()           # pdf at y
    Phi = dist.cdf(y)                      # cdf at y
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / (3.14159265 ** 0.5))
    return crps.mean()


# ---------------------------------------------------------------------------
# 10. Full GridCFN Model
# ---------------------------------------------------------------------------
class GridCFN(nn.Module):
    """
    Full GridCFN pipeline.

    Hyper-parameters (defaults match paper: λ=0.5, L_SCG=3, d_hidden=64):
      in_dim      : input feature dimension F
      gcn_hidden  : GCN hidden dim
      tcn_hidden  : TCN output dim  (= D in paper)
      env_dim     : He dimension  (De)
      stoch_dim   : Hs dimension  (Ds)
      ms_out_dim  : H'e dimension after multi-scale context
      n_scg_layers: L_SCG
      out_dim     : number of output variables (Fout)
      lambda_mi   : weight for MI regularisation loss
    """
    def __init__(
        self,
        in_dim: int      = 7,
        gcn_hidden: int  = 64,
        tcn_hidden: int  = 64,
        env_dim: int     = 32,
        stoch_dim: int   = 32,
        ms_out_dim: int  = 32,
        n_scg_layers: int = 3,
        out_dim: int     = 1,
        lambda_mi: float = 0.5,
        gcn_layers: int  = 2,
        tcn_layers: int  = 4,
    ):
        super().__init__()
        self.lambda_mi = lambda_mi

        self.backbone    = Backbone(in_dim, gcn_hidden, tcn_hidden,
                                    gcn_layers, tcn_layers)
        self.disentangler = CausalDisentangler(tcn_hidden, env_dim, stoch_dim)
        self.mine         = MINEEstimator(env_dim, stoch_dim)
        self.ms_context   = MultiScaleContext(env_dim, ms_out_dim)
        self.scgmp        = SCGMP(stoch_dim, env_dim, n_scg_layers)
        self.predictor    = ProbabilisticPredictor(ms_out_dim + stoch_dim, out_dim)

    @staticmethod
    def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalisation: D^{-1/2} (A+I) D^{-1/2}."""
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(deg, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        D = torch.diag(d_inv_sqrt)
        return D @ adj @ D

    @staticmethod
    def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
        """Convert adjacency matrix to edge_index [2, E]."""
        return adj.nonzero(as_tuple=False).t().contiguous()

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        x   : [B, T, N, F]    raw input features
        adj : [N, N]           adjacency matrix (0/1 or weighted)
        returns:
          mu    : [B, N, out_dim]
          sigma : [B, N, out_dim]
          mi_loss: scalar  (MI regularisation term)
        """
        adj_norm   = self.normalize_adj(adj)
        edge_index = self.adj_to_edge_index(adj)

        # --- Backbone ---
        H = self.backbone(x, adj_norm)        # [B, T, N, D]

        # --- Causal Disentanglement ---
        He, Hs, H_seq = self.disentangler(H)  # He:[B,N,De], Hs:[B,N,Ds]

        # --- MI loss (to be minimised) ---
        mi_loss = self.mine(He, Hs)

        # --- Multi-scale context on He ---
        # Build He sequence: project each time step of H to env space
        B, T, N, D = H_seq.shape
        He_seq = self.disentangler.env_proj(
            H_seq.reshape(B * T * N, D)
        ).reshape(B, T, N, -1)               # [B, T, N, De]
        He_prime = self.ms_context(He_seq)   # [B, N, ms_out_dim]

        # --- Spatial Causal Gated MP ---
        Hs_prime = self.scgmp(Hs, He, edge_index)   # [B, N, Ds]

        # --- Feature Fusion + Prediction ---
        H_final = torch.cat([He_prime, Hs_prime], dim=-1)  # [B, N, ms+Ds]
        mu, sigma = self.predictor(H_final)

        return mu, sigma, mi_loss

    def compute_loss(self, mu, sigma, y, mi_loss):
        """
        Total loss = L_NLL + lambda * L_MI   (paper eq. 12)
        y : [B, N, out_dim]
        """
        l_nll = nll_gaussian_loss(mu, sigma, y)
        l_total = l_nll + self.lambda_mi * mi_loss
        return l_total, l_nll, mi_loss
