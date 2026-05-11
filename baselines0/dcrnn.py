"""
baselines/dcrnn.py
DCRNN: Diffusion Convolutional Recurrent Neural Network
论文: Li et al., ICLR 2018  https://arxiv.org/abs/1707.01926
参考: https://github.com/liyaguang/DCRNN

修复:
  [Bug1] DiffusionConv 补回 T0=x（恒等项），每个 support 收集 K 项 [T0..T_{K-1}]
  [Bug8] build_supports 加自环后行归一化，前向用 adj，后向用 adj.T
"""
import torch
import torch.nn as nn


# ── 扩散卷积 ──────────────────────────────────────────────────────────────

class DiffusionConv(nn.Module):
    """
    K 阶双向扩散卷积。
    对每个 support 收集 [T0=x, T1=A@x, ..., T_{K-1}]，共 K 项。
    两个 support（前向/后向）共 2K 项拼接后线性映射。
    weight shape: [2*K*in_channels, out_channels]
    """
    def __init__(self, in_channels: int, out_channels: int, K: int = 2):
        super().__init__()
        self.K = K
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(
            torch.empty(2 * K * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, supports: list) -> torch.Tensor:
        """
        x        : [B, N, in_channels]
        supports : [fwd_adj, bwd_adj]，每个 [N, N]
        returns  : [B, N, out_channels]
        """
        diffused = []
        for sup in supports:
            h = x                             # T0 = x（恒等）
            diffused.append(h)
            for _ in range(self.K - 1):       # T1 … T_{K-1}
                h = torch.einsum('nm,bmc->bnc', sup, h)
                diffused.append(h)
        feat = torch.cat(diffused, dim=-1)    # [B, N, 2*K*in_channels]
        return feat @ self.weight             # [B, N, out_channels]


# ── DCRNN Cell ────────────────────────────────────────────────────────────

class DCRNNCell(nn.Module):
    """GRU，线性变换替换为 DiffusionConv。"""
    def __init__(self, in_channels: int, hidden_dim: int,
                 K: int, bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_rz = DiffusionConv(in_channels + hidden_dim, hidden_dim * 2, K)
        self.conv_c  = DiffusionConv(in_channels + hidden_dim, hidden_dim, K)
        if bias:
            self.bias_rz = nn.Parameter(torch.zeros(hidden_dim * 2))
            self.bias_c  = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.bias_rz = self.bias_c = None

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                supports: list) -> torch.Tensor:
        inp = torch.cat([x, h], dim=-1)
        rz  = self.conv_rz(inp, supports)
        if self.bias_rz is not None:
            rz = rz + self.bias_rz
        rz = torch.sigmoid(rz)
        r, z = rz.chunk(2, dim=-1)

        inp_r = torch.cat([x, r * h], dim=-1)
        c = self.conv_c(inp_r, supports)
        if self.bias_c is not None:
            c = c + self.bias_c
        c = torch.tanh(c)
        return z * h + (1 - z) * c


# ── Encoder ───────────────────────────────────────────────────────────────

class DCRNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int,
                 n_layers: int, K: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            ic = in_channels if i == 0 else hidden_dim
            self.cells.append(DCRNNCell(ic, hidden_dim, K))

    def forward(self, x_seq: torch.Tensor, supports: list) -> list:
        """
        x_seq : [B, T, N, F]
        returns: list of [B, N, hidden_dim], one per layer
        """
        B, T, N, _ = x_seq.shape
        hiddens = [
            torch.zeros(B, N, self.hidden_dim, device=x_seq.device)
            for _ in range(self.n_layers)
        ]
        for t in range(T):
            inp = x_seq[:, t]
            for i, cell in enumerate(self.cells):
                hiddens[i] = cell(inp, hiddens[i], supports)
                inp = hiddens[i]
        return hiddens


# ── DCRNN（T_out=1） ───────────────────────────────────────────────────────

class DCRNN(nn.Module):
    """单步预测 DCRNN (T_out=1)。"""
    def __init__(self,
                 in_dim:     int,
                 hidden_dim: int = 64,
                 n_layers:   int = 2,
                 K:          int = 2,
                 out_dim:    int = 1):
        super().__init__()
        self.encoder     = DCRNNEncoder(in_dim, hidden_dim, n_layers, K)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def build_supports(adj: torch.Tensor) -> list:
        """
        构建前向和后向扩散支持矩阵（加自环后行归一化）。
        adj: [N, N]
        注：对称邻接矩阵下 fwd == bwd，双向退化为重复计算，
            这是无向图场景下的已知局限（不影响运行）。
        """
        def _row_norm(a: torch.Tensor) -> torch.Tensor:
            a = a + torch.eye(a.shape[0], device=a.device)
            d = a.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return a / d

        fwd = _row_norm(adj)
        bwd = _row_norm(adj.T.contiguous())
        return [fwd, bwd]

    def forward(self, x: torch.Tensor, supports: list) -> torch.Tensor:
        """
        x        : [B, T_in, N, F]
        supports : list of [N, N]（已移到正确 device）
        returns  : [B, N, out_dim]
        """
        hiddens = self.encoder(x, supports)
        return self.output_proj(hiddens[-1])