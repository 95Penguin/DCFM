"""
baselines/stgcn.py
STGCN: Spatio-Temporal Graph Convolutional Networks
论文: Yu et al., IJCAI 2018  https://arxiv.org/abs/1709.04875
参考: https://github.com/hazdzz/STGCN

修复:
  [Bug9]  compute_laplacian 用 torch.linalg.eigvalsh 计算真实 lambda_max，
          正确做 L_tilde = (2 / lambda_max) * L_sym - I
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Chebyshev 图卷积 ──────────────────────────────────────────────────────

class ChebConv(nn.Module):
    """
    K 阶 Chebyshev 谱图卷积。
    输入/输出格式: [B, N, C]（在时间循环内逐步调用）
    """
    def __init__(self, in_channels: int, out_channels: int,
                 K: int = 3, bias: bool = True):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.empty(K, in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """
        x       : [B, N, in_channels]
        L_tilde : [N, N]
        returns : [B, N, out_channels]
        """
        Tx = [x]
        if self.K > 1:
            Tx.append(torch.einsum('nm,bmc->bnc', L_tilde, x))
        for _ in range(2, self.K):
            Tk = (2 * torch.einsum('nm,bmc->bnc', L_tilde, Tx[-1]) - Tx[-2])
            Tx.append(Tk)
        out = sum(
            torch.einsum('bni,io->bno', Tx[k], self.weight[k])
            for k in range(self.K)
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    @staticmethod
    def compute_laplacian(adj: torch.Tensor) -> torch.Tensor:
        """
        计算归一化 Chebyshev Laplacian:
          L_sym   = I - D^{-1/2} A D^{-1/2}
          L_tilde = (2 / lambda_max) * L_sym - I
        [Bug9 修复] 用 eigvalsh 计算真实 lambda_max，不再硬编码为 2。
        """
        N = adj.shape[0]
        d = adj.sum(dim=1).clamp(min=1e-6)
        d_inv_sqrt = d.pow(-0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        L_sym = torch.eye(N, device=adj.device) - D_inv_sqrt @ adj @ D_inv_sqrt
        try:
            lambda_max = torch.linalg.eigvalsh(L_sym).max().item()
            lambda_max = max(lambda_max, 1e-6)
        except Exception:
            lambda_max = 2.0   # fallback
        L_tilde = (2.0 / lambda_max) * L_sym - torch.eye(N, device=adj.device)
        return L_tilde


# ── Temporal Gated Conv ───────────────────────────────────────────────────

class TemporalGatedConv(nn.Module):
    """
    时序门控卷积 (GLU): output = tanh(P) ⊙ sigmoid(Q)
    时间维缩短: T_out = T_in - kernel_size + 1
    输入/输出格式: [B, C, N, T]
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * out_channels,
                              kernel_size=(1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        P, Q = out.chunk(2, dim=1)
        return torch.tanh(P) * torch.sigmoid(Q)


# ── ST-Conv Block ─────────────────────────────────────────────────────────

class STConvBlock(nn.Module):
    """
    ST-Conv Block: TCN → GraphConv → TCN + 残差 + BN
    时间维在两次 TCN 后缩短: T_out = T - 2*(kernel_size-1)
    """
    def __init__(self, in_channels: int, spatial_channels: int,
                 out_channels: int, kernel_size: int, K: int):
        super().__init__()
        self.tgc1 = TemporalGatedConv(in_channels,      spatial_channels, kernel_size)
        self.cheb = ChebConv(spatial_channels, spatial_channels, K)
        self.tgc2 = TemporalGatedConv(spatial_channels, out_channels,     kernel_size)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.res_proj = (nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
                         if in_channels != out_channels else None)

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """
        x       : [B, C_in, N, T]
        L_tilde : [N, N]
        returns : [B, C_out, N, T - 2*(kernel_size-1)]
        """
        # Temporal 1
        out = self.tgc1(x)                           # [B, C_s, N, T1]
        B, Cs, N, T1 = out.shape

        # Graph conv（逐时间步）
        out_t = out.permute(0, 3, 2, 1).reshape(B * T1, N, Cs)
        out_t = F.relu(self.cheb(out_t, L_tilde))
        out   = out_t.reshape(B, T1, N, Cs).permute(0, 3, 2, 1)  # [B, C_s, N, T1]

        # Temporal 2
        out = self.tgc2(out)                          # [B, C_out, N, T2]
        T2  = out.shape[-1]

        # 残差：裁剪到 T2
        res = x if self.res_proj is None else self.res_proj(x)
        res = res[..., -T2:]
        return self.bn(out + res)


# ── STGCN ─────────────────────────────────────────────────────────────────

class STGCN(nn.Module):
    """
    STGCN 单步预测版本 (T_out=1)。
    T_in=168, kernel_size=3, n_blocks=2 时:
      Block1: 168 → 164, Block2: 164 → 160，取最后时间步输出。
    """
    def __init__(self,
                 in_dim:      int,
                 hidden_dim:  int = 64,
                 kernel_size: int = 3,
                 K:           int = 3,
                 n_blocks:    int = 2,
                 out_dim:     int = 1):
        super().__init__()
        self.blocks = nn.ModuleList()
        c_in = in_dim
        for _ in range(n_blocks):
            self.blocks.append(
                STConvBlock(c_in, hidden_dim, hidden_dim, kernel_size, K))
            c_in = hidden_dim
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """
        x       : [B, T_in, N, F]
        L_tilde : [N, N]
        returns : [B, N, out_dim]
        """
        out = x.permute(0, 3, 2, 1)              # [B, F, N, T]
        for block in self.blocks:
            out = block(out, L_tilde)             # [B, hidden, N, T']
        out = out[..., -1].permute(0, 2, 1)       # [B, N, hidden]
        out = self.ln(out)
        return self.fc(out)                        # [B, N, out_dim]
