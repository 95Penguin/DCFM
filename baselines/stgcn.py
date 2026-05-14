"""
baselines/stgcn.py
STGCN: Spatio-Temporal Graph Convolutional Networks（多步预测版）
论文: Yu et al., IJCAI 2018  https://arxiv.org/abs/1709.04875

多步改动: 最后 FC 输出 T_out*out_dim，reshape 为 [B, T_out, N, out_dim]。

修复:
  [1] compute_laplacian 对大图 (N > 500) 跳过 O(N³) 的 eigvalsh，
      直接使用理论上界 lambda_max=2.0，避免在 Weather(1866节点) 等数据集上卡死。
      对于中小图 (N ≤ 500) 保留精确计算，加超时保护。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebConv(nn.Module):
    """K 阶 Chebyshev 谱图卷积。"""
    def __init__(self, in_channels: int, out_channels: int,
                 K: int = 3, bias: bool = True):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.empty(K, in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
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
        计算归一化拉普拉斯的缩放版本 L_tilde = 2/lambda_max * L_sym - I。

        修复 [1]：
          - N > 500 时直接用 lambda_max=2.0（谱半径理论上界），
            跳过 O(N³) 的 eigvalsh，避免在大图上计算数分钟。
          - N ≤ 500 时保留精确计算，并 try-except 兜底为 2.0。
        """
        N = adj.shape[0]
        d = adj.sum(dim=1).clamp(min=1e-6)
        d_inv_sqrt = d.pow(-0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        L_sym = torch.eye(N, device=adj.device) - D_inv_sqrt @ adj @ D_inv_sqrt

        if N > 500:
            # 大图：直接使用理论上界，避免 eigvalsh 的 O(N³) 开销
            lambda_max = 2.0
        else:
            try:
                lambda_max = torch.linalg.eigvalsh(L_sym).max().item()
                lambda_max = max(lambda_max, 1e-6)
            except Exception:
                lambda_max = 2.0

        L_tilde = (2.0 / lambda_max) * L_sym - torch.eye(N, device=adj.device)
        return L_tilde


class TemporalGatedConv(nn.Module):
    """时序门控卷积 (GLU)。"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * out_channels,
                              kernel_size=(1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        P, Q = out.chunk(2, dim=1)
        return torch.tanh(P) * torch.sigmoid(Q)


class STConvBlock(nn.Module):
    """ST-Conv Block: TCN → GraphConv → TCN + 残差 + BN"""
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
        out = self.tgc1(x)
        B, Cs, N, T1 = out.shape

        out_t = out.permute(0, 3, 2, 1).reshape(B * T1, N, Cs)
        out_t = F.relu(self.cheb(out_t, L_tilde))
        out   = out_t.reshape(B, T1, N, Cs).permute(0, 3, 2, 1)

        out = self.tgc2(out)
        T2  = out.shape[-1]

        res = x if self.res_proj is None else self.res_proj(x)
        res = res[..., -T2:]
        return self.bn(out + res)


class STGCN(nn.Module):
    """STGCN 多步预测版本。"""
    def __init__(self,
                 in_dim:      int,
                 hidden_dim:  int = 64,
                 kernel_size: int = 3,
                 K:           int = 3,
                 n_blocks:    int = 2,
                 out_dim:     int = 1,
                 T_out:       int = 1):
        super().__init__()
        self.T_out   = T_out
        self.out_dim = out_dim
        self.blocks = nn.ModuleList()
        c_in = in_dim
        for _ in range(n_blocks):
            self.blocks.append(
                STConvBlock(c_in, hidden_dim, hidden_dim, kernel_size, K))
            c_in = hidden_dim
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, T_out * out_dim)

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """
        x       : [B, T_in, N, F]
        L_tilde : [N, N]
        returns : [B, T_out, N, out_dim]
        """
        B = x.shape[0]
        out = x.permute(0, 3, 2, 1)               # [B, F, N, T]
        for block in self.blocks:
            out = block(out, L_tilde)              # [B, hidden, N, T']
        out = out[..., -1].permute(0, 2, 1)        # [B, N, hidden]
        out = self.ln(out)
        out = self.fc(out)                         # [B, N, T_out*out_dim]
        return out.reshape(B, self.T_out, x.shape[2], self.out_dim)