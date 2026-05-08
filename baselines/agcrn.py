"""
baselines/agcrn.py
AGCRN: Adaptive Graph Convolutional Recurrent Network
论文: Bai et al., NeurIPS 2020  https://arxiv.org/abs/2007.02842
参考: https://github.com/LeiBAI/AGCRN

修复:
  [Bug6]  AVWGCN 改用官方 DAGG（softmax ReLU E@E^T）而非 Chebyshev Laplacian。
          support_set = [I, A, A^2, ..., A^{K-1}]（直接幂次，不用 Chebyshev 递推）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── AVWGCN ────────────────────────────────────────────────────────────────

class AVWGCN(nn.Module):
    """
    Adaptive View-Wise GCN:
      DAGG: A = softmax(ReLU(E @ E^T))
      NAPL: W_node = einsum(E, weights_pool)
      support_set = [I, A, A^2, ..., A^{K-1}]
    """
    def __init__(self, dim_in: int, dim_out: int,
                 embed_dim: int, cheb_k: int = 2):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, dim_out))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, X: torch.Tensor,
                node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        X               : [B, N, dim_in]
        node_embeddings : [N, embed_dim]
        returns         : [B, N, dim_out]
        """
        node_num = node_embeddings.shape[0]

        # DAGG
        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.T)), dim=1)

        # support_set = [I, A, A^2, ...]
        support_set = [torch.eye(node_num, device=supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.mm(supports, support_set[-1]))

        # NAPL
        weights = torch.einsum('nd,dkio->nkio', node_embeddings,
                               self.weights_pool)   # [N, K, dim_in, dim_out]
        bias    = torch.matmul(node_embeddings, self.bias_pool)  # [N, dim_out]

        x_g_list = [torch.einsum('nm,bmi->bni', sup, X) for sup in support_set]
        x_g = torch.stack(x_g_list, dim=1)          # [B, K, N, dim_in]
        x_gconv = torch.einsum('bkni,nkio->bno', x_g, weights) + bias
        return x_gconv


# ── AGCRN Cell ────────────────────────────────────────────────────────────

class AGCRNCell(nn.Module):
    def __init__(self, node_num: int, dim_in: int, dim_out: int,
                 embed_dim: int, cheb_k: int):
        super().__init__()
        self.node_num   = node_num
        self.hidden_dim = dim_out
        self.gate   = AVWGCN(dim_in + dim_out, 2 * dim_out, embed_dim, cheb_k)
        self.update = AVWGCN(dim_in + dim_out, dim_out,     embed_dim, cheb_k)

    def forward(self, x: torch.Tensor, state: torch.Tensor,
                node_embeddings: torch.Tensor) -> torch.Tensor:
        x_state = torch.cat([x, state], dim=-1)
        z_r = torch.sigmoid(self.gate(x_state, node_embeddings))
        z, r = z_r.chunk(2, dim=-1)
        x_r_state = torch.cat([x, r * state], dim=-1)
        c = torch.tanh(self.update(x_r_state, node_embeddings))
        return z * state + (1 - z) * c

    def init_hidden_state(self, batch_size: int,
                          device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num,
                           self.hidden_dim, device=device)


# ── AGCRN ─────────────────────────────────────────────────────────────────

class AGCRN(nn.Module):
    """AGCRN 单步预测版本 (T_out=1)。"""
    def __init__(self,
                 num_nodes:  int,
                 in_dim:     int,
                 hidden_dim: int = 64,
                 n_layers:   int = 2,
                 embed_dim:  int = 10,
                 cheb_k:     int = 2,
                 out_dim:    int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embed_dim), requires_grad=True)

        self.encoder = nn.ModuleList([
            AGCRNCell(
                node_num  = num_nodes,
                dim_in    = in_dim    if i == 0 else hidden_dim,
                dim_out   = hidden_dim,
                embed_dim = embed_dim,
                cheb_k    = cheb_k,
            )
            for i in range(n_layers)
        ])

        self.end_conv = nn.Conv2d(
            in_channels  = n_layers * hidden_dim,
            out_channels = out_dim,
            kernel_size  = (1, 1),
            bias         = True,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x       : [B, T_in, N, F]
        returns : [B, N, out_dim]
        """
        B, T, N, _ = x.shape
        device = x.device

        hidden_list = [cell.init_hidden_state(B, device)
                       for cell in self.encoder]

        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.encoder):
                hidden_list[i] = cell(inp, hidden_list[i],
                                      self.node_embeddings)
                inp = hidden_list[i]

        h_cat = torch.cat(hidden_list, dim=-1)          # [B, N, layers*hidden]
        h_cat = h_cat.permute(0, 2, 1).unsqueeze(-1)   # [B, layers*hidden, N, 1]
        out   = self.end_conv(h_cat)                    # [B, out_dim, N, 1]
        return out.squeeze(-1).permute(0, 2, 1)         # [B, N, out_dim]
