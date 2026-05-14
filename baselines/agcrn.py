"""
baselines/agcrn.py
AGCRN: Adaptive Graph Convolutional Recurrent Network（多步预测版）
论文: Bai et al., NeurIPS 2020  https://arxiv.org/abs/2007.02842

多步改动: end_conv 输出 T_out*out_dim，reshape 为 [B, T_out, N, out_dim]。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    """Adaptive View-Wise GCN: DAGG + NAPL"""
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
        node_num = node_embeddings.shape[0]

        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.T)), dim=1)

        support_set = [torch.eye(node_num, device=supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.mm(supports, support_set[-1]))

        weights = torch.einsum('nd,dkio->nkio', node_embeddings,
                               self.weights_pool)
        bias    = torch.matmul(node_embeddings, self.bias_pool)

        x_g_list = [torch.einsum('nm,bmi->bni', sup, X) for sup in support_set]
        x_g = torch.stack(x_g_list, dim=1)
        x_gconv = torch.einsum('bkni,nkio->bno', x_g, weights) + bias
        return x_gconv


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


class AGCRN(nn.Module):
    """AGCRN 多步预测版本。"""
    def __init__(self,
                 num_nodes:  int,
                 in_dim:     int,
                 hidden_dim: int = 64,
                 n_layers:   int = 2,
                 embed_dim:  int = 10,
                 cheb_k:     int = 2,
                 out_dim:    int = 1,
                 T_out:      int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.T_out      = T_out
        self.out_dim    = out_dim
        self.num_nodes  = num_nodes

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
            out_channels = T_out * out_dim,
            kernel_size  = (1, 1),
            bias         = True,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x       : [B, T_in, N, F]
        returns : [B, T_out, N, out_dim]
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

        h_cat = torch.cat(hidden_list, dim=-1)           # [B, N, layers*hidden]
        h_cat = h_cat.permute(0, 2, 1).unsqueeze(-1)    # [B, layers*hidden, N, 1]
        out   = self.end_conv(h_cat)                     # [B, T_out*out_dim, N, 1]
        out   = out.squeeze(-1).permute(0, 2, 1)         # [B, N, T_out*out_dim]
        return out.reshape(B, self.T_out, N, self.out_dim)
