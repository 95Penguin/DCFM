"""
baselines/mtgnn.py
MTGNN: Connecting the Dots - Multivariate Time Series Forecasting with GNNs
论文: Wu et al., KDD 2020  https://arxiv.org/abs/2005.11650
参考: https://github.com/nnzhan/MTGNN  (layer.py + net.py)

多步改动: end_conv2 输出 T_out*out_dim，reshape 为 [B, T_out, N, out_dim]。

修复:
  [1] GraphLearner.forward 中 s1.fill_(1) 改为 torch.ones_like(s1)，
      避免原地修改 s1 张量（虽然功能上不影响结果，但会破坏 autograd 图，
      在某些 PyTorch 版本下可能引起梯度计算错误）。
  [2] training 分支中仅用 adj_noisy 决定 topk 索引，mask 值写入全 1，
      与原论文保持一致（mask 是二值的，不需要保留 noisy 权重值）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from torch.nn import init


# ── nconv ─────────────────────────────────────────────────────────────────

class nconv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """x: [B,C,N,T], A: [N,N] → [B,C,N,T]"""
        return torch.einsum('ncwl,vw->ncvl', x, A).contiguous()


# ── GraphLearner ──────────────────────────────────────────────────────────

class GraphLearner(nn.Module):
    """
    对齐官方 graph_constructor:
      nv1 = tanh(alpha * lin1(E1))
      nv2 = tanh(alpha * lin2(E2))
      A   = ReLU(tanh(alpha * (nv1@nv2^T - nv2@nv1^T)))  ← 反对称差
      topk 稀疏化
    """
    def __init__(self, num_nodes: int, embed_dim: int = 40,
                 alpha: float = 3.0, top_k: int = 20):
        super().__init__()
        self.emb1  = nn.Embedding(num_nodes, embed_dim)
        self.emb2  = nn.Embedding(num_nodes, embed_dim)
        self.lin1  = nn.Linear(embed_dim, embed_dim)
        self.lin2  = nn.Linear(embed_dim, embed_dim)
        self.alpha = alpha
        self.top_k = top_k

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        nv1 = torch.tanh(self.alpha * self.lin1(self.emb1(idx)))
        nv2 = torch.tanh(self.alpha * self.lin2(self.emb2(idx)))
        a   = torch.mm(nv1, nv2.T) - torch.mm(nv2, nv1.T)
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros_like(adj)
        if self.training:
            # 训练时加微小随机扰动以打破 topk 的确定性，提升探索性
            adj_noisy = adj + torch.rand_like(adj) * 0.01
        else:
            adj_noisy = adj
        _, t1 = adj_noisy.topk(self.top_k, dim=1)
        # 修复 [1][2]：使用 torch.ones(...) 而非 s1.fill_(1)
        # s1.fill_(1) 会原地修改 s1 破坏 autograd 图；
        # 此处 mask 只需二值 0/1，不需要保留 topk 的实际权重。
        mask.scatter_(1, t1, torch.ones(mask.shape[0], self.top_k,
                                        device=mask.device))
        return adj * mask


# ── mixprop ───────────────────────────────────────────────────────────────

class mixprop(nn.Module):
    """
    官方 mix-hop propagation，数据格式 [B,C,N,T]。
    h_0=x, h_k=alpha*x+(1-alpha)*(A_norm@h_{k-1})
    out = MLP(cat([h_0,...,h_{gdep}]))
    """
    def __init__(self, c_in: int, c_out: int,
                 gdep: int, dropout: float, alpha: float):
        super().__init__()
        self.nc    = nconv()
        self.mlp   = nn.Conv2d((gdep + 1) * c_in, c_out, kernel_size=(1, 1))
        self.gdep  = gdep
        self.alpha = alpha

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        d   = adj.sum(dim=1)
        a   = adj / d.view(-1, 1)
        h   = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nc(h, a)
            out.append(h)
        return self.mlp(torch.cat(out, dim=1))


# ── dilated_inception ─────────────────────────────────────────────────────

class dilated_inception(nn.Module):
    """
    官方 dilated_inception: 4 个 kernel (2,3,6,7) 并行，输出 cat。
    时间维会自然缩短（不做 same-padding），最短的输出决定时间维长度。
    cout 必须是 4 的倍数。
    """
    def __init__(self, cin: int, cout: int, dilation_factor: int = 1):
        super().__init__()
        self.kernel_set = [2, 3, 6, 7]
        assert cout % len(self.kernel_set) == 0, \
            f"cout={cout} 必须是 {len(self.kernel_set)} 的倍数"
        cout_each = cout // len(self.kernel_set)
        self.tconv = nn.ModuleList([
            nn.Conv2d(cin, cout_each, kernel_size=(1, k),
                      dilation=(1, dilation_factor))
            for k in self.kernel_set
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs  = [conv(x) for conv in self.tconv]
        T_min = outs[-1].size(3)
        outs  = [o[..., -T_min:] for o in outs]
        return torch.cat(outs, dim=1)


# ── 官方自定义 LayerNorm ──────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """官方自定义 LayerNorm，支持按节点索引 idx 切片 affine 参数。"""
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape   = tuple(normalized_shape)
        self.eps                = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(*normalized_shape))
            self.bias   = nn.Parameter(torch.empty(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias',   None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if self.elementwise_affine:
            w = self.weight[:, idx, :]
            b = self.bias[:,   idx, :]
            return F.layer_norm(x, tuple(x.shape[1:]), w, b, self.eps)
        return F.layer_norm(x, tuple(x.shape[1:]), None, None, self.eps)


# ── MTGNN ─────────────────────────────────────────────────────────────────

class MTGNN(nn.Module):
    """
    MTGNN 多步预测版本，对齐官方 gtnet 架构。

    注意: hidden_dim 必须是 4 的倍数（dilated_inception 要求）。
    推荐: hidden_dim=32, skip_dim=64, end_dim=128。
    """
    def __init__(self,
                 num_nodes:  int,
                 in_dim:     int,
                 hidden_dim: int   = 32,
                 skip_dim:   int   = 64,
                 end_dim:    int   = 128,
                 n_layers:   int   = 3,
                 depth:      int   = 2,
                 dropout:    float = 0.3,
                 propalpha:  float = 0.05,
                 tanhalpha:  float = 3.0,
                 embed_dim:  int   = 40,
                 top_k:      int   = 20,
                 out_dim:    int   = 1,
                 T_out:      int   = 1,
                 seq_length: int   = 168):
        super().__init__()
        self.num_nodes  = num_nodes
        self.dropout    = dropout
        self.n_layers   = n_layers
        self.seq_length = seq_length
        self.T_out      = T_out
        self.out_dim    = out_dim

        kernel_size = 7
        self.receptive_field = n_layers * (kernel_size - 1) + 1

        self.start_conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1))
        self.gc = GraphLearner(num_nodes, embed_dim, tanhalpha,
                               min(top_k, num_nodes - 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs   = nn.ModuleList()
        self.gconv1       = nn.ModuleList()
        self.gconv2       = nn.ModuleList()
        self.norm         = nn.ModuleList()

        # skip_convs 统一用 (1,1) kernel，forward 中取最后时间步
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, skip_dim, kernel_size=(1, 1))
            for _ in range(n_layers)
        ])
        self.skip0 = nn.Conv2d(in_dim,    skip_dim, kernel_size=(1, 1))
        self.skipE = nn.Conv2d(hidden_dim, skip_dim, kernel_size=(1, 1))

        rf_size = 1
        for j in range(1, n_layers + 1):
            rf_size_j = rf_size + (kernel_size - 1)
            self.filter_convs.append(dilated_inception(hidden_dim, hidden_dim))
            self.gate_convs.append(  dilated_inception(hidden_dim, hidden_dim))
            self.gconv1.append(mixprop(hidden_dim, hidden_dim, depth,
                                       dropout, propalpha))
            self.gconv2.append(mixprop(hidden_dim, hidden_dim, depth,
                                       dropout, propalpha))
            if seq_length > self.receptive_field:
                norm_t = seq_length - rf_size_j + 1
            else:
                norm_t = self.receptive_field - rf_size_j + 1
            norm_t = max(norm_t, 1)
            self.norm.append(
                LayerNorm((hidden_dim, num_nodes, norm_t),
                          elementwise_affine=True))
            rf_size = rf_size_j

        self.end_conv1 = nn.Conv2d(skip_dim, end_dim,          kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(end_dim,  T_out * out_dim, kernel_size=(1, 1))
        self.register_buffer('idx', torch.arange(num_nodes))

    def forward(self, x: torch.Tensor,
                A_fixed: torch.Tensor = None) -> torch.Tensor:
        """
        x       : [B, T_in, N, F]
        A_fixed : 保留接口兼容性，内部不使用
        returns : [B, T_out, N, out_dim]
        """
        B = x.shape[0]
        inp = x.permute(0, 3, 2, 1).contiguous()   # [B, F, N, T]

        seq_len = inp.size(3)
        if seq_len < self.receptive_field:
            inp = F.pad(inp, (self.receptive_field - seq_len, 0, 0, 0))

        adp = self.gc(self.idx)                     # [N, N]

        x_res = self.start_conv(inp)                # [B, hidden, N, T']

        skip = self.skip0(
            F.dropout(inp, self.dropout, training=self.training)
        )[..., -1:]                                 # [B, skip_dim, N, 1]

        for i in range(self.n_layers):
            residual = x_res
            fil = torch.tanh(self.filter_convs[i](x_res))
            gat = torch.sigmoid(self.gate_convs[i](x_res))
            x_res = fil * gat
            x_res = F.dropout(x_res, self.dropout, training=self.training)

            s    = self.skip_convs[i](x_res)[..., -1:]   # [B, skip_dim, N, 1]
            skip = s + skip

            x_res = (self.gconv1[i](x_res, adp) +
                     self.gconv2[i](x_res, adp.T.contiguous()))

            x_res = x_res + residual[:, :, :, -x_res.size(3):]
            x_res = self.norm[i](x_res, self.idx)

        skip = self.skipE(x_res)[..., -1:] + skip   # [B, skip_dim, N, 1]

        x_out = F.relu(skip)
        x_out = F.relu(self.end_conv1(x_out))        # [B, end_dim, N, 1]
        x_out = self.end_conv2(x_out)                # [B, T_out*out_dim, N, 1]
        x_out = x_out.squeeze(-1).permute(0, 2, 1)   # [B, N, T_out*out_dim]
        return x_out.reshape(B, self.T_out, self.num_nodes, self.out_dim)