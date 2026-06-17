"""
baselines/stid.py
STID: Spatial-Temporal Identity - A Simple yet Effective Baseline（多步预测版）
论文: Shao et al., CIKM 2022  https://arxiv.org/abs/2208.05233

多步改动: regression_layer 输出 T_out*out_dim，reshape 为 [B, T_out, N, out_dim]。

修复:
  [1] time_emb 从 mean(dim=0) 改为 x[-1]（最近时刻嵌入）。
      原来 mean 把 T_in 个位置嵌入压缩为标量向量，完全丢弃时间位置信息，
      导致 time_emb 退化为一个与输入无关的固定偏置，对 T_in 较长时（如 168）影响明显。
      修复：取最后一个时刻的嵌入作为"当前时刻"的时间表示，保留位置语义。
"""
import torch
import torch.nn as nn


class STID(nn.Module):
    """STID 多步预测版本。"""
    def __init__(self,
                 num_nodes:  int,
                 in_dim:     int,
                 T_in:       int,
                 hidden_dim: int = 32,
                 n_layers:   int = 3,
                 embed_dim:  int = 32,
                 out_dim:    int = 2,
                 T_out:      int = 1):
        super().__init__()
        self.T_in     = T_in
        self.T_out    = T_out
        self.out_dim  = out_dim
        self.num_nodes = num_nodes

        self.node_emb = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_emb)

        self.time_emb = nn.Parameter(torch.empty(T_in, embed_dim))
        nn.init.xavier_uniform_(self.time_emb)

        self.time_series_emb_layer = nn.Linear(T_in * in_dim, hidden_dim)

        first_in = hidden_dim + 2 * embed_dim
        self.mlp_layers = nn.ModuleList()
        for i in range(n_layers):
            in_h = first_in if i == 0 else hidden_dim
            self.mlp_layers.append(
                nn.Sequential(nn.Linear(in_h, hidden_dim), nn.ReLU(inplace=True))
            )

        self.regression_layer = nn.Linear(hidden_dim, T_out * out_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x       : [B, T_in, N, F]
        returns : [B, T_out, N, out_dim]
        """
        B, T, N, F = x.shape

        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, T * F)
        h = self.time_series_emb_layer(x_flat)                 # [B, N, hidden]

        node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)  # [B, N, embed]

        # 修复 [1]：取最近时刻（T_in-1）的嵌入作为时间表示。
        # 原来 mean(dim=0) 将所有时刻平均，丢失位置信息，退化为固定偏置。
        time_emb = self.time_emb[-1]                              # [embed]
        time_emb = time_emb.unsqueeze(0).unsqueeze(0).expand(B, N, -1)

        h = torch.cat([h, node_emb, time_emb], dim=-1)          # [B, N, hidden+2*embed]

        for layer in self.mlp_layers:
            h = layer(h)                                        # [B, N, hidden]

        out = self.regression_layer(h)                          # [B, N, T_out*out_dim]
        out = out.reshape(B, N, self.T_out, self.out_dim)      # [B, N, T_out, out_dim]
        return out.permute(0, 2, 1, 3)                          # [B, T_out, N, out_dim]