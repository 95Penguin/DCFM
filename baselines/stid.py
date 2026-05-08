"""
baselines/stid.py
STID: Spatial-Temporal Identity - A Simple yet Effective Baseline
论文: Shao et al., CIKM 2022  https://arxiv.org/abs/2208.05233
参考: https://github.com/GestaltCogTeam/STID

修复:
  [Bug7]  nn.Sequential 不接受嵌套的 nn.Sequential unpack，
          改用 nn.ModuleList 显式前向传播。
  改进:   时序特征用完整展平 [T*F] → Linear 投影，而非只用最后时间步。
"""
import torch
import torch.nn as nn


class STID(nn.Module):
    """
    STID: 空间-时间 ID 嵌入 + MLP。

    时序特征 : x 展平为 [B, N, T*F]，Linear 投影到 hidden_dim
    空间嵌入 : 可学习节点 ID 嵌入 [N, embed_dim]
    时间嵌入 : 可学习时间步 ID 嵌入 [T, embed_dim]，对时间维取均值
    MLP 第一层输入: hidden_dim + 2*embed_dim
    """
    def __init__(self,
                 num_nodes:  int,
                 in_dim:     int,
                 T_in:       int,
                 hidden_dim: int = 32,
                 n_layers:   int = 3,
                 embed_dim:  int = 32,
                 out_dim:    int = 1):
        super().__init__()
        self.T_in = T_in

        # 空间 ID 嵌入
        self.node_emb = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_emb.unsqueeze(0))

        # 时间步 ID 嵌入
        self.time_emb = nn.Parameter(torch.empty(T_in, embed_dim))
        nn.init.xavier_uniform_(self.time_emb.unsqueeze(0))

        # 时序特征投影
        self.time_series_emb_layer = nn.Linear(T_in * in_dim, hidden_dim)

        # MLP（ModuleList 避免嵌套 Sequential 问题）
        first_in = hidden_dim + 2 * embed_dim
        self.mlp_layers = nn.ModuleList()
        for i in range(n_layers):
            in_h = first_in if i == 0 else hidden_dim
            self.mlp_layers.append(
                nn.Sequential(nn.Linear(in_h, hidden_dim), nn.ReLU(inplace=True))
            )

        self.regression_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x       : [B, T_in, N, F]
        returns : [B, N, out_dim]
        """
        B, T, N, F = x.shape

        # 时序特征：展平后投影
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, T * F)   # [B, N, T*F]
        h = self.time_series_emb_layer(x_flat)                  # [B, N, hidden]

        # 空间嵌入
        node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)  # [B, N, embed]

        # 时间嵌入（对时间步取均值）
        time_emb = self.time_emb.mean(dim=0)                      # [embed]
        time_emb = time_emb.unsqueeze(0).unsqueeze(0).expand(B, N, -1)  # [B,N,embed]

        # 拼接
        h = torch.cat([h, node_emb, time_emb], dim=-1)   # [B, N, hidden+2*embed]

        # MLP
        for layer in self.mlp_layers:
            h = layer(h)                                   # [B, N, hidden]

        return self.regression_layer(h)                    # [B, N, out_dim]
