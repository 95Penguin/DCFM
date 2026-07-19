"""
baselines/patchtst.py
PatchTST: A Time Series is Worth 64 Words — Nie et al., ICLR 2023
论文: https://arxiv.org/abs/2211.14730
参考: https://github.com/yuqinie98/PatchTST

核心思路：
  将时序切成 patch（类似 ViT），patch 作为 token 送入 Transformer Encoder，
  然后 flatten 后接 MLP 输出多步预测。
  - 每个节点/变量独立建模（Channel-Independence，CI 策略）
  - 无图结构依赖

多步改动: head 输出 T_out * out_dim，reshape 为 [B, T_out, N, out_dim]。

"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Positional Encoding ────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """标准正弦位置编码，支持 dropout。

    Bug 修复 [B]：d_model 为奇数时，sin 列（0::2）数量 = (d_model+1)//2，
    cos 列（1::2）数量 = d_model//2，两者不等。分别用各自长度的 div_term，
    防止赋值时列数越界或末列全零。
    """
    def __init__(self, d_model: int, dropout: float = 0.0,
                 max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        n_sin = (d_model + 1) // 2   # 0::2 列数
        n_cos = d_model // 2          # 1::2 列数
        div_sin = torch.exp(
            torch.arange(n_sin, dtype=torch.float32)
            * (-math.log(10000.0) / (2 * n_sin))
        )
        div_cos = torch.exp(
            torch.arange(n_cos, dtype=torch.float32)
            * (-math.log(10000.0) / (2 * n_cos))
        )
        pe[:, 0::2] = torch.sin(position * div_sin)
        pe[:, 1::2] = torch.cos(position * div_cos)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── PatchTST 主模型 ────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    PatchTST 多步预测版本（Channel-Independence）。

    参数:
        in_dim      : 输入特征维度 F（每节点）
        T_in        : 历史时间步数
        T_out       : 预测步数
        patch_len   : patch 长度（需满足 patch_len <= T_in）
        stride      : patch 步长（需满足 1 <= stride <= patch_len）
        d_model     : Transformer 隐层维度
        n_heads     : 注意力头数（需整除 d_model）
        n_layers    : Transformer Encoder 层数
        d_ff        : FFN 内部维度
        dropout     : Dropout 率
        attn_dropout: Attention 内部 Dropout
        out_dim     : 输出特征维度（通常=1）
        padding_patch: True 时在时序尾部补 stride 长度的 padding
    """
    def __init__(self,
                 in_dim:        int   = 1,
                 T_in:          int   = 168,
                 T_out:         int   = 12,
                 patch_len:     int   = 16,
                 stride:        int   = 8,
                 d_model:       int   = 128,
                 n_heads:       int   = 8,
                 n_layers:      int   = 3,
                 d_ff:          int   = 256,
                 dropout:       float = 0.1,
                 attn_dropout:  float = 0.0,
                 out_dim:       int   = 2,
                 padding_patch: bool  = True):
        super().__init__()
        self.T_out        = T_out
        self.out_dim      = out_dim
        self.in_dim       = in_dim
        self.patch_len    = patch_len
        self.stride       = stride
        self.padding_patch = padding_patch

        # n_patches 计算与原仓库 utils.py 严格对齐：
        #   no-pad:  floor((T_in - patch_len) / stride + 1)
        #   padding: 末尾补 stride 步，等价于再 +1
        if padding_patch:
            self.pad_len = stride
            n_patches = math.floor((T_in - patch_len) / stride + 1) + 1
        else:
            self.pad_len = 0
            n_patches = math.floor((T_in - patch_len) / stride + 1)
        n_patches = max(n_patches, 1)
        self.n_patches = n_patches

        # Patch 线性投影：patch_len * in_dim → d_model
        self.patch_proj = nn.Linear(patch_len * in_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout,
                                             max_len=n_patches + 16)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = attn_dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,   # Pre-Norm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))

        # Prediction Head:
        #   enc [B*N, n_patches, d_model]
        #   → Flatten(-2) → [B*N, n_patches*d_model]
        #   → Linear → [B*N, T_out*out_dim]
        head_in = n_patches * d_model
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(head_in, T_out * out_dim),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x       : [B, T_in, N, F_dim]
        returns : [B, T_out, N, out_dim]
        """
        B, T, N, F_dim = x.shape   # 注意：用 F_dim 避免遮蔽模块级 F = nn.functional

        # Channel-Independence: 每节点独立走 Transformer
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, F_dim)   # [B*N, T, F_dim]

        # Padding（时间维尾部补零）
        if self.padding_patch:
            h = F.pad(h, (0, 0, 0, self.pad_len))              # [B*N, T+pad, F_dim]

        # Patch 切分
        # unfold(dim=1) on [B*N, T', F_dim] → [B*N, n_p, F_dim, patch_len]
        # 说明：unfold 在 dim=1 上滑窗，输出保留原 dim2(F_dim)，新维 patch_len 追加到末尾
        patches = h.unfold(1, self.patch_len, self.stride)      # [B*N, n_p, F_dim, patch_len]
        patches = patches.permute(0, 1, 3, 2)                   # [B*N, n_p, patch_len, F_dim]
        n_p = patches.shape[1]
        patches = patches.reshape(B * N, n_p, self.patch_len * F_dim)  # [B*N, n_p, patch_len*F_dim]

        # Patch 投影 + 位置编码 + Transformer Encoder
        tok = self.patch_proj(patches)    # [B*N, n_p, d_model]
        tok = self.pos_enc(tok)
        enc = self.encoder(tok)           # [B*N, n_p, d_model]

        # Prediction Head
        out = self.head(enc)                              # [B*N, T_out*out_dim]
        out = out.reshape(B, N, self.T_out, self.out_dim)
        return out.permute(0, 2, 1, 3).contiguous()      # [B, T_out, N, out_dim]


# ── 训练入口（baselines 统一签名）─────────────────────────────────────────

def run_patchtst(loaders, adj, cfg, device, save_dir, logger,
                 in_dim=None, num_nodes=None, scaler=None, null_val=None):
    """
    PatchTST 训练 + 测试（与 baselines 框架统一签名）。
    loaders = (train_loader, val_loader, test_loader)

    Bug 修复 [A]：
      先将 patch_len clip 到 max(1, T_in//4)，
      再将 stride clamp 到 [1, clipped_patch_len]，
      最后一起传给 PatchTST 构造函数，保证：
        ① n_patches 计算值 == unfold 运行时实际产出
        ② head Linear in_features 在 forward 中不会 mismatch
    """
    from baselines.utils import train_model

    train_loader, val_loader, test_loader = loaders
    d, m, t_cfg = cfg.data, cfg.model, cfg.train

    # 读取 cfg 原始值
    patch_len_cfg = getattr(m, "patch_len",    16)
    stride_cfg    = getattr(m, "patch_stride", patch_len_cfg // 2)
    d_model       = getattr(m, "d_model",      128)
    n_heads       = getattr(m, "n_heads",      8)
    n_layers      = getattr(m, "n_layers",     3)
    d_ff          = getattr(m, "d_ff",         256)
    dropout       = getattr(m, "dropout",      0.1)
    attn_dropout  = getattr(m, "attn_dropout", 0.0)

    # Bug 修复 [A]：先 clip patch_len，再 clamp stride（stride <= patch_len）
    patch_len = max(1, min(patch_len_cfg, d.T_in // 4))
    stride    = max(1, min(stride_cfg, patch_len))
    if patch_len != patch_len_cfg or stride != stride_cfg:
        logger.info(f"[PatchTST] patch_len: {patch_len_cfg}→{patch_len}, "
                    f"stride: {stride_cfg}→{stride} "
                    f"(上界 T_in//4={d.T_in // 4})")

    # d_model 必须能被 n_heads 整除
    if d_model % n_heads != 0:
        d_model = (d_model // n_heads) * n_heads
        logger.info(f"[PatchTST] d_model 调整为 {d_model}（n_heads={n_heads} 的倍数）")

    model = PatchTST(
        in_dim        = in_dim or 1,
        T_in          = d.T_in,
        T_out         = d.T_out,
        patch_len     = patch_len,
        stride        = stride,
        d_model       = d_model,
        n_heads       = n_heads,
        n_layers      = n_layers,
        d_ff          = d_ff,
        dropout       = dropout,
        attn_dropout  = attn_dropout,
        out_dim       = 2,          # mu + log_sigma (Gaussian head)
        padding_patch = True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[PatchTST] Parameters: {n_params:,}")
    logger.info(f"[PatchTST] n_patches={model.n_patches}, "
                f"patch_len={model.patch_len}, stride={model.stride}")
    logger.info(f"[PatchTST] Gaussian head: out_dim=2 (mu, log_sigma)")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=t_cfg.lr_decay_factor,
        patience=t_cfg.lr_decay_patience)

    return train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        scaler     = scaler,
        null_val   = null_val,
        max_epochs = t_cfg.max_epochs,
        patience   = t_cfg.patience,
        grad_clip  = t_cfg.grad_clip,
        save_path  = os.path.join(save_dir, "patchtst_best.pt"),
        logger     = logger,
        prob       = True,
        mc_samples_test = getattr(t_cfg, "mc_samples_test", 200),
        mc_chunk_size   = getattr(t_cfg, "mc_chunk_size", 0),
    )