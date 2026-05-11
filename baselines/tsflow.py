"""
TSFlow Baseline — ICLR 2025
"Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting"
Kollovieh et al., 2025  https://github.com/marcelkollovieh/TSFlow

架构说明（按论文忠实复现核心设计）：
  - 条件 CFM：历史序列 x_past → condition encoder → 向量场网络
  - GP 先验（OU 核）：x0 ~ GP(0, K_OU)，比各向同性高斯更贴近时序结构
  - OT-CFM 训练目标：L = ||u_theta(t, x_t) - (x1 - x0)||^2
  - Euler 采样（论文默认 NFE=20）

与 GridCFN 的关键区别（写进 baseline 注释，方便论文分析章节引用）：
  1. 无图结构：N 个节点作为独立通道处理，不建模节点间空间依赖
  2. 无因果解耦：无 He/Hs 分离，无 CLUB 互信息最小化
  3. 无多尺度上下文：无 MultiScaleContext / SCGMP
  4. GP 先验 vs 各向同性高斯先验（本文贡献）

接口适配：与 GridCFN 共享 dataset.py / train.py 中的 load_data、Scaler、
评估指标（MAE/RMSE/CRPS/PICP/PINAW），可直接替换 model 参数传入 run_tsflow()。
"""

import math
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 1. GP 先验采样（OU 核，即 Ornstein-Uhlenbeck）
# ---------------------------------------------------------------------------

class OUKernel:
    """
    Ornstein-Uhlenbeck 核：K(tau, tau') = exp(-|tau - tau'| / ell)
    适合具有粗糙随机游走结构的时序数据（论文 Sec 3.1.1 推荐用于大多数数据集）。
    """
    def __init__(self, ell: float = 1.0):
        self.ell = ell

    def matrix(self, T: int, device: torch.device) -> torch.Tensor:
        """返回 [T, T] 核矩阵"""
        tau = torch.arange(T, dtype=torch.float32, device=device)
        dist = (tau.unsqueeze(0) - tau.unsqueeze(1)).abs()        # [T, T]
        return torch.exp(-dist / self.ell)


def sample_gp_prior(B: int, N: int, T: int, kernel: OUKernel,
                    device: torch.device) -> torch.Tensor:
    """
    从 GP(0, K) 采样。
    返回 [B, N, T]，每个 (b, n) 是一条独立的 GP 样本。

    实现：Cholesky 分解 + 标准正态映射
        K = L L^T  →  x = L z, z ~ N(0, I)
    """
    K = kernel.matrix(T, device)                                   # [T, T]
    jitter = 1e-5 * torch.eye(T, device=device)
    L = torch.linalg.cholesky(K + jitter)                         # [T, T]
    z = torch.randn(B, N, T, device=device)                        # [B, N, T]
    # x = L z: [T, T] x [B*N, T, 1] → [B, N, T]
    x = (L @ z.reshape(B * N, T, 1)).reshape(B, N, T)
    return x


# ---------------------------------------------------------------------------
# 2. Condition Encoder：把历史窗口 x_past → context 向量
# ---------------------------------------------------------------------------

class ConditionEncoder(nn.Module):
    """
    轻量 TCN 编码器，将历史 [B, N, T_in, F] 压缩为 context [B, N, hidden_dim]。

    按论文附录：使用 WaveNet 风格的因果卷积 + 全局平均池化。
    这里保持简洁（与 TSFlow 原版 transformer encoder 等效但更轻），
    方便与 GridCFN backbone 做消融对比。
    """
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 4,
                 T_in: int = 168):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                          padding=2 ** i, dilation=2 ** i),
                nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                nn.GELU(),
            )
            for i in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_past: torch.Tensor) -> torch.Tensor:
        """
        x_past : [B, T_in, N, F]
        return  : [B, N, hidden_dim]
        """
        B, T_in, N, F = x_past.shape
        # [B, N, T_in, F] → [B*N, T_in, F]
        h = x_past.permute(0, 2, 1, 3).reshape(B * N, T_in, F)
        h = self.input_proj(h)                                     # [B*N, T_in, hidden]
        h = h.permute(0, 2, 1)                                     # [B*N, hidden, T_in]
        for layer in self.layers:
            h = h + layer(h)[..., :T_in]                           # residual + causal trim
        h = h.mean(dim=-1)                                         # [B*N, hidden] 全局池化
        h = self.out_proj(h)
        return h.reshape(B, N, -1)                                 # [B, N, hidden]


# ---------------------------------------------------------------------------
# 3. TSFlow 向量场网络（条件版）
# ---------------------------------------------------------------------------

class TSFlowVectorField(nn.Module):
    """
    向量场 u_theta(t, x_t | context)。

    输入：
      x_t     : [B, N, T_out * F]  — 当前流时刻的噪声样本（展平多步输出）
      t       : [B]                — 流时间 t ∈ [0, 1]
      context : [B, N, hidden_dim] — 历史编码

    架构：
      时间嵌入（正弦） + context 投影 → 三层 MLP + AdaLN 条件
    """
    def __init__(self, out_dim: int, context_dim: int,
                 hidden_dim: int = 256, time_emb_dim: int = 16):
        super().__init__()
        self.out_dim = out_dim

        # 时间嵌入
        half = time_emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half) / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)
        self.time_proj = nn.Linear(time_emb_dim, 4 * hidden_dim)  # scale×2, bias×2

        # context 投影 → shift/scale for AdaLN
        self.ctx_proj = nn.Linear(context_dim, 4 * hidden_dim)

        self.input_proj = nn.Linear(out_dim, hidden_dim)
        self.layer1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _time_embed(self, t: torch.Tensor, B: int, N: int) -> torch.Tensor:
        angles = t.reshape(B, 1) * self.freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb.unsqueeze(1).expand(B, N, -1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_t.shape
        t_emb = self._time_embed(t, B, N)                         # [B, N, time_emb]
        t_out = self.time_proj(t_emb)                              # [B, N, 4H]
        c_out = self.ctx_proj(context)                             # [B, N, 4H]

        ts1, tb1, ts2, tb2 = t_out.chunk(4, dim=-1)
        cs1, cb1, cs2, cb2 = c_out.chunk(4, dim=-1)

        h = self.input_proj(x_t)

        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.layer1(h_norm * (1.0 + ts1 + cs1) + (tb1 + cb1))

        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.layer2(h_norm * (1.0 + ts2 + cs2) + (tb2 + cb2))

        return self.out_proj(h)


# ---------------------------------------------------------------------------
# 4. TSFlow 主模型
# ---------------------------------------------------------------------------

class TSFlow(nn.Module):
    """
    TSFlow — 条件 CFM + GP(OU) 先验，无图结构。

    参数：
      in_dim      : 输入特征维度（通常=1）
      T_in        : 历史窗口长度
      T_out       : 预测步长
      hidden_dim  : 隐层宽度
      n_enc_layers: Condition Encoder 的 TCN 层数
      ell         : OU 核的长度尺度
    """
    def __init__(self, in_dim: int = 1, T_in: int = 168, T_out: int = 12,
                 hidden_dim: int = 256, n_enc_layers: int = 4,
                 time_emb_dim: int = 16, ell: float = 1.0):
        super().__init__()
        self.T_out     = T_out
        self.feat_dim  = in_dim
        self.cfm_dim   = T_out * in_dim
        self.ou_kernel = OUKernel(ell=ell)

        self.encoder = ConditionEncoder(in_dim, hidden_dim, n_enc_layers, T_in)
        self.vector_field = TSFlowVectorField(
            out_dim=self.cfm_dim,
            context_dim=hidden_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
        )

    # ------------------------------------------------------------------
    # 训练损失（OT-CFM，GP 先验 x0）
    # ------------------------------------------------------------------

    def cfm_loss(self, x_past: torch.Tensor, y_target: torch.Tensor,
                 n_t_samples: int = 4, sigma_min: float = 0.01) -> torch.Tensor:
        """
        OT-CFM 损失。

        x_past   : [B, T_in, N, F]
        y_target : [B, N, T_out*F]  (已展平)

        分层 t 采样（与 GridCFN 保持一致）：
          t_k ~ Uniform(k/n, (k+1)/n)，k = 0,...,n-1
        """
        B, N, D = y_target.shape
        device  = y_target.device
        context = self.encoder(x_past)                             # [B, N, hidden]
        losses  = []

        for k in range(n_t_samples):
            # GP(OU) 先验采样
            x0 = sample_gp_prior(B, N, self.T_out, self.ou_kernel, device)
            # 展平为 [B, N, T_out*F]（F=1 时直接是 [B,N,T_out]）
            x0 = x0.reshape(B, N, self.cfm_dim)

            # 分层 t 采样
            t = (k + torch.rand(B, device=device)) / n_t_samples
            t_bc  = t.reshape(B, 1, 1)

            # OT-CFM 插值：x_t = (1 - t)*x0 + t*x1  （sigma_min 噪声）
            x_t   = (1.0 - (1.0 - sigma_min) * t_bc) * x0 + t_bc * y_target
            u_t   = y_target - (1.0 - sigma_min) * x0            # 目标向量场

            v_pred = self.vector_field(x_t, t, context)
            losses.append(F.mse_loss(v_pred, u_t))

        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # 采样（Euler，论文默认 NFE=20）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, x_past: torch.Tensor, n_samples: int = 50,
               n_steps: int = 20, sigma_min: float = 0.01) -> torch.Tensor:
        """
        x_past : [B, T_in, N, F]
        return : [n_samples, B, N, T_out, feat_dim]  — 与 GridCFN.sample 接口一致
        """
        B = x_past.shape[0]
        N = x_past.shape[2]
        S = n_samples
        device = x_past.device
        dt = 1.0 / n_steps

        context = self.encoder(x_past)                             # [B, N, hidden]
        # 复制 S 份
        ctx = context.repeat_interleave(S, dim=0)                  # [B*S, N, hidden]

        # GP 先验初始样本
        x = sample_gp_prior(B * S, N, self.T_out, self.ou_kernel, device)
        x = x.reshape(B * S, N, self.cfm_dim)                     # [B*S, N, cfm_dim]

        for step in range(n_steps):
            t_val = step * dt
            t_vec = torch.full((B * S,), t_val, device=device)
            v = self.vector_field(x, t_vec, ctx)
            x = x + dt * v                                         # Euler step

        # [B*S, N, T_out*F] → [S, B, N, T_out, F]
        x = x.reshape(B, S, N, self.T_out, self.feat_dim)
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        return x


# ---------------------------------------------------------------------------
# 5. 训练 / 评估入口（与 GridCFN 接口对齐）
# ---------------------------------------------------------------------------

def run_tsflow(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    scaler,
    cfg,                     # 直接传入 GridCFN 的 Config 对象
    device:       torch.device,
    logger:       logging.Logger = None,
) -> dict:
    """
    TSFlow 训练 + 测试。

    返回格式与 GridCFN 的 train() 完全一致：
      {train_loss, val_crps, val_mae, val_rmse, test_metrics, ...}

    注意：TSFlow 不使用 adj / edge_index（无图结构），
          故意不传入，以凸显与 GridCFN 的架构差异。
    """
    if logger is None:
        logger = logging.getLogger("tsflow")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)

    # ── 超参从 cfg 中读取，保持与 GridCFN 一致的训练条件 ──────────────
    d, m, t_cfg = cfg.data, cfg.model, cfg.train

    model = TSFlow(
        in_dim     = getattr(m, "in_dim", 1),
        T_in       = d.T_in,
        T_out      = d.T_out,
        hidden_dim = getattr(m, "cfm_hidden", 256),
        n_enc_layers = getattr(m, "tcn_layers", 4),
        time_emb_dim = getattr(m, "cfm_time_emb_dim", 16),
        ell        = 1.0,              # OU 核长度尺度（论文默认）
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[TSFlow] Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=t_cfg.lr_decay_factor,
        patience=t_cfg.lr_decay_patience,
    )

    n_samples_val  = getattr(t_cfg, "cfm_n_samples",      50)
    n_samples_test = getattr(t_cfg, "cfm_n_samples_test", 200)
    n_steps        = getattr(t_cfg, "cfm_n_steps",        20)
    n_t_samples    = getattr(t_cfg, "cfm_n_t_samples",    4)
    sigma_min      = getattr(t_cfg, "cfm_sigma_min",      0.01)
    save_path      = t_cfg.save_path.replace(".pt", "_tsflow.pt")

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history = {
        "train_loss": [], "val_crps": [], "val_mae": [], "val_rmse": []
    }

    logger.info("[TSFlow] 开始训练 (无图结构, GP-OU 先验)")
    logger.info(f"{'Epoch':>6} | {'Loss':>8} | {'Val MAE':>8} | "
                f"{'Val RMSE':>9} | {'Val CRPS':>9} | {'LR':>8} | {'Time':>6}")

    for epoch in range(1, t_cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for x, y in train_loader:
            x = x.to(device)                                       # [B, T_in, N, F]
            y = y.to(device)                                       # [B, T_out, N, F]
            B, T_out, N, F = y.shape
            y_flat = y.permute(0, 2, 1, 3).reshape(B, N, T_out * F)

            loss = model.cfm_loss(x, y_flat, n_t_samples, sigma_min)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # ── 验证（归一化域）──────────────────────────────────────────
        val_m = _evaluate(model, val_loader, device, scaler,
                          n_samples_val, n_steps, sigma_min,
                          inverse_transform=False)
        scheduler.step(val_m["CRPS"])
        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(avg_loss)
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])
        history["val_rmse"].append(val_m["RMSE"])

        logger.info(
            f"{epoch:>6} | {avg_loss:>8.4f} | {val_m['MAE']:>8.4f} | "
            f"{val_m['RMSE']:>9.4f} | {val_m['CRPS']:>9.4f} | "
            f"{cur_lr:>8.2e} | {elapsed:>5.1f}s"
        )

        if val_m["CRPS"] < best_val_crps:
            best_val_crps     = val_m["CRPS"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= t_cfg.patience:
                logger.info(f"[TSFlow] 早停于 epoch {epoch}")
                break

    # ── 最终测试（反归一化域）────────────────────────────────────────
    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )
    test_m = _evaluate(model, test_loader, device, scaler,
                       n_samples_test, n_steps, sigma_min,
                       inverse_transform=True)

    sep = "=" * 55
    logger.info(f"\n{sep}")
    logger.info(f"[TSFlow] TEST SET RESULTS (反归一化域, T_out={d.T_out})")
    logger.info(sep)
    for k in ["MAE", "RMSE", "CRPS", "PICP", "PINAW"]:
        logger.info(f"  {k:<8}: {test_m[k]:.4f}")
    logger.info(sep)

    history["test_metrics"] = test_m
    return history


# ---------------------------------------------------------------------------
# 6. 内部评估函数
# ---------------------------------------------------------------------------

def _evaluate(model: TSFlow, loader: DataLoader, device: torch.device,
              scaler, n_samples: int, n_steps: int, sigma_min: float,
              inverse_transform: bool = False) -> dict:
    """与 train.py 中的 evaluate() 返回同样结构的 dict。"""
    model.eval()
    samples_list, y_list = [], []

    for x, y in loader:
        x = x.to(device)
        raw = model.sample(x, n_samples=n_samples,
                           n_steps=n_steps, sigma_min=sigma_min)
        # raw: [S, B, N, T_out, F]
        samples_list.append(raw.cpu().numpy())
        # y: [B, T_out, N, F] → [B, N, T_out, F]
        y_list.append(y.permute(0, 2, 1, 3).numpy())

    samples_all = np.concatenate(samples_list, axis=1)  # [S, total, N, T_out, F]
    y_all       = np.concatenate(y_list,       axis=0)  # [total, N, T_out, F]

    if inverse_transform and scaler is not None:
        shape = samples_all.shape
        samples_all = scaler.inverse_transform(
            samples_all.reshape(-1)).reshape(shape)
        y_all = scaler.inverse_transform(
            y_all.reshape(-1)).reshape(y_all.shape)

    return _compute_metrics(samples_all, y_all)


def _compute_metrics(samples: np.ndarray, y: np.ndarray) -> dict:
    """samples: [S,...], y: [...]"""
    mu = samples.mean(axis=0)

    def mae(p, t):  return float(np.abs(p - t).mean())
    def rmse(p, t): return float(np.sqrt(((p - t) ** 2).mean()))

    def crps(s, t):
        S = s.shape[0]
        mae_term = np.abs(s - t[None]).mean(axis=0)
        n_perm = min(10, S - 1)
        if n_perm <= 0:
            return float(mae_term.mean())
        rng = np.random.default_rng(0)
        spreads = []
        for _ in range(n_perm):
            perm = rng.permutation(S)
            clash = np.where(perm == np.arange(S))[0]
            for idx in clash:
                swap = (idx + 1) % S
                perm[idx], perm[swap] = perm[swap], perm[idx]
            spreads.append(np.abs(s - s[perm]).mean(axis=0))
        return float((mae_term - 0.5 * np.mean(spreads, axis=0)).mean())

    def picp(s, t, conf=0.95):
        lo = np.quantile(s, (1 - conf) / 2,     axis=0)
        hi = np.quantile(s, 1 - (1 - conf) / 2, axis=0)
        return float(((t >= lo) & (t <= hi)).astype(float).mean())

    def pinaw(s, t, conf=0.95):
        lo = np.quantile(s, (1 - conf) / 2,     axis=0)
        hi = np.quantile(s, 1 - (1 - conf) / 2, axis=0)
        return float(((hi - lo) / (t.max() - t.min() + 1e-8)).mean())

    metrics = {
        "MAE":   mae(mu, y),
        "RMSE":  rmse(mu, y),
        "CRPS":  crps(samples, y),
        "PICP":  picp(samples, y),
        "PINAW": pinaw(samples, y),
    }
    T_out = y.shape[2]
    for h in range(T_out):
        sh = samples[:, :, :, h, :]
        yh = y[:, :, h, :]
        mh = sh.mean(axis=0)
        metrics[f"MAE_h{h+1}"]  = mae(mh, yh)
        metrics[f"RMSE_h{h+1}"] = rmse(mh, yh)
        metrics[f"CRPS_h{h+1}"] = crps(sh, yh)
    return metrics
