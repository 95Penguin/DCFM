"""
baselines/tsflow.py
TSFlow: Flow Matching with Gaussian Process Priors for Probabilistic
Time Series Forecasting  — Kollovieh et al., ICLR 2025

github:https://github.com/marcelkollovieh/TSFlow

多步预测版: T_out 支持，GP(OU) 先验 + OT-CFM + Euler 采样。
"""
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── GP 先验（OU 核）────────────────────────────────────────────────────────

class OUKernel:
    """Ornstein-Uhlenbeck 核: K(τ, τ') = exp(-|τ - τ'| / ell)"""
    def __init__(self, ell: float = 1.0):
        self.ell = ell

    def matrix(self, T: int, device: torch.device) -> torch.Tensor:
        tau = torch.arange(T, dtype=torch.float32, device=device)
        dist = (tau.unsqueeze(0) - tau.unsqueeze(1)).abs()
        return torch.exp(-dist / self.ell)


def sample_gp_prior(B: int, N: int, T: int, kernel: OUKernel,
                    device: torch.device) -> torch.Tensor:
    """Cholesky 分解 + 标准正态 → [B, N, T]"""
    K = kernel.matrix(T, device)
    jitter = 1e-5 * torch.eye(T, device=device)
    L = torch.linalg.cholesky(K + jitter)
    z = torch.randn(B, N, T, device=device)
    x = (L @ z.reshape(B * N, T, 1)).reshape(B, N, T)
    return x


# ── Condition Encoder ──────────────────────────────────────────────────────

class ConditionEncoder(nn.Module):
    """TCN 编码器: [B, T_in, N, F] → [B, N, hidden_dim]"""
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
        B, T_in, N, F = x_past.shape
        h = x_past.permute(0, 2, 1, 3).reshape(B * N, T_in, F)
        h = self.input_proj(h)
        h = h.permute(0, 2, 1)
        for layer in self.layers:
            h = h + layer(h)[..., :T_in]
        h = h.mean(dim=-1)
        h = self.out_proj(h)
        return h.reshape(B, N, -1)


# ── 向量场网络 ─────────────────────────────────────────────────────────────

class TSFlowVectorField(nn.Module):
    """u_θ(t, x_t | context) — AdaLN 条件注入"""
    def __init__(self, out_dim: int, context_dim: int,
                 hidden_dim: int = 256, time_emb_dim: int = 16):
        super().__init__()
        self.out_dim = out_dim

        half = time_emb_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("freqs", freqs)
        self.time_proj = nn.Linear(time_emb_dim, 4 * hidden_dim)
        self.ctx_proj  = nn.Linear(context_dim, 4 * hidden_dim)

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
        t_emb = self._time_embed(t, B, N)
        t_out = self.time_proj(t_emb)
        c_out = self.ctx_proj(context)
        ts1, tb1, ts2, tb2 = t_out.chunk(4, dim=-1)
        cs1, cb1, cs2, cb2 = c_out.chunk(4, dim=-1)

        h = self.input_proj(x_t)
        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.layer1(h_norm * (1.0 + ts1 + cs1) + (tb1 + cb1))
        h_norm = F.layer_norm(h, [h.shape[-1]])
        h = h + self.layer2(h_norm * (1.0 + ts2 + cs2) + (tb2 + cb2))
        return self.out_proj(h)


# ── TSFlow 主模型 ──────────────────────────────────────────────────────────

class TSFlow(nn.Module):
    """条件 CFM + GP(OU) 先验，无图结构"""
    def __init__(self, in_dim: int = 1, out_feat: int = 1,
                 T_in: int = 168, T_out: int = 12,
                 hidden_dim: int = 256, n_enc_layers: int = 4,
                 time_emb_dim: int = 16, ell: float = 1.0):
        super().__init__()
        self.T_out    = T_out
        self.feat_dim = out_feat
        self.cfm_dim  = T_out * out_feat
        self.ou_kernel = OUKernel(ell=ell)
        self.encoder = ConditionEncoder(in_dim, hidden_dim, n_enc_layers, T_in)
        self.vector_field = TSFlowVectorField(
            out_dim=self.cfm_dim, context_dim=hidden_dim,
            hidden_dim=hidden_dim, time_emb_dim=time_emb_dim)

    def cfm_loss(self, x_past: torch.Tensor, y_target: torch.Tensor,
                 n_t_samples: int = 4, sigma_min: float = 0.01) -> torch.Tensor:
        """OT-CFM 损失。x_past: [B, T_in, N, F], y_target: [B, N, T_out*F]"""
        B, N, D = y_target.shape
        device  = y_target.device
        context = self.encoder(x_past)
        losses  = []

        for k in range(n_t_samples):
            x0 = sample_gp_prior(B, N, self.T_out, self.ou_kernel, device)  # [B, N, T_out]
            x0 = x0.unsqueeze(-1).expand(-1, -1, -1, self.feat_dim)          # [B, N, T_out, feat_dim]
            x0 = x0.contiguous().reshape(B, N, self.cfm_dim)                 # [B, N, T_out*feat_dim]

            t = (k + torch.rand(B, device=device)) / n_t_samples
            t_bc = t.reshape(B, 1, 1)

            x_t = (1.0 - (1.0 - sigma_min) * t_bc) * x0 + t_bc * y_target
            u_t = y_target - (1.0 - sigma_min) * x0

            v_pred = self.vector_field(x_t, t, context)
            losses.append(F.mse_loss(v_pred, u_t))

        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(self, x_past: torch.Tensor, n_samples: int = 50,
               n_steps: int = 20, sigma_min: float = 0.01) -> torch.Tensor:
        """Euler 采样，返回 [n_samples, B, N, T_out, feat_dim]"""
        B = x_past.shape[0]
        N = x_past.shape[2]
        S = n_samples
        device = x_past.device
        dt = 1.0 / n_steps

        context = self.encoder(x_past)
        ctx = context.repeat_interleave(S, dim=0)

        x = sample_gp_prior(B * S, N, self.T_out, self.ou_kernel, device)  # [B*S, N, T_out]
        x = x.unsqueeze(-1).expand(-1, -1, -1, self.feat_dim)               # [B*S, N, T_out, feat_dim]
        x = x.contiguous().reshape(B * S, N, self.cfm_dim)                  # [B*S, N, T_out*feat_dim]

        for step in range(n_steps):
            t_val = step * dt
            t_vec = torch.full((B * S,), t_val, device=device)
            v = self.vector_field(x, t_vec, ctx)
            x = x + dt * v

        x = x.reshape(B, S, N, self.T_out, self.feat_dim)
        return x.permute(1, 0, 2, 3, 4).contiguous()


# ── 内部评估 ────────────────────────────────────────────────────────────────

def _evaluate_tsflow(model: TSFlow, loader, device, scaler,
                     n_samples: int, n_steps: int, sigma_min: float,
                     null_val: float = None) -> dict:
    from baselines.utils import compute_prob_metrics

    model.eval()
    samples_list, y_list = [], []

    for x, y in loader:
        x = x.to(device)
        raw = model.sample(x, n_samples=n_samples,
                           n_steps=n_steps, sigma_min=sigma_min)
        samples_list.append(raw.cpu().numpy())
        y_list.append(y.permute(0, 2, 1, 3).numpy())

    samples_all = np.concatenate(samples_list, axis=1)
    y_all       = np.concatenate(y_list,       axis=0)

    metrics_norm = compute_prob_metrics(samples_all, y_all)

    if scaler is not None:
        shape = samples_all.shape
        # 修复：若 Scaler 训练于多特征（SDWPF 有 4 维），预测只含第 0 维（功率），
        # 需将 mean/std 切片至第 0 维，否则 Scaler.inverse_transform 中的
        # reshape(-1, F) 会错误混用其他特征尺度
        s_mean = scaler.mean[..., :1] if scaler.mean.shape[-1] > 1 else scaler.mean
        s_std  = scaler.std[..., :1]  if scaler.std.shape[-1] > 1  else scaler.std
        samples_all = (samples_all.reshape(-1) * s_std + s_mean).reshape(shape)
        y_all = (y_all.reshape(-1) * s_std + s_mean).reshape(y_all.shape)

    metrics = compute_prob_metrics(samples_all, y_all)
    for k, v in metrics_norm.items():
        metrics[f"{k}_norm"] = v
    return metrics


# ── 训练入口（baselines 统一签名）─────────────────────────────────────────

def run_tsflow(loaders, adj, cfg, device, save_dir, logger,
               in_dim=None, num_nodes=None, scaler=None, null_val=None):
    """
    TSFlow 训练 + 测试。
    loaders = (train_loader, val_loader, test_loader)
    """
    train_loader, val_loader, test_loader = loaders
    d, m, t_cfg = cfg.data, cfg.model, cfg.train
    # 推断输出特征维度：SlidingWindowDataset 的 y 取 data[..., :1]，F_out = 1
    for x_batch, y_batch in train_loader:
        out_feat = y_batch.shape[3]
        break

    model = TSFlow(
        in_dim      = in_dim or 1,
        out_feat    = out_feat,
        T_in        = d.T_in,
        T_out       = d.T_out,
        hidden_dim  = getattr(m, "cfm_hidden", 256),
        n_enc_layers = getattr(m, "tcn_layers", 4),
        time_emb_dim = getattr(m, "cfm_time_emb_dim", 16),
        ell         = 1.0,
    ).to(device)
    logger.info(f"[TSFlow] in_dim={in_dim}, out_feat={out_feat}, cfm_dim={model.cfm_dim}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[TSFlow] Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=t_cfg.lr_decay_factor,
        patience=t_cfg.lr_decay_patience)

    n_samples_val  = getattr(t_cfg, "cfm_n_samples",      50)
    n_samples_test = getattr(t_cfg, "cfm_n_samples_test", 200)
    n_steps        = getattr(t_cfg, "cfm_n_steps",        20)
    n_t_samples    = getattr(t_cfg, "cfm_n_t_samples",    4)
    sigma_min      = getattr(t_cfg, "cfm_sigma_min",      0.01)
    save_path      = os.path.join(save_dir, "tsflow_best.pt")

    best_val_crps  = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_crps": [], "val_mae": [], "val_rmse": []}

    logger.info("[TSFlow] 开始训练 (无图结构, GP-OU 先验)")
    logger.info(f"{'Epoch':>6} | {'Loss':>8} | {'Val MAE':>8} | "
                f"{'Val RMSE':>9} | {'Val CRPS':>9} | {'LR':>8} | {'Time':>6}")

    for epoch in range(1, t_cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
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

        val_m = _evaluate_tsflow(model, val_loader, device, scaler,
                                 n_samples_val, n_steps, sigma_min,
                                 null_val=null_val)
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
            f"{cur_lr:>8.2e} | {elapsed:>5.1f}s")

        if val_m["CRPS"] < best_val_crps:
            best_val_crps     = val_m["CRPS"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= t_cfg.patience:
                logger.info(f"[TSFlow] 早停于 epoch {epoch}")
                break

    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True))
    test_m = _evaluate_tsflow(model, test_loader, device, scaler,
                              n_samples_test, n_steps, sigma_min,
                              null_val=null_val)

    sep = "=" * 55
    logger.info(f"\n{sep}")
    logger.info(f"[TSFlow] TEST SET RESULTS (归一化域, T_out={d.T_out})")
    logger.info(sep)
    for k in ["MAE", "RMSE", "CRPS", "PICP", "PINAW"]:
        logger.info(f"  {k:<8}: {test_m[f'{k}_norm']:.4f}")
    logger.info(sep)
    logger.info(f"[TSFlow] TEST SET RESULTS (反归一化域, T_out={d.T_out})")
    logger.info(sep)
    for k in ["MAE", "RMSE", "CRPS", "PICP", "PINAW"]:
        logger.info(f"  {k:<8}: {test_m[k]:.4f}")
    logger.info(f"\n  {'Step':<6}  {'MAE':>8}  {'RMSE':>8}  {'CRPS':>8}")
    for h in range(d.T_out):
        mae_h  = test_m.get(f"MAE_h{h+1}",  float("nan"))
        rmse_h = test_m.get(f"RMSE_h{h+1}", float("nan"))
        crps_h = test_m.get(f"CRPS_h{h+1}", float("nan"))
        logger.info(f"  h={h+1:<4}  {mae_h:>8.4f}  {rmse_h:>8.4f}  {crps_h:>8.4f}")
    logger.info(sep)

    history["test_metrics"] = test_m
    history["test_mae"]  = test_m["MAE"]
    history["test_rmse"] = test_m["RMSE"]
    history["test_mape"] = test_m["MAPE"]
    return history
