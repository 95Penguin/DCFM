"""
baselines/utils.py
共享工具：指标计算 + 通用训练循环（多步预测版）

多步改动：
  - 模型输出 [B, T_out, N, F]，target [B, T_out, N, F]
  - 损失和指标在所有维度上计算（时间维也参与平均）
  - 支持 prob 模式（输出 mu/sigma 双通道）

null_val 说明：
  传入归一化后的零值（如 Solar 的 -0.648）时，用 true > null_val 做 mask，
  只保留原始值 > 0 的时间点，与论文口径一致。
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 指标（null_val=None 时不 mask）
# ---------------------------------------------------------------------------

def _get_mask(true, null_val):
    if null_val is None:
        return torch.ones_like(true)
    return (true > null_val).float()


def masked_mae(pred, true, null_val=None):
    mask = _get_mask(true, null_val)
    mask = mask / mask.mean().clamp(min=1e-5)
    return (torch.abs(pred - true) * mask).mean()


def masked_mse(pred, true, null_val=None):
    mask = _get_mask(true, null_val)
    mask = mask / mask.mean().clamp(min=1e-5)
    return (((pred - true) ** 2) * mask).mean()


def masked_rmse(pred, true, null_val=None):
    return torch.sqrt(masked_mse(pred, true, null_val))


def masked_mape(pred, true, null_val=None, eps=1e-8):
    mask = _get_mask(true, null_val)
    mask = mask / mask.mean().clamp(min=1e-5)
    return (torch.abs((pred - true) / (true.abs() + eps)) * mask).mean()


def compute_metrics(pred, true, null_val=None):
    """返回 (MAE, RMSE, MAPE%)，支持任意维度广播"""
    return (masked_mae(pred, true, null_val).item(),
            masked_rmse(pred, true, null_val).item(),
            masked_mape(pred, true, null_val).item() * 100.0)


def inverse_torch(tensor: torch.Tensor, scaler):
    """Inverse-transform a tensor while preserving shape and returning CPU float."""
    shape = tensor.shape
    inv = scaler.inverse_transform(tensor.detach().cpu().numpy().reshape(-1))
    return torch.from_numpy(inv.reshape(shape)).float()


def gaussian_nll_loss(mu, sigma_raw, target, null_val=None):
    """高斯 NLL loss，支持 null_val mask"""
    sigma = F.softplus(sigma_raw) + 1e-4
    nll = 0.5 * ((target - mu) ** 2) / (sigma ** 2) + sigma.log()
    mask = _get_mask(target, null_val)
    mask = mask / mask.mean().clamp(min=1e-5)
    return (nll * mask).mean()


def compute_crps_gaussian(mu, sigma_raw, target, null_val=None, n_samples: int = 100):
    """蒙特卡洛估计 CRPS"""
    sigma = F.softplus(sigma_raw) + 1e-4
    eps = torch.randn(n_samples, *mu.shape, device=mu.device)
    samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    mae_term  = (samples - target.unsqueeze(0)).abs().mean(0)
    diff_term = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().mean([0, 1])
    crps_per_point = mae_term - 0.5 * diff_term
    mask = _get_mask(target, null_val)
    mask = mask / mask.mean().clamp(min=1e-5)
    return (crps_per_point * mask).mean().item()


def compute_prob_metrics(samples: np.ndarray, y: np.ndarray,
                         null_val: float = None) -> dict:
    """
    概率预测指标（多步版）。
    samples : [S, total, N, T_out, F]  或  [S, ...]
    y       : [total, N, T_out, F]
    null_val: 若不为 None，mask 掉 true <= null_val 的位置（原始量纲）
    返回 dict 包含 MAE/RMSE/CRPS/PICP/PINAW + per-step MAE_h1 RMSE_h1 等。
    """
    if null_val is not None:
        mask = (y > null_val).astype(np.float32)
        mask = mask / (mask.mean() + 1e-8)
    else:
        mask = np.ones_like(y)

    mu = samples.mean(axis=0)

    def _mae(p, t, m=mask):  return float((np.abs(p - t) * m).mean())
    def _rmse(p, t, m=mask): return float(np.sqrt(((p - t) ** 2 * m).mean()))
    def _mape(p, t, m=mask): return float((np.abs((p - t) / (np.abs(t) + 1e-8)) * m).mean() * 100.0)

    def _crps(s, t, m=mask):
        S = s.shape[0]
        mae_term = np.abs(s - t[None]).mean(axis=0)
        rng = np.random.default_rng()
        n_perm = min(10, S - 1)
        if n_perm <= 0:
            return float((mae_term * m).mean())
        spreads = []
        indices = np.arange(S)
        for _ in range(n_perm):
            perm = rng.permutation(S)
            clash = np.where(perm == indices)[0]
            if len(clash) > 1:
                perm[clash] = perm[np.roll(clash, -1)]
            elif len(clash) == 1:
                other = (clash[0] + 1) % S
                perm[clash[0]], perm[other] = perm[other], perm[clash[0]]
            spreads.append(np.abs(s - s[perm]).mean(axis=0))
        crps_pt = mae_term - 0.5 * np.mean(spreads, axis=0)
        return float((crps_pt * m).mean())

    def _picp(s, t, conf=0.95):
        lo = np.quantile(s, (1 - conf) / 2,     axis=0)
        hi = np.quantile(s, 1 - (1 - conf) / 2, axis=0)
        covered = ((t >= lo) & (t <= hi)).astype(np.float32)
        return float((covered * mask).mean())

    def _pinaw(s, t, conf=0.95):
        lo = np.quantile(s, (1 - conf) / 2,     axis=0)
        hi = np.quantile(s, 1 - (1 - conf) / 2, axis=0)
        rng = t.max() - t.min() + 1e-8
        return float((((hi - lo) / rng) * mask).mean())

    metrics = {
        "MAE":   _mae(mu, y),
        "RMSE":  _rmse(mu, y),
        "MAPE":  _mape(mu, y),
        "CRPS":  _crps(samples, y, mask),
        "PICP":  _picp(samples, y),
        "PINAW": _pinaw(samples, y),
    }
    T_out = y.shape[2]
    for h in range(T_out):
        sh = samples[:, :, :, h, :]
        yh = y[:, :, h, :]
        mh = sh.mean(axis=0)
        mh_mask = mask[:, :, h, :]
        metrics[f"MAE_h{h+1}"]  = _mae(mh, yh, mh_mask)
        metrics[f"RMSE_h{h+1}"] = _rmse(mh, yh, mh_mask)
        metrics[f"MAPE_h{h+1}"] = _mape(mh, yh, mh_mask)
        metrics[f"CRPS_h{h+1}"] = _crps(sh, yh, mh_mask)
    return metrics


# ---------------------------------------------------------------------------
# 通用训练循环（多步版）
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, test_loader,
                optimizer, scheduler, device, scaler=None,
                max_epochs=200, patience=20, grad_clip=1.0,
                save_path="best_baseline.pt", logger=None,
                extra_forward_kwargs=None,
                log_batch_interval=50,
                prob=False,
                null_val=None):
    """
    多步预测版通用训练循环。

    x: [B, T_in, N, F]
    y: [B, T_out, N, F]
    model(x) → [B, T_out, N, out_dim]

    null_val : None（不 mask）或归一化后的零值
    prob     : True 时输出 [B, T_out, N, 2*F]，用 NLL loss，测试额外报告 CRPS
    """
    def _log(msg):
        (logger.info if logger else print)(msg)

    ekw = extra_forward_kwargs or {}
    best_val, no_improve = float("inf"), 0
    history = {"train_loss": [], "val_mae": [],
               "test_mae": 0., "test_rmse": 0., "test_mape": 0., "test_crps": 0.}
    n_batches = len(train_loader)

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        losses = []

        for batch_idx, (x, y) in enumerate(train_loader, 1):
            # x: [B, T_in, N, F],  y: [B, T_out, N, F]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, **ekw)           # [B, T_out, N, out_dim]

            if prob:
                F_dim = y.shape[-1]
                mu, sigma_raw = out[..., :F_dim], out[..., F_dim:]
                loss = gaussian_nll_loss(mu, sigma_raw, y, null_val)
            else:
                loss = masked_mae(out, y, null_val)

            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(loss.item())

            if log_batch_interval > 0 and batch_idx % log_batch_interval == 0:
                elapsed = time.time() - t0
                eta = elapsed / batch_idx * (n_batches - batch_idx)
                _log(f"  Epoch {epoch:3d} [{batch_idx:4d}/{n_batches}] "
                     f"loss={float(np.mean(losses)):.4f}  "
                     f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        tl = float(np.mean(losses))
        history["train_loss"].append(tl)

        model.eval()
        vm = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, **ekw)                       # [B, T_out, N, out_dim]
                mu = out[..., :y.shape[-1]] if prob else out
                vm.append(masked_mae(mu, y, null_val).item())
        val_mae = float(np.mean(vm))
        history["val_mae"].append(val_mae)

        if scheduler is not None:
            try:    scheduler.step(val_mae)
            except: scheduler.step()

        if val_mae < best_val:
            best_val, no_improve = val_mae, 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        _log(f"  Epoch {epoch:3d} | train={tl:.4f} | val={val_mae:.4f} | "
             f"best={best_val:.4f} | {time.time()-t0:.1f}s")
        if no_improve >= patience:
            _log(f"  Early stop @ epoch {epoch}")
            break

    # ── 测试 ──────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    all_mu, all_sigma_raw, all_true = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x, **ekw)
            if prob:
                F_dim = y.shape[-1]
                all_mu.append(out[..., :F_dim].cpu())
                all_sigma_raw.append(out[..., F_dim:].cpu())
            else:
                all_mu.append(out.cpu())
            all_true.append(y.cpu())

    pred_mu  = torch.cat(all_mu)     # [total, T_out, N, F]
    true_cat = torch.cat(all_true)
    null_val_eval = null_val
    if scaler is not None:
        pred_mu = inverse_torch(pred_mu, scaler)
        true_cat = inverse_torch(true_cat, scaler)
        null_val_eval = 0.0 if null_val is not None else None

    mae, rmse, mape = compute_metrics(pred_mu, true_cat, null_val_eval)

    if prob:
        pred_sigma = torch.cat(all_sigma_raw)
        crps = compute_crps_gaussian(pred_mu, pred_sigma, true_cat, null_val_eval)
        _log(f"  [Test] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  CRPS={crps:.4f}")
        history.update({"test_mae": mae, "test_rmse": rmse,
                         "test_mape": mape, "test_crps": crps})
    else:
        _log(f"  [Test] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
        history.update({"test_mae": mae, "test_rmse": rmse, "test_mape": mape})

    return history
