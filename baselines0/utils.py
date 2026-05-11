"""
baselines/utils.py
共享工具：指标计算 + 通用训练循环

null_val 说明：
  传入归一化后的零值（如 Solar 的 -0.648）时，用 true > null_val 做 mask，
  只保留原始值 > 0 的时间点（白天有发电的时段），与论文口径一致。
  默认 null_val=None 表示不 mask（对 Electricity/Weather 数据集使用）。
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 指标（null_val=None 时不 mask，null_val 为数值时过滤掉 <= null_val 的点）
# ---------------------------------------------------------------------------

def _get_mask(true, null_val):
    """null_val=None: 不过滤; 否则只保留 true > null_val 的点"""
    if null_val is None:
        return torch.ones_like(true)
    return (true > null_val).float()


def masked_mae(pred, true, null_val=None):
    mask = _get_mask(true, null_val)
    mask /= mask.mean().clamp(min=1e-5)
    return (torch.abs(pred - true) * mask).mean()


def masked_mse(pred, true, null_val=None):
    mask = _get_mask(true, null_val)
    mask /= mask.mean().clamp(min=1e-5)
    return (((pred - true) ** 2) * mask).mean()


def masked_rmse(pred, true, null_val=None):
    return torch.sqrt(masked_mse(pred, true, null_val))


def masked_mape(pred, true, null_val=None, eps=1e-8):
    if null_val is None:
        mask = (true.abs() > eps).float()
    else:
        mask = (true > null_val).float()
    mask /= mask.mean().clamp(min=1e-5)
    return (torch.abs((pred - true) / (true.abs() + eps)) * mask).mean()


def compute_metrics(pred, true, null_val=None):
    """返回 (MAE, RMSE, MAPE%)"""
    return (masked_mae(pred, true, null_val).item(),
            masked_rmse(pred, true, null_val).item(),
            masked_mape(pred, true, null_val).item() * 100.0)


def gaussian_nll_loss(mu, sigma_raw, target, null_val=None):
    """高斯 NLL loss，支持 null_val mask（只在非零时段计算）"""
    sigma = F.softplus(sigma_raw) + 1e-4
    nll = 0.5 * ((target - mu) ** 2) / (sigma ** 2) + sigma.log()
    mask = _get_mask(target, null_val)
    mask /= mask.mean().clamp(min=1e-5)
    return (nll * mask).mean()


def compute_crps_gaussian(mu, sigma_raw, target, null_val=None, n_samples: int = 100):
    """蒙特卡洛估计 CRPS，支持 null_val mask"""
    sigma = F.softplus(sigma_raw) + 1e-4
    eps = torch.randn(n_samples, *mu.shape, device=mu.device)
    samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    mae_term  = (samples - target.unsqueeze(0)).abs().mean(0)
    diff_term = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().mean([0, 1])
    crps_per_point = mae_term - 0.5 * diff_term
    mask = _get_mask(target, null_val)
    mask /= mask.mean().clamp(min=1e-5)
    return (crps_per_point * mask).mean().item()


# ---------------------------------------------------------------------------
# 通用训练循环
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
    null_val : None（默认，不 mask）或归一化后的零值（如 Solar 传 -0.648）
               只在 true > null_val 的时间点计算 loss 和指标。
    prob     : True 时输出 [B,N,2F]，用 NLL loss，测试额外报告 CRPS。
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
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, **ekw)

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
                out = model(x, **ekw)
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
    model.load_state_dict(torch.load(save_path, map_location=device))
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

    pred_mu  = torch.cat(all_mu)
    true_cat = torch.cat(all_true)
    mae, rmse, mape = compute_metrics(pred_mu, true_cat, null_val)

    if prob:
        pred_sigma = torch.cat(all_sigma_raw)
        crps = compute_crps_gaussian(pred_mu, pred_sigma, true_cat, null_val)
        _log(f"  [Test] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  CRPS={crps:.4f}")
        history.update({"test_mae": mae, "test_rmse": rmse,
                         "test_mape": mape, "test_crps": crps})
    else:
        _log(f"  [Test] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
        history.update({"test_mae": mae, "test_rmse": rmse, "test_mape": mape})

    return history