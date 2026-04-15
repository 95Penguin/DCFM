"""
GridCFN – 训练循环（GMM v2）

相对于 v1 的改动：
  [改动] 新增 use_grin 参数
  use_grin=True 时，y_target 经 scaler 还原到原始尺度，
  与 GRIN-denorm 后的 mu/sigma 对齐计算损失和评估指标。
"""

import logging
import math
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_gmm import GridCFN


def mae(pred, true):
    return float(np.abs(pred - true).mean())


def rmse(pred, true):
    return float(np.sqrt(((pred - true) ** 2).mean()))


def crps_score_gmm(w_all, mu_all, sigma_all, y_all, n_samples=500):
    M, K   = mu_all.shape
    w_soft = np.exp(w_all - w_all.max(axis=-1, keepdims=True))
    w_soft = w_soft / w_soft.sum(axis=-1, keepdims=True)
    def draw(n):
        k_idx = np.array([np.random.choice(K, size=n, p=w_soft[i]) for i in range(M)])
        return np.random.normal(mu_all[np.arange(M)[:, None], k_idx],
                                sigma_all[np.arange(M)[:, None], k_idx])
    X, Xp = draw(n_samples), draw(n_samples)
    return float((np.abs(X - y_all[:, None]).mean(-1)
                  - 0.5 * np.abs(X - Xp).mean(-1)).mean())


def _gmm_cdf(x_grid, w_soft, mu, sigma):
    from scipy.special import ndtr
    z = (x_grid[None, None, :] - mu[:, :, None]) / (sigma[:, :, None] + 1e-8)
    return (w_soft[:, :, None] * ndtr(z)).sum(axis=1)


def picp_gmm(w_all, mu_all, sigma_all, y_all, confidence=0.95, n_grid=200):
    M, K   = mu_all.shape
    w_soft = np.exp(w_all - w_all.max(axis=-1, keepdims=True))
    w_soft = w_soft / w_soft.sum(axis=-1, keepdims=True)
    alpha  = (1 - confidence) / 2
    mu_mean = (w_soft * mu_all).sum(axis=-1)
    sig_max = sigma_all.max(axis=-1)
    xg = np.linspace((mu_mean - 6*sig_max).min(), (mu_mean + 6*sig_max).max(), n_grid)
    cdf = _gmm_cdf(xg, w_soft, mu_all, sigma_all)
    lo = np.array([np.interp(alpha,   cdf[i], xg) for i in range(M)])
    hi = np.array([np.interp(1-alpha, cdf[i], xg) for i in range(M)])
    return float(((y_all >= lo) & (y_all <= hi)).astype(float).mean())


def pinaw_gmm(w_all, mu_all, sigma_all, y_all, confidence=0.95, n_grid=200):
    M, K   = mu_all.shape
    w_soft = np.exp(w_all - w_all.max(axis=-1, keepdims=True))
    w_soft = w_soft / w_soft.sum(axis=-1, keepdims=True)
    alpha  = (1 - confidence) / 2
    mu_mean = (w_soft * mu_all).sum(axis=-1)
    sig_max = sigma_all.max(axis=-1)
    xg = np.linspace((mu_mean - 6*sig_max).min(), (mu_mean + 6*sig_max).max(), n_grid)
    cdf = _gmm_cdf(xg, w_soft, mu_all, sigma_all)
    lo = np.array([np.interp(alpha,   cdf[i], xg) for i in range(M)])
    hi = np.array([np.interp(1-alpha, cdf[i], xg) for i in range(M)])
    return float(((hi - lo) / (y_all.max() - y_all.min() + 1e-8)).mean())


def evaluate_all_gmm(w_all, mu_all, sigma_all, y_all):
    w_soft  = np.exp(w_all - w_all.max(axis=-1, keepdims=True))
    w_soft  = w_soft / w_soft.sum(axis=-1, keepdims=True)
    mu_mean = (w_soft * mu_all).sum(axis=-1)
    return {
        "MAE":   mae(mu_mean, y_all),
        "RMSE":  rmse(mu_mean, y_all),
        "CRPS":  crps_score_gmm(w_all, mu_all, sigma_all, y_all),
        "PICP":  picp_gmm(w_all, mu_all, sigma_all, y_all),
        "PINAW": pinaw_gmm(w_all, mu_all, sigma_all, y_all),
    }


def _inv_transform(y_np, scaler):
    """将 numpy array 从归一化域还原到原始尺度。"""
    out = y_np * scaler.std + scaler.mean
    if scaler.log_transform:
        out = np.expm1(out)
    return out


def _inv_transform_tensor(y_t, scaler, device):
    """将 tensor 从归一化域还原到原始尺度（在 device 上操作）。"""
    out = y_t * scaler.std + scaler.mean
    if scaler.log_transform:
        out = out.exp() - 1.0
    return out


def train_one_epoch(
    model, loader, optimizer, club_optimizer,
    adj_norm, edge_index, device, epoch,
    grad_clip=1.0, warmup_epochs=5,
    scaler=None, use_grin=True,
) -> Dict[str, float]:
    model.train()
    total_loss = total_nll = total_mi = total_var = 0.0
    total_mean = total_weight = 0.0
    n_batches  = 0

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        w, mu, sigma, He, Hs, _ = model(x, adj_norm_, edge_idx_)

        var_loss = model.club.variational_loss(He.detach(), Hs.detach())
        club_optimizer.zero_grad()
        var_loss.backward()
        club_optimizer.step()

        mi_loss  = model.club(He, Hs)
        y_target = y.unsqueeze(-1) if y.dim() == 2 else y   # [B, N, 1]

        if use_grin and scaler is not None:
            y_target = _inv_transform_tensor(y_target, scaler, device)

        loss, l_nll, l_mean, l_weight = model.compute_loss(
            w, mu, sigma, y_target, mi_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss   += loss.item()
        total_nll    += l_nll.item()
        total_mi     += mi_loss.item()
        total_var    += var_loss.item()
        total_mean   += l_mean.item()
        total_weight += l_weight.item()
        n_batches    += 1

    return {
        "loss":     total_loss   / n_batches,
        "nll":      total_nll    / n_batches,
        "mi":       total_mi     / n_batches,
        "var_loss": total_var    / n_batches,
        "l_mean":   total_mean   / n_batches,
        "l_weight": total_weight / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, adj_norm, edge_index, device,
             scaler=None, use_grin=True, return_preds=False):
    model.eval()
    w_list, mu_list, sigma_list, y_list = [], [], [], []

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    for x, y in loader:
        w, mu, sigma, _, _, _ = model(x.to(device), adj_norm_, edge_idx_)
        w_list.append(w.cpu().numpy())
        mu_list.append(mu.cpu().numpy())
        sigma_list.append(sigma.cpu().numpy())
        y_list.append(y.numpy())

    w_all     = np.concatenate(w_list,     axis=0)
    mu_all    = np.concatenate(mu_list,    axis=0)
    sigma_all = np.concatenate(sigma_list, axis=0)
    y_all     = np.concatenate(y_list,     axis=0)

    if use_grin and scaler is not None:
        y_all = _inv_transform(y_all, scaler)

    K        = w_all.shape[-1]
    w_flat   = w_all.reshape(-1, K)
    mu_flat  = mu_all.reshape(-1, K)
    sig_flat = sigma_all.reshape(-1, K)
    y_flat   = y_all.reshape(-1) if y_all.ndim <= 2 else y_all[..., 0].reshape(-1)

    metrics = evaluate_all_gmm(w_flat, mu_flat, sig_flat, y_flat)

    if return_preds:
        return metrics, w_flat, mu_flat, sig_flat, y_flat
    return metrics


def train(model, train_loader, val_loader, test_loader,
          adj_norm, edge_index, device, cfg_train,
          scaler=None, logger=None, use_grin=True) -> Dict:

    if logger is None:
        logger = logging.getLogger("gridcfn.train")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                             datefmt="%H:%M:%S"))
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg_train.lr, weight_decay=cfg_train.weight_decay)
    club_optimizer = torch.optim.Adam(
        model.club.parameters(), lr=cfg_train.lr * 5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg_train.lr_decay_factor,
        patience=cfg_train.lr_decay_patience)

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history = {
        "train_loss": [], "train_nll": [], "train_mi": [], "train_var_loss": [],
        "train_l_mean": [], "train_l_weight": [],
        "val_crps": [], "val_mae": [], "val_rmse": [],
    }

    header = (f"{'Epoch':>6} | {'Loss':>8} | {'NLL':>8} | {'MI':>8} | "
              f"{'VarLoss':>8} | {'L_mean':>7} | {'L_wt':>6} | "
              f"{'ValMAE':>7} | {'ValRMSE':>8} | {'ValCRPS':>8} | "
              f"{'LR':>8} | {'Time':>6}")
    logger.info(header)
    logger.info("-" * len(header))

    for epoch in range(1, cfg_train.max_epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, club_optimizer,
            adj_norm, edge_index, device, epoch,
            cfg_train.grad_clip, getattr(cfg_train, "warmup_epochs", 5),
            scaler=scaler, use_grin=use_grin,
        )
        val_m = evaluate(
            model, val_loader, adj_norm, edge_index, device,
            scaler=scaler, use_grin=use_grin,
        )
        scheduler.step(val_m["CRPS"])
        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["train_nll"].append(train_m["nll"])
        history["train_mi"].append(train_m["mi"])
        history["train_var_loss"].append(train_m["var_loss"])
        history["train_l_mean"].append(train_m["l_mean"])
        history["train_l_weight"].append(train_m["l_weight"])
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])
        history["val_rmse"].append(val_m["RMSE"])

        logger.info(
            f"{epoch:>6} | {train_m['loss']:>8.4f} | {train_m['nll']:>8.4f} | "
            f"{train_m['mi']:>8.4f} | {train_m['var_loss']:>8.4f} | "
            f"{train_m['l_mean']:>7.4f} | {train_m['l_weight']:>6.4f} | "
            f"{val_m['MAE']:>7.4f} | {val_m['RMSE']:>8.4f} | "
            f"{val_m['CRPS']:>8.4f} | {cur_lr:>8.2e} | {elapsed:>5.1f}s"
        )

        if val_m["CRPS"] < best_val_crps:
            best_val_crps     = val_m["CRPS"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg_train.save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg_train.patience:
                logger.info(f"\n早停于 epoch {epoch}（最佳 val CRPS={best_val_crps:.4f}）")
                break

    model.load_state_dict(
        torch.load(cfg_train.save_path, map_location=device, weights_only=True))
    test_m, w_t, mu_t, sig_t, y_t = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler, use_grin=use_grin, return_preds=True,
    )

    sep = "=" * 52
    logger.info(f"\n{sep}\nTEST SET RESULTS\n{sep}")
    for k, v in test_m.items():
        logger.info(f"  {k:<8}: {v:.4f}")
    logger.info(sep)

    w_soft_t  = np.exp(w_t - w_t.max(axis=-1, keepdims=True))
    w_soft_t  = w_soft_t / w_soft_t.sum(axis=-1, keepdims=True)
    mu_mean_t = (w_soft_t * mu_t).sum(axis=-1)

    history["test_metrics"] = test_m
    history["test_mu"]      = mu_mean_t.tolist()
    history["test_sigma"]   = sig_t.tolist()
    history["test_w"]       = w_soft_t.tolist()
    history["test_y"]       = y_t.tolist()
    history["test_shape"]   = list(mu_t.shape)
    return history
