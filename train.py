"""
GridCFN-Improved – 训练循环

改动：
  - forward 返回 (alpha, mu, sigma, commit_loss, mi_loss)
  - 训练 loss = NLL_GMM + beta_vq * commit + beta_mi * mi
  - 评估用 GMM 期望 + GMM 标准差
  - 日志列改为 NLL | Commit | MI
  - 无 MINE / GRL / warmup / 双 optimizer，纯单 optimizer 训练
"""

import logging
import math
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import norm as scipy_norm

from model import GridCFN, gmm_mean, gmm_variance


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------

def mae(pred, true):
    return float(np.abs(pred - true).mean())

def rmse(pred, true):
    return float(np.sqrt(((pred - true) ** 2).mean()))

def crps_gaussian_np(mu, sigma, y):
    z   = (y - mu) / (sigma + 1e-8)
    phi = scipy_norm.pdf(z)
    Phi = scipy_norm.cdf(z)
    return float(
        (sigma * (z * (2 * Phi - 1) + 2 * phi
                  - 1.0 / math.sqrt(math.pi))).mean())

def picp(mu, sigma, y, confidence=0.95):
    z = scipy_norm.ppf((1 + confidence) / 2)
    return float(((y >= mu - z*sigma) & (y <= mu + z*sigma)).astype(float).mean())

def pinaw(mu, sigma, y, confidence=0.95):
    z = scipy_norm.ppf((1 + confidence) / 2)
    return float((2 * z * sigma / (y.max() - y.min() + 1e-8)).mean())

def evaluate_all(mu_all, sigma_all, y_all):
    return {
        "MAE":   mae(mu_all, y_all),
        "RMSE":  rmse(mu_all, y_all),
        "CRPS":  crps_gaussian_np(mu_all, sigma_all, y_all),
        "PICP":  picp(mu_all, sigma_all, y_all),
        "PINAW": pinaw(mu_all, sigma_all, y_all),
    }


# ---------------------------------------------------------------------------
# 单轮训练
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, adj, device,
                    grad_clip=1.0):
    model.train()
    total_nll = total_commit = total_mi = total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x    = x.to(device)
        y    = y.to(device)
        adj_ = adj.to(device)

        optimizer.zero_grad()

        alpha, mu, sigma, commit_loss, mi_loss = model(x, adj_)

        y_target = y[..., :model.predictor.out_dim]
        l_total, l_nll, l_commit, l_mi = model.compute_loss(
            alpha, mu, sigma, y_target, commit_loss, mi_loss)

        l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_nll    += l_nll.item()
        total_commit += l_commit.item()
        total_mi     += l_mi.item()
        total_loss   += l_total.item()
        n_batches    += 1

    return {
        "loss":   total_loss   / n_batches,
        "nll":    total_nll    / n_batches,
        "commit": total_commit / n_batches,
        "mi":     total_mi     / n_batches,
    }


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, adj, device, scaler=None):
    model.eval()
    mu_list, sigma_list, y_list = [], [], []

    for x, y in loader:
        alpha, mu, sigma, _, _ = model(x.to(device), adj.to(device))
        mu_list.append(gmm_mean(alpha, mu).cpu().numpy())
        sigma_list.append(gmm_variance(alpha, mu, sigma).cpu().numpy())
        y_list.append(y.numpy())

    mu_all    = np.concatenate(mu_list,    axis=0)
    sigma_all = np.concatenate(sigma_list, axis=0)
    y_all     = np.concatenate(y_list,     axis=0)[..., :mu_all.shape[-1]]

    if scaler is not None:
        mu_all    = scaler.inverse_transform(mu_all)
        sigma_all = sigma_all * scaler.std
        y_all     = scaler.inverse_transform(y_all)

    return evaluate_all(mu_all, sigma_all, y_all)


# ---------------------------------------------------------------------------
# 主训练函数（接口与原版完全一致）
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, test_loader,
          adj, device, cfg_train, scaler=None,
          logger: Optional[logging.Logger] = None) -> Dict:

    if logger is None:
        logger = logging.getLogger("gridcfn.train")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                "%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg_train.lr_decay_factor,
        patience=cfg_train.lr_decay_patience)

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history           = {"train_loss": [], "val_crps": [], "val_mae": []}

    header = (f"{'Epoch':>6} | {'NLL':>8} | {'Commit':>7} | {'MI':>7} | "
              f"{'Val MAE':>8} | {'Val CRPS':>9} | {'LR':>8} | {'Time':>6}")
    logger.info(header)
    logger.info("-" * len(header))

    for epoch in range(1, cfg_train.max_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, adj, device,
            cfg_train.grad_clip)

        val_m   = evaluate(model, val_loader, adj, device, scaler)
        scheduler.step(val_m["CRPS"])
        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["nll"])
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])

        logger.info(
            f"{epoch:>6} | {train_m['nll']:>8.4f} | {train_m['commit']:>7.4f} | "
            f"{train_m['mi']:>7.4f} | "
            f"{val_m['MAE']:>8.4f} | {val_m['CRPS']:>9.4f} | "
            f"{cur_lr:>8.2e} | {elapsed:>5.1f}s")

        if val_m["CRPS"] < best_val_crps:
            best_val_crps     = val_m["CRPS"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg_train.save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg_train.patience:
                logger.info(
                    f"\n早停于 epoch {epoch}"
                    f"（最佳 val CRPS = {best_val_crps:.4f}）")
                break

    model.load_state_dict(
        torch.load(cfg_train.save_path, map_location=device))
    test_m = evaluate(model, test_loader, adj, device, scaler)

    sep = "=" * 52
    logger.info(f"\n{sep}")
    logger.info("TEST SET RESULTS")
    logger.info(sep)
    for k, v in test_m.items():
        logger.info(f"  {k:<8}: {v:.4f}")
    logger.info(sep)

    history["test_metrics"] = test_m
    return history
