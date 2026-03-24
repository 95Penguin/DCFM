"""
GridCFN – 训练循环、评估与指标
所有训练超参数通过 TrainConfig 传入，不在此处硬编码。
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from model import GridCFN, nll_gaussian_loss


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------

def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).mean())


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(((pred - true) ** 2).mean()))


def crps_score(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Gaussian CRPS 闭合解，越小越好。"""
    from scipy.stats import norm
    z   = (y - mu) / (sigma + 1e-8)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    return float((sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))).mean())


def picp(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
         confidence: float = 0.95) -> float:
    """预测区间覆盖概率（越接近 confidence 越好）。"""
    from scipy.stats import norm
    z = norm.ppf((1 + confidence) / 2)
    covered = ((y >= mu - z * sigma) & (y <= mu + z * sigma)).astype(float)
    return float(covered.mean())


def pinaw(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
          confidence: float = 0.95) -> float:
    """归一化平均区间宽度（越小越好）。"""
    from scipy.stats import norm
    z = norm.ppf((1 + confidence) / 2)
    width   = 2 * z * sigma
    y_range = y.max() - y.min() + 1e-8
    return float((width / y_range).mean())


def evaluate_all(mu_all: np.ndarray, sigma_all: np.ndarray,
                 y_all: np.ndarray) -> Dict[str, float]:
    return {
        "MAE":   mae(mu_all, y_all),
        "RMSE":  rmse(mu_all, y_all),
        "CRPS":  crps_score(mu_all, sigma_all, y_all),
        "PICP":  picp(mu_all, sigma_all, y_all),
        "PINAW": pinaw(mu_all, sigma_all, y_all),
    }


# ---------------------------------------------------------------------------
# 单轮训练
# ---------------------------------------------------------------------------

def train_one_epoch(model: GridCFN, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    adj: torch.Tensor, device: torch.device,
                    grad_clip: float = 5.0) -> Dict[str, float]:
    model.train()
    total_loss = total_nll = total_mi = 0.0
    n_batches = 0

    for x, y in loader:
        x    = x.to(device)
        y    = y.to(device)
        adj_ = adj.to(device)

        optimizer.zero_grad()
        mu, sigma, mi_loss = model(x, adj_)

        y_target = y[..., :mu.shape[-1]]
        loss, l_nll, l_mi = model.compute_loss(mu, sigma, y_target, mi_loss)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_nll  += l_nll.item()
        total_mi   += l_mi.item()
        n_batches  += 1

    return {
        "loss": total_loss / n_batches,
        "nll":  total_nll  / n_batches,
        "mi":   total_mi   / n_batches,
    }


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: GridCFN, loader: DataLoader,
             adj: torch.Tensor, device: torch.device,
             scaler=None) -> Dict[str, float]:
    model.eval()
    mu_list, sigma_list, y_list = [], [], []

    for x, y in loader:
        mu, sigma, _ = model(x.to(device), adj.to(device))
        mu_list.append(mu.cpu().numpy())
        sigma_list.append(sigma.cpu().numpy())
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
# 主训练函数（接收 TrainConfig）
# ---------------------------------------------------------------------------

def train(model: GridCFN,
          train_loader: DataLoader,
          val_loader: DataLoader,
          test_loader: DataLoader,
          adj: torch.Tensor,
          device: torch.device,
          cfg_train,          # TrainConfig
          scaler=None) -> Dict:
    """
    完整训练循环，含早停和学习率衰减。
    所有超参数从 cfg_train (TrainConfig) 读取。
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg_train.lr_decay_factor,
        patience=cfg_train.lr_decay_patience,
    )

    best_val_crps    = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_crps": [], "val_mae": []}

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val MAE':>8} | "
          f"{'Val RMSE':>9} | {'Val CRPS':>9} | {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, cfg_train.max_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, adj, device, cfg_train.grad_clip)
        val_m   = evaluate(model, val_loader, adj, device, scaler)

        scheduler.step(val_m["CRPS"])
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])

        print(f"{epoch:>6} | {train_m['loss']:>10.4f} | "
              f"{val_m['MAE']:>8.4f} | {val_m['RMSE']:>9.4f} | "
              f"{val_m['CRPS']:>9.4f} | {elapsed:>5.1f}s")

        if val_m["CRPS"] < best_val_crps:
            best_val_crps     = val_m["CRPS"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg_train.save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg_train.patience:
                print(f"\n早停于 epoch {epoch}（最佳 val CRPS={best_val_crps:.4f}）")
                break

    # 加载最优权重，在测试集上评估
    model.load_state_dict(torch.load(cfg_train.save_path, map_location=device))
    test_m = evaluate(model, test_loader, adj, device, scaler)

    print("\n" + "=" * 52)
    print("TEST SET RESULTS")
    print("=" * 52)
    for k, v in test_m.items():
        print(f"  {k:<8}: {v:.4f}")
    print("=" * 52)

    history["test_metrics"] = test_m
    return history
