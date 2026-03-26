"""
GridCFN – 训练循环、评估与指标

修复说明：
  [Bug-1 修复] MINE minimax 训练：
    - 新增独立的 mine_optimizer（Adam，对 MINE 网络参数做梯度上升）
    - 主 optimizer 只更新非 MINE 参数
    - 每个 batch 先更新 MINE（最大化 MI 估计），再更新主网络（最小化 NLL + λ·MI）

  [Bug-4 修复] 热身逻辑接入：
    - 前 warmup_epochs（默认 5）个 epoch，lambda_mi=0，只训练骨干预测
    - 热身结束后逐渐引入 MI 正则化
    - 通过 compute_loss(lambda_mi=...) 参数动态传入，不修改 model.lambda_mi
"""

import logging
import math
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    return float(
        (sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))).mean()
    )


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
    z       = norm.ppf((1 + confidence) / 2)
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

def train_one_epoch(
    model:          GridCFN,
    loader:         DataLoader,
    optimizer:      torch.optim.Optimizer,
    mine_optimizer: torch.optim.Optimizer,
    adj:            torch.Tensor,
    device:         torch.device,
    epoch:          int,
    grad_clip:      float = 1.0,
    warmup_epochs:  int   = 5,
) -> Dict[str, float]:
    """
    [Bug-1 修复] MINE minimax 训练：
      Step 1：固定主网络，只更新 MINE（最大化 MI 估计，即梯度上升）
      Step 2：固定 MINE，只更新主网络（最小化 NLL + λ·MI）

    [Bug-4 修复] 热身逻辑：
      前 warmup_epochs 个 epoch，curr_lambda_mi=0，不传递 MI 正则化梯度到主网络，
      让模型先学好基础预测，避免 MINE 还未收敛时错误的 MI 估计干扰主网络。
    """
    model.train()

    # 热身：前几个 epoch MI 正则权重为 0
    curr_lambda_mi = 0.0 if epoch <= warmup_epochs else model.lambda_mi

    total_loss = total_nll = total_mi = 0.0
    n_batches  = 0

    for x, y in loader:
        x    = x.to(device)
        y    = y.to(device)
        adj_ = adj.to(device)

        # ── Step 1: 更新 MINE（最大化 MI 估计 = 梯度上升）──────────────
        # 先 forward 一次获得 He, Hs（不需要主网络梯度）
        with torch.no_grad():
            mu_detach, sigma_detach, _ = model(x, adj_)
        # 重新 forward 只为 MINE（只对 He, Hs 计算，主网络参数不更新）
        # 通过 detach 将主网络输出从计算图中断开，只让梯度流向 MINE 网络
        model_out = _forward_for_mine(model, x, adj_)
        mi_for_mine = model_out  # MINE 估计值

        mine_optimizer.zero_grad()
        # MINE 需要最大化 MI 估计（即最小化负 MI 估计）
        (-mi_for_mine).backward()
        mine_optimizer.step()

        # ── Step 2: 更新主网络（最小化 NLL + λ·MI）────────────────────
        optimizer.zero_grad()
        mu, sigma, mi_loss = model(x, adj_)

        y_target = y[..., :mu.shape[-1]]
        loss, l_nll, l_mi = model.compute_loss(
            mu, sigma, y_target, mi_loss, lambda_mi=curr_lambda_mi
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.main_parameters(), max_norm=grad_clip)
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


def _forward_for_mine(model: GridCFN, x: torch.Tensor,
                      adj: torch.Tensor) -> torch.Tensor:
    """
    只为 MINE 的梯度上升步骤做 forward：
    将主网络输出（He, Hs）detach 后传入 MINE，
    使梯度只流向 MINE 网络参数，不影响主网络。
    """
    adj_norm   = model.normalize_adj(adj)
    H          = model.backbone(x, adj_norm)
    He, Hs, _  = model.disentangler(H)
    # detach：断开与主网络的计算图，确保梯度只更新 MINE
    mi_est = model.mine(He.detach(), Hs.detach())
    return mi_est


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: GridCFN,
             loader: DataLoader,
             adj: torch.Tensor,
             device: torch.device,
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
# 主训练函数
# ---------------------------------------------------------------------------

def train(model: GridCFN,
          train_loader: DataLoader,
          val_loader:   DataLoader,
          test_loader:  DataLoader,
          adj:          torch.Tensor,
          device:       torch.device,
          cfg_train,
          scaler=None,
          logger: Optional[logging.Logger] = None) -> Dict:
    """
    完整训练循环，含 MINE minimax 训练、热身、早停和学习率衰减。

    cfg_train : TrainConfig
    logger    : 传入 main.py 创建的 logger
    """
    if logger is None:
        logger = logging.getLogger("gridcfn.train")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                             datefmt="%H:%M:%S"))
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)

    # [Bug-1 修复] 主 optimizer 只更新非 MINE 参数
    optimizer = torch.optim.Adam(
        model.main_parameters(),
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
    )
    # [Bug-1 修复] 独立的 MINE optimizer（通常用稍大的学习率帮助 MINE 快速收敛）
    mine_optimizer = torch.optim.Adam(
        model.mine_parameters(),
        lr=cfg_train.lr * 2,   # MINE 收敛相对容易，稍大 lr 加速
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg_train.lr_decay_factor,
        patience=cfg_train.lr_decay_patience,
    )

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history           = {"train_loss": [], "val_crps": [], "val_mae": []}

    warmup_epochs = getattr(cfg_train, 'warmup_epochs', 5)

    header = (f"{'Epoch':>6} | {'Train Loss':>10} | {'Val MAE':>8} | "
              f"{'Val RMSE':>9} | {'Val CRPS':>9} | {'LR':>8} | {'Time':>6}")
    logger.info(header)
    logger.info("-" * len(header))

    for epoch in range(1, cfg_train.max_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, mine_optimizer,
            adj, device, epoch, cfg_train.grad_clip, warmup_epochs
        )

        val_m   = evaluate(model, val_loader, adj, device, scaler)
        scheduler.step(val_m["CRPS"])

        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])

        logger.info(
            f"{epoch:>6} | {train_m['loss']:>10.4f} | "
            f"{val_m['MAE']:>8.4f} | {val_m['RMSE']:>9.4f} | "
            f"{val_m['CRPS']:>9.4f} | {cur_lr:>8.2e} | {elapsed:>5.1f}s"
        )

        if val_m["CRPS"] < best_val_crps:
            best_val_crps     = val_m["CRPS"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg_train.save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg_train.patience:
                logger.info(
                    f"\n早停于 epoch {epoch}（最佳 val CRPS = {best_val_crps:.4f}）"
                )
                break

    # 加载最优权重并在测试集上评估
    model.load_state_dict(torch.load(cfg_train.save_path, map_location=device))
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
