"""
GridCFN – 训练循环

训练策略：
  - warmup 期间完全跳过 CLUB（包括 forward），让主网络先稳定
  - warmup 结束后用独立 club_optimizer 更新变分网络
  - MI loss clamp(-1, +∞)：CLUB 是上界估计，偶发小负值属正常，
    但 < -1 通常表示变分网络本次估计严重失效，此时跳过 MI 项
  - Val 在归一化域评估（用于早停判断）；Test 在反归一化域报告
"""

import logging
import math
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import GridCFN


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------

def mae(pred, true):
    return float(np.abs(pred - true).mean())


def rmse(pred, true):
    return float(np.sqrt(((pred - true) ** 2).mean()))


def crps_empirical(samples: np.ndarray, y: np.ndarray) -> float:
    """
    经验 CRPS（Gneiting & Raftery 2007）：
      CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|，X, X' iid ~ F

    使用不放回置换近似 E|X - X'|，避免自配对（|X-X|=0）压低 spread 导致 CRPS 虚高。

    samples : [S, ...] S 个粒子
    y       : [...] 真实值
    """
    S        = samples.shape[0]
    mae_term = np.abs(samples - y[None]).mean(axis=0)

    n_rep   = min(10, S - 1)
    spreads = []
    rng     = np.random.default_rng(seed=0)
    for _ in range(n_rep):
        perm  = rng.permutation(S)
        clash = np.where(perm == np.arange(S))[0]
        for idx in clash:
            swap = (idx + 1) % S
            perm[idx], perm[swap] = perm[swap], perm[idx]
        spreads.append(np.abs(samples - samples[perm]).mean(axis=0))

    spread        = np.mean(spreads, axis=0)
    crps_per_node = mae_term - 0.5 * spread
    return float(crps_per_node.mean())


def picp_empirical(samples: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> float:
    """基于经验分位数的区间覆盖率（PICP），不依赖高斯假设。"""
    alpha   = (1.0 - confidence) / 2.0
    lower   = np.quantile(samples, alpha,     axis=0)
    upper   = np.quantile(samples, 1 - alpha, axis=0)
    covered = ((y >= lower) & (y <= upper)).astype(float)
    return float(covered.mean())


def pinaw_empirical(samples: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> float:
    """基于经验分位数的归一化区间宽度（PINAW）。"""
    alpha   = (1.0 - confidence) / 2.0
    lower   = np.quantile(samples, alpha,     axis=0)
    upper   = np.quantile(samples, 1 - alpha, axis=0)
    width   = upper - lower
    y_range = y.max() - y.min() + 1e-8
    return float((width / y_range).mean())


def evaluate_all(samples: np.ndarray, y_all: np.ndarray) -> dict:
    """
    统一评估入口，所有概率指标走经验路径（不依赖高斯假设）。

    samples : [S, total, N, out_dim]
    y_all   : [total, N, out_dim]
    """
    mu_all = samples.mean(axis=0)
    return {
        "MAE":   mae(mu_all, y_all),
        "RMSE":  rmse(mu_all, y_all),
        "CRPS":  crps_empirical(samples, y_all),
        "PICP":  picp_empirical(samples, y_all),
        "PINAW": pinaw_empirical(samples, y_all),
    }


# ---------------------------------------------------------------------------
# 反归一化辅助函数
# ---------------------------------------------------------------------------

def _inverse_samples(samples: np.ndarray, scaler) -> np.ndarray:
    """对 [S, ...] 格式的采样粒子批量反归一化。"""
    shape = samples.shape
    return scaler.inverse_transform(samples.reshape(-1)).reshape(shape)


def _inverse_y(y: np.ndarray, scaler) -> np.ndarray:
    """对真实值 [total, N, D] 反归一化。"""
    return scaler.inverse_transform(y.reshape(-1)).reshape(y.shape)


# ---------------------------------------------------------------------------
# Temperature Calibration
# ---------------------------------------------------------------------------

def calibrate_temperature(samples: np.ndarray, y_all: np.ndarray,
                           target_coverage: float = 0.95,
                           grid: np.ndarray = None) -> float:
    """
    后验温度缩放：以均值为中心放缩粒子，搜索使经验 PICP 最接近目标覆盖率的 T。

    samples_scaled = mu + T * (samples - mu)

    使用经验分位数 PICP，对非高斯分布（如 Electricity 重尾）也适用。
    搜索范围 [0.5, 5.0]，T>1 表示放宽置信区间，T<1 表示收紧。
    """
    if grid is None:
        grid = np.linspace(0.5, 5.0, 91)

    mu       = samples.mean(axis=0)
    best_T   = 1.0
    best_gap = float("inf")

    for T in grid:
        samples_scaled = mu[None] + T * (samples - mu[None])
        coverage = picp_empirical(samples_scaled, y_all, target_coverage)
        gap      = abs(coverage - target_coverage)
        if gap < best_gap:
            best_gap = gap
            best_T   = float(T)

    return best_T


# ---------------------------------------------------------------------------
# 单轮训练
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:           GridCFN,
    loader:          DataLoader,
    optimizer:       torch.optim.Optimizer,
    club_optimizer:  torch.optim.Optimizer,
    adj_norm:        torch.Tensor,
    edge_index:      torch.Tensor,
    device:          torch.device,
    epoch:           int,
    grad_clip:       float = 1.0,
    warmup_epochs:   int = 5,
    cfm_n_t_samples: int = 4,
    sigma_min:       float = 0.01,
) -> Dict[str, float]:
    model.train()
    total_loss = total_cfm = total_mi = total_var = 0.0
    n_batches  = 0

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    train_club = (epoch > warmup_epochs)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        He_prime, Hs_prime, He, Hs, _ = model(x, adj_norm_, edge_idx_)

        # Step 1: 更新 CLUB 变分网络（主网络参数 detach）
        if train_club:
            var_loss = model.club.variational_loss(He.detach(), Hs.detach())
            club_optimizer.zero_grad()
            var_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.club.parameters(), max_norm=1.0)
            club_optimizer.step()
            var_loss_val = var_loss.item()
        else:
            var_loss_val = 0.0

        # Step 2: 更新主网络
        y_target = y[..., :model.out_dim]
        cfm_l    = model.cfm_loss(He_prime, Hs_prime, y_target,
                                   n_t_samples=cfm_n_t_samples,
                                   sigma_min=sigma_min)

        if train_club:
            mi_loss = model.club(He, Hs)
            if mi_loss.item() < -1.0:
                # 变分网络估计失效，跳过 MI 项，仍记录用于监控
                loss   = cfm_l
                mi_val = mi_loss.item()
            else:
                mi_loss_clamped = mi_loss.clamp(min=-0.5)
                loss   = cfm_l + model.lambda_mi * mi_loss_clamped
                mi_val = mi_loss.item()
        else:
            loss   = cfm_l
            mi_val = 0.0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_cfm  += cfm_l.item()
        total_mi   += mi_val
        total_var  += var_loss_val
        n_batches  += 1

    return {
        "loss":     total_loss / n_batches,
        "cfm":      total_cfm  / n_batches,
        "mi":       total_mi   / n_batches,
        "var_loss": total_var  / n_batches,
    }


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: GridCFN, loader: DataLoader,
             adj_norm: torch.Tensor, edge_index: torch.Tensor,
             device: torch.device,
             scaler=None, return_preds: bool = False,
             n_samples: int = 50, n_steps: int = 20,
             temperature: float = 1.0,
             inverse_transform: bool = True,
             sigma_min: float = 0.01,
             x0_scale: float = 1.0):
    """
    CFM 推断评估。

    inverse_transform=False 时在归一化域评估（用于验证集早停）；
    inverse_transform=True  时反归一化后评估（用于最终测试报告）。
    temperature 作用于粒子（以 mu 为中心放缩），与 calibrate_temperature 定义一致。
    """
    model.eval()
    samples_list, y_list = [], []

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    for x, y in loader:
        x = x.to(device)
        He_prime, Hs_prime, _, _, _ = model(x, adj_norm_, edge_idx_)

        raw_samples = model.sample(
            He_prime, Hs_prime,
            n_samples=n_samples,
            n_steps=n_steps,
            sigma_min=sigma_min,
            x0_scale=x0_scale,
        ).cpu().numpy()

        samples_list.append(raw_samples)
        y_list.append(y.numpy())

    samples_all = np.concatenate(samples_list, axis=1)              # [S, total, N, D]
    y_all       = np.concatenate(y_list, axis=0)[..., :model.out_dim]

    if inverse_transform and scaler is not None:
        samples_all = _inverse_samples(samples_all, scaler)
        y_all       = _inverse_y(y_all, scaler)

    if temperature != 1.0:
        mu          = samples_all.mean(axis=0, keepdims=True)
        samples_all = mu + temperature * (samples_all - mu)

    metrics = evaluate_all(samples_all, y_all)

    if return_preds:
        return metrics, samples_all, y_all
    return metrics


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(model: GridCFN, train_loader, val_loader, test_loader,
          adj_norm, edge_index, device, cfg_train,
          scaler=None, logger=None) -> Dict:

    if logger is None:
        logger = logging.getLogger("gridcfn.train")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                             datefmt="%H:%M:%S"))
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
    )
    # club_optimizer 使用较小学习率，防止变分网络过拟合崩溃
    club_optimizer = torch.optim.Adam(
        model.club.parameters(),
        lr=cfg_train.lr * 0.5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg_train.lr_decay_factor,
        patience=cfg_train.lr_decay_patience,
    )

    n_samples_val   = getattr(cfg_train, "cfm_n_samples",      50)
    n_steps         = getattr(cfg_train, "cfm_n_steps",        20)
    cfm_n_t_samples = getattr(cfg_train, "cfm_n_t_samples",     4)
    n_samples_test  = getattr(cfg_train, "cfm_n_samples_test", 200)
    warmup_epochs   = getattr(cfg_train, "warmup_epochs",        5)
    sigma_min       = getattr(cfg_train, "cfm_sigma_min",      0.01)
    x0_scale        = getattr(cfg_train, "cfm_x0_scale",       1.0)

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history = {
        "train_loss": [], "train_cfm": [], "train_mi": [], "train_var_loss": [],
        "val_crps": [], "val_mae": [], "val_rmse": [],
    }

    logger.info("Val 指标在归一化域（用于早停），Test 指标在反归一化域（实际量纲）")
    logger.info(f"CLUB warmup: {warmup_epochs} epochs，club_lr={cfg_train.lr * 0.5:.2e}")
    header = (f"{'Epoch':>6} | {'Loss':>8} | {'CFM':>8} | {'MI':>8} | "
              f"{'VarLoss':>9} | {'Val MAE':>8} | {'Val RMSE':>9} | "
              f"{'Val CRPS':>9} | {'LR':>8} | {'Time':>6}")
    logger.info(header)
    logger.info("-" * len(header))

    for epoch in range(1, cfg_train.max_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, club_optimizer,
            adj_norm, edge_index, device, epoch,
            cfg_train.grad_clip, warmup_epochs,
            cfm_n_t_samples=cfm_n_t_samples,
            sigma_min=sigma_min,
        )
        val_m = evaluate(
            model, val_loader, adj_norm, edge_index, device,
            scaler=scaler,
            n_samples=n_samples_val, n_steps=n_steps,
            temperature=1.0,
            inverse_transform=False,
            sigma_min=sigma_min, x0_scale=x0_scale,
        )
        scheduler.step(val_m["CRPS"])

        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["train_cfm"].append(train_m["cfm"])
        history["train_mi"].append(train_m["mi"])
        history["train_var_loss"].append(train_m["var_loss"])
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])
        history["val_rmse"].append(val_m["RMSE"])

        logger.info(
            f"{epoch:>6} | {train_m['loss']:>8.4f} | {train_m['cfm']:>8.4f} | "
            f"{train_m['mi']:>8.4f} | {train_m['var_loss']:>9.4f} | "
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
                    f"\n早停于 epoch {epoch}（最佳 val CRPS={best_val_crps:.4f}，归一化域）"
                )
                break

    model.load_state_dict(
        torch.load(cfg_train.save_path, map_location=device, weights_only=True)
    )

    # Temperature Calibration（在验证集反归一化域上搜索）
    logger.info("\n正在验证集上做 Temperature Calibration...")
    _, samples_val, y_val = evaluate(
        model, val_loader, adj_norm, edge_index, device,
        scaler=scaler, return_preds=True,
        n_samples=n_samples_test, n_steps=n_steps,
        temperature=1.0, inverse_transform=True,
        sigma_min=sigma_min, x0_scale=x0_scale,
    )
    best_T      = calibrate_temperature(samples_val, y_val, target_coverage=0.95)
    picp_before = picp_empirical(samples_val, y_val)
    mu_val      = samples_val.mean(axis=0)
    scaled_val  = mu_val[None] + best_T * (samples_val - mu_val[None])
    picp_after  = picp_empirical(scaled_val, y_val)
    logger.info(
        f"最优 Temperature: {best_T:.3f}  "
        f"（验证集 PICP@T=1.0: {picp_before:.4f} → PICP@T={best_T:.2f}: {picp_after:.4f}）"
    )

    # 测试集最终评估（反归一化域 + 归一化域各报一次）
    test_m, samples_test, y_test = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler, return_preds=True,
        n_samples=n_samples_test, n_steps=n_steps,
        temperature=best_T, inverse_transform=True,
        sigma_min=sigma_min, x0_scale=x0_scale,
    )
    test_m_norm, _, _ = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler, return_preds=True,
        n_samples=n_samples_test, n_steps=n_steps,
        temperature=best_T, inverse_transform=False,
        sigma_min=sigma_min, x0_scale=x0_scale,
    )

    sep = "=" * 60
    logger.info(f"\n{sep}")
    logger.info(f"TEST SET RESULTS — 反归一化域 (Temperature={best_T:.3f})")
    logger.info(sep)
    for k, v in test_m.items():
        logger.info(f"  {k:<8}: {v:.4f}")

    logger.info(f"\n{sep}")
    logger.info("TEST SET RESULTS — 归一化域")
    logger.info(sep)
    for k, v in test_m_norm.items():
        logger.info(f"  {k:<8}: {v:.4f}")
    logger.info(sep)

    history["test_metrics"]      = test_m
    history["test_metrics_norm"] = test_m_norm
    history["best_temperature"]  = best_T
    history["test_shape"]        = list(samples_test.shape)
    return history