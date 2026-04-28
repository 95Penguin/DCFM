"""
GridCFN – 训练循环（CFM 版 v4，全量 bug 修复）

修复列表：
  Fix-ClubLR      : club_optimizer lr 从 lr*5 降到 lr*2
  Fix-ClubClip    : CLUB step 前加 grad clip (max_norm=1.0)
  Fix-WarmupClub  : warmup 期间完全跳过 CLUB 的 forward + backward
                    （包括 mi_loss 计算，不只是 loss 中的 lambda 项）
  Fix-SigmaFloor  : sigma 下界改为自适应（mean * 1e-4），防止反归一化后下界失效
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


def crps_score(mu, sigma, y):
    from scipy.stats import norm
    z   = (y - mu) / (sigma + 1e-8)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    return float((sigma * (z * (2*Phi - 1) + 2*phi - 1/math.sqrt(math.pi))).mean())


def picp(mu, sigma, y, confidence=0.95):
    from scipy.stats import norm
    z       = norm.ppf((1 + confidence) / 2)
    covered = ((y >= mu - z*sigma) & (y <= mu + z*sigma)).astype(float)
    return float(covered.mean())


def pinaw(mu, sigma, y, confidence=0.95):
    from scipy.stats import norm
    z       = norm.ppf((1 + confidence) / 2)
    width   = 2 * z * sigma
    y_range = y.max() - y.min() + 1e-8
    return float((width / y_range).mean())


def evaluate_all(mu_all, sigma_all, y_all):
    return {
        "MAE":   mae(mu_all, y_all),
        "RMSE":  rmse(mu_all, y_all),
        "CRPS":  crps_score(mu_all, sigma_all, y_all),
        "PICP":  picp(mu_all, sigma_all, y_all),
        "PINAW": pinaw(mu_all, sigma_all, y_all),
    }


# ---------------------------------------------------------------------------
# 反归一化辅助函数
# ---------------------------------------------------------------------------

def _inv_zscore_mean(arr: np.ndarray, scaler) -> np.ndarray:
    """只做反 Z-score（乘 std + 加 mean），不做 expm1。"""
    return arr * scaler.std + scaler.mean


def _inv_zscore_sigma(arr: np.ndarray, scaler) -> np.ndarray:
    """对 sigma 只乘 std（不加 mean）。sigma 是尺度量，不加偏移。"""
    return arr * scaler.std


# ---------------------------------------------------------------------------
# Temperature Calibration
# ---------------------------------------------------------------------------

def calibrate_temperature(mu_all: np.ndarray, sigma_all: np.ndarray,
                           y_all: np.ndarray,
                           target_coverage: float = 0.95,
                           grid: Optional[np.ndarray] = None) -> float:
    """在验证集上 grid search 最优 temperature T*，使 PICP ≈ target_coverage。"""
    if grid is None:
        grid = np.linspace(0.5, 3.0, 51)

    best_T   = 1.0
    best_gap = float("inf")

    for T in grid:
        coverage = picp(mu_all, sigma_all * T, y_all, confidence=target_coverage)
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
) -> Dict[str, float]:
    """
    [Fix-WarmupClub]
    原版问题：warmup 期间虽然 loss = cfm_l（不加 MI 正则），但仍调用了
    model.club(He, Hs) 并触发 backward，让主优化器把未训练的 CLUB 梯度
    混入主网络更新中。

    修复：warmup 期间完全跳过 CLUB 的所有计算（forward + backward 全部跳过），
    只做纯 CFM 训练。warmup 结束后再引入 CLUB。

    [Fix-ClubLR] club_optimizer lr = lr*2（在 train() 里设置）。
    [Fix-ClubClip] CLUB backward 后 clip grad norm=1.0。
    """
    model.train()
    total_loss = total_cfm = total_mi = total_var = 0.0
    n_batches  = 0

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    train_club = (epoch > warmup_epochs)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        context_feat, He, Hs, _ = model(x, adj_norm_, edge_idx_)

        # ── Step 1: CLUB 变分网络更新 ─────────────────────────────────────
        if train_club:
            var_loss = model.club.variational_loss(He.detach(), Hs.detach())
            club_optimizer.zero_grad()
            var_loss.backward()
            # [Fix-ClubClip] 去掉 normalize 后 CLUB 梯度量级增大，需要 clip
            torch.nn.utils.clip_grad_norm_(model.club.parameters(), max_norm=1.0)
            club_optimizer.step()
            var_loss_val = var_loss.item()
        else:
            var_loss_val = 0.0

        # ── Step 2: 主网络更新 ───────────────────────────────────────────
        y_target = y[..., :model.out_dim]
        cfm_l    = model.cfm_loss(context_feat, y_target, n_t_samples=cfm_n_t_samples)

        if train_club:
            # [Fix-WarmupClub] warmup 后再计算 mi_loss，避免未训练的 CLUB 梯度污染主网络
            mi_loss = model.club(He, Hs)

            # [Fix-MIClamp] MI 理论上 >= 0，但 CLUB 是上界估计，偶尔可为小负数。
            # 若 mi_loss < -1，说明变分网络本次估计严重失效（过拟合），
            # 跳过 MI 项，只用 cfm_l 更新主网络，避免极端负梯度冲垮表征。
            if mi_loss.item() < -1.0:
                loss   = cfm_l
                mi_val = mi_loss.item()   # 仍记录用于监控，但不加入 loss
            else:
                # clamp 到 [-0.5, 正无穷]，允许小幅负值但阻断极端崩溃
                mi_loss_clamped = mi_loss.clamp(min=-0.5)
                loss   = cfm_l + model.lambda_mi * mi_loss_clamped
                mi_val = mi_loss.item()
        else:
            # [Fix-WarmupClub] warmup 期间完全跳过，mi_loss = 0，无任何 CLUB forward
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
             inverse_transform: bool = True):
    """
    CFM 推断评估。

    [Fix-SigmaFloor]
    原版：sigma_cal = np.maximum(sigma_all * temperature, 1e-6)
    问题：inverse_transform=True 时 sigma_all 已乘以 scaler.std，
          1e-6 的下界在反归一化域几乎无意义（Solar std≈0.3，Electricity log-std≈1.2）。
    修复：下界改为自适应 max(sigma_all) * 1e-4，保证相对有效。
          两种 transform 模式均适用。
    """
    model.eval()
    mu_list, sigma_list, y_list = [], [], []

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    for x, y in loader:
        x = x.to(device)
        context_feat, _, _, _ = model(x, adj_norm_, edge_idx_)

        # 并行采样 [S, B, N, out_dim]
        samples = model.sample(context_feat, n_samples=n_samples, n_steps=n_steps)

        # 无偏估计（Bessel 校正）
        mu_t    = samples.mean(dim=0).cpu().numpy()
        sigma_t = samples.std(dim=0, correction=1).cpu().numpy()

        mu_list.append(mu_t)
        sigma_list.append(sigma_t)
        y_list.append(y.numpy())

    mu_all    = np.concatenate(mu_list,    axis=0)   # [total, N, out_dim]
    sigma_all = np.concatenate(sigma_list, axis=0)
    y_all     = np.concatenate(y_list,     axis=0)[..., :mu_all.shape[-1]]

    # ── 反归一化（只反 Z-score，不做 expm1）────────────────────────────────
    if inverse_transform and scaler is not None:
        mu_all    = _inv_zscore_mean(mu_all,    scaler)
        sigma_all = _inv_zscore_sigma(sigma_all, scaler)
        y_all     = _inv_zscore_mean(y_all,     scaler)

    # ── Temperature 校准 + [Fix-SigmaFloor] 自适应下界 ─────────────────────
    sigma_floor = float(np.abs(sigma_all).mean()) * 1e-4   # [Fix-SigmaFloor]
    sigma_cal   = np.maximum(sigma_all * temperature, sigma_floor)

    metrics = evaluate_all(mu_all, sigma_cal, y_all)

    if return_preds:
        # 返回未乘 temperature 的原始 sigma，供 calibrate_temperature 使用
        return metrics, mu_all, sigma_all, y_all
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
    # [Fix-ClubLR2] 进一步从 lr*2 降到 lr*0.5，防止变分网络过拟合崩溃
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

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history = {
        "train_loss": [], "train_cfm": [], "train_mi": [], "train_var_loss": [],
        "val_crps": [], "val_mae": [], "val_rmse": [],
    }

    logger.info("注意：训练中 Val 指标在归一化域（用于早停判断），"
                "最终 Test 指标在反 Z-score 域（实际量纲）。")
    logger.info(f"[v5-fix] CLUB 在 warmup ({warmup_epochs} epochs) 后开始训练，"
                f"club_lr={cfg_train.lr * 0.5:.2e}，MI clamp 保护已启用")
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
            cfg_train.grad_clip,
            warmup_epochs,
            cfm_n_t_samples=cfm_n_t_samples,
        )
        val_m = evaluate(
            model, val_loader, adj_norm, edge_index, device,
            scaler=scaler,
            n_samples=n_samples_val, n_steps=n_steps,
            temperature=1.0,
            inverse_transform=False,   # 归一化域，快速验证，用于早停
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

    # ── 加载最优权重 ────────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(cfg_train.save_path, map_location=device, weights_only=True)
    )

    # ── Temperature Calibration ─────────────────────────────────────────────
    logger.info("\n正在验证集上做 Temperature Calibration（反 Z-score 域）...")
    _, mu_val, sigma_val, y_val = evaluate(
        model, val_loader, adj_norm, edge_index, device,
        scaler=scaler,
        return_preds=True,
        n_samples=n_samples_test,
        n_steps=n_steps,
        temperature=1.0,
        inverse_transform=True,
    )
    best_T = calibrate_temperature(mu_val, sigma_val, y_val, target_coverage=0.95)
    logger.info(
        f"最优 Temperature: {best_T:.3f}  "
        f"（验证集 PICP@T=1.0: {picp(mu_val, sigma_val, y_val):.4f} → "
        f"PICP@T={best_T:.2f}: {picp(mu_val, sigma_val*best_T, y_val):.4f}）"
    )

    # ── 测试集最终评估 ──────────────────────────────────────────────────────
    test_m, mu_all, sigma_all, y_all = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler,
        return_preds=True,
        n_samples=n_samples_test,
        n_steps=n_steps,
        temperature=best_T,
        inverse_transform=True,
    )

    test_m_norm, _, _, _ = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler,
        return_preds=True,
        n_samples=n_samples_test,
        n_steps=n_steps,
        temperature=best_T,
        inverse_transform=False,
    )

    sep = "=" * 60
    logger.info(f"\n{sep}")
    logger.info(f"TEST SET RESULTS — 反 Z-score 域 (Temperature={best_T:.3f})")
    logger.info(sep)
    for k, v in test_m.items():
        logger.info(f"  {k:<8}: {v:.4f}")

    logger.info(f"\n{sep}")
    logger.info("TEST SET RESULTS — 归一化域（对应训练时的 Val CRPS 尺度）")
    logger.info(sep)
    for k, v in test_m_norm.items():
        logger.info(f"  {k:<8}: {v:.4f}")
    logger.info(sep)

    history["test_metrics"]          = test_m
    history["test_metrics_norm"]     = test_m_norm
    history["best_temperature"]      = best_T
    history["test_mu"]               = mu_all.flatten().tolist()
    history["test_sigma_calibrated"] = (sigma_all * best_T).flatten().tolist()
    history["test_y"]                = y_all.flatten().tolist()
    history["test_shape"]            = list(mu_all.shape)
    return history