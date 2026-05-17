"""
GridCFN – 训练循环（多步预测版）

多步改动说明：
  - y 形状从 [B, N, 1] 变为 [B, T_out, N, F]，内部 reshape 为 [B, N, T_out*F]
    再传给 cfm_loss。
  - evaluate 中 sample 返回 [S, B, N, T_out, F]，按步分解指标
    (MAE/RMSE/CRPS/PICP/PINAW)，并汇报 avg 及各步数值。
  - 早停监控 val_crps_avg（所有步平均 CRPS）。
  - calibrate_temperature 在展平的 [S, total*T_out, N, F] 上搜索，
    保持与原版一致的后验校准逻辑。

训练策略（不变）：
  - warmup 期间跳过 CLUB
  - MI loss clamp(-1, +∞)，< -1 时跳过
  - Val 在归一化域评估；Test 在反归一化域报告
"""

import logging
import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import GridCFN


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)


# ---------------------------------------------------------------------------
# 评估指标（逐元素，支持任意形状末尾维度）
# ---------------------------------------------------------------------------

def mae(pred, true):
    return float(np.abs(pred - true).mean())


def rmse(pred, true):
    return float(np.sqrt(((pred - true) ** 2).mean()))


def mape(pred, true, eps=1e-8):
    return float((np.abs((pred - true) / (np.abs(true) + eps))).mean() * 100.0)


def crps_empirical(samples: np.ndarray, y: np.ndarray) -> float:
    """
    经验 CRPS。
    samples : [S, ...]   S 个粒子
    y       : [...]      真实值
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

    spread = np.mean(spreads, axis=0)
    return float((mae_term - 0.5 * spread).mean())


def picp_empirical(samples: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> float:
    alpha   = (1.0 - confidence) / 2.0
    lower   = np.quantile(samples, alpha,     axis=0)
    upper   = np.quantile(samples, 1 - alpha, axis=0)
    covered = ((y >= lower) & (y <= upper)).astype(float)
    return float(covered.mean())


def pinaw_empirical(samples: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> float:
    alpha   = (1.0 - confidence) / 2.0
    lower   = np.quantile(samples, alpha,     axis=0)
    upper   = np.quantile(samples, 1 - alpha, axis=0)
    width   = upper - lower
    y_range = y.max() - y.min() + 1e-8
    return float((width / y_range).mean())


def evaluate_all(samples: np.ndarray, y_all: np.ndarray) -> dict:
    """
    统一评估入口（多步版）。

    samples : [S, total, N, T_out, F]
    y_all   : [total, N, T_out, F]

    返回 avg 指标 + 各预测步指标（key 格式: MAE_h1, MAE_h2, ...）
    """
    mu_all = samples.mean(axis=0)   # [total, N, T_out, F]
    T_out  = y_all.shape[2]

    metrics = {
        "MAE":   mae(mu_all, y_all),
        "RMSE":  rmse(mu_all, y_all),
        "MAPE":  mape(mu_all, y_all),
        "CRPS":  crps_empirical(samples, y_all),
        "PICP":  picp_empirical(samples, y_all),
        "PINAW": pinaw_empirical(samples, y_all),
    }

    # 按预测步分解
    for h in range(T_out):
        s_h = samples[:, :, :, h, :]   # [S, total, N, F]
        y_h = y_all[:, :, h, :]        # [total, N, F]
        mu_h = s_h.mean(axis=0)
        metrics[f"MAE_h{h+1}"]  = mae(mu_h, y_h)
        metrics[f"RMSE_h{h+1}"] = rmse(mu_h, y_h)
        metrics[f"CRPS_h{h+1}"] = crps_empirical(s_h, y_h)

    return metrics


# ---------------------------------------------------------------------------
# 反归一化辅助
# ---------------------------------------------------------------------------

def _inverse_samples(samples: np.ndarray, scaler) -> np.ndarray:
    """samples: [S, ...] → 反归一化"""
    shape = samples.shape
    return scaler.inverse_transform(samples.reshape(-1)).reshape(shape)


def _inverse_y(y: np.ndarray, scaler) -> np.ndarray:
    return scaler.inverse_transform(y.reshape(-1)).reshape(y.shape)


# ---------------------------------------------------------------------------
# Temperature Calibration（展平后搜索，与原版逻辑一致）
# ---------------------------------------------------------------------------

def calibrate_temperature(samples: np.ndarray, y_all: np.ndarray,
                           target_coverage: float = 0.95,
                           grid: np.ndarray = None) -> float:
    """
    后验温度缩放。
    samples : [S, total, N, T_out, F]
    y_all   : [total, N, T_out, F]
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
    main_params,
    grad_clip:       float = 1.0,
    warmup_epochs:   int = 5,
    cfm_n_t_samples: int = 4,
    sigma_min:       float = 0.01,
    club_inner_steps: int = 3,       # D: CLUB 每步内循环更新次数
) -> Dict[str, float]:
    model.train()
    total_loss = total_cfm = total_mi = total_mi_prime = total_var = total_gnorm = 0.0
    n_batches  = 0

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    train_club = (epoch > warmup_epochs)

    # CLUB 激活后前 3 个 epoch 线性升温 lambda_mi，避免突然引入大正则导致不稳定
    if train_club:
        ramp_epochs  = 3
        epochs_since = epoch - warmup_epochs          # 1, 2, 3, 4, ...
        ramp_factor  = min(1.0, epochs_since / ramp_epochs)
    else:
        ramp_factor  = 0.0

    for x, y in loader:
        # x: [B, T_in, N, F]
        # y: [B, T_out, N, F]  ← 多步版，保留时间维
        x = x.to(device)
        y = y.to(device)

        He_prime, Hs_prime, He, Hs, _ = model(x, adj_norm_, edge_idx_)

        # Step 1: 更新 CLUB 变分网络（内循环多次，降低 MI 估计方差）
        if train_club:
            var_loss_val = 0.0
            for _ in range(club_inner_steps):
                var_loss = model.club.variational_loss(He.detach(), Hs.detach())
                club_optimizer.zero_grad()
                var_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.club.parameters(), max_norm=1.0)
                club_optimizer.step()
                var_loss_val = var_loss.item()
        else:
            var_loss_val = 0.0

        # Step 2: 更新主网络
        # y: [B, T_out, N, F] → [B, N, T_out*F]（CFM 期望的格式）
        B, T_out, N, F = y.shape
        y_target = y.permute(0, 2, 1, 3).reshape(B, N, T_out * F)

        cfm_l = model.cfm_loss(He_prime, Hs_prime, y_target,
                               n_t_samples=cfm_n_t_samples,
                               sigma_min=sigma_min)

        if train_club:
            _set_requires_grad(model.club, False)
            try:
                mi_loss = model.club(He, Hs)
                mi_val  = mi_loss.item()
                # 只惩罚正的 MI 估计；负估计不再作为奖励项降低主损失。
                mi_penalty = mi_loss.clamp(min=0.0)
                loss = cfm_l + model.lambda_mi * ramp_factor * mi_penalty

                with torch.no_grad():
                    if He_prime.shape[-1] == He.shape[-1] and Hs_prime.shape[-1] == Hs.shape[-1]:
                        mi_prime_val = model.club(He_prime.detach(), Hs_prime.detach()).item()
                    else:
                        mi_prime_val = float("nan")
            finally:
                _set_requires_grad(model.club, True)
        else:
            loss   = cfm_l
            mi_val = 0.0
            mi_prime_val = 0.0

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(main_params, max_norm=grad_clip)
        optimizer.step()

        total_loss  += loss.item()
        total_cfm   += cfm_l.item()
        total_mi    += mi_val
        total_mi_prime += mi_prime_val
        total_var   += var_loss_val
        total_gnorm += grad_norm.item()
        n_batches   += 1

    return {
        "loss":      total_loss  / n_batches,
        "cfm":       total_cfm   / n_batches,
        "mi":        total_mi    / n_batches,
        "mi_prime":  total_mi_prime / n_batches,
        "var_loss":  total_var   / n_batches,
        "grad_norm": total_gnorm / n_batches,
    }


# ---------------------------------------------------------------------------
# 评估（多步版）
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
    CFM 推断评估（多步版）。

    返回 metrics dict，key 含 avg 指标 + 各步指标（MAE_h1, ..., CRPS_h1, ...）。

    samples_all : [S, total, N, T_out, feat_dim]
    y_all       : [total, N, T_out, feat_dim]
    """
    model.eval()
    samples_list, y_list = [], []

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    for x, y in loader:
        # x: [B, T_in, N, F]
        # y: [B, T_out, N, F]
        x = x.to(device)
        He_prime, Hs_prime, _, _, _ = model(x, adj_norm_, edge_idx_)

        # raw_samples: [S, B, N, T_out, feat_dim]
        raw_samples = model.sample(
            He_prime, Hs_prime,
            n_samples=n_samples,
            n_steps=n_steps,
            sigma_min=sigma_min,
            x0_scale=x0_scale,
        ).cpu().numpy()

        samples_list.append(raw_samples)
        # y: [B, T_out, N, F] → [B, N, T_out, F]
        y_np = y.permute(0, 2, 1, 3).numpy()
        y_list.append(y_np)

    # samples_all: [S, total, N, T_out, feat_dim]
    samples_all = np.concatenate(samples_list, axis=1)
    # y_all:       [total, N, T_out, feat_dim]
    y_all       = np.concatenate(y_list, axis=0)

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

    club_param_ids = {id(p) for p in model.club.parameters()}
    main_params = [p for p in model.parameters() if id(p) not in club_param_ids]

    optimizer = torch.optim.Adam(
        main_params,
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
    )
    club_optimizer = torch.optim.Adam(
        model.club.parameters(),
        lr=cfg_train.lr * 0.5,
        weight_decay=cfg_train.weight_decay,   # 防止 CLUB 网络在小图数据集上过拟合
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg_train.lr_decay_factor,
        patience=cfg_train.lr_decay_patience,
    )

    # 训练阶段 val 只需相对排序准确，减半采样数可显著加速每 epoch 时间。
    # 最终 test 评估仍使用 cfm_n_samples_test（默认 200），不受影响。
    n_samples_val    = max(20, getattr(cfg_train, "cfm_n_samples",      50) // 2)
    n_steps          = getattr(cfg_train, "cfm_n_steps",        20)
    cfm_n_t_samples  = getattr(cfg_train, "cfm_n_t_samples",     4)
    n_samples_test   = getattr(cfg_train, "cfm_n_samples_test", 200)
    warmup_epochs    = getattr(cfg_train, "warmup_epochs",        5)
    sigma_min        = getattr(cfg_train, "cfm_sigma_min",      0.01)
    x0_scale         = getattr(cfg_train, "cfm_x0_scale",       1.0)
    club_inner_steps = getattr(cfg_train, "club_inner_steps",     3)

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history = {
        "train_loss": [], "train_cfm": [], "train_mi": [], "train_mi_prime": [], "train_var_loss": [],
        "val_crps": [], "val_mae": [], "val_rmse": [],
    }

    logger.info("Val 指标在归一化域（用于早停），Test 指标在反归一化域（实际量纲）")
    logger.info(f"多步预测 T_out={model.T_out}")
    logger.info(f"CLUB warmup: {warmup_epochs} epochs，club_lr={cfg_train.lr * 0.5:.2e}")
    header = (f"{'Epoch':>6} | {'Loss':>8} | {'CFM':>8} | {'MI':>8} | {'MIp':>8} | "
              f"{'VarLoss':>9} | {'GradNorm':>9} | {'Val MAE':>8} | {'Val RMSE':>9} | "
              f"{'Val CRPS':>9} | {'LR':>8} | {'Time':>6}")
    logger.info(header)
    logger.info("-" * len(header))

    for epoch in range(1, cfg_train.max_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, club_optimizer,
            adj_norm, edge_index, device, epoch,
            main_params, cfg_train.grad_clip, warmup_epochs,
            cfm_n_t_samples=cfm_n_t_samples,
            sigma_min=sigma_min,
            club_inner_steps=club_inner_steps,
        )
        val_m = evaluate(
            model, val_loader, adj_norm, edge_index, device,
            scaler=scaler,
            n_samples=n_samples_val, n_steps=n_steps,
            temperature=1.0,
            inverse_transform=False,
            sigma_min=sigma_min, x0_scale=x0_scale,
        )
        # 早停监控：所有步平均 CRPS
        val_crps_avg = val_m["CRPS"]
        scheduler.step(val_crps_avg)

        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["train_cfm"].append(train_m["cfm"])
        history["train_mi"].append(train_m["mi"])
        history["train_mi_prime"].append(train_m["mi_prime"])
        history["train_var_loss"].append(train_m["var_loss"])
        history["val_crps"].append(val_crps_avg)
        history["val_mae"].append(val_m["MAE"])
        history["val_rmse"].append(val_m["RMSE"])

        logger.info(
            f"{epoch:>6} | {train_m['loss']:>8.4f} | {train_m['cfm']:>8.4f} | "
            f"{train_m['mi']:>8.4f} | {train_m['mi_prime']:>8.4f} | "
            f"{train_m['var_loss']:>9.4f} | "
            f"{train_m['grad_norm']:>9.3f} | "
            f"{val_m['MAE']:>8.4f} | {val_m['RMSE']:>9.4f} | "
            f"{val_crps_avg:>9.4f} | {cur_lr:>8.2e} | {elapsed:>5.1f}s"
        )

        if val_crps_avg < best_val_crps:
            best_val_crps     = val_crps_avg
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

    # Temperature Calibration
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

    # 测试集：只跑一次采样，复用 samples 计算三套指标
    _, samples_test, y_test = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler, return_preds=True,
        n_samples=n_samples_test, n_steps=n_steps,
        temperature=1.0, inverse_transform=True,   # 先拿未校准的反归一化样本
        sigma_min=sigma_min, x0_scale=x0_scale,
    )
    _, samples_test_norm, y_test_norm = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler, return_preds=True,
        n_samples=n_samples_test, n_steps=n_steps,
        temperature=1.0, inverse_transform=False,  # 归一化域
        sigma_min=sigma_min, x0_scale=x0_scale,
    )

    # 从同一批 samples 派生四套指标，不重复跑 ODE
    # 1. 未校准反归一化（与 baselines 对比用）
    test_m_raw = evaluate_all(samples_test, y_test)

    # 2. 校准后反归一化
    mu_test = samples_test.mean(axis=0, keepdims=True)
    samples_calibrated = mu_test + best_T * (samples_test - mu_test)
    test_m = evaluate_all(samples_calibrated, y_test)

    # 3. 校准后归一化域
    mu_norm = samples_test_norm.mean(axis=0, keepdims=True)
    samples_calibrated_norm = mu_norm + best_T * (samples_test_norm - mu_norm)
    test_m_norm = evaluate_all(samples_calibrated_norm, y_test_norm)

    # 4. 未校准归一化域
    test_m_raw_norm = evaluate_all(samples_test_norm, y_test_norm)

    sep = "=" * 60
    avg_keys = ["MAE", "RMSE", "MAPE", "CRPS", "PICP", "PINAW"]

    logger.info(f"\n{sep}")
    logger.info(f"TEST SET RESULTS — 反归一化域 (Temperature={best_T:.3f}, T_out={model.T_out})")
    logger.info(sep)
    for k in avg_keys:
        logger.info(f"  {k:<8}: {test_m[k]:.4f}")
    logger.info(f"\n  {'Step':<6}  {'MAE':>8}  {'RMSE':>8}  {'CRPS':>8}")
    for h in range(model.T_out):
        mae_h  = test_m.get(f"MAE_h{h+1}",  float("nan"))
        rmse_h = test_m.get(f"RMSE_h{h+1}", float("nan"))
        crps_h = test_m.get(f"CRPS_h{h+1}", float("nan"))
        logger.info(f"  h={h+1:<4}  {mae_h:>8.4f}  {rmse_h:>8.4f}  {crps_h:>8.4f}")

    logger.info(f"\n{sep}")
    logger.info(f"TEST SET RESULTS — 归一化域 (Temperature={best_T:.3f}, T_out={model.T_out})")
    logger.info(sep)
    for k in avg_keys:
        logger.info(f"  {k:<8}: {test_m_norm[k]:.4f}")

    logger.info(f"\n{sep}")
    logger.info(f"TEST SET RESULTS — 未校准反归一化 / 与 baselines 对比用 (Temperature=1.0)")
    logger.info(sep)
    for k in avg_keys:
        logger.info(f"  {k:<8}: {test_m_raw[k]:.4f}")

    logger.info(f"\n{sep}")
    logger.info(f"TEST SET RESULTS — 未校准归一化域 (Temperature=1.0)")
    logger.info(sep)
    for k in avg_keys:
        logger.info(f"  {k:<8}: {test_m_raw_norm[k]:.4f}")
    logger.info(sep)

    history["test_metrics"]          = test_m
    history["test_metrics_norm"]     = test_m_norm
    history["test_metrics_raw"]      = test_m_raw       # 未校准反归一化，供与 baselines 对比
    history["test_metrics_raw_norm"] = test_m_raw_norm  # 未校准归一化域
    history["best_temperature"]      = best_T
    history["test_shape"]            = list(samples_test.shape)
    return history
