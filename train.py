"""
GridCFN-DMSD – 训练循环

相对原版的改动：
  1. model.forward 新签名：返回 6 个值
       He_prime, Hs_prime, He, Hs, X_low_pooled, X_high_pooled
  2. CLUB 分为两个实例 club_e / club_s：
       club_e: MI(He, X_low_pooled)   — 环境表征不应含残差信息
       club_s: MI(Hs, X_high_pooled)  — 因果表征不应含趋势信息
     相应地：
       - club_param_ids 同时覆盖 model.club_e 和 model.club_s
       - club_optimizer 管理两个 CLUB 模块的参数
       - variational_loss 分别对两个实例调用
       - MI 损失 = club_e(He, X_low_pooled) + club_s(Hs, X_high_pooled)
  3. 新增 L_rank = model.rank_loss()，从 warmup 第一个 epoch 起就加入总损失
     （秩正则不依赖 CLUB 的一致性约束，可以更早起效）。
  4. 日志列名调整：MI 列监控新 CLUB 目标；MIp 列改为监控 L_rank。
  5. 其余逻辑（早停、温度校准、评估、历史记录）完全不变。
"""

import logging
import math
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import GridCFN


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)


# ---------------------------------------------------------------------------
# 评估指标（不变）
# ---------------------------------------------------------------------------

def mae(pred, true):
    return float(np.abs(pred - true).mean())


def rmse(pred, true):
    return float(np.sqrt(((pred - true) ** 2).mean()))


def mape(pred, true, eps=1e-8):
    return float((np.abs((pred - true) / (np.abs(true) + eps))).mean() * 100.0)


def crps_empirical(samples: np.ndarray, y: np.ndarray) -> float:
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
    mu_all = samples.mean(axis=0)
    T_out  = y_all.shape[2]
    metrics = {
        "MAE":   mae(mu_all, y_all),
        "RMSE":  rmse(mu_all, y_all),
        "MAPE":  mape(mu_all, y_all),
        "CRPS":  crps_empirical(samples, y_all),
        "PICP":  picp_empirical(samples, y_all),
        "PINAW": pinaw_empirical(samples, y_all),
    }
    for h in range(T_out):
        s_h  = samples[:, :, :, h, :]
        y_h  = y_all[:, :, h, :]
        mu_h = s_h.mean(axis=0)
        metrics[f"MAE_h{h+1}"]  = mae(mu_h, y_h)
        metrics[f"RMSE_h{h+1}"] = rmse(mu_h, y_h)
        metrics[f"CRPS_h{h+1}"] = crps_empirical(s_h, y_h)
    return metrics


# ---------------------------------------------------------------------------
# 反归一化辅助（不变）
# ---------------------------------------------------------------------------

def _inverse_samples(samples: np.ndarray, scaler) -> np.ndarray:
    shape = samples.shape
    return scaler.inverse_transform(samples.reshape(-1)).reshape(shape)


def _inverse_y(y: np.ndarray, scaler) -> np.ndarray:
    return scaler.inverse_transform(y.reshape(-1)).reshape(y.shape)


# ---------------------------------------------------------------------------
# Temperature Calibration（不变）
# ---------------------------------------------------------------------------

def calibrate_temperature(samples: np.ndarray, y_all: np.ndarray,
                           target_coverage: float = 0.95,
                           grid: np.ndarray = None) -> float:
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
    model:            GridCFN,
    loader:           DataLoader,
    optimizer:        torch.optim.Optimizer,
    club_optimizer:   torch.optim.Optimizer,
    adj_norm:         torch.Tensor,
    edge_index:       torch.Tensor,
    device:           torch.device,
    epoch:            int,
    main_params,
    grad_clip:        float = 1.0,
    warmup_epochs:    int = 5,
    cfm_n_t_samples:  int = 4,
    sigma_min:        float = 0.01,
    club_inner_steps: int = 3,
) -> Dict[str, float]:
    model.train()
    total_loss = total_cfm = total_mi = total_rank = total_var = total_gnorm = 0.0
    n_batches  = 0

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    train_club = (epoch > warmup_epochs)

    # CLUB 激活后前 3 个 epoch 线性升温，避免突然引入大正则导致不稳定
    if train_club:
        ramp_epochs  = 3
        epochs_since = epoch - warmup_epochs
        ramp_factor  = min(1.0, epochs_since / ramp_epochs)
    else:
        ramp_factor  = 0.0

    for x, y in loader:
        x = x.to(device)   # [B, T_in, N, F]
        y = y.to(device)   # [B, T_out, N, F]

        # ── Forward ──────────────────────────────────────────────────────
        He_prime, Hs_prime, He, Hs, X_low_pooled, X_high_pooled = \
            model(x, adj_norm_, edge_idx_)

        # ── Step 1：更新 CLUB 变分网络（内循环，降低 MI 估计方差）────────
        # 新目标：club_e(He, X_low_pooled) + club_s(Hs, X_high_pooled)
        if train_club:
            var_loss_val = 0.0
            for _ in range(club_inner_steps):
                var_e = model.club_e.variational_loss(
                    He.detach(), X_low_pooled.detach()
                )
                var_s = model.club_s.variational_loss(
                    Hs.detach(), X_high_pooled.detach()
                )
                var_loss = var_e + var_s
                club_optimizer.zero_grad()
                var_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.club_e.parameters()) + list(model.club_s.parameters()),
                    max_norm=1.0,
                )
                club_optimizer.step()
                var_loss_val += var_loss.item()
            var_loss_val /= club_inner_steps   # 记录内循环平均值，而非最后一步
        else:
            var_loss_val = 0.0

        # ── Step 2：更新主网络 ─────────────────────────────────────────
        B, T_out, N, Fy = y.shape
        y_target = y.permute(0, 2, 1, 3).reshape(B, N, T_out * Fy)

        cfm_l = model.cfm_loss(He_prime, Hs_prime, y_target,
                               n_t_samples=cfm_n_t_samples,
                               sigma_min=sigma_min)

        # L_rank：从 warmup 第一个 epoch 起就加入（不依赖 CLUB）
        rank_l     = model.rank_loss()
        rank_val   = rank_l.item()

        # 默认 loss（train_club=True 时在 try 块内覆盖；异常时退回纯 CFM+rank 损失）
        loss   = cfm_l + model.lambda_rank * rank_l
        mi_val = 0.0

        if train_club:
            _set_requires_grad(model.club_e, False)
            _set_requires_grad(model.club_s, False)
            try:
                mi_e    = model.club_e(He, X_low_pooled)
                mi_s    = model.club_s(Hs, X_high_pooled)
                mi_loss = mi_e + mi_s
                mi_val  = mi_loss.item()
                # 只惩罚正的 MI 估计
                mi_penalty = mi_loss.clamp(min=0.0)
                loss = (cfm_l
                        + model.lambda_rank * rank_l
                        + model.lambda_club * ramp_factor * mi_penalty)
            finally:
                # 无论是否异常，都恢复 CLUB 参数梯度
                _set_requires_grad(model.club_e, True)
                _set_requires_grad(model.club_s, True)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(main_params, max_norm=grad_clip)
        optimizer.step()

        total_loss  += loss.item()
        total_cfm   += cfm_l.item()
        total_mi    += mi_val
        total_rank  += rank_val
        total_var   += var_loss_val
        total_gnorm += grad_norm.item()
        n_batches   += 1

    return {
        "loss":      total_loss  / n_batches,
        "cfm":       total_cfm   / n_batches,
        "mi":        total_mi    / n_batches,
        "rank":      total_rank  / n_batches,   # 原 mi_prime 列，现在记录 L_rank
        "var_loss":  total_var   / n_batches,
        "grad_norm": total_gnorm / n_batches,
    }


# ---------------------------------------------------------------------------
# 评估（多步版，不变）
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
    model.eval()
    samples_list, y_list = [], []

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    for x, y in loader:
        x = x.to(device)
        # forward 现在返回 6 个值，只取前两个供采样使用
        He_prime, Hs_prime, *_ = model(x, adj_norm_, edge_idx_)

        raw_samples = model.sample(
            He_prime, Hs_prime,
            n_samples=n_samples,
            n_steps=n_steps,
            sigma_min=sigma_min,
            x0_scale=x0_scale,
        ).cpu().numpy()

        samples_list.append(raw_samples)
        # y 不参与 GPU 计算，直接在 CPU 上 permute
        y_np = y.permute(0, 2, 1, 3).numpy()
        y_list.append(y_np)

    samples_all = np.concatenate(samples_list, axis=1)
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

    # CLUB 参数：同时覆盖 club_e 和 club_s
    club_param_ids = (
        {id(p) for p in model.club_e.parameters()} |
        {id(p) for p in model.club_s.parameters()}
    )
    main_params = [p for p in model.parameters() if id(p) not in club_param_ids]

    optimizer = torch.optim.Adam(
        main_params,
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
    )
    club_optimizer = torch.optim.Adam(
        list(model.club_e.parameters()) + list(model.club_s.parameters()),
        lr=cfg_train.lr * 0.5,
        weight_decay=cfg_train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg_train.lr_decay_factor,
        patience=cfg_train.lr_decay_patience,
    )

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
        "train_loss": [], "train_cfm": [], "train_mi": [],
        "train_rank": [], "train_var_loss": [],
        "val_crps": [], "val_mae": [], "val_rmse": [],
    }

    logger.info("Val 指标在归一化域（用于早停），Test 指标在反归一化域（实际量纲）")
    logger.info(f"多步预测 T_out={model.T_out}")
    logger.info(f"DMSD CLUB warmup: {warmup_epochs} epochs，目标: MI(He,X_low)+MI(Hs,X_high)")
    logger.info(f"lambda_rank={model.lambda_rank}，lambda_club={model.lambda_club}")
    # 日志列：MI 监控新 CLUB 目标；Rank 监控 L_rank
    header = (f"{'Epoch':>6} | {'Loss':>8} | {'CFM':>8} | {'MI':>8} | {'Rank':>8} | "
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
        val_crps_avg = val_m["CRPS"]
        scheduler.step(val_crps_avg)

        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["train_cfm"].append(train_m["cfm"])
        history["train_mi"].append(train_m["mi"])
        history["train_rank"].append(train_m["rank"])
        history["train_var_loss"].append(train_m["var_loss"])
        history["val_crps"].append(val_crps_avg)
        history["val_mae"].append(val_m["MAE"])
        history["val_rmse"].append(val_m["RMSE"])

        logger.info(
            f"{epoch:>6} | {train_m['loss']:>8.4f} | {train_m['cfm']:>8.4f} | "
            f"{train_m['mi']:>8.4f} | {train_m['rank']:>8.4f} | "
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

    # 测试集评估
    _, samples_test, y_test = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler, return_preds=True,
        n_samples=n_samples_test, n_steps=n_steps,
        temperature=1.0, inverse_transform=True,
        sigma_min=sigma_min, x0_scale=x0_scale,
    )
    _, samples_test_norm, y_test_norm = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler, return_preds=True,
        n_samples=n_samples_test, n_steps=n_steps,
        temperature=1.0, inverse_transform=False,
        sigma_min=sigma_min, x0_scale=x0_scale,
    )

    test_m_raw  = evaluate_all(samples_test, y_test)
    mu_test     = samples_test.mean(axis=0, keepdims=True)
    test_m      = evaluate_all(mu_test + best_T * (samples_test - mu_test), y_test)
    mu_norm     = samples_test_norm.mean(axis=0, keepdims=True)
    test_m_norm = evaluate_all(mu_norm + best_T * (samples_test_norm - mu_norm), y_test_norm)
    test_m_raw_norm = evaluate_all(samples_test_norm, y_test_norm)

    sep      = "=" * 60
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
    history["test_metrics_raw"]      = test_m_raw
    history["test_metrics_raw_norm"] = test_m_raw_norm
    history["best_temperature"]      = best_T
    history["test_shape"]            = list(samples_test.shape)
    return history