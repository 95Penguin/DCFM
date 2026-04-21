"""
GridCFN – 训练循环（CFM 版 v3，修正指标域）

[本版核心修改：指标报告域统一]

问题背景：
  Electricity 数据做了 log1p + Z-score 双重变换。
  原版 evaluate() 直接在 Z-score 后的 log 域算指标，导致：
    · MAE/RMSE 数值偏大（≈ 论文的 2x），失去原始物理意义
    · 与 Solar（只做 Z-score）的指标尺度不统一，无法直接对比

修复方案（方案 B：反 Z-score，在 log1p 域或原始归一化域报告）：
  对每个数据集，在 evaluate() 里对 mu/sigma/y 做 scaler.inverse_transform()：
    · Solar:       inverse_transform = 反 Z-score（无 log 变换）→ 原始量纲域
    · Electricity: inverse_transform = 反 Z-score（无 expm1）→ log1p 域
                   注意：Scaler.inverse_transform 在 log_transform=True 时会再做 expm1，
                   这里我们不想做 expm1（保留在 log 域），所以用 _inv_zscore() 直接反 Z-score。
    · Weather:     inverse_transform = 反 Z-score（无 log 变换）→ 原始量纲域

  这样所有数据集的指标都在"经过最终预处理后的域"里，彼此可比。

CRPS/PICP/PINAW 的注意事项：
  反归一化后 mu/sigma/y 的量纲一致，CRPS/PICP 计算仍然有效。
  但注意：sigma 是在归一化域学到的，inverse_transform 只能线性缩放 mu（加偏移），
  如果 Scaler 是 Z-score（线性变换），sigma 只需乘以 std 即可（不加 mean）。
  代码里用 _scale_sigma() 单独处理 sigma（只乘 std，不加 mean）。

Temperature Calibration：
  在反归一化后的域做校准，更有物理意义（校准后 PICP 对应真实覆盖率）。
"""

import logging
import math
import time
from typing import Dict, Optional, Tuple

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
    """
    只做反 Z-score（乘 std + 加 mean），不做 expm1。
    用于 mu 和 y 的反归一化。
    适用于所有数据集（Solar/Electricity/Weather）。
    """
    return arr * scaler.std + scaler.mean


def _inv_zscore_sigma(arr: np.ndarray, scaler) -> np.ndarray:
    """
    对 sigma 只乘 std（不加 mean）。
    因为 sigma 是标准差（尺度量），线性变换只改变尺度，不改变位置。
    """
    return arr * scaler.std


# ---------------------------------------------------------------------------
# Temperature Calibration
# ---------------------------------------------------------------------------

def calibrate_temperature(mu_all: np.ndarray, sigma_all: np.ndarray,
                           y_all: np.ndarray,
                           target_coverage: float = 0.95,
                           grid: Optional[np.ndarray] = None) -> float:
    """
    在验证集上 grid search 最优 temperature T*，使 PICP ≈ target_coverage。
    sigma_calibrated = sigma * T*
    """
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
# 单轮训练（训练在归一化域进行，不做反归一化）
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
    训练在归一化域进行（无需反归一化）。
    CFM loss = MSE(v_θ(x_t, t, c), u_t) 在归一化域是尺度无关的。
    """
    model.train()
    total_loss = total_cfm = total_mi = total_var = 0.0
    n_batches  = 0

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        context_feat, He, Hs, _ = model(x, adj_norm_, edge_idx_)

        # Step 1: 变分网络更新
        var_loss = model.club.variational_loss(He.detach(), Hs.detach())
        club_optimizer.zero_grad()
        var_loss.backward()
        club_optimizer.step()

        # Step 2: 主网络更新
        mi_loss  = model.club(He, Hs)
        y_target = y[..., :model.out_dim]
        cfm_l    = model.cfm_loss(context_feat, y_target, n_t_samples=cfm_n_t_samples)
        loss     = cfm_l + model.lambda_mi * mi_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_cfm  += cfm_l.item()
        total_mi   += mi_loss.item()
        total_var  += var_loss.item()
        n_batches  += 1

    return {
        "loss":     total_loss / n_batches,
        "cfm":      total_cfm  / n_batches,
        "mi":       total_mi   / n_batches,
        "var_loss": total_var  / n_batches,
    }


# ---------------------------------------------------------------------------
# 评估（修正：反 Z-score 后再计算指标）
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
    CFM 推断评估（v3+，修正指标域）。

    inverse_transform=True（默认，推荐）：
      对 mu、sigma、y 做反 Z-score，在各数据集的"原始预处理域"里报告指标：
        · Solar/Weather: 反 Z-score → 接近原始量纲（MW / 气象单位）
        · Electricity:   反 Z-score → log1p 域（不做 expm1，保持在 log 域）

      sigma 的反归一化：只乘 std（不加 mean），因为 sigma 是尺度量。

    inverse_transform=False：
      保持在 Z-score 归一化域，用于训练时的快速验证（不改变相对排名，
      可以用来做早停判断，速度更快因为不需要 numpy 转换）。

    注意：早停用的 val_m["CRPS"] 不管哪种模式都在同一域内单调对应，
    所以用 inverse_transform=False 做早停是安全的（只是绝对值不同）。
    最终测试结果用 inverse_transform=True 报告。
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

    # ── 反归一化（只反 Z-score，不做 expm1）──────────────────────────────────
    if inverse_transform and scaler is not None:
        mu_all    = _inv_zscore_mean(mu_all,    scaler)
        sigma_all = _inv_zscore_sigma(sigma_all, scaler)
        y_all     = _inv_zscore_mean(y_all,     scaler)

    # ── Temperature 校准 ────────────────────────────────────────────────────
    sigma_cal = np.maximum(sigma_all * temperature, 1e-6)

    metrics = evaluate_all(mu_all, sigma_cal, y_all)

    if return_preds:
        # 返回反归一化后的原始 sigma（未乘 temperature），供校准函数使用
        return metrics, mu_all, sigma_all, y_all
    return metrics


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(model: GridCFN, train_loader, val_loader, test_loader,
          adj_norm, edge_index, device, cfg_train,
          scaler=None, logger=None) -> Dict:
    """
    训练流程：
      · 训练 epoch 的验证：inverse_transform=False（快速，只用于早停判断）
        注意：此时 val CRPS 是在归一化域的，绝对值比最终测试结果小，
        但用于早停的相对大小判断是正确的。
      · 训练结束后的 temperature 校准和测试：inverse_transform=True（最终报告域）

    这样设计的原因：
      训练中频繁调用 evaluate（每 epoch 一次），inverse_transform 额外增加
      numpy 运算开销（对大数据集约 +10%），意义不大因为只用于早停。
      最终报告才需要准确的量纲。
    """
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
    club_optimizer = torch.optim.Adam(
        model.club.parameters(),
        lr=cfg_train.lr * 5,
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

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history = {
        "train_loss": [], "train_cfm": [], "train_mi": [], "train_var_loss": [],
        "val_crps": [], "val_mae": [], "val_rmse": [],
    }

    # 说明：训练时 Val CRPS 是在归一化域的（inverse_transform=False）
    logger.info("注意：训练中 Val 指标在归一化域（用于早停判断），"
                "最终 Test 指标在反 Z-score 域（实际量纲）。")
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
            getattr(cfg_train, "warmup_epochs", 5),
            cfm_n_t_samples=cfm_n_t_samples,
        )
        # 训练时不做反归一化（快速，用于早停）
        val_m = evaluate(
            model, val_loader, adj_norm, edge_index, device,
            scaler=scaler,
            n_samples=n_samples_val, n_steps=n_steps,
            temperature=1.0,
            inverse_transform=False,   # 归一化域，快速验证
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

    # ── Temperature Calibration（在反归一化后的验证集上校准） ────────────────
    logger.info("\n正在验证集上做 Temperature Calibration（反 Z-score 域）...")
    _, mu_val, sigma_val, y_val = evaluate(
        model, val_loader, adj_norm, edge_index, device,
        scaler=scaler,
        return_preds=True,
        n_samples=n_samples_test,
        n_steps=n_steps,
        temperature=1.0,
        inverse_transform=True,   # 反归一化后校准，有物理意义
    )
    best_T = calibrate_temperature(mu_val, sigma_val, y_val, target_coverage=0.95)
    logger.info(
        f"最优 Temperature: {best_T:.3f}  "
        f"（验证集 PICP@T=1.0: {picp(mu_val, sigma_val, y_val):.4f} → "
        f"PICP@T={best_T:.2f}: {picp(mu_val, sigma_val*best_T, y_val):.4f}）"
    )

    # ── 测试集最终评估（反归一化 + temperature 校准） ────────────────────────
    test_m, mu_all, sigma_all, y_all = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler,
        return_preds=True,
        n_samples=n_samples_test,
        n_steps=n_steps,
        temperature=best_T,
        inverse_transform=True,   # 最终结果在实际量纲域
    )

    # 同时报告归一化域的指标（便于与旧版对比）
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