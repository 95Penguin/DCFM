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


def crps_empirical(samples: np.ndarray, y: np.ndarray) -> float:
    """
    [Fix-CRPS] 经验 CRPS，直接用 CFM 采样粒子计算，不依赖高斯假设。

    经验 CRPS 公式（Gneiting & Raftery 2007）：
      CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
      其中 X, X' iid ~ F（用 samples 近似），且 X ≠ X'（不放回）

    [Fix-Spread] 原版用有放回随机抽对，idx1==idx2 时 |X-X'|=0，
    人为压低 spread，导致 CRPS 虚高。
    修复：用不放回随机排列保证每对 i≠j，消除自配对偏差。

    参数：
      samples : [S, ...] S 个粒子
      y       : [...] 真实值，与 samples[i] shape 相同
    """
    S        = samples.shape[0]
    mae_term = np.abs(samples - y[None]).mean(axis=0)   # E|X - y|

    # [Fix-Spread] 不放回配对：对 samples 做随机置换，与原始对齐后取差
    # 每次 shuffle 得到一组 (samples[i], samples[perm[i]]) 且 i≠perm[i]（大概率）
    # 重复 n_pairs_per_S 次取均值，近似 E|X - X'|
    n_rep   = min(10, S - 1)   # 重复次数，S 小时少重复
    spreads = []
    rng     = np.random.default_rng(seed=0)   # 固定 seed 保证复现
    for _ in range(n_rep):
        perm = rng.permutation(S)
        # 保证 perm[i] != i（derangement 近似：若有碰撞，循环移一位）
        clash = np.where(perm == np.arange(S))[0]
        for idx in clash:
            swap = (idx + 1) % S
            perm[idx], perm[swap] = perm[swap], perm[idx]
        spreads.append(np.abs(samples - samples[perm]).mean(axis=0))

    spread        = np.mean(spreads, axis=0)   # E|X - X'|（无放回近似）
    crps_per_node = mae_term - 0.5 * spread
    return float(crps_per_node.mean())


def picp(mu, sigma, y, confidence=0.95):
    from scipy.stats import norm
    z       = norm.ppf((1 + confidence) / 2)
    covered = ((y >= mu - z*sigma) & (y <= mu + z*sigma)).astype(float)
    return float(covered.mean())


def picp_empirical(samples: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> float:
    """
    [Fix-PICP] 基于经验分位数的区间覆盖率，不依赖高斯假设。

    lower = quantile(samples, (1-confidence)/2)
    upper = quantile(samples, (1+confidence)/2)
    PICP  = mean(lower <= y <= upper)
    """
    alpha = (1.0 - confidence) / 2.0
    lower = np.quantile(samples, alpha,     axis=0)   # [B, N, D]
    upper = np.quantile(samples, 1 - alpha, axis=0)
    covered = ((y >= lower) & (y <= upper)).astype(float)
    return float(covered.mean())


def pinaw_empirical(samples: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> float:
    """
    [Fix-PINAW] 基于经验分位数的归一化区间宽度。
    """
    alpha = (1.0 - confidence) / 2.0
    lower = np.quantile(samples, alpha,     axis=0)
    upper = np.quantile(samples, 1 - alpha, axis=0)
    width   = upper - lower
    y_range = y.max() - y.min() + 1e-8
    return float((width / y_range).mean())


def evaluate_all(samples: np.ndarray, y_all: np.ndarray) -> dict:
    """
    [Fix-EvalAll] 统一入口，接收原始 samples，所有指标走经验计算路径。

    samples : [S, total, N, out_dim]（concatenate 后的完整测试集采样）
    y_all   : [total, N, out_dim]

    内部计算 mu/sigma 用于记录，但 CRPS/PICP/PINAW 全用经验版。
    """
    mu_all    = samples.mean(axis=0)                             # [total, N, D]
    sigma_all = samples.std(axis=0, ddof=1)                     # Bessel 校正
    return {
        "MAE":   mae(mu_all, y_all),
        "RMSE":  rmse(mu_all, y_all),
        "CRPS":  crps_empirical(samples, y_all),                 # [Fix-CRPS]
        "PICP":  picp_empirical(samples, y_all),                 # [Fix-PICP]
        "PINAW": pinaw_empirical(samples, y_all),                # [Fix-PINAW]
    }


# ---------------------------------------------------------------------------
# 反归一化辅助函数
# ---------------------------------------------------------------------------

def _inverse_samples(samples: np.ndarray, scaler) -> np.ndarray:
    """
    [Fix-InvTransform] 对 [S, ...] 格式的采样粒子做批量反归一化。

    修复：
      原版逐样本 for 循环调用 scaler.inverse_transform，性能差。
      scaler 的反变换是逐元素线性操作（乘 std + 加 mean，可选 expm1），
      对任意 shape 的 ndarray 均可广播，直接对整个 [S*rest] 做一次调用即可。
    """
    shape    = samples.shape           # [S, total, N, D]
    flat     = samples.reshape(-1)     # [S*total*N*D]
    inv_flat = scaler.inverse_transform(flat)
    return inv_flat.reshape(shape)


def _inverse_y(y: np.ndarray, scaler) -> np.ndarray:
    """对真实值 y [total, N, D] 做反归一化，走 scaler.inverse_transform()。"""
    shape = y.shape
    return scaler.inverse_transform(y.reshape(-1)).reshape(shape)


# ---------------------------------------------------------------------------
# Temperature Calibration
# ---------------------------------------------------------------------------

def calibrate_temperature(samples: np.ndarray, y_all: np.ndarray,
                           target_coverage: float = 0.95,
                           grid: np.ndarray = None) -> float:
    """
    [Fix-CalibTemp] Temperature Calibration 改为走经验 PICP，不再用高斯公式。

    原版问题：
      calibrate_temperature(mu, sigma, y) 用高斯区间 [mu±z*sigma*T]，
      当真实分布非高斯时（Electricity 重尾），T 的最优值偏大（如 3.0），
      但即使 T=3.0 PICP 也只有 83%，无法达到 95%。

    修复：
      直接对 samples 做分位数缩放：scale t → samples_scaled = mu + t*(samples-mu)，
      等效于以 mu 为中心放缩粒子，再用经验分位数计算 PICP。
      这样 T 的搜索空间与真实分布形状匹配，能找到更准确的校准值。
    """
    if grid is None:
        grid = np.linspace(0.5, 5.0, 91)           # 搜索范围扩大到 5.0

    mu = samples.mean(axis=0)                       # [B, N, D]
    best_T   = 1.0
    best_gap = float("inf")

    for T in grid:
        samples_scaled = mu[None] + T * (samples - mu[None])   # [S, B, N, D]
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

        # [v6-DualStream] forward 返回 (He_prime, Hs_prime, He, Hs, mi_loss_raw)
        He_prime, Hs_prime, He, Hs, _ = model(x, adj_norm_, edge_idx_)

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
        # [v6-DualStream] cfm_loss 接收分开的 He_prime 和 Hs_prime
        cfm_l    = model.cfm_loss(He_prime, Hs_prime, y_target,
                                   n_t_samples=cfm_n_t_samples,
                                   sigma_min=sigma_min)

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
             inverse_transform: bool = True,
             sigma_min: float = 0.01,
             x0_scale: float = 1.0):
    """
    CFM 推断评估（v6：经验指标 + 统一反归一化）。

    [Fix-InvTransform] 反归一化统一走 scaler.inverse_transform()，
      支持纯 Z-score 和 log+Z-score 两种模式，不再手动乘 std + mean。

    [Fix-CRPS] evaluate_all 内部走经验 CRPS/PICP/PINAW，不依赖高斯假设。

    [Fix-Temperature] temperature 作用于粒子（以 mu 为中心放缩），
      而非乘在 sigma 上，与 calibrate_temperature 的定义对齐。

    返回值：
      metrics                  : dict（MAE/RMSE/CRPS/PICP/PINAW）
      (可选) samples_inv, y_inv: 反归一化后的粒子和真实值，供 calibrate_temperature 用
    """
    model.eval()
    samples_list, y_list = [], []

    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    for x, y in loader:
        x = x.to(device)
        # [v6-DualStream] forward 返回 (He_prime, Hs_prime, He, Hs, mi_loss_raw)
        He_prime, Hs_prime, _, _, _ = model(x, adj_norm_, edge_idx_)

        # [S, B, N, out_dim]
        raw_samples = model.sample(
            He_prime, Hs_prime,
            n_samples=n_samples,
            n_steps=n_steps,
            sigma_min=sigma_min,
            x0_scale=x0_scale,
        ).cpu().numpy()

        samples_list.append(raw_samples)
        y_list.append(y.numpy())

    # 沿 batch 维拼接：[S, total, N, D]
    samples_all = np.concatenate(samples_list, axis=1)
    y_all       = np.concatenate(y_list,       axis=0)[..., :model.out_dim]  # [total, N, D]

    # ── 反归一化 ───────────────────────────────────────────────────────────
    if inverse_transform and scaler is not None:
        # [Fix-InvTransform] 走 scaler，支持 log+Z-score
        samples_all = _inverse_samples(samples_all, scaler)
        y_all       = _inverse_y(y_all, scaler)

    # ── Temperature 校准：以 mu 为中心放缩粒子 ────────────────────────────
    if temperature != 1.0:
        mu = samples_all.mean(axis=0, keepdims=True)            # [1, total, N, D]
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
    sigma_min       = getattr(cfg_train, "cfm_sigma_min",      0.01)
    x0_scale        = getattr(cfg_train, "cfm_x0_scale",       1.0)

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
            sigma_min=sigma_min,
        )
        val_m = evaluate(
            model, val_loader, adj_norm, edge_index, device,
            scaler=scaler,
            n_samples=n_samples_val, n_steps=n_steps,
            temperature=1.0,
            inverse_transform=False,   # 归一化域，快速验证，用于早停
            sigma_min=sigma_min,
            x0_scale=x0_scale,
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
    _, samples_val, y_val = evaluate(
        model, val_loader, adj_norm, edge_index, device,
        scaler=scaler,
        return_preds=True,
        n_samples=n_samples_test,
        n_steps=n_steps,
        temperature=1.0,
        inverse_transform=True,
        sigma_min=sigma_min,
        x0_scale=x0_scale,
    )
    best_T = calibrate_temperature(samples_val, y_val, target_coverage=0.95)
    picp_before = picp_empirical(samples_val, y_val)
    mu_val      = samples_val.mean(axis=0)
    scaled_val  = mu_val[None] + best_T * (samples_val - mu_val[None])
    picp_after  = picp_empirical(scaled_val, y_val)
    logger.info(
        f"最优 Temperature: {best_T:.3f}  "
        f"（验证集 PICP@T=1.0: {picp_before:.4f} → "
        f"PICP@T={best_T:.2f}: {picp_after:.4f}）"
    )

    # ── 测试集最终评估 ──────────────────────────────────────────────────────
    test_m, samples_test, y_test = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler,
        return_preds=True,
        n_samples=n_samples_test,
        n_steps=n_steps,
        temperature=best_T,
        inverse_transform=True,
        sigma_min=sigma_min,
        x0_scale=x0_scale,
    )

    test_m_norm, _, _ = evaluate(
        model, test_loader, adj_norm, edge_index, device,
        scaler=scaler,
        return_preds=True,
        n_samples=n_samples_test,
        n_steps=n_steps,
        temperature=best_T,
        inverse_transform=False,
        sigma_min=sigma_min,
        x0_scale=x0_scale,
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
    history["test_shape"]            = list(samples_test.shape)
    return history