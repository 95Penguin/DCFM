"""
GridCFN – 训练循环、评估与指标

修复说明：
  [Fix-A/B] MINE minimax 训练重写：
    删除无效的 _forward_for_mine；改为单次 forward 复用计算图。
    Step1：mine_optimizer 对 MINE 参数梯度上升（maximize MI），
           用 retain_graph=True 保留计算图供 Step2 使用。
    Step2：mi_loss.detach() 后传入 compute_loss，使梯度只更新主网络，
           不回传到 MINE 参数，两步优化彻底隔离。

  [Fix-I] forward() 签名变更：
    接收预计算的 adj_norm 和 edge_index，不再在 forward 内部重算。
    train.py 统一传入这两个预计算张量。
"""

import logging
import math
import time
from typing import Dict, Optional

import numpy as np
import torch
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
    """高斯 CRPS 闭合解，越小越好。"""
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
    adj_norm:       torch.Tensor,
    edge_index:     torch.Tensor,
    device:         torch.device,
    epoch:          int,
    grad_clip:      float = 1.0,
    warmup_epochs:  int   = 5,
) -> Dict[str, float]:
    """
    [Fix-A/B] 修复后的 MINE minimax 训练：

    Step1 — MINE 梯度上升（maximize MI 估计）：
      · 执行一次完整 forward，获得 mi_loss（计算图连接 MINE 参数）
      · retain_graph=True 保留计算图，供 Step2 继续使用
      · 只 step mine_optimizer，主网络参数不动

    Step2 — 主网络梯度下降（minimize NLL + λ·MI）：
      · 用 mi_loss.detach() 切断与 MINE 参数的梯度路径
      · loss.backward() 梯度只流向主网络参数
      · 只 step optimizer，MINE 参数不动

    [Fix-I] 接收预计算的 adj_norm 和 edge_index，不在内部重算。

    热身逻辑（warmup_epochs）：
      · 前 warmup_epochs 个 epoch，curr_lambda_mi=0，主网络不受 MI 正则影响
      · 确保 MINE 有足够时间收敛到合理估计，再让主网络依赖它优化
    """
    model.train()

    curr_lambda_mi = 0.0 if epoch <= warmup_epochs else model.lambda_mi

    total_loss = total_nll = total_mi = 0.0
    n_batches  = 0

    for x, y in loader:
        x          = x.to(device)
        y          = y.to(device)
        adj_norm_  = adj_norm.to(device)
        edge_idx_  = edge_index.to(device)

        # ── Step1：MINE 梯度上升 ────────────────────────────────────────────
        # 单次 forward，计算图同时连接主网络参数和 MINE 参数
        mu, sigma, mi_loss = model(x, adj_norm_, edge_idx_)

        mine_optimizer.zero_grad()
        # 最大化 MI 估计 = 最小化负 MI；retain_graph 保留计算图供 Step2 用
        (-mi_loss).backward(retain_graph=True)
        mine_optimizer.step()

        # ── Step2：主网络梯度下降 ───────────────────────────────────────────
        # mi_loss.detach()：切断梯度路径，Step2 的 backward 不会更新 MINE 参数
        y_target = y[..., :mu.shape[-1]]
        loss, l_nll, l_mi = model.compute_loss(
            mu, sigma, y_target,
            mi_loss=mi_loss.detach(),   # [Fix-A/B] 关键：detach 隔离 MINE 梯度
            lambda_mi=curr_lambda_mi,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.main_parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_nll  += l_nll.item()
        total_mi   += mi_loss.item()   # 记录原始（未 detach 的）MI 估计值用于日志
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
def evaluate(model: GridCFN,
             loader: DataLoader,
             adj_norm:   torch.Tensor,
             edge_index: torch.Tensor,
             device:     torch.device,
             scaler=None) -> Dict[str, float]:
    """
    在归一化尺度下计算所有指标，与论文 Table II 的报告口径一致。
    scaler 参数保留供外部调用方使用，evaluate 内部不做反归一化。
    """
    model.eval()
    mu_list, sigma_list, y_list = [], [], []

    adj_norm_  = adj_norm.to(device)
    edge_idx_  = edge_index.to(device)

    for x, y in loader:
        mu, sigma, _ = model(x.to(device), adj_norm_, edge_idx_)
        mu_list.append(mu.cpu().numpy())
        sigma_list.append(sigma.cpu().numpy())
        y_list.append(y.numpy())

    mu_all    = np.concatenate(mu_list,    axis=0)
    sigma_all = np.concatenate(sigma_list, axis=0)
    y_all     = np.concatenate(y_list,     axis=0)[..., :mu_all.shape[-1]]

    # 在归一化尺度下计算，与论文 Table II 报告口径一致
    return evaluate_all(mu_all, sigma_all, y_all)


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(model:        GridCFN,
          train_loader: DataLoader,
          val_loader:   DataLoader,
          test_loader:  DataLoader,
          adj_norm:     torch.Tensor,
          edge_index:   torch.Tensor,
          device:       torch.device,
          cfg_train,
          scaler=None,
          logger: Optional[logging.Logger] = None) -> Dict:
    """
    完整训练循环。

    参数变更（对比旧版）：
      · adj 拆分为 adj_norm + edge_index（由 main.py 预计算后传入）
      · 移除 _forward_for_mine 相关逻辑
    """
    if logger is None:
        logger = logging.getLogger("gridcfn.train")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                             datefmt="%H:%M:%S"))
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)

    # 主 optimizer：只更新非 MINE 参数
    optimizer = torch.optim.Adam(
        model.main_parameters(),
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
    )
    # MINE optimizer：稍大的 lr 帮助 MINE 快速收敛
    mine_optimizer = torch.optim.Adam(
        model.mine_parameters(),
        lr=cfg_train.lr * 2,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg_train.lr_decay_factor,
        patience=cfg_train.lr_decay_patience,
    )

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history           = {"train_loss": [], "val_crps": [], "val_mae": []}
    warmup_epochs     = getattr(cfg_train, "warmup_epochs", 5)

    header = (f"{'Epoch':>6} | {'Loss':>8} | {'NLL':>8} | {'MI':>7} | "
              f"{'Val MAE':>8} | {'Val RMSE':>9} | {'Val CRPS':>9} | "
              f"{'LR':>8} | {'Time':>6}")
    logger.info(header)
    logger.info("-" * len(header))

    for epoch in range(1, cfg_train.max_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, mine_optimizer,
            adj_norm, edge_index, device,
            epoch, cfg_train.grad_clip, warmup_epochs,
        )
        val_m   = evaluate(model, val_loader, adj_norm, edge_index, device, scaler)
        scheduler.step(val_m["CRPS"])

        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])

        warmup_tag = " [warmup]" if epoch <= warmup_epochs else ""
        logger.info(
            f"{epoch:>6} | {train_m['loss']:>8.4f} | {train_m['nll']:>8.4f} | "
            f"{train_m['mi']:>7.4f} | {val_m['MAE']:>8.4f} | "
            f"{val_m['RMSE']:>9.4f} | {val_m['CRPS']:>9.4f} | "
            f"{cur_lr:>8.2e} | {elapsed:>5.1f}s{warmup_tag}"
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

    # 加载最优权重并测试
    model.load_state_dict(
        torch.load(cfg_train.save_path, map_location=device, weights_only=True)
    )
    test_m = evaluate(model, test_loader, adj_norm, edge_index, device, scaler)

    sep = "=" * 52
    logger.info(f"\n{sep}")
    logger.info("TEST SET RESULTS")
    logger.info(sep)
    for k, v in test_m.items():
        logger.info(f"  {k:<8}: {v:.4f}")
    logger.info(sep)

    history["test_metrics"] = test_m
    return history