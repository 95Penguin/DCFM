"""
GridCFN – 训练循环、评估与指标（CLUB 版本）

与原版（MINE）的主要变更：
  [CLUB-1] 删除 mine_optimizer 和两步 minimax 训练逻辑：
    原版 MINE 需要：
      Step1: mine_optimizer 对 MINE 参数做梯度上升（retain_graph=True）
      Step2: 主 optimizer 用 mi_loss.detach() 更新主网络
    原因：MINE 估计下界，最大化下界 vs 最小化总损失方向相反，必须隔离。

    CLUB 版本：
      变分网络参数与主网络参数更新方向一致（都在最小化 CLUB 上界），
      统一由单个 optimizer 更新。mi_loss 无需 detach，一次 backward 搞定。
    → train_one_epoch 从两步降为一步，代码大幅简化。

  [CLUB-2] 删除 warmup_epochs 的 MI 热身逻辑：
    原版热身是因为 MINE 网络在随机初始化时估计值不稳定，
    过早引入会给主网络错误信号。
    CLUB 变分网络在训练初期即能提供有方向性的梯度（推动解耦），
    无需热身阶段。warmup_epochs 参数保留在 config 中但不再使用。

  [Fix-I] 保留：接收预计算的 adj_norm 和 edge_index，不在内部重算。

  [Fix-CLUB-v2] 修复：
    · 删除错误的 clamp(mi_raw, max=0) 逻辑：
        原代码：mi_loss_clipped = torch.clamp(mi_raw, max=0.0)
        错误原因：
          1. clamp(max=0) 把正值截成 0，负值原样保留
          2. CLUB 返回值本来就是很大的负数（-10^6 量级）
          3. 负值进入 loss → loss = NLL + λ*(-10^6) = -874K → 梯度方向完全错误
        正确做法：
          CLUBEstimator.forward 内部已 clamp(min=0)，mi_loss 恒 ≥ 0
          train.py 直接传入 mi_raw（实际上是已截断的 ≥ 0 值），无需再处理
    · 日志新增 mi_loss 监控（应在 [0, ~5] 范围，过大说明解耦效果差）
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
    model:      GridCFN,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    adj_norm:   torch.Tensor,
    edge_index: torch.Tensor,
    device:     torch.device,
    epoch:      int,
    grad_clip:  float = 1.0,
    warmup_epochs: int = 5,      # 保留参数签名兼容性，CLUB 版本中不使用
) -> Dict[str, float]:
    """
    [CLUB-1] 单步训练（原版 MINE 为两步 minimax）：

      一次 forward → 得到 mu, sigma, mi_loss（CLUB 上界，已 clamp(min=0)）
      L_total = L_NLL + λ · mi_loss
      一次 backward → 更新所有参数（含 CLUB 变分网络）

    [Fix-CLUB-v2] 关键修复（删除错误的 clamp 逻辑）：
      原版错误：
        mi_loss_clipped = torch.clamp(mi_raw, max=0.0)
          → clamp(max=0) 让负值原样通过，正值截成 0
          → CLUB 返回 -10^6 → loss = NLL + 0.5*(-10^6) = -500K → 梯度崩溃

      正确做法：
        CLUBEstimator.forward() 内部已执行 clamp(min=0)：
          club_upper_bound = (pos_term - neg_term).clamp(min=0.0)
        因此 mi_loss 从 model.forward() 返回时已经 ≥ 0。
        train.py 直接使用，无需再做任何 clamp 处理。

      为何 CLUB 可以单步：
        · MINE 估计下界，主网络要最小化 MI，MINE 要最大化 MI 估计，方向相反，
          必须用两个 optimizer 隔离更新。
        · CLUB 估计上界（≥0），主网络最小化总损失（含 CLUB 项）= 让上界估计变小，
          同时也在让变分网络 q(Hs|He) 更准确（上界更紧），两者方向一致，
          单个 optimizer 统一更新即可。

    正常训练时日志参考值（Solar 数据集）：
      NLL  : 0.8 ~ 1.5（高斯负对数似然，与 sigma 量级相关）
      MI   : 0.0 ~ 5.0（CLUB 上界，0 = 完全解耦；初期可能较大）
      Loss : NLL + 0.5*MI ≈ 0.8 ~ 4.0（正数，随训练下降）
    """
    model.train()

    total_loss = total_nll = total_mi = 0.0
    n_batches  = 0

    for x, y in loader:
        x          = x.to(device)
        y          = y.to(device)
        adj_norm_  = adj_norm.to(device)
        edge_idx_  = edge_index.to(device)

        # ── 单步前向 + 反向 ─────────────────────────────────────────────────
        mu, sigma, mi_loss = model(x, adj_norm_, edge_idx_)

        # [Fix-CLUB-v2] mi_loss 从 CLUBEstimator.forward() 返回时已 clamp(min=0)
        # 直接使用，不需要任何额外处理
        # mi_loss ∈ [0, +∞)，语义：0 = 完全解耦，>0 = 存在互信息残留

        y_target = y[..., :mu.shape[-1]]

        loss, l_nll = model.compute_loss(
            mu, sigma, y_target,
            mi_loss=mi_loss,
            lambda_mi=model.lambda_mi,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_nll  += l_nll.item()
        total_mi   += mi_loss.item()
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
             scaler=None,
             return_preds: bool = False):
    """
    在归一化尺度下计算所有指标，与论文 Table II 的报告口径一致。
    scaler 参数保留供外部调用方使用，evaluate 内部不做反归一化。

    return_preds=True 时额外返回 (metrics, mu_all, sigma_all, y_all)，
    供画图使用（预测区间、校准图、误差分布）。
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

    metrics = evaluate_all(mu_all, sigma_all, y_all)

    if return_preds:
        return metrics, mu_all, sigma_all, y_all
    return metrics


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
    完整训练循环（CLUB 版本）。

    [CLUB-1] 删除 mine_optimizer：
      原版有两个 optimizer：
        optimizer      → 主网络参数（排除 MINE）
        mine_optimizer → MINE 参数（lr * 2，梯度上升）
      CLUB 版本只需一个 optimizer 覆盖全部参数。

    [CLUB-2] 删除 warmup_epochs 热身：
      原版热身期间 curr_lambda_mi=0，避免未收敛的 MINE 干扰主网络。
      CLUB 从第 1 个 epoch 即可提供稳定梯度，无需热身。

    [Fix-CLUB-v2] 修复说明：
      · train_one_epoch 不再执行任何 clamp(max=0) 操作
      · mi 日志列应显示 [0, ~5] 范围内的正数，而不是 -10^6 这样的负数
      · Loss 列应显示正数（NLL + λ*CLUB ≥ 0）
    """
    if logger is None:
        logger = logging.getLogger("gridcfn.train")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                             datefmt="%H:%M:%S"))
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)

    # [CLUB-1] 单个 optimizer 覆盖所有参数（含 CLUB 变分网络）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg_train.lr_decay_factor,
        patience=cfg_train.lr_decay_patience,
    )

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history           = {
        "train_loss": [], "train_nll": [], "train_mi": [],
        "val_crps": [], "val_mae": [], "val_rmse": [],
    }

    header = (f"{'Epoch':>6} | {'Loss':>8} | {'NLL':>8} | {'MI':>8} | "
              f"{'Val MAE':>8} | {'Val RMSE':>9} | {'Val CRPS':>9} | "
              f"{'LR':>8} | {'Time':>6}")
    logger.info(header)
    logger.info("-" * len(header))

    for epoch in range(1, cfg_train.max_epochs + 1):
        t0 = time.time()

        # [CLUB-1] train_one_epoch 不再需要 mine_optimizer
        train_m = train_one_epoch(
            model, train_loader, optimizer,
            adj_norm, edge_index, device,
            epoch, cfg_train.grad_clip,
            getattr(cfg_train, "warmup_epochs", 5),
        )
        val_m   = evaluate(model, val_loader, adj_norm, edge_index, device, scaler)
        scheduler.step(val_m["CRPS"])

        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["train_nll"].append(train_m["nll"])
        history["train_mi"].append(train_m["mi"])
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])
        history["val_rmse"].append(val_m["RMSE"])

        logger.info(
            f"{epoch:>6} | {train_m['loss']:>8.4f} | {train_m['nll']:>8.4f} | "
            f"{train_m['mi']:>8.4f} | {val_m['MAE']:>8.4f} | "
            f"{val_m['RMSE']:>9.4f} | {val_m['CRPS']:>9.4f} | "
            f"{cur_lr:>8.2e} | {elapsed:>5.1f}s"
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

    # 加载最优权重并在测试集上推理
    model.load_state_dict(
        torch.load(cfg_train.save_path, map_location=device, weights_only=True)
    )
    test_m, mu_all, sigma_all, y_all = evaluate(
        model, test_loader, adj_norm, edge_index, device, scaler,
        return_preds=True,
    )

    sep = "=" * 52
    logger.info(f"\n{sep}")
    logger.info("TEST SET RESULTS")
    logger.info(sep)
    for k, v in test_m.items():
        logger.info(f"  {k:<8}: {v:.4f}")
    logger.info(sep)

    history["test_metrics"] = test_m
    history["test_mu"]    = mu_all.flatten().tolist()
    history["test_sigma"] = sigma_all.flatten().tolist()
    history["test_y"]     = y_all.flatten().tolist()
    history["test_shape"] = list(mu_all.shape)
    return history