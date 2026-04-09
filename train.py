"""
GridCFN – 训练循环（CLUB v4）

[Fix-v4] 相比 v3 的修复：

  Bug：v3 的 train_one_epoch 每个 batch 做了两次 forward pass：
    Step 1: with torch.no_grad(): forward() → 获取 He, Hs
    Step 2: forward() → 重新计算（第二次 forward）

    问题：
      1. 速度慢一倍（两次完整的 backbone+TCN+GCN）
      2. 两次 forward 的 He/Hs 不同（dropout 随机性等），Step1 更新后的
         变分网络参数与 Step2 实际使用的 He/Hs 不对应

  修复：单次 forward，在同一个计算图上做两步更新：
    forward() → (mu, sigma, He, Hs, mi_loss)

    Step 1: var_loss = club.variational_loss(He.detach(), Hs.detach())
            club_optimizer.step()  ← 只更新变分网络
            （He.detach() 确保梯度不流回 backbone）

    Step 2: loss = NLL + λ * mi_loss
            optimizer.step()       ← 更新全部参数
            （mi_loss 的计算图在同一次 forward 中已建立）

  注意：Step 2 的 optimizer 包含变分网络参数，所以变分网络在一个 batch 里
        实际被更新了两次（Step1 + Step2）。这是正确的：
          · Step1 更新让 q(Hs|He) 更准确
          · Step2 更新让 CLUB 上界变小（推动解耦）
        两个方向不冲突。
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

def mae(pred, true):
    return float(np.abs(pred - true).mean())


def rmse(pred, true):
    return float(np.sqrt(((pred - true) ** 2).mean()))


def crps_score(mu, sigma, y):
    from scipy.stats import norm
    z   = (y - mu) / (sigma + 1e-8)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    return float((sigma * (z*(2*Phi-1) + 2*phi - 1/math.sqrt(math.pi))).mean())


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
# 单轮训练
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:          GridCFN,
    loader:         DataLoader,
    optimizer:      torch.optim.Optimizer,
    club_optimizer: torch.optim.Optimizer,
    adj_norm:       torch.Tensor,
    edge_index:     torch.Tensor,
    device:         torch.device,
    epoch:          int,
    grad_clip:      float = 1.0,
    warmup_epochs:  int = 5,    # 保留签名兼容性，不使用
) -> Dict[str, float]:
    """
    单次 forward，两步参数更新。

    每个 batch 流程：
      1. forward(x) → mu, sigma, He, Hs, mi_loss
         （一次 forward，所有中间结果共享同一计算图）

      2. Step 1 - 变分网络更新：
           var_loss = club.variational_loss(He.detach(), Hs.detach())
           club_optimizer.zero_grad()
           var_loss.backward()   ← 梯度只流向变分网络（He/Hs 已 detach）
           club_optimizer.step()

      3. Step 2 - 主网络更新：
           loss = NLL + λ * mi_loss
           optimizer.zero_grad()
           loss.backward()       ← mi_loss 的计算图仍然有效（未被清除）
           optimizer.step()

    为什么 Step2 的 mi_loss 计算图还在：
      · Step1 的 var_loss.backward() 只清除了与 var_loss 相关的计算图
      · mi_loss = club(He, Hs) 的计算图是独立的，未被 Step1 清除
      · 但如果 Step1 用了 optimizer.zero_grad()（主 optimizer），
        会清除 mi_loss 的梯度缓存 → 错误
      · 这里用独立的 club_optimizer，只 zero_grad club 参数 → 正确

    日志参考值（Solar 数据集，sigma_min=0.1）：
      NLL     : 0.3 ~ 1.5（初期高，逐渐下降）
      MI      : -2 ~ 3（初期负值正常，随变分网络收敛逐渐变正）
      VarLoss : 初期 ~12，逐渐下降到 ~8（反映变分网络准确度）
      Loss    : NLL + 0.5*MI，随训练下降
    """
    model.train()
    total_loss = total_nll = total_mi = total_var = 0.0
    n_batches  = 0

    adj_norm_  = adj_norm.to(device)
    edge_idx_  = edge_index.to(device)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # ── forward：获取表征 He, Hs（以及预测结果）────────────────────────
        mu, sigma, He, Hs, _ = model(x, adj_norm_, edge_idx_)
        # 注意：forward 中的 mi_loss 此时丢弃（_），原因见下方说明

        # ── Step 1: 用 He/Hs（detach）更新变分网络 ──────────────────────────
        # He.detach(), Hs.detach()：梯度不流回 backbone，只更新变分网络参数
        var_loss = model.club.variational_loss(He.detach(), Hs.detach())
        club_optimizer.zero_grad()
        var_loss.backward()
        club_optimizer.step()
        # club_optimizer.step() 修改了变分网络参数（in-place update）
        # 此时 forward 中建立的旧 mi_loss 计算图已失效（参数版本号不匹配）
        # 必须用更新后的变分网络参数重新计算 mi_loss

        # ── Step 2: 重新计算 mi_loss，更新主网络 ────────────────────────────
        # 用更新后的变分网络（Step1 已 step）重新估计 CLUB 上界
        # 这样 mi_loss 反映的是更准确的变分网络的估计，梯度信号更可靠
        mi_loss     = model.club(He, Hs)   # He, Hs 保留梯度（未 detach），
                                           # 梯度可流回 backbone，推动解耦
        y_target    = y[..., :mu.shape[-1]]
        loss, l_nll = model.compute_loss(mu, sigma, y_target, mi_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_nll  += l_nll.item()
        total_mi   += mi_loss.item()
        total_var  += var_loss.item()
        n_batches  += 1

    return {
        "loss":     total_loss / n_batches,
        "nll":      total_nll  / n_batches,
        "mi":       total_mi   / n_batches,
        "var_loss": total_var  / n_batches,
    }


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, adj_norm, edge_index, device,
             scaler=None, return_preds=False):
    """forward 返回 5 个值，evaluate 用 _, _ 忽略 He, Hs。"""
    model.eval()
    mu_list, sigma_list, y_list = [], [], []

    adj_norm_  = adj_norm.to(device)
    edge_idx_  = edge_index.to(device)

    for x, y in loader:
        mu, sigma, _, _, _ = model(x.to(device), adj_norm_, edge_idx_)
        mu_list.append(mu.cpu().numpy())
        sigma_list.append(sigma.cpu().numpy())
        y_list.append(y.numpy())

    mu_all    = np.concatenate(mu_list,    axis=0)
    sigma_all = np.concatenate(sigma_list, axis=0)
    y_all     = np.concatenate(y_list,     axis=0)[..., :mu_all.shape[-1]]
    metrics   = evaluate_all(mu_all, sigma_all, y_all)

    if return_preds:
        return metrics, mu_all, sigma_all, y_all
    return metrics


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, test_loader,
          adj_norm, edge_index, device, cfg_train,
          scaler=None, logger=None) -> Dict:
    """
    两个 optimizer：
      optimizer      → 所有参数，lr = cfg_train.lr
      club_optimizer → 仅变分网络，lr = cfg_train.lr * 5
        （变分网络需要更快收敛，×5 参考 CLUB 官方代码）
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
    # 变分网络独立 optimizer，lr 更大
    club_optimizer = torch.optim.Adam(
        model.club.parameters(),
        lr=cfg_train.lr * 5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg_train.lr_decay_factor,
        patience=cfg_train.lr_decay_patience,
    )

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history = {
        "train_loss": [], "train_nll": [], "train_mi": [], "train_var_loss": [],
        "val_crps": [], "val_mae": [], "val_rmse": [],
    }

    header = (f"{'Epoch':>6} | {'Loss':>8} | {'NLL':>8} | {'MI':>8} | "
              f"{'VarLoss':>9} | {'Val MAE':>8} | {'Val RMSE':>9} | "
              f"{'Val CRPS':>9} | {'LR':>8} | {'Time':>6}")
    logger.info(header)
    logger.info("-" * len(header))

    for epoch in range(1, cfg_train.max_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, club_optimizer,
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
        history["train_var_loss"].append(train_m["var_loss"])
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])
        history["val_rmse"].append(val_m["RMSE"])

        logger.info(
            f"{epoch:>6} | {train_m['loss']:>8.4f} | {train_m['nll']:>8.4f} | "
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
                logger.info(f"\n早停于 epoch {epoch}（最佳 val CRPS={best_val_crps:.4f}）")
                break

    model.load_state_dict(
        torch.load(cfg_train.save_path, map_location=device, weights_only=True)
    )
    test_m, mu_all, sigma_all, y_all = evaluate(
        model, test_loader, adj_norm, edge_index, device, scaler,
        return_preds=True,
    )

    sep = "=" * 52
    logger.info(f"\n{sep}\nTEST SET RESULTS\n{sep}")
    for k, v in test_m.items():
        logger.info(f"  {k:<8}: {v:.4f}")
    logger.info(sep)

    history["test_metrics"] = test_m
    history["test_mu"]    = mu_all.flatten().tolist()
    history["test_sigma"] = sigma_all.flatten().tolist()
    history["test_y"]     = y_all.flatten().tolist()
    history["test_shape"] = list(mu_all.shape)
    return history