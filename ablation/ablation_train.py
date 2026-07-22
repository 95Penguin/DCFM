# ablation_train.py
"""
消融实验训练循环
────────────────────────────────────────────────────────────────────────
与 train.py::train_one_epoch / train.py::train 逻辑基本一致，区别仅在于：
  1. 当 ablation.use_club=False 时，完全跳过 CLUB 相关的优化器、变分内层
     更新和互信息惩罚项，loss 中也不包含 lambda_club * mi_penalty；
  2. 当 ablation.use_rank_loss=False 时，loss 中不包含 lambda_rank * rank_loss
     （AblationDCFM.rank_loss() 在该情况下本身就返回 0，这里加一层显式判断
     只是为了让日志更清楚，去掉对推门那一项的依赖）；
  3. 其余（MultiScaleContext / SCGMP / wind_mask）的开关只影响 forward 内部
     计算路径，不影响这里的训练循环结构，所以这个文件不需要为它们写特判。

复用 train.py 里现成的：
  - evaluate()         ：验证/测试集采样 + 指标计算
  - evaluate_all()      ：MAE/RMSE/MAPE/CRPS/PICP/PINAW 等指标
  - calibrate_temperature() ：温度校准
都是纯函数，不依赖具体是 DCFM 还是 AblationDCFM，可以直接拿来用。
"""

import os
import sys

# ── 路径处理：把项目根目录插入 sys.path ──────────────────────────────────────
# 本文件位于项目根目录的 ablation/ablation_train.py，上一级是项目根目录
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import time
import logging
import gc
from typing import Dict

import torch
import numpy as np

from train import evaluate, calibrate_temperature, evaluate_all
from ablation_model import AblationDCFM, AblationConfig


def ablation_train_one_epoch(
    model:            AblationDCFM,
    loader,
    optimizer:        torch.optim.Optimizer,
    club_optimizer,   # 可能为 None（use_club=False 时）
    adj_norm:         torch.Tensor,
    edge_index:       torch.Tensor,
    device:           torch.device,
    epoch:            int,
    main_params,
    grad_clip:        float = 1.0,
    warmup_epochs:    int = 5,
    cfm_n_t_samples:  int = 4,
    sigma_min:        float = 0.01,
    club_inner_steps: int = 5,
) -> Dict[str, float]:
    model.train()
    total_loss = total_cfm = total_mi = total_rank = total_var = total_gnorm = 0.0
    n_batches  = 0

    use_club = model.ablation.use_club
    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)

    train_club = use_club and (epoch > warmup_epochs)
    if train_club:
        ramp_epochs  = 3
        epochs_since = epoch - warmup_epochs
        ramp_factor  = min(1.0, epochs_since / ramp_epochs)
    else:
        ramp_factor  = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        He_prime, Hs_prime, He, Hs, X_low_pooled, X_high_pooled = \
            model(x, adj_norm_, edge_idx_)

        var_loss_val = 0.0
        if train_club:
            He_d, Hs_d = He.detach(), Hs.detach()
            X_low_d, X_high_d = X_low_pooled.detach(), X_high_pooled.detach()
            for _ in range(club_inner_steps):
                var_e = model.club_e.variational_loss(He_d, X_high_d)
                var_s = model.club_s.variational_loss(Hs_d, X_low_d)
                var_loss = var_e + var_s
                club_optimizer.zero_grad()
                var_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.club_e.parameters()) + list(model.club_s.parameters()),
                    max_norm=1.0,
                )
                club_optimizer.step()
                var_loss_val += var_loss.item()
            var_loss_val /= club_inner_steps

        B, T_out, N, Fy = y.shape
        y_target = y.permute(0, 2, 1, 3).reshape(B, N, T_out * Fy)

        cfm_l = model.cfm_loss(He_prime, Hs_prime, y_target,
                                n_t_samples=cfm_n_t_samples, sigma_min=sigma_min)
        rank_l   = model.rank_loss().clamp(min=-10.0, max=10.0)
        rank_val = rank_l.item()
        loss     = cfm_l + model.lambda_rank * rank_l
        mi_val   = 0.0

        if train_club:
            mi_e = model.club_e(He, X_high_pooled)
            mi_s = model.club_s(Hs, X_low_pooled)
            mi_penalty = torch.clamp(mi_e + mi_s, min=0.0)
            mi_val = mi_penalty.item()
            loss = (cfm_l
                    + model.lambda_rank * rank_l
                    + model.lambda_club * ramp_factor * mi_penalty)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(main_params, max_norm=grad_clip)
        optimizer.step()

        if use_club:
            # 与 train.py 保持一致：清掉 mi_penalty 反传时残留在 club 参数上的梯度
            club_optimizer.zero_grad()

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
        "rank":      total_rank  / n_batches,
        "var_loss":  total_var   / n_batches,
        "grad_norm": total_gnorm / n_batches,
    }


def ablation_train(
    model: AblationDCFM,
    train_loader, val_loader, test_loader,
    adj_norm, edge_index, device, cfg_train,
    scaler=None, logger=None,
    save_path: str = "ablation_best.pt",
) -> Dict:
    """
    与 train.py::train 结构一致，但：
      - 当 use_club=False 时不创建 club_optimizer，main_params 包含全部参数
        （此时 club_e/club_s 的参数虽然存在，但从不参与 forward 也从不被更新，
        不影响结果，只是占一点显存，可接受）；
      - 早停标准、温度校准、四组指标计算逻辑与原版完全一致，保证消融实验和
        主实验的评估口径一致，结果才能放在同一张表里比较。
    """
    if logger is None:
        logger = logging.getLogger("dcfm.ablation")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)

    use_club = model.ablation.use_club

    # club_e/club_s 的参数始终从 main_params 里排除：use_club=True 时它们由
    # club_optimizer 单独更新；use_club=False 时它们完全不参与 forward。
    # 同时过滤 requires_grad=False 的参数（如替代头实验里被冻结的 vector_field），
    # 避免它们进入 clip_grad_norm_ 带来无效开销。
    club_param_ids = (
        {id(p) for p in model.club_e.parameters()} |
        {id(p) for p in model.club_s.parameters()}
    )
    main_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in club_param_ids
    ]

    if use_club:
        club_optimizer = torch.optim.Adam(
            list(model.club_e.parameters()) + list(model.club_s.parameters()),
            lr=cfg_train.lr * 2.0, weight_decay=cfg_train.weight_decay,
        )
    else:
        club_optimizer = None

    optimizer = torch.optim.Adam(main_params, lr=cfg_train.lr, weight_decay=cfg_train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg_train.lr_decay_factor, patience=cfg_train.lr_decay_patience,
    )

    n_samples_val   = max(5, getattr(cfg_train, "cfm_n_samples_val", 10))
    n_steps_val     = max(5, getattr(cfg_train, "cfm_n_steps_val", 10))
    n_steps         = getattr(cfg_train, "cfm_n_steps", 20)
    cfm_n_t_samples = getattr(cfg_train, "cfm_n_t_samples", 4)
    n_samples_test  = getattr(cfg_train, "cfm_n_samples_test", 200)
    warmup_epochs   = getattr(cfg_train, "warmup_epochs", 5)
    sigma_min       = getattr(cfg_train, "cfm_sigma_min", 0.01)
    x0_scale        = getattr(cfg_train, "cfm_x0_scale", 1.0)
    club_inner_steps = getattr(cfg_train, "club_inner_steps", 5)

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history = {
        "train_loss": [], "train_cfm": [], "train_mi": [],
        "train_rank": [], "train_var_loss": [],
        "val_crps": [], "val_mae": [], "val_rmse": [],
    }

    logger.info(f"[Ablation:{model.ablation.tag()}] 开始训练  cfm_head={model.ablation.cfm_head}")
    for epoch in range(1, cfg_train.max_epochs + 1):
        t0 = time.time()
        train_m = ablation_train_one_epoch(
            model, train_loader, optimizer, club_optimizer,
            adj_norm, edge_index, device, epoch,
            main_params, cfg_train.grad_clip, warmup_epochs,
            cfm_n_t_samples=cfm_n_t_samples, sigma_min=sigma_min,
            club_inner_steps=club_inner_steps,
        )
        val_m = evaluate(
            model, val_loader, adj_norm, edge_index, device,
            scaler=scaler, n_samples=n_samples_val, n_steps=n_steps_val,
            temperature=1.0, inverse_transform=False,
            sigma_min=sigma_min, x0_scale=x0_scale,
        )
        val_crps_avg = val_m["CRPS"]
        scheduler.step(val_crps_avg)
        cur_lr = optimizer.param_groups[0]["lr"]
        if club_optimizer is not None:
            for pg in club_optimizer.param_groups:
                pg["lr"] = cur_lr * 2.0
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
            f"[{model.ablation.tag()}] Epoch {epoch:>4} | loss {train_m['loss']:.4f} | "
            f"cfm {train_m['cfm']:.4f} | mi {train_m['mi']:.4f} | "
            f"val_MAE {val_m['MAE']:.4f} | val_RMSE {val_m['RMSE']:.4f} | "
            f"val_CRPS {val_crps_avg:.4f} | lr {cur_lr:.2e} | {elapsed:.1f}s"
        )

        if val_crps_avg < best_val_crps:
            best_val_crps = val_crps_avg
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg_train.patience:
                logger.info(
                    f"[{model.ablation.tag()}] 早停于 epoch {epoch}（最佳 val CRPS={best_val_crps:.4f}）"
                )
                break

    # ── 加载最佳模型权重，并释放内存 ──
    logger.info(f"[{model.ablation.tag()}] 加载最佳模型权重: {save_path}")
    t0_load = time.time()
    try:
        state_dict = torch.load(save_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"[{model.ablation.tag()}] 权重加载成功，耗时 {time.time()-t0_load:.2f}s")
    except Exception as e:
        logger.error(f"[{model.ablation.tag()}] 权重加载失败: {e}")
        raise
    
    # 清理内存，释放 GPU 缓存
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info(f"[{model.ablation.tag()}] GPU 缓存已清空")

    # ── 温度校准（与 train.py 完全一致的口径） ──
    # 确定性头（cfm_head="deterministic"）的所有 S 份样本完全相同，
    # mu + T*(samples - mu) = mu 对任意 T 恒成立，calibration 无意义，直接用 T=1.0。
    is_deterministic = (model.ablation.cfm_head == "deterministic")

    logger.info(f"[{model.ablation.tag()}] 开始验证集评估...")
    t0_val = time.time()
    try:
        _, samples_val_norm, y_val_norm = evaluate(
            model, val_loader, adj_norm, edge_index, device,
            scaler=scaler, return_preds=True,
            n_samples=n_samples_test, n_steps=n_steps,
            temperature=1.0, inverse_transform=False,
            sigma_min=sigma_min, x0_scale=x0_scale,
        )
        logger.info(f"[{model.ablation.tag()}] 验证集评估完成，耗时 {time.time()-t0_val:.2f}s")
    except Exception as e:
        logger.error(f"[{model.ablation.tag()}] 验证集评估失败: {e}")
        raise
    if is_deterministic:
        best_T = 1.0
        logger.info(f"[{model.ablation.tag()}] 确定性头，跳过温度校准，best_T 固定为 1.0")
    else:
        logger.info(f"[{model.ablation.tag()}] 进行温度校准...")
        t0_calib = time.time()
        best_T = calibrate_temperature(samples_val_norm, y_val_norm, target_coverage=0.95)
        logger.info(f"[{model.ablation.tag()}] 温度校准完成: best_T={best_T:.4f}，耗时 {time.time()-t0_calib:.2f}s")

    # 清理中间数据
    del samples_val_norm, y_val_norm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 测试集评估 ──
    logger.info(f"[{model.ablation.tag()}] 开始测试集评估...")
    t0_test = time.time()
    try:
        _, samples_test_norm, y_test_norm = evaluate(
            model, test_loader, adj_norm, edge_index, device,
            scaler=scaler, return_preds=True,
            n_samples=n_samples_test, n_steps=n_steps,
            temperature=1.0, inverse_transform=False,
            sigma_min=sigma_min, x0_scale=x0_scale,
        )
        logger.info(f"[{model.ablation.tag()}] 测试集评估完成，耗时 {time.time()-t0_test:.2f}s")
    except Exception as e:
        logger.error(f"[{model.ablation.tag()}] 测试集评估失败: {e}")
        raise
    logger.info(f"[{model.ablation.tag()}] 进行反归一化和指标计算...")
    t0_metrics = time.time()
    
    samples_test = scaler.inverse_transform(samples_test_norm) if scaler is not None else samples_test_norm
    y_test       = scaler.inverse_transform(y_test_norm) if scaler is not None else y_test_norm

    mu_norm = samples_test_norm.mean(axis=0, keepdims=True)
    test_m_norm = evaluate_all(mu_norm + best_T * (samples_test_norm - mu_norm), y_test_norm)

    mu_test = samples_test.mean(axis=0, keepdims=True)
    test_m  = evaluate_all(mu_test + best_T * (samples_test - mu_test), y_test)

    test_m_raw_norm = evaluate_all(samples_test_norm, y_test_norm)
    test_m_raw       = evaluate_all(samples_test, y_test)
    
    logger.info(f"[{model.ablation.tag()}] 指标计算完成，耗时 {time.time()-t0_metrics:.2f}s")

    history["test_metrics"]          = test_m
    history["test_metrics_norm"]     = test_m_norm
    history["test_metrics_raw"]      = test_m_raw
    history["test_metrics_raw_norm"] = test_m_raw_norm
    history["best_temperature"]      = best_T
    history["best_val_crps"]         = best_val_crps
    history["ablation_tag"]          = model.ablation.tag()
    history["ablation_cfm_head"]     = model.ablation.cfm_head
    history["n_epochs_run"]          = epoch
    return history
