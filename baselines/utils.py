"""
baselines/utils.py
共享工具：指标计算 + 通用训练循环（多步预测版）

多步改动：
  - 模型输出 [B, T_out, N, F]，target [B, T_out, N, F]
  - 损失和指标在所有维度上计算（时间维也参与平均）
  - 支持 prob 模式（输出 mu/sigma 双通道）

评估策略（2025-05 修订）：
  统一无 mask 全量评估，与 DCFM 主模型对齐，方便与文献直接比较。
  所有函数保留 null_val 参数接口，但传入任何值均不做 mask。

修复:
  [5] train_model 训练循环中增加 NaN/Inf 防护：
        · 计算 loss 后立即检测 torch.isfinite(loss)，若为 NaN 或 Inf
          则跳过本 batch（zero_grad + continue），不执行 backward / step，
          防止 NaN 通过参数更新污染整个模型权重。
        · 同时在 batch 日志中统计并打印 NaN 跳过次数，方便排查根因。
        · 对已收集的有效 losses 做均值时加 `or [float('nan')]` 保护，
          避免全 NaN epoch 时 np.mean([]) 引发 RuntimeWarning。
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 指标（无 mask，全量计算；null_val 参数保留接口兼容性，不使用）
# ---------------------------------------------------------------------------

def masked_mae(pred, true, null_val=None):
    return torch.abs(pred - true).mean()


def masked_mse(pred, true, null_val=None):
    return ((pred - true) ** 2).mean()


def masked_rmse(pred, true, null_val=None):
    return torch.sqrt(masked_mse(pred, true))


def masked_mape(pred, true, null_val=None, eps=1e-8):
    return (torch.abs((pred - true) / (true.abs() + eps))).mean()


def compute_metrics(pred, true, null_val=None):
    """返回 (MAE, RMSE, MAPE%)，无 mask 全量计算。"""
    return (masked_mae(pred, true).item(),
            masked_rmse(pred, true).item(),
            masked_mape(pred, true).item() * 100.0)


def inverse_torch(tensor: torch.Tensor, scaler):
    """Inverse-transform a tensor while preserving shape and returning CPU float."""
    shape = tensor.shape
    # 修复：若 Scaler 训练于多特征（F > 1）但 tensor 只有单特征（F = 1），
    # 需手动反归一化，避免 Scaler.inverse_transform 内部 reshape(-1, F) 误用特征尺度
    if scaler.mean.shape[-1] > 1 and tensor.shape[-1] == 1:
        data_flat = tensor.detach().cpu().numpy().reshape(-1)
        s_mean = scaler.mean[..., :1].ravel().item()
        s_std  = scaler.std[..., :1].ravel().item()
        inv = data_flat * s_std + s_mean
        return torch.from_numpy(inv.reshape(shape)).float()
    inv = scaler.inverse_transform(tensor.detach().cpu().numpy().reshape(-1))
    return torch.from_numpy(inv.reshape(shape)).float()


def gaussian_nll_loss(mu, sigma_raw, target, null_val=None):
    """高斯 NLL loss，无 mask 全量计算。"""
    sigma = F.softplus(sigma_raw) + 1e-4
    nll = 0.5 * ((target - mu) ** 2) / (sigma ** 2) + sigma.log()
    return nll.mean()


def compute_crps_gaussian(mu, sigma, target, null_val=None):
    """解析高斯 CRPS（闭式解），O(1) 内存，无采样。
    公式: CRPS = σ · [z · (2Φ(z)-1) + 2φ(z) − 1/√π]
    其中 z = (target − mu) / σ, Φ=标准正态 CDF, φ=标准正态 PDF。
    sigma 应为已处理好的标准差（调用方负责 softplus + 数值稳定）。"""
    sigma = sigma + 1e-6
    z = (target - mu) / sigma
    # 标准正态 PDF: φ(z) = exp(-z²/2) / √(2π)
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    # 标准正态 CDF: Φ(z) = 0.5 · (1 + erf(z/√2))
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return crps.mean().item()


def compute_prob_metrics(samples: np.ndarray, y: np.ndarray,
                         null_val: float = None) -> dict:
    """
    概率预测指标（多步版），无 mask 全量计算。

    samples  : [S, total, N, T_out, F]
    y        : [total, N, T_out, F]
    null_val : 保留接口兼容性，不使用。
    """
    mu = samples.mean(axis=0)

    def _mae(p, t):  return float(np.abs(p - t).mean())
    def _rmse(p, t): return float(np.sqrt(((p - t) ** 2).mean()))
    def _mape(p, t): return float((np.abs((p - t) / (np.abs(t) + 1e-8))).mean() * 100.0)

    def _crps(s, t):
        S = s.shape[0]
        mae_term = np.abs(s - t[None]).mean(axis=0)
        rng = np.random.default_rng(seed=0)
        n_perm = min(10, S - 1)
        if n_perm <= 0:
            return float(mae_term.mean())
        spreads = []
        indices = np.arange(S)
        for _ in range(n_perm):
            perm = rng.permutation(S)
            clash = np.where(perm == indices)[0]
            if len(clash) > 1:
                perm[clash] = perm[np.roll(clash, -1)]
            elif len(clash) == 1:
                other = (clash[0] + 1) % S
                perm[clash[0]], perm[other] = perm[other], perm[clash[0]]
            spreads.append(np.abs(s - s[perm]).mean(axis=0))
        crps_pt = mae_term - 0.5 * np.mean(spreads, axis=0)
        return float(crps_pt.mean())

    def _picp(s, t, conf=0.95):
        lo = np.quantile(s, (1 - conf) / 2,     axis=0)
        hi = np.quantile(s, 1 - (1 - conf) / 2, axis=0)
        return float(((t >= lo) & (t <= hi)).astype(np.float32).mean())

    def _pinaw(s, t, conf=0.95):
        lo = np.quantile(s, (1 - conf) / 2,     axis=0)
        hi = np.quantile(s, 1 - (1 - conf) / 2, axis=0)
        rng = t.max() - t.min() + 1e-8
        return float(((hi - lo) / rng).mean())

    metrics = {
        "MAE":   _mae(mu, y),
        "RMSE":  _rmse(mu, y),
        "MAPE":  _mape(mu, y),
        "CRPS":  _crps(samples, y),
        "PICP":  _picp(samples, y),
        "PINAW": _pinaw(samples, y),
    }
    T_out = y.shape[2]
    for h in range(T_out):
        sh = samples[:, :, :, h, :]
        yh = y[:, :, h, :]
        mh = sh.mean(axis=0)
        metrics[f"MAE_h{h+1}"]  = _mae(mh, yh)
        metrics[f"RMSE_h{h+1}"] = _rmse(mh, yh)
        metrics[f"MAPE_h{h+1}"] = _mape(mh, yh)
        metrics[f"CRPS_h{h+1}"] = _crps(sh, yh)
    return metrics


# ---------------------------------------------------------------------------
# 通用训练循环（多步版）
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, test_loader,
                optimizer, scheduler, device, scaler=None,
                max_epochs=200, patience=20, grad_clip=1.0,
                save_path="best_baseline.pt", logger=None,
                extra_forward_kwargs=None,
                log_batch_interval=50,
                prob=False,
                null_val=None,
                mc_samples_test: int = None,
                mc_chunk_size: int = 0):
    """
    多步预测版通用训练循环。

    x: [B, T_in, N, F]
    y: [B, T_out, N, F]
    model(x) → [B, T_out, N, out_dim]

    null_val : 保留接口兼容性，不使用，统一无 mask 全量评估。
    prob     : True 时输出 [B, T_out, N, 2*F]，用 NLL loss，测试额外报告 CRPS

    修复 [5]：每个 batch 计算 loss 后先检测 isfinite，NaN/Inf 时跳过该 batch，
    防止梯度爆炸污染模型权重，同时在日志中打印跳过次数便于排查。
    """
    def _log(msg):
        (logger.info if logger else print)(msg)

    ekw = extra_forward_kwargs or {}
    best_val, no_improve = float("inf"), 0
    history = {"train_loss": [], "val_mae": [],
               "test_mae": 0., "test_rmse": 0., "test_mape": 0., "test_crps": 0.}
    n_batches = len(train_loader)

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        nan_skipped = 0  # 修复 [5]：记录本 epoch 跳过的 NaN batch 数

        for batch_idx, (x, y) in enumerate(train_loader, 1):
            # x: [B, T_in, N, F],  y: [B, T_out, N, F]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, **ekw)           # [B, T_out, N, out_dim]

            if prob:
                F_dim = y.shape[-1]
                mu, sigma_raw = out[..., :F_dim], out[..., F_dim:]
                loss = gaussian_nll_loss(mu, sigma_raw, y)
            else:
                loss = masked_mae(out, y)

            # 修复 [5]：检测 NaN / Inf，若出现则跳过本 batch。
            # 不执行 backward / step，防止 NaN 梯度更新污染模型权重。
            # 原因：梯度爆炸首先体现为 loss=NaN，一旦 step() 执行，
            # 权重也会变 NaN，之后所有 batch 的前向传播均输出 NaN，无法恢复。
            if not torch.isfinite(loss):
                nan_skipped += 1
                optimizer.zero_grad()
                continue

            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(loss.item())

            if log_batch_interval > 0 and batch_idx % log_batch_interval == 0:
                elapsed = time.time() - t0
                eta = elapsed / batch_idx * (n_batches - batch_idx)
                _log(f"  Epoch {epoch:3d} [{batch_idx:4d}/{n_batches}] "
                     f"loss={float(np.mean(losses or [float('nan')])):.4f}  "
                     f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        # 修复 [5]：若本 epoch 出现过 NaN，在 epoch 行统一打印，不淹没 batch 日志
        nan_warn = f"  [!] {nan_skipped} NaN batch(es) skipped" if nan_skipped > 0 else ""

        # 用 `or [float('nan')]` 防止 losses 为空列表（全 epoch NaN）时 np.mean 报错
        tl = float(np.mean(losses or [float("nan")]))
        history["train_loss"].append(tl)

        model.eval()
        vm = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, **ekw)                       # [B, T_out, N, out_dim]
                mu = out[..., :y.shape[-1]] if prob else out
                vm.append(masked_mae(mu, y).item())
        val_mae = float(np.mean(vm))
        history["val_mae"].append(val_mae)

        if scheduler is not None:
            try:    scheduler.step(val_mae)
            except: scheduler.step()

        if val_mae < best_val:
            best_val, no_improve = val_mae, 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        _log(f"  Epoch {epoch:3d} | train={tl:.4f} | val={val_mae:.4f} | "
             f"best={best_val:.4f} | {time.time()-t0:.1f}s{nan_warn}")
        if no_improve >= patience:
            _log(f"  Early stop @ epoch {epoch}")
            break

    # ── 测试 ──────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    all_mu, all_sigma_raw, all_true = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x, **ekw)
            if prob:
                F_dim = y.shape[-1]
                all_mu.append(out[..., :F_dim].cpu())
                all_sigma_raw.append(out[..., F_dim:].cpu())
            else:
                all_mu.append(out.cpu())
            all_true.append(y.cpu())

    pred_mu  = torch.cat(all_mu)
    true_cat = torch.cat(all_true)

    # ================= 保存预测结果 =================
    history["prediction"] = pred_mu.detach().cpu().numpy()
    history["ground_truth"] = true_cat.detach().cpu().numpy()

    if prob:
        pred_sigma_norm = torch.cat(all_sigma_raw)
        mae_norm, rmse_norm, mape_norm = compute_metrics(pred_mu, true_cat)
        crps_norm = compute_crps_gaussian(pred_mu, F.softplus(pred_sigma_norm), true_cat)
    else:
        mae_norm, rmse_norm, mape_norm = compute_metrics(pred_mu, true_cat)

    if scaler is not None:
        pred_mu  = inverse_torch(pred_mu,  scaler)
        true_cat = inverse_torch(true_cat, scaler)

    history["prediction_inv"] = pred_mu.detach().cpu().numpy()
    history["ground_truth_inv"] = true_cat.detach().cpu().numpy()

    mae, rmse, mape = compute_metrics(pred_mu, true_cat)

    if prob:
        pred_sigma_raw = torch.cat(all_sigma_raw)
        if scaler is not None and hasattr(scaler, "scale_"):
            # sigma 只需除以 std（scale），不减 mean
            scale = torch.tensor(scaler.scale_, dtype=torch.float32)
            pred_sigma = F.softplus(pred_sigma_raw) * scale
        else:
            pred_sigma = F.softplus(pred_sigma_raw)
        crps = compute_crps_gaussian(pred_mu, pred_sigma, true_cat)
        _log(f"  [Test] 归一化域   MAE={mae_norm:.4f}  RMSE={rmse_norm:.4f}  MAPE={mape_norm:.2f}%  CRPS={crps_norm:.4f}")
        _log(f"  [Test] 反归一化域 MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  CRPS={crps:.4f}")
        history.update({"test_mae": mae, "test_rmse": rmse,
                         "test_mape": mape, "test_crps": crps,
                         "test_mae_norm": mae_norm, "test_rmse_norm": rmse_norm,
                         "test_mape_norm": mape_norm, "test_crps_norm": crps_norm})

        # Monte-Carlo sampling to compute empirical PICP / PINAW (for probabilistic baselines)
        try:
            S = int(mc_samples_test) if mc_samples_test is not None else 200
            mu_norm = torch.cat(all_mu)                # [total, T_out, N, F]
            sigma_norm = F.softplus(torch.cat(all_sigma_raw))  # same shape

            # y in normalized domain for metrics
            y_norm = torch.cat(all_true).numpy().transpose(0, 2, 1, 3)  # [total, N, T_out, F]

            # If chunking requested, write samples to a temporary memmap to avoid large peak RAM
            if mc_chunk_size and mc_chunk_size > 0 and S > mc_chunk_size:
                import tempfile, os
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
                tmpf.close()
                try:
                    shape_mem = (S, mu_norm.shape[0], mu_norm.shape[2], mu_norm.shape[1], mu_norm.shape[3])
                    samples_mem = np.memmap(tmpf.name, dtype="float32", mode="w+", shape=shape_mem)
                    for start in range(0, S, mc_chunk_size):
                        cur = min(mc_chunk_size, S - start)
                        eps = torch.randn((cur,) + mu_norm.shape)
                        samples_chunk = (mu_norm.unsqueeze(0) + eps * sigma_norm.unsqueeze(0)).numpy()
                        samples_chunk = samples_chunk.transpose(0, 1, 3, 2, 4)  # to [cur, total, N, T_out, F]
                        samples_mem[start:start+cur] = samples_chunk.astype("float32")
                    samples_norm = np.array(samples_mem)  # load as ndarray for metrics
                finally:
                    try:
                        os.unlink(tmpf.name)
                    except Exception:
                        pass
            else:
                eps = torch.randn((S,) + mu_norm.shape)
                samples_norm = (mu_norm.unsqueeze(0) + eps * sigma_norm.unsqueeze(0)).numpy()
                samples_norm = samples_norm.transpose(0, 1, 3, 2, 4)  # [S, total, N, T_out, F]

            # Prepare physical-domain copies if scaler provided
            samples_phys = samples_norm.copy()
            y_phys = y_norm.copy()
            if scaler is not None:
                shape_s = samples_phys.shape
                shape_y = y_phys.shape
                s_mean = scaler.mean[..., :1] if scaler.mean.shape[-1] > 1 else scaler.mean
                s_std = scaler.std[..., :1] if scaler.std.shape[-1] > 1 else scaler.std
                samples_phys = (samples_phys.reshape(-1) * s_std + s_mean).reshape(shape_s)
                y_phys = (y_phys.reshape(-1) * s_std + s_mean).reshape(shape_y)

            mc_norm = compute_prob_metrics(samples_norm, y_norm)
            mc_phys = compute_prob_metrics(samples_phys, y_phys)
            history.update({
                "PICP": mc_phys.get("PICP", float("nan")),
                "PINAW": mc_phys.get("PINAW", float("nan")),
                "PICP_norm": mc_norm.get("PICP", float("nan")),
                "PINAW_norm": mc_norm.get("PINAW", float("nan")),
            })
            _log(f"  [Test] Empirical PICP={history['PICP']:.4f}  PINAW={history['PINAW']:.4f}")
        except Exception as e:
            _log(f"  [ProbMetrics] MC sampling failed: {e}; skipping empirical PICP/PINAW")
    else:
        _log(f"  [Test] 归一化域   MAE={mae_norm:.4f}  RMSE={rmse_norm:.4f}  MAPE={mape_norm:.2f}%")
        _log(f"  [Test] 反归一化域 MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
        history.update({"test_mae": mae, "test_rmse": rmse, "test_mape": mape,
                         "test_mae_norm": mae_norm, "test_rmse_norm": rmse_norm,
                         "test_mape_norm": mape_norm})

    return history
