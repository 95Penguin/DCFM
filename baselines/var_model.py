"""
baselines/var_model.py
VAR baseline — 对每个节点独立拟合 AR(max_lags) 模型（多步预测版）。

多步改动：用迭代预测替代单步预测，前一步预测值作为下一步的输入历史。

修复：
  [1] recent = hist[-lags_n:] 当历史长度不足 lags_n 时（边界情况），
      用均值填充补齐，避免 dot 积维度不匹配导致 ValueError。
  [2] recent[::-1] 转为 list 后执行 np.dot，避免负步长 slice 的隐式 copy 警告。
"""
import numpy as np
import torch
from .utils import compute_metrics


def run_var(train_loader, test_loader, T_in: int, T_out: int,
            device, max_lags: int = 3, logger=None, scaler=None, null_val=None):  #24->3
    def _log(msg):
        (logger.info if logger else print)(msg)

    try:
        from statsmodels.tsa.ar_model import AutoReg
    except ImportError:
        _log("[VAR] 需要安装 statsmodels: pip install statsmodels")
        return {}

    _log("[VAR] 收集训练集时序数据（保持时间顺序）...")

    dataset = train_loader.dataset
    all_x = []
    for i in range(len(dataset)):
        x, _ = dataset[i]
        all_x.append(x[..., 0].numpy())          # [T_in, N]

    # 滑动窗口步长=1，首段全取，后续只取最后一步（新信息）
    ts = np.concatenate([all_x[0]] + [a[-1:] for a in all_x[1:]], axis=0)
    N = ts.shape[1]

    lags = min(max_lags, T_in - 1)
    _log(f"[VAR] 对 N={N} 个节点拟合 AR({lags}) 模型，时序长度 T={ts.shape[0]}...")

    ar_params = []
    for n in range(N):
        try:
            res = AutoReg(ts[:, n], lags=lags, old_names=False).fit()
            intercept = float(res.params[0])
            ar_coef   = res.params[1:].astype(np.float32)
            ar_params.append((intercept, ar_coef))
        except Exception:
            ar_params.append(None)
        if (n + 1) % 100 == 0:
            _log(f"  [VAR] 已拟合 {n + 1}/{N} 个节点")

    _log(f"[VAR] 在测试集上做 {T_out} 步迭代预测...")
    preds, trues = [], []

    for x, y in test_loader:
        B, T, Nn, F = x.shape
        x_np = x[..., 0].numpy()                     # [B, T_in, N]
        batch_pred = np.zeros((B, T_out, Nn, 1), dtype=np.float32)

        for n in range(Nn):
            for b in range(B):
                hist = list(x_np[b, :, n])            # 初始历史
                for step in range(T_out):
                    if ar_params[n] is not None:
                        intercept, ar_coef = ar_params[n]
                        lags_n = len(ar_coef)
                        # 修复 [1]：历史不足时用均值填充，保证维度匹配
                        if len(hist) >= lags_n:
                            recent = list(reversed(hist[-lags_n:]))
                        else:
                            # 历史长度不足（理论上不会发生，但加保护）
                            pad_len = lags_n - len(hist)
                            fill_val = float(np.mean(hist)) if hist else 0.0
                            recent = list(reversed(hist)) + [fill_val] * pad_len
                        # 修复 [2]：recent 已是 list，np.dot 无负步长 slice 问题
                        val = intercept + float(np.dot(ar_coef, recent))
                    else:
                        val = hist[-1]                 # fallback
                    batch_pred[b, step, n, 0] = val
                    hist.append(val)                   # 迭代：预测值加入历史

        preds.append(torch.tensor(batch_pred))
        trues.append(y[..., :1])

    pred = torch.cat(preds, dim=0)     # [total, T_out, N, 1]
    true = torch.cat(trues, dim=0)     # [total, T_out, N, 1]

    mae_norm, rmse_norm, mape_norm = compute_metrics(pred, true)

    if scaler is not None:
        pred = scaler.inverse_transform(pred.numpy().reshape(-1)).reshape(pred.shape)
        pred = torch.from_numpy(pred).float()
        true_np = scaler.inverse_transform(true.numpy().reshape(-1)).reshape(true.shape)
        true = torch.from_numpy(true_np).float()

    mae, rmse, mape = compute_metrics(pred, true)
    _log(f"[VAR] 归一化域   MAE={mae_norm:.4f}  RMSE={rmse_norm:.4f}  MAPE={mape_norm:.2f}%")
    _log(f"[VAR] 反归一化域 MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
    return {"test_mae": mae, "test_rmse": rmse, "test_mape": mape,
            "test_mae_norm": mae_norm, "test_rmse_norm": rmse_norm,
            "test_mape_norm": mape_norm}