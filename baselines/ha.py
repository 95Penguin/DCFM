"""
baselines/ha.py
HA: Historical Average baseline（多步预测版）。

预测逻辑：用输入窗口均值作为所有预测步的预测值（持久性均值），
即 y_hat[:, t, :, :] = mean(x, dim=1) for all t in 1..T_out。
"""
import torch
from .utils import compute_metrics, inverse_torch


def run_ha(test_loader, device, logger=None, scaler=None, null_val=None):
    def _log(msg):
        (logger.info if logger else print)(msg)

    _log("[HA] Running Historical Average baseline (multi-step)...")
    preds, trues = [], []
    for x, y in test_loader:
        # x: [B, T_in, N, F],  y: [B, T_out, N, F]
        B, T_in, N, F = x.shape
        T_out = y.shape[1]
        # 对输入时间维取均值，重复 T_out 次
        pred = x.mean(dim=1, keepdim=True)           # [B, 1, N, F]
        pred = pred.expand(-1, T_out, -1, -1)         # [B, T_out, N, F]
        preds.append(pred)
        trues.append(y)

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)

    null_val_eval = null_val
    if scaler is not None:
        pred = inverse_torch(pred, scaler)
        true = inverse_torch(true, scaler)
        null_val_eval = 0.0 if null_val is not None else None

    mae, rmse, mape = compute_metrics(pred, true, null_val_eval)
    _log(f"[HA] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
    return {"test_mae": mae, "test_rmse": rmse, "test_mape": mape}
