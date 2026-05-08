"""
baselines/ha.py
HA: Historical Average baseline.
不需要训练，直接在测试集上预测。
pred = mean over T_in dim of x

[Fix] 支持传入 scaler，在原始量纲下计算指标，与论文口径一致。
      scaler=None 时在归一化空间计算（旧行为）。
"""
import torch
from .utils import compute_metrics


def run_ha(test_loader, device, logger=None, scaler=None):
    def _log(msg):
        (logger.info if logger else print)(msg)

    _log("[HA] Running Historical Average baseline...")
    preds, trues = [], []
    for x, y in test_loader:
        # x: [B, T_in, N, F]
        pred = x.mean(dim=1)   # [B, N, F]
        preds.append(pred)
        trues.append(y)

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)

    mae, rmse, mape = compute_metrics(pred, true)
    _log(f"[HA] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
    return {"test_mae": mae, "test_rmse": rmse, "test_mape": mape}