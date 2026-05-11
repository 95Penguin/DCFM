"""
baselines/var_model.py
VAR baseline — 对每个节点独立拟合 AR(max_lags) 模型。

修复:
  [Bug12] 删除无效的 AutoReg.predict() 调用，
          统一用 AR 系数手动做 1 步预测。
          hist[-lags:][::-1] 从近到远排列后与 ar_coef 做点积。
  [Fix]   支持传入 scaler，在原始量纲下计算指标，与论文口径一致。
"""
import numpy as np
import torch
from .utils import compute_metrics


def run_var(train_loader, test_loader, T_in: int,
            device, max_lags: int = 24, logger=None, scaler=None):
    def _log(msg):
        (logger.info if logger else print)(msg)

    try:
        from statsmodels.tsa.ar_model import AutoReg
    except ImportError:
        _log("[VAR] 需要安装 statsmodels: pip install statsmodels")
        return {}

    _log("[VAR] 收集训练集时序数据...")

    # 从 DataLoader 重建全量时序
    all_segs = []
    for x, y in train_loader:
        # x: [B, T_in, N, F]，取第一个特征
        all_segs.append(x[..., 0].numpy())   # [B, T_in, N]

    segs = np.concatenate(all_segs, axis=0)  # [n_windows, T_in, N]
    N = segs.shape[2]

    # 重建时序：第 0 窗口完整，后续窗口取最后 1 步
    ts_list = [segs[0, :, :]]               # [T_in, N]
    for i in range(1, segs.shape[0]):
        ts_list.append(segs[i, -1:, :])     # [1, N]
    ts = np.concatenate(ts_list, axis=0)    # [T_total, N]

    lags = min(max_lags, T_in - 1)
    _log(f"[VAR] 对 N={N} 个节点拟合 AR({lags}) 模型，时序长度 T={ts.shape[0]}...")

    # 拟合每个节点，记录系数
    ar_params = []   # list of (intercept, ar_coefs) or None
    for n in range(N):
        try:
            res = AutoReg(ts[:, n], lags=lags, old_names=False).fit()
            intercept = float(res.params[0])
            ar_coef   = res.params[1:].astype(np.float32)  # [lags]
            ar_params.append((intercept, ar_coef))
        except Exception as e:
            _log(f"  [VAR] 节点 {n} 拟合失败: {e}，使用 last-value fallback")
            ar_params.append(None)
        if (n + 1) % 100 == 0:
            _log(f"  [VAR] 已拟合 {n + 1}/{N} 个节点")

    # 在测试集上预测
    _log("[VAR] 在测试集上预测...")
    preds, trues = [], []

    for x, y in test_loader:
        B, T, Nn, F = x.shape
        x_np = x[..., 0].numpy()                 # [B, T_in, N]
        batch_pred = np.zeros((B, Nn, 1), dtype=np.float32)

        for n in range(Nn):
            if ar_params[n] is not None:
                intercept, ar_coef = ar_params[n]
                lags_n = len(ar_coef)
                for b in range(B):
                    hist = x_np[b, :, n]          # [T_in]
                    # AR 预测：intercept + ar[0]*hist[-1] + ar[1]*hist[-2] + ...
                    hist_lag = hist[-lags_n:][::-1]   # [lags]，从近到远
                    val = intercept + float(np.dot(ar_coef, hist_lag))
                    batch_pred[b, n, 0] = val
            else:
                batch_pred[:, n, 0] = x_np[:, -1, n]  # fallback: last value

        preds.append(torch.tensor(batch_pred))    # [B, N, 1]
        trues.append(y[..., :1])                  # [B, N, 1]

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)

    mae, rmse, mape = compute_metrics(pred, true)
    _log(f"[VAR] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
    return {"test_mae": mae, "test_rmse": rmse, "test_mape": mape}