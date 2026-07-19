"""
baselines/run_baselines.py
统一 baseline 运行入口（多步预测版），与 GridCFN 的 main.py 接口一致。

用法:
  uv run baselines/run_baselines.py --preset solar --models tsdiff stid
  uv run baselines/run_baselines.py --preset electricity --models dcrnn mtgnn stid
  uv run baselines/run_baselines.py --preset weather --models ha stid --gpu_id 0
  uv run baselines/run_baselines.py --preset sdwpf --models dcrnn tsflow k2vae
  uv run baselines/run_baselines.py --preset pjm --models ha var dcrnn mtgnn

所有结果保存在 result/baselines/<dataset>/<timestamp>/

修复：
  [1] load_weather 调用补全 feature_idx 参数
  [2] null_val 统一传 None：全量无 mask 评估，与 GridCFN 主模型对齐，
      方便与文献直接对比。
  [3] 补全 pjm 数据集支持（import、load_data 分支、--preset choices）
  [6] run_agcrn 中学习率由 3e-3 降至 5e-4，并收紧 weight_decay 至 1e-3：
        · 日志显示 Epoch 1 后半段 loss 从 0.41 升至 0.61，是 lr 过大导致
          参数开始震荡的典型表现。原 lr=3e-3 在图卷积梯度尚不稳定时
          （配合修复 [4] 后仍需保守起步）仍可能引发震荡。
        · 降至 5e-4 与 DCRNN/STGCN 等其他基线保持同一量级，
          同时加强 weight_decay 抑制嵌入向量的范数无限增大。
        · scheduler patience 从 10 降至 8，让 ReduceLROnPlateau 更快介入。
"""

import argparse
import csv
import json
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from dataset import load_solar_energy, load_electricity, load_weather, load_sdwpf, load_pjm


# ── 工具函数 ──────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    if gpu_id < 0:
        return torch.device("cuda:0")
    if gpu_id >= n_gpu:
        raise ValueError(f"gpu_id={gpu_id} 超出范围，共 {n_gpu} 块 GPU")
    return torch.device(f"cuda:{gpu_id}")


def setup_logger(log_path: str, name: str = "baselines") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_data(cfg):
    """
    修复 [1]：load_weather 补全 feature_idx 参数。
    修复 [3]：补全 pjm 数据集支持，与 main.py 保持一致。
    """
    d = cfg.data
    if d.dataset == "solar":
        return load_solar_energy(d.data_path, d.T_in, d.T_out,
                                 d.adj_threshold, d.batch_size)
    elif d.dataset == "electricity":
        return load_electricity(d.data_path, d.T_in, d.T_out,
                                d.adj_threshold, d.batch_size)
    elif d.dataset == "weather":
        # 修复 [1]：补全 feature_idx 参数，与 main.py 保持一致
        return load_weather(d.data_path, d.T_in, d.T_out,
                            d.adj_threshold, d.batch_size,
                            feature_idx=getattr(d, "weather_feature_idx", 0))
    elif d.dataset == "sdwpf":
        return load_sdwpf(d.data_path, d.T_in, d.T_out,
                          d.adj_threshold, d.batch_size)
    elif d.dataset == "pjm":
        # 修复 [3]：补全 pjm 分支
        return load_pjm(d.data_path, d.T_in, d.T_out,
                        d.adj_threshold, d.batch_size)
    else:
        raise ValueError(d.dataset)


def _get_null_val(dataset: str, scaler) -> None:
    """
    统一返回 None：所有数据集均无 mask，全量评估，与 GridCFN 对齐。
    null_val 参数在各函数中保留接口但不使用。
    """
    return None


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def _metric(res: dict, key: str, norm: bool = False):
    metrics = res.get("test_metrics", {}) if isinstance(res, dict) else {}
    if norm:
        direct_key = f"test_{key.lower()}_norm"
        metric_key = f"{key}_norm"
    else:
        direct_key = f"test_{key.lower()}"
        metric_key = key
    return res.get(direct_key, metrics.get(metric_key, float("nan")))


def _summary_rows(all_results: dict):
    rows = []
    for name, res in all_results.items():
        row = {"model": name}
        if not isinstance(res, dict):
            row.update({"status": "ERROR", "error": "invalid result"})
        elif "error" in res:
            row.update({"status": "ERROR", "error": res["error"]})
        elif not res or (
            "test_metrics" not in res
            and not any(k.startswith("test_") for k in res)
        ):
            row.update({"status": "ERROR", "error": "no test metrics returned"})
        else:
            row.update({
                "status": "OK",
                "error": "",
                "MAE": _metric(res, "MAE"),
                "RMSE": _metric(res, "RMSE"),
                "MAPE": _metric(res, "MAPE"),
                "CRPS": _metric(res, "CRPS"),
                "MAE_norm": _metric(res, "MAE", norm=True),
                "RMSE_norm": _metric(res, "RMSE", norm=True),
                "MAPE_norm": _metric(res, "MAPE", norm=True),
                "CRPS_norm": _metric(res, "CRPS", norm=True),
            })
        rows.append(row)
    return rows


def _format_value(value):
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{value:.4f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return value


def save_summaries(all_results: dict, save_dir: str, dataset: str, ts: str, logger=None):
    rows = _summary_rows(all_results)
    result_path = os.path.join(save_dir, f"baselines_results_{dataset}_{ts}.json")
    csv_path = os.path.join(save_dir, "summary.csv")
    md_path = os.path.join(save_dir, "summary.md")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=_json_default)

    fieldnames = [
        "model", "status",
        "MAE", "RMSE", "MAPE", "CRPS",
        "MAE_norm", "RMSE_norm", "MAPE_norm", "CRPS_norm",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_value(row.get(k, "")) for k in fieldnames})

    table_rows = [
        {k: _format_value(row.get(k, "")) for k in fieldnames}
        for row in rows
    ]
    widths = {
        k: max(len(k), *(len(str(row.get(k, ""))) for row in table_rows))
        for k in fieldnames
    }

    def md_row(row):
        return "| " + " | ".join(str(row.get(k, "")).ljust(widths[k]) for k in fieldnames) + " |"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Baseline 汇总（{dataset}, {ts}）\n\n")
        f.write(md_row({k: k for k in fieldnames}) + "\n")
        f.write("| " + " | ".join("-" * widths[k] for k in fieldnames) + " |\n")
        for row in table_rows:
            f.write(md_row(row) + "\n")
        f.write("\n注：无后缀指标为反归一化域；`*_norm` 指标为归一化域。ERROR 行表示该模型运行失败，但其他模型结果已保留。\n")

    if logger:
        logger.info(f"Summary saved: {csv_path}")
        logger.info(f"Summary saved: {md_path}")
        logger.info(f"Results saved: {result_path}")
    return result_path, csv_path, md_path


class _Timeout:
    def __init__(self, seconds: int, label: str):
        self.seconds = int(seconds or 0)
        self.label = label
        self.enabled = self.seconds > 0 and hasattr(signal, "SIGALRM")
        self.prev_handler = None
        self.prev_alarm = 0

    def __enter__(self):
        if not self.enabled:
            return self
        self.prev_handler = signal.getsignal(signal.SIGALRM)
        self.prev_alarm = signal.alarm(0)

        def _handler(signum, frame):
            raise TimeoutError(f"{self.label} exceeded {self.seconds}s")

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.prev_handler)
            if self.prev_alarm:
                signal.alarm(self.prev_alarm)
        return False


# ── 各 baseline 工厂函数 ──────────────────────────────────────────────────

def run_ha(loaders, adj, cfg, device, save_dir, logger,
           in_dim=None, num_nodes=None, scaler=None, null_val=None):
    from baselines.ha import run_ha as _ha
    logger.info("=" * 52 + "\n[HA] Historical Average")
    _, _, test_loader = loaders
    return _ha(test_loader, device, logger, scaler=scaler, null_val=null_val)


def run_var(loaders, adj, cfg, device, save_dir, logger,
            in_dim=None, num_nodes=None, scaler=None, null_val=None):
    from baselines.var_model import run_var as _var
    logger.info("=" * 52 + "\n[VAR] Vector AutoRegression (AR per node)")
    train_loader, _, test_loader = loaders
    return _var(train_loader, test_loader, cfg.data.T_in, cfg.data.T_out,
                device, logger=logger, scaler=scaler, null_val=null_val)


def run_dcrnn(loaders, adj, cfg, device, save_dir, logger,
              in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.dcrnn import DCRNN
    from baselines.utils import train_model

    logger.info("=" * 52 + "\n[DCRNN] Diffusion Convolutional RNN")
    train_loader, val_loader, test_loader = loaders

    model = DCRNN(in_dim=in_dim, hidden_dim=64, n_layers=2,
                  K=2, out_dim=1, T_out=cfg.data.T_out).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    supports = DCRNN.build_supports(adj)
    supports = [s.to(device) for s in supports]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    return train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        scaler     = scaler,
        null_val   = null_val,
        max_epochs = cfg.train.max_epochs,
        patience   = cfg.train.patience,
        grad_clip  = cfg.train.grad_clip,
        save_path  = os.path.join(save_dir, "dcrnn_best.pt"),
        logger     = logger,
        mc_samples_test = getattr(cfg.train, "mc_samples_test", 200),
        mc_chunk_size   = getattr(cfg.train, "mc_chunk_size", 0),
        extra_forward_kwargs = {"supports": supports},
    )


def run_stgcn(loaders, adj, cfg, device, save_dir, logger,
              in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.stgcn import STGCN, ChebConv
    from baselines.utils import train_model

    logger.info("=" * 52 + "\n[STGCN] Spatio-Temporal Graph Conv Net")
    train_loader, val_loader, test_loader = loaders

    model = STGCN(in_dim=in_dim, hidden_dim=64, kernel_size=3,
                  K=3, n_blocks=2, out_dim=1, T_out=cfg.data.T_out).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    L_tilde = ChebConv.compute_laplacian(adj).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    return train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        scaler     = scaler,
        null_val   = null_val,
        max_epochs = cfg.train.max_epochs,
        patience   = cfg.train.patience,
        grad_clip  = cfg.train.grad_clip,
        save_path  = os.path.join(save_dir, "stgcn_best.pt"),
        logger     = logger,
        mc_samples_test = getattr(cfg.train, "mc_samples_test", 200),
        mc_chunk_size   = getattr(cfg.train, "mc_chunk_size", 0),
        extra_forward_kwargs = {"L_tilde": L_tilde},
    )


def run_mtgnn(loaders, adj, cfg, device, save_dir, logger,
              in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.mtgnn import MTGNN
    from baselines.utils import train_model

    logger.info("=" * 52 + "\n[MTGNN] Multivariate Time Series GNN")
    train_loader, val_loader, test_loader = loaders

    model = MTGNN(
        num_nodes  = num_nodes,
        in_dim     = in_dim,
        hidden_dim = 32,
        skip_dim   = 64,
        end_dim    = 128,
        n_layers   = 3,
        depth      = 2,
        dropout    = 0.3,
        propalpha  = 0.05,
        tanhalpha  = 3.0,
        embed_dim  = 40,
        top_k      = min(20, num_nodes - 1),
        out_dim    = 1,
        T_out      = cfg.data.T_out,
        seq_length = cfg.data.T_in,
    ).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    return train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        scaler     = scaler,
        null_val   = null_val,
        max_epochs = cfg.train.max_epochs,
        patience   = cfg.train.patience,
        grad_clip  = cfg.train.grad_clip,
        save_path  = os.path.join(save_dir, "mtgnn_best.pt"),
        logger     = logger,
        mc_samples_test = getattr(cfg.train, "mc_samples_test", 200),
        mc_chunk_size   = getattr(cfg.train, "mc_chunk_size", 0),
        extra_forward_kwargs = {},
    )


def run_agcrn(loaders, adj, cfg, device, save_dir, logger,
              in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.agcrn import AGCRN
    from baselines.utils import train_model

    logger.info("=" * 52 + "\n[AGCRN] Adaptive Graph Conv RNN")
    train_loader, val_loader, test_loader = loaders

    model = AGCRN(
        num_nodes  = num_nodes,
        in_dim     = in_dim,
        hidden_dim = 64,
        n_layers   = 2,
        embed_dim  = 10,
        cheb_k     = 2,
        out_dim    = 1,
        T_out      = cfg.data.T_out,
    ).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5)

    return train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        scaler     = scaler,
        null_val   = null_val,
        max_epochs = cfg.train.max_epochs,
        patience   = cfg.train.patience,
        grad_clip  = cfg.train.grad_clip,
        save_path  = os.path.join(save_dir, "agcrn_best.pt"),
        logger     = logger,
        mc_samples_test = getattr(cfg.train, "mc_samples_test", 200),
        mc_chunk_size   = getattr(cfg.train, "mc_chunk_size", 0),
    )


def run_stid(loaders, adj, cfg, device, save_dir, logger,
             in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.stid import STID
    from baselines.utils import train_model

    logger.info("=" * 52 + "\n[STID] Spatial-Temporal Identity MLP")
    train_loader, val_loader, test_loader = loaders

    model = STID(
        num_nodes  = num_nodes,
        in_dim     = in_dim,
        T_in       = cfg.data.T_in,
        hidden_dim = 32,
        n_layers   = 3,
        embed_dim  = 32,
        out_dim    = 2,          # mu + log_sigma (Gaussian head)
        T_out      = cfg.data.T_out,
    ).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Gaussian head: out_dim=2 (mu, log_sigma)")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    return train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        scaler     = scaler,
        null_val   = null_val,
        max_epochs = cfg.train.max_epochs,
        patience   = cfg.train.patience,
        grad_clip  = cfg.train.grad_clip,
        save_path  = os.path.join(save_dir, "stid_best.pt"),
        logger     = logger,
        prob       = True,
        mc_samples_test = getattr(cfg.train, "mc_samples_test", 200),
        mc_chunk_size   = getattr(cfg.train, "mc_chunk_size", 0),
    )


def run_csdi(loaders, adj, cfg, device, save_dir, logger,
             in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.csdi import CSDI
    from baselines.utils import compute_prob_metrics, masked_mae

    logger.info("=" * 52 + "\n[CSDI] Conditional Score-based Diffusion")
    train_loader, val_loader, test_loader = loaders

    model = CSDI(
        num_nodes       = num_nodes,
        in_dim          = in_dim,
        T_in            = cfg.data.T_in,
        T_out           = cfg.data.T_out,
        channels        = 64,
        n_layers        = 4,
        nheads          = 8,
        diffusion_steps = 100,
        n_samples       = 10,
        out_dim         = 1,
    ).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    best_val, no_improve = float("inf"), 0
    patience   = cfg.train.patience
    max_epochs = cfg.train.max_epochs
    save_path  = os.path.join(save_dir, "csdi_best.pt")
    history    = {"train_loss": [], "val_mae": []}
    # 验证时最多跑这么多 batch，防止大图上的 DDPM 逆向采样把 val 卡住。
    # 可在 cfg.train 中设置 csdi_val_max_batches=N 覆盖。
    val_max_batches = getattr(cfg.train, "csdi_val_max_batches", 50)

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        n_batches = len(train_loader)
        log_every = max(1, n_batches // 10)
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if (i + 1) % log_every == 0 or i == 0:
                logger.info(f"  [CSDI] Epoch {epoch:3d} | batch {i+1:4d}/{n_batches} | "
                            f"loss={loss.item():.4f} | {time.time() - t0:.1f}s")

        tl = float(np.mean(losses))
        history["train_loss"].append(tl)
        train_time = time.time() - t0

        model.n_samples = 1
        model.eval()
        vm = []
        t_val = time.time()
        with torch.no_grad():
            for batch_i, (x, y) in enumerate(val_loader):
                if batch_i >= val_max_batches:
                    break
                x, y = x.to(device), y.to(device)
                pred = model(x)                              # [B, T_out, N, 1]
                vm.append(masked_mae(pred, y[..., :1], null_val).item())
        val_mae = float(np.mean(vm))
        model.n_samples = 10
        history["val_mae"].append(val_mae)

        try:
            scheduler.step(val_mae)
        except TypeError:
            scheduler.step()

        if val_mae < best_val:
            best_val, no_improve = val_mae, 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        logger.info(f"  Epoch {epoch:3d} | train={tl:.4f} | val={val_mae:.4f} | "
                    f"best={best_val:.4f} | train={train_time:.1f}s | "
                    f"val={time.time() - t_val:.1f}s")
        if no_improve >= patience:
            logger.info(f"  Early stop @ epoch {epoch}")
            break

    # 测试
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    samples_list, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            samples_list.append(model.sample(x, return_samples=True).cpu())
            trues.append(y.cpu().permute(0, 2, 1, 3))  # [B, N, T_out, F]

    samples_all = torch.cat(samples_list, dim=1).numpy()  # [S, total, N, T_out, F]
    true_all = torch.cat(trues, dim=0).numpy()            # [total, N, T_out, F]

    test_m_norm = compute_prob_metrics(samples_all, true_all)

    null_val_eval = None
    if scaler is not None:
        shape = samples_all.shape
        samples_all = scaler.inverse_transform(samples_all.reshape(-1)).reshape(shape)
        true_all = scaler.inverse_transform(true_all.reshape(-1)).reshape(true_all.shape)

    test_m = compute_prob_metrics(samples_all, true_all, null_val_eval)
    for k, v in test_m_norm.items():
        test_m[f"{k}_norm"] = v
    mae, rmse, mape = test_m["MAE"], test_m["RMSE"], test_m["MAPE"]
    logger.info(f"  [Test] 归一化域   MAE={test_m['MAE_norm']:.4f}  RMSE={test_m['RMSE_norm']:.4f}  "
                f"MAPE={test_m['MAPE_norm']:.2f}%  CRPS={test_m['CRPS_norm']:.4f}")
    logger.info(f"  [Test] 反归一化域 MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  "
                f"CRPS={test_m['CRPS']:.4f}")
    T_out = cfg.data.T_out
    logger.info(f"\n  {'Step':<6}  {'MAE':>8}  {'RMSE':>8}  {'CRPS':>8}")
    for h in range(T_out):
        mae_h  = test_m.get(f"MAE_h{h+1}",  float("nan"))
        rmse_h = test_m.get(f"RMSE_h{h+1}", float("nan"))
        crps_h = test_m.get(f"CRPS_h{h+1}", float("nan"))
        logger.info(f"  h={h+1:<4}  {mae_h:>8.4f}  {rmse_h:>8.4f}  {crps_h:>8.4f}")
    history.update({"test_metrics": test_m, "test_mae": mae,
                    "test_rmse": rmse, "test_mape": mape,
                    "test_crps": test_m["CRPS"]})
    return history


def run_tsflow(loaders, adj, cfg, device, save_dir, logger,
               in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.tsflow import run_tsflow as _tsflow
    logger.info("=" * 52 + "\n[TSFlow] CFM + GP(OU) prior (no graph)")
    return _tsflow(loaders, adj, cfg, device, save_dir, logger,
                   in_dim=in_dim, num_nodes=num_nodes,
                   scaler=scaler, null_val=null_val)


def run_k2vae(loaders, adj, cfg, device, save_dir, logger,
              in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.k2vae import run_k2vae as _k2vae
    logger.info("=" * 52 + "\n[K2VAE] Koopman-Kalman VAE (no graph)")
    return _k2vae(loaders, adj, cfg, device, save_dir, logger,
                  in_dim=in_dim, num_nodes=num_nodes,
                  scaler=scaler, null_val=null_val)


def run_patchtst(loaders, adj, cfg, device, save_dir, logger,
                 in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.patchtst import run_patchtst as _patchtst
    logger.info("=" * 52 + "\n[PatchTST] Patch Time Series Transformer (no graph)")
    return _patchtst(loaders, adj, cfg, device, save_dir, logger,
                     in_dim=in_dim, num_nodes=num_nodes,
                     scaler=scaler, null_val=null_val)


def run_tsdiff(loaders, adj, cfg, device, save_dir, logger,
               in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.tsdiff import run_tsdiff as _tsdiff
    logger.info("=" * 52 + "\n[TSDiff] Unconditional Diffusion + Replacement Guidance (no graph)")
    return _tsdiff(loaders, adj, cfg, device, save_dir, logger,
                   in_dim=in_dim, num_nodes=num_nodes,
                   scaler=scaler, null_val=null_val)


def run_diffstg(loaders, adj, cfg, device, save_dir, logger,
                in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.diffstg import run_diffstg as _diffstg
    logger.info("=" * 52 + "\n[DiffSTG] DDPM + UGnet (graph-aware)")
    return _diffstg(loaders, adj, cfg, device, save_dir, logger,
                    in_dim=in_dim, num_nodes=num_nodes,
                    scaler=scaler, null_val=null_val)


# ── 模型注册表 ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "ha":       run_ha,
    "var":      run_var,
    "diffstg":  run_diffstg,
    "dcrnn":    run_dcrnn,
    "stgcn":    run_stgcn,
    "mtgnn":    run_mtgnn,
    "agcrn":    run_agcrn,
    "stid":     run_stid,
    "tsflow":   run_tsflow,
    "patchtst": run_patchtst,
    "k2vae":    run_k2vae,
    "tsdiff":   run_tsdiff,
    "csdi":     run_csdi,
}


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GridCFN Baselines Runner (Multi-Step)")
    parser.add_argument("--preset", type=str, default="solar",
                        choices=["solar", "electricity", "weather", "sdwpf", "pjm"])
    parser.add_argument("--models", nargs="+",
                        default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help="要运行的 baseline 列表")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--model_timeout_minutes", type=float, default=0.0,
                        help="单个 baseline 的最长运行分钟数；0 表示不启用超时")
    args = parser.parse_args()

    cfg     = get_config(args.preset)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = cfg.data.dataset
    save_dir = os.path.join("result", "baselines", dataset, ts)
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(
        os.path.join(save_dir, f"baselines_{dataset}_{ts}.log"))
    set_seed(cfg.train.seed)
    device = get_device(args.gpu_id)

    logger.info(f"Dataset  : {dataset}")
    logger.info(f"T_out    : {cfg.data.T_out} (multi-step)")
    logger.info(f"Device   : {device}")
    logger.info(f"Models   : {args.models}")
    logger.info(f"Timeout  : {args.model_timeout_minutes} min per model"
                if args.model_timeout_minutes > 0 else "Timeout  : disabled")
    logger.info(f"Save dir : {save_dir}")

    # 加载数据
    train_loader, val_loader, test_loader, adj, scaler, in_dim = load_data(cfg)
    num_nodes = adj.shape[0]
    loaders   = (train_loader, val_loader, test_loader)
    logger.info(f"Nodes={num_nodes}, in_dim={in_dim}, T_in={cfg.data.T_in}")

    # 统一无 mask：null_val=None，全量评估，与 GridCFN 对齐
    null_val = _get_null_val(dataset, scaler)
    logger.info(f"null_val: None (无 mask，全量评估，与 GridCFN 统一)")

    all_results = {}
    timeout_seconds = int(args.model_timeout_minutes * 60)

    for model_name in args.models:
        try:
            fn = MODEL_REGISTRY[model_name]
            # 为 baseline 调用准备一个本地 cfg 副本，按 dataset 注入 baseline 专用的安全默认值，
            # 避免修改全局 `cfg`（保持你的方法配置不受影响）。
            local_cfg = deepcopy(cfg)
            dataset = getattr(local_cfg.data, "dataset", "").lower()
            if model_name == "tsdiff":
                # model-level diffusion_steps 优先使用用户配置，否则按 dataset 选默认
                if not hasattr(local_cfg.model, "diffusion_steps") or getattr(local_cfg.model, "diffusion_steps") is None:
                    if dataset in ("sdwpf", "electricity"):
                        local_cfg.model.diffusion_steps = 20
                    elif dataset == "weather":
                        local_cfg.model.diffusion_steps = 50
                    else:
                        local_cfg.model.diffusion_steps = 100

                # train-level val/test batch 限制：若未配置则按 dataset 选较小上限以加速验证
                if not hasattr(local_cfg.train, "tsdiff_val_max_batches") or getattr(local_cfg.train, "tsdiff_val_max_batches") is None:
                    local_cfg.train.tsdiff_val_max_batches = 2 if dataset in ("sdwpf", "electricity") else 50
                if not hasattr(local_cfg.train, "tsdiff_test_max_batches") or getattr(local_cfg.train, "tsdiff_test_max_batches") is None:
                    local_cfg.train.tsdiff_test_max_batches = 10 if dataset in ("sdwpf", "electricity") else 50

                # 打印实际注入的参数，便于确认运行时使用的默认值
                logger.info(f"  [TSDiff run defaults] dataset={dataset} | diffusion_steps={local_cfg.model.diffusion_steps} | "
                            f"val_max_batches={local_cfg.train.tsdiff_val_max_batches} | "
                            f"test_max_batches={local_cfg.train.tsdiff_test_max_batches}")

            with _Timeout(timeout_seconds, model_name):
                result = fn(loaders, adj, local_cfg, device, save_dir, logger,
                            in_dim=in_dim, num_nodes=num_nodes,
                            scaler=scaler, null_val=null_val)
            all_results[model_name] = result

            if isinstance(result, dict):
                if "prediction_inv" in result:
                    np.save(
                        os.path.join(save_dir, f"{model_name}_prediction.npy"),
                        result["prediction_inv"]
                    )
                if "ground_truth_inv" in result:
                    gt_path = os.path.join(save_dir, "ground_truth.npy")
                    if not os.path.exists(gt_path):
                        np.save(gt_path, result["ground_truth_inv"])

            save_summaries(all_results, save_dir, dataset, ts, logger=logger)
        except Exception as e:
            logger.error(f"[{model_name}] FAILED: {e}", exc_info=True)
            all_results[model_name] = {"error": str(e)}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            save_summaries(all_results, save_dir, dataset, ts, logger=logger)

    # ── 汇总表 ────────────────────────────────────────────────────────────
    W = 84
    logger.info("\n" + "=" * W)
    logger.info(f"{'Model':<12} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>10} {'CRPS':>10}  (反归一化域)")
    logger.info("-" * W)
    for name, res in all_results.items():
        if "error" in res:
            logger.info(f"{name:<12}  ERROR: {res['error']}")
        else:
            metrics = res.get("test_metrics", {}) if isinstance(res, dict) else {}
            mae  = res.get("test_mae",  metrics.get("MAE",  float("nan")))
            rmse = res.get("test_rmse", metrics.get("RMSE", float("nan")))
            mape = res.get("test_mape", metrics.get("MAPE", float("nan")))
            crps = res.get("test_crps", metrics.get("CRPS", float("nan")))
            logger.info(f"{name:<12} {mae:>10.4f} {rmse:>10.4f} {mape:>10.2f} {crps:>10.4f}")
    logger.info("=" * W)

    logger.info(f"{'Model':<12} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>10} {'CRPS':>10}  (归一化域)")
    logger.info("-" * W)
    for name, res in all_results.items():
        if "error" in res:
            logger.info(f"{name:<12}  ERROR: {res['error']}")
        else:
            metrics = res.get("test_metrics", {}) if isinstance(res, dict) else {}
            mae  = res.get("test_mae_norm",  metrics.get("MAE_norm",  float("nan")))
            rmse = res.get("test_rmse_norm", metrics.get("RMSE_norm", float("nan")))
            mape = res.get("test_mape_norm", metrics.get("MAPE_norm", float("nan")))
            crps = res.get("test_crps_norm", metrics.get("CRPS_norm", float("nan")))
            logger.info(f"{name:<12} {mae:>10.4f} {rmse:>10.4f} {mape:>10.2f} {crps:>10.4f}")
    logger.info("=" * W)

    save_summaries(all_results, save_dir, dataset, ts, logger=logger)


if __name__ == "__main__":
    main()
