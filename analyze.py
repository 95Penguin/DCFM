#!/usr/bin/env python3
"""
analyze.py — GridCFN 论文 Section 4.5 (概率校准) 和 4.6 (案例分析) 可视化脚本

用法:
  # GridCFN 校准分析
  python analyze.py --checkpoint result/sdwpf/20260613_105929/gridcfn_sdwpf_20260613_105929.pt \\
                    --preset sdwpf --output_dir analysis_figures

  # 含 DiffSTG 对比的案例分析
  python analyze.py --checkpoint <gridcfn_ckpt> --preset sdwpf \\
                    --diffstg_checkpoint <diffstg_ckpt> \\
                    --output_dir analysis_figures

  # 指定 T_out（如果 checkpoint 与 preset 默认不同）
  python analyze.py --checkpoint <ckpt> --preset solar --T_out 24
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from matplotlib.patches import Polygon

from config import Config, get_config
from dataset import load_solar_energy, load_electricity, load_weather, load_sdwpf, load_pjm
from model import GridCFN
from train import (
    evaluate as gridcfn_evaluate,
    calibrate_temperature,
    evaluate_all,
    picp_empirical,
    pinaw_empirical,
    crps_empirical,
)

# ───────────────────────────────────────────────────────────────────────────
# 全局样式
# ───────────────────────────────────────────────────────────────────────────

BASE_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

COLORS = {
    "gridcfn": "#1f77b4",
    "gridcfn_ci": "#1f77b4",
    "diffstg": "#d62728",
    "diffstg_ci": "#d62728",
    "ground_truth": "#2ca02c",
    "calibration": "#9467bd",
}


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GridCFN 校准 / 案例分析")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="GridCFN checkpoint .pt 路径")
    p.add_argument("--preset", type=str, default="solar",
                   choices=["solar", "electricity", "weather", "sdwpf", "pjm"])
    p.add_argument("--T_out", type=int, default=None,
                   help="预测步长，覆盖 preset 默认值")
    p.add_argument("--output_dir", type=str, default="analysis_figures",
                   help="输出目录")
    p.add_argument("--diffstg_checkpoint", type=str, default=None,
                   help="DiffSTG 模型 checkpoint（可选，用于 4.6 对比）")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda:0", "cuda:1"])
    p.add_argument("--n_samples", type=int, default=200,
                   help="CFM 采样数 (默认 200)")
    p.add_argument("--n_steps", type=int, default=20,
                   help="CFM ODE 步数 (默认 20)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ───────────────────────────────────────────────────────────────────────────
# 设备 / 种子
# ───────────────────────────────────────────────────────────────────────────

def resolve_device(device_str: str):
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ───────────────────────────────────────────────────────────────────────────
# 数据加载
# ───────────────────────────────────────────────────────────────────────────

DATASET_LOADERS = {
    "solar": load_solar_energy,
    "electricity": load_electricity,
    "weather": load_weather,
    "sdwpf": load_sdwpf,
    "pjm": load_pjm,
}


def load_data(cfg):
 d = cfg.data
 loader_fn = DATASET_LOADERS.get(d.dataset)
 if loader_fn is None:
     raise ValueError(f"未知数据集: {d.dataset}")
 train_loader, val_loader, test_loader, adj, scaler, in_dim = \
     loader_fn(d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size)
 def _rebind(loader):
     from torch.utils.data import DataLoader
     if loader.num_workers > 0:
         return DataLoader(
             loader.dataset, batch_size=loader.batch_size, shuffle=False,
             num_workers=0, pin_memory=False, persistent_workers=False,
         )
     return loader
 logging.info("DataLoaders wrapped with num_workers=0 (sandbox-safe mode)")
 return _rebind(train_loader), _rebind(val_loader), _rebind(test_loader), adj, scaler, in_dim


# ───────────────────────────────────────────────────────────────────────────
# 模型加载
# ───────────────────────────────────────────────────────────────────────────

def build_model(cfg, in_dim, n_nodes, wind_mask=None, device=None):
    m = cfg.model
    model = GridCFN(
        n_nodes=n_nodes,
        in_dim=in_dim if in_dim is not None else m.in_dim,
        gcn_hidden=m.gcn_hidden, gcn_layers=m.gcn_layers,
        tcn_hidden=m.tcn_hidden, tcn_layers=m.tcn_layers,
        env_dim=m.env_dim, stoch_dim=m.stoch_dim, ms_out_dim=m.ms_out_dim,
        n_scg_layers=m.n_scg_layers, out_dim=m.out_dim, lambda_mi=m.lambda_mi,
        cfm_hidden=m.cfm_hidden, cfm_time_emb_dim=m.cfm_time_emb_dim,
        chunk_size=m.chunk_size, ms_dilations=m.ms_dilations,
        T_out=cfg.data.T_out, T_in=cfg.data.T_in,
        rank_r=m.rank_r, lambda_rank=m.lambda_rank,
        freq_candidates=m.freq_candidates,
        wind_mask=wind_mask,
    )
    if device is not None:
        model = model.to(device)
    return model


def load_model_from_checkpoint(checkpoint_path, cfg, in_dim, n_nodes, wind_mask, device):
    model = build_model(cfg, in_dim, n_nodes, wind_mask=wind_mask, device=device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # 处理 state_dict 前缀（DDP 包装等情况）
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    logging.info(f"Model loaded from {checkpoint_path}")
    return model


# ───────────────────────────────────────────────────────────────────────────
# DiffSTG 模型加载
# ───────────────────────────────────────────────────────────────────────────

def load_diffstg_from_checkpoint(checkpoint_path, cfg, in_dim, n_nodes, device):
    from baselines.diffstg import DiffSTG
    # 确定 out_feat
    d = cfg.data
    _, _, _, adj, _, _ = load_data(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_loader, _, _, adj, _, _ = load_data(cfg)
    for _, y_batch in train_loader:
        out_feat = y_batch.shape[3]
        break
    del train_loader

    model = DiffSTG(
        in_dim=in_dim, out_feat=out_feat,
        T_in=d.T_in, T_out=d.T_out,
        n_nodes=n_nodes, hidden_size=min(64, n_nodes),
        N=200, diff_emb_dim=128, n_down=2, kernel_size=3,
    ).to(device)
    model.scheduler.to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    agcn = DiffSTG.build_agcn(adj_t).to(device)
    logging.info(f"DiffSTG model loaded from {checkpoint_path}")
    return model, agcn


# ───────────────────────────────────────────────────────────────────────────
# 辅助：构建 wind_mask
# ───────────────────────────────────────────────────────────────────────────

def _build_wind_mask(cfg, n_nodes):
    wind_mask = None
    if cfg.data.dataset == "sdwpf":
        from wind_mask_utils import build_wind_mask_from_csv
        coord_path = os.path.join(os.path.dirname(cfg.data.data_path),
                                  "sdwpf_turb_location.csv")
        if os.path.exists(coord_path):
            wind_mask = build_wind_mask_from_csv(coord_path, angle_tol=45.0)
            if wind_mask.shape[0] != n_nodes:
                raise ValueError(
                    f"wind_mask 台数 {wind_mask.shape[0]} != 节点数 {n_nodes}"
                )
    return wind_mask


# ───────────────────────────────────────────────────────────────────────────
# GridCFN 推理（含 Temperature Calibration）
# ───────────────────────────────────────────────────────────────────────────

def run_gridcfn_inference(model, train_loader, val_loader, test_loader,
                          adj_norm, edge_index, device, scaler,
                          n_samples=200, n_steps=20, sigma_min=0.01, x0_scale=1.0):
    """运行 inference，返回 calibrated samples 和 ground truth。"""

    # ── Temperature Calibration ──
    logging.info("Temperature calibration on validation set...")
    _, samples_val_norm, y_val_norm = gridcfn_evaluate(
        model, val_loader, adj_norm, edge_index, device,
        scaler=scaler, return_preds=True,
        n_samples=n_samples, n_steps=n_steps,
        temperature=1.0, inverse_transform=False,
        sigma_min=sigma_min, x0_scale=x0_scale,
    )
    best_T = calibrate_temperature(samples_val_norm, y_val_norm, target_coverage=0.95)
    logging.info(f"Optimal Temperature: {best_T:.3f}")

    # ── Test set inference ──
    logging.info("Test set inference (this may take a while)...")
    model.eval()
    samples_list, y_list, x_list = [], [], []
    adj_norm_ = adj_norm.to(device)
    edge_idx_ = edge_index.to(device)
    for xb, yb in test_loader:
        xb = xb.to(device)
        He_prime, Hs_prime, *_ = model(xb, adj_norm_, edge_idx_)
        raw_samples = model.sample(He_prime, Hs_prime, n_samples=n_samples, n_steps=n_steps,
                                   sigma_min=sigma_min, x0_scale=x0_scale)
        x_list.append(xb.detach().cpu().numpy())
        samples_list.append(raw_samples.cpu().numpy())
        y_list.append(yb.permute(0, 2, 1, 3).numpy())
        del He_prime, Hs_prime, xb
    if device.type == "cuda":
        torch.cuda.empty_cache()
    samples_test_norm = np.concatenate(samples_list, axis=1)
    y_test_norm = np.concatenate(y_list, axis=0)
    x_test_norm = np.concatenate(x_list, axis=0)
    logging.info(f"Test set collected: samples {samples_test_norm.shape}, x {x_test_norm.shape}")
def run_diffstg_inference(model, agcn, test_loader, device, scaler,
                          n_samples=200, n_steps=200):
    from baselines.utils import compute_prob_metrics
    model.eval()
    samples_list, y_list = [], []
    for x, y in test_loader:
        x = x.to(device)
        raw = model.sample(x, agcn, n_samples=n_samples, n_steps=n_steps)
        samples_list.append(raw.cpu().numpy())
        y_list.append(y.permute(0, 2, 1, 3).numpy())
    samples_all = np.concatenate(samples_list, axis=1)
    y_all = np.concatenate(y_list, axis=0)

    # inverse transform
    shape = samples_all.shape
    s_mean = scaler.mean[..., :1] if scaler.mean.shape[-1] > 1 else scaler.mean
    s_std = scaler.std[..., :1] if scaler.std.shape[-1] > 1 else scaler.std
    samples_inv = (samples_all.reshape(-1) * s_std + s_mean).reshape(shape)
    y_inv = (y_all.reshape(-1) * s_std + s_mean).reshape(y_all.shape)
    return samples_inv, y_inv


# ═══════════════════════════════════════════════════════════════════════════
# Section 4.5 — 概率校准分析
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# 4.5a — Reliability Diagram
# ───────────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(samples, y, output_dir, confidence_levels=None):
    """
    Reliability diagram: for each nominal confidence level (e.g. 0.5~0.99),
    compute empirical coverage (PICP).
    """
    if confidence_levels is None:
        confidence_levels = np.arange(0.5, 0.995, 0.025)

    empirical_cov = []
    for conf in confidence_levels:
        picp = picp_empirical(samples, y, confidence=conf)
        empirical_cov.append(picp)
    empirical_cov = np.array(empirical_cov)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.plot(confidence_levels, empirical_cov, "o-",
            color=COLORS["gridcfn"], label="GridCFN", markersize=4)
    ax.plot([0.5, 1.0], [0.5, 1.0], "--", color="gray", linewidth=1,
            label="Perfect calibration")

    # 填充偏差区域
    ax.fill_between(confidence_levels,
                    np.minimum(confidence_levels, empirical_cov),
                    np.maximum(confidence_levels, empirical_cov),
                    alpha=0.1, color=COLORS["gridcfn"])

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage (PICP)")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "reliability_diagram.png")
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"Saved: {path}")

    # 计算 ECE
    ece = np.mean(np.abs(empirical_cov - confidence_levels))
    logging.info(f"Expected Calibration Error (ECE): {ece:.4f}")
    return fig


# ───────────────────────────────────────────────────────────────────────────
# 4.5b — PICP / PINAW vs 预测步长
# ───────────────────────────────────────────────────────────────────────────

def plot_picp_pinaw_by_horizon(samples, y, output_dir):
    T_out = y.shape[2]
    picp_h, pinaw_h = [], []
    for h in range(T_out):
        s_h = samples[:, :, :, h, :]
        y_h = y[:, :, h, :]
        picp_h.append(picp_empirical(s_h, y_h, confidence=0.9))
        pinaw_h.append(pinaw_empirical(s_h, y_h, confidence=0.9))
    picp_h = np.array(picp_h)
    pinaw_h = np.array(pinaw_h)

    # 再加 95% CI
    picp_95_h = []
    for h in range(T_out):
        s_h = samples[:, :, :, h, :]
        y_h = y[:, :, h, :]
        picp_95_h.append(picp_empirical(s_h, y_h, confidence=0.95))
    picp_95_h = np.array(picp_95_h)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    steps = np.arange(1, T_out + 1)

    # PICP
    ax1.plot(steps, picp_h, "o-", color=COLORS["gridcfn"], label="90% CI",
             markersize=4)
    ax1.plot(steps, picp_95_h, "s--", color=COLORS["calibration"],
             label="95% CI", markersize=4)
    ax1.axhline(0.9, color="gray", linestyle=":", alpha=0.5,
                label="Target 90%")
    ax1.set_xlabel("Prediction horizon")
    ax1.set_ylabel("PICP")
    ax1.set_title("PICP vs Prediction Horizon")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # PINAW
    ax2.plot(steps, pinaw_h, "o-", color=COLORS["gridcfn"], markersize=4)
    ax2.set_xlabel("Prediction horizon")
    ax2.set_ylabel("PINAW")
    ax2.set_title("PINAW vs Prediction Horizon")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "picp_pinaw_by_horizon.png")
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"Saved: {path}")
    return fig


# ───────────────────────────────────────────────────────────────────────────
# 4.5c — 预测分布可视化（典型场景）
# ───────────────────────────────────────────────────────────────────────────

def plot_distribution_example(samples, y, output_dir, n_examples=3):
    """
    选几个典型 (节点, 时间步) 展示预测分布直方图 vs 真实值。
    """
    S, B, N, T_out, F = samples.shape
    choices = []
    rng = np.random.RandomState(42)

    # 随机选 n_examples 个 (batch, node, step) 组合
    for _ in range(n_examples):
        b = rng.randint(0, B)
        n = rng.randint(0, N)
        t = rng.randint(0, T_out)
        choices.append((b, n, t))

    fig, axes = plt.subplots(1, n_examples, figsize=(4.5 * n_examples, 3.5))
    if n_examples == 1:
        axes = [axes]

    for ax, (b, n, t) in zip(axes, choices):
        s_vals = samples[:, b, n, t, 0]
        truth = y[b, n, t, 0]
        mu_pred = s_vals.mean()

        ax.hist(s_vals, bins=40, color=COLORS["gridcfn"], alpha=0.7,
                density=True, edgecolor="white", linewidth=0.3)
        ax.axvline(truth, color=COLORS["ground_truth"], linewidth=2,
                   label=f"Truth={truth:.2f}")
        ax.axvline(mu_pred, color=COLORS["gridcfn"], linestyle="--",
                   linewidth=1.5, label=f"Mean={mu_pred:.2f}")

        # 90% CI
        lo, hi = np.quantile(s_vals, [0.05, 0.95])
        ax.axvspan(lo, hi, alpha=0.15, color=COLORS["gridcfn"],
                   label="90% CI")

        ax.set_xlabel("Predicted value")
        ax.set_ylabel("Density")
        ax.set_title(f"Node {n}, Step {t+1}")
        ax.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(output_dir, "distribution_example.png")
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"Saved: {path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Section 4.6 — 案例分析
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# 辅助：选取极端场景
# ───────────────────────────────────────────────────────────────────────────

def _find_extreme_scenarios(y, top_k=2):
    """
    从 test set 中找到波动最大 / 骤降最剧烈的 (batch, node) 组合。
    返回 [(b, n, score, label), ...]。
    """
    B, N, T_out, F = y.shape
    scenarios = []

    for b in range(B):
        for n in range(N):
            ts = y[b, n, :, 0]
            # 波动剧烈程度：相邻差分绝对值之和
            fluctuation = np.sum(np.abs(np.diff(ts)))
            # 骤降剧烈程度：最大下降幅度
            drops = -np.diff(ts)
            max_drop = np.max(drops) if len(drops) > 0 else 0

            scenarios.append({
                "b": b, "n": n,
                "fluctuation": fluctuation,
                "max_drop": max_drop,
            })

    # 按波动排序
    scenarios.sort(key=lambda x: x["fluctuation"], reverse=True)
    top_fluctuation = [
        (s["b"], s["n"], s["fluctuation"], "Strong Fluctuation")
        for s in scenarios[:top_k]
    ]

    # 按最大降幅排序
    scenarios.sort(key=lambda x: x["max_drop"], reverse=True)
    top_sharp_drop = [
        (s["b"], s["n"], s["max_drop"], "Sharp Drop")
        for s in scenarios[:top_k]
    ]

    return top_fluctuation + top_sharp_drop


# ───────────────────────────────────────────────────────────────────────────
# 4.6 — 主绘图
# ───────────────────────────────────────────────────────────────────────────

def plot_case_study(samples_gridcfn, y, output_dir,
                    samples_diffstg=None, x_history=None, tin=168):
    scenarios = _find_extreme_scenarios(y, top_k=2)

    n_rows = len(scenarios)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 2.8 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for ax, (b, n, score, label) in zip(axes, scenarios):
        ts_truth = y[b, n, :, 0]  # (T_out,)

        # GridCFN
        gc_samples = samples_gridcfn[:, b, n, :, 0]  # (S, T_out)
        gc_mean = gc_samples.mean(axis=0)
        gc_lo = np.quantile(gc_samples, 0.05, axis=0)
        gc_hi = np.quantile(gc_samples, 0.95, axis=0)

        # ── History context ──
        if x_history is not None:
            n_hist = min(tin, x_history.shape[1])
            ts_hist = x_history[b, :n_hist, n, 0]
            hist_steps = np.arange(-n_hist, 0)
            ax.plot(hist_steps, ts_hist, "-", color="gray", linewidth=1.5,
                    label="Input history")
            # vertical separator
            ax.axvline(-0.5, color="gray", linestyle=":", linewidth=0.8)

        fore_steps = np.arange(len(ts_truth))
        all_steps = np.arange(-n_hist if x_history is not None else 0, len(ts_truth))

        # GridCFN CI
        ax.fill_between(fore_steps, gc_lo, gc_hi, alpha=0.25,
                        color=COLORS["gridcfn_ci"], label="GridCFN 90% CI")
        ax.plot(fore_steps, gc_mean, "o-", color=COLORS["gridcfn"],
                label="GridCFN Mean", markersize=3, linewidth=1.5)

        # DiffSTG 对比（可选）
        if samples_diffstg is not None:
            d_samples = samples_diffstg[:, b, n, :, 0]
            d_mean = d_samples.mean(axis=0)
            d_lo = np.quantile(d_samples, 0.05, axis=0)
            d_hi = np.quantile(d_samples, 0.95, axis=0)

            ax.fill_between(fore_steps, d_lo, d_hi, alpha=0.2,
                            color=COLORS["diffstg_ci"], label="DiffSTG 90% CI")
            ax.plot(fore_steps, d_mean, "s--", color=COLORS["diffstg"],
                    label="DiffSTG Mean", markersize=3, linewidth=1.2)

        ax.plot(fore_steps, ts_truth, "-", color=COLORS["ground_truth"],
                label="Ground Truth", linewidth=2)

        ax.set_xlabel("Time step (within forecast horizon)")
        ax.set_ylabel("Value")
        ax.set_title(f"{label} — Batch {b}, Node {n} "
                     f"(score={score:.2f})")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "case_study.png")
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"Saved: {path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "analyze.log")),
        ],
    )
    logger = logging.getLogger("analyze")

    # ── Load config ──
    cfg = get_config(args.preset)
    if args.T_out is not None:
        cfg.data.T_out = args.T_out

    logger.info(f"Preset: {args.preset}, T_out={cfg.data.T_out}")
    logger.info(f"Output dir: {args.output_dir}")

    # ── Load data ──
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, adj, scaler, in_dim = load_data(cfg)

    adj_norm = GridCFN.normalize_adj(adj)
    edge_index = GridCFN.adj_to_edge_index(adj)
    n_nodes = adj.shape[0]
    logger.info(f"Graph: {n_nodes} nodes")

    wind_mask = _build_wind_mask(cfg, n_nodes)

    # ── Load GridCFN model ──
    logger.info(f"Loading GridCFN from {args.checkpoint}...")
    model_gc = load_model_from_checkpoint(
        args.checkpoint, cfg, in_dim, n_nodes, wind_mask, device
    )

    # ── GridCFN inference ──
    samples_gc, y_test, x_test = run_gridcfn_inference(
        model_gc, train_loader, val_loader, test_loader,
        adj_norm, edge_index, device, scaler,
        n_samples=args.n_samples, n_steps=args.n_steps,
    )
    logger.info(f"GridCFN samples shape: {samples_gc.shape}")
    logger.info(f"Input x shape: {x_test.shape}")

    # ── DiffSTG inference (optional) ──
    samples_diffstg = None
    if args.diffstg_checkpoint:
        logger.info("Loading DiffSTG model...")
        model_ds, agcn = load_diffstg_from_checkpoint(
            args.diffstg_checkpoint, cfg, in_dim, n_nodes, device
        )
        samples_diffstg, _ = run_diffstg_inference(
            model_ds, agcn, test_loader, device, scaler,
            n_samples=args.n_samples, n_steps=200,
        )
        logger.info(f"DiffSTG samples shape: {samples_diffstg.shape}")

    # ══════════════════════════════════════════════════════════════════════
    # Section 4.5 — 概率校准分析
    # ══════════════════════════════════════════════════════════════════════
    with plt.style.context(BASE_STYLE):
        logger.info("\n" + "=" * 52)
        logger.info("Section 4.5 — Probability Calibration Analysis")
        logger.info("=" * 52)

        # 4.5a — Reliability diagram
        logger.info("[4.5a] Reliability diagram...")
        plot_reliability_diagram(samples_gc, y_test, args.output_dir)

        # 4.5b — PICP / PINAW vs horizon
        logger.info("[4.5b] PICP / PINAW by horizon...")
        plot_picp_pinaw_by_horizon(samples_gc, y_test, args.output_dir)

        # 4.5c — Distribution example
        logger.info("[4.5c] Distribution examples...")
        plot_distribution_example(samples_gc, y_test, args.output_dir)

    # ══════════════════════════════════════════════════════════════════════
    # Section 4.6 — 案例分析
    # ══════════════════════════════════════════════════════════════════════
    with plt.style.context(BASE_STYLE):
        logger.info("\n" + "=" * 52)
        logger.info("Section 4.6 — Case Study")
        logger.info("=" * 52)

        logger.info("[4.6] Extreme scenario analysis...")
        plot_case_study(samples_gc, y_test, args.output_dir,
                        x_history=x_test, tin=cfg.data.T_in,
                        samples_diffstg=samples_diffstg)

    # ── 打印最终指标摘要 ──
    metrics = evaluate_all(samples_gc, y_test)
    logger.info("\n" + "=" * 52)
    logger.info("Final Metrics on Test Set (calibrated, physical domain)")
    logger.info("=" * 52)
    for k in ["MAE", "RMSE", "MAPE", "CRPS", "PICP", "PINAW"]:
        logger.info(f"  {k:<8}: {metrics[k]:.4f}")

    logger.info(f"\nAll figures saved to: {os.path.abspath(args.output_dir)}")
    print(f"\nDone. All figures saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
