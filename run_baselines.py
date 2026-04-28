"""
run_baselines.py – 一键运行所有对比算法
用法：
  python run_baselines.py --preset solar
  python run_baselines.py --preset electricity
  python run_baselines.py --preset weather
  python run_baselines.py --preset solar --models ha var dcrnn stgcn mtgnn agcrn
  python run_baselines.py --preset solar --gpu_id 0   # 指定 GPU
  uv run run_baselines.py --preset solar --gpu_id 0 --models ha var 

修改说明（v2）：
  [Fix-1] 结果目录改为 result/<dataset>/baselines/<timestamp>/
          与 GridCFN 的 result/<dataset>/<timestamp>/ 平行，不混在一起
  [Fix-2] get_device() 新增 --gpu_id 参数说明：
            gpu_id = -1（默认）→ 自动选 cuda:0（有 GPU 就用）
            gpu_id =  0,1,2... → 指定某块 GPU
          命令行传 --gpu_id 0 即可启用第一块 GPU
  [Fix-3] 指标域对齐：与 GridCFN train.py 一致，同时输出
            · 归一化域（normalized）：用于与论文 Table II 数值直接对比
            · 反 Z-score 域（unnormalized）：实际物理量纲
          之前只输出反归一化域，与论文数值差距 ~50x，导致误解
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

from config import get_config
from dataset import load_solar_energy, load_electricity, load_weather
from model import GridCFN
from baselines import (
    run_ha, run_var,
    build_baseline,
    train_deep_baseline,
    evaluate_deep_baseline,
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id: int) -> torch.device:
    """
    [Fix-2] GPU 选择逻辑：
      · 无 CUDA → CPU
      · gpu_id == -1 → 自动选 cuda:0（有 GPU 就用）
      · gpu_id >= 0  → 指定 cuda:gpu_id
    命令行加 --gpu_id 0 即可启用第一块 GPU。
    """
    if not torch.cuda.is_available():
        print("[Device] 未检测到 CUDA，使用 CPU")
        return torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    if gpu_id < 0:
        idx = 0
    elif gpu_id >= n_gpu:
        raise ValueError(f"--gpu_id={gpu_id} 超出范围，共 {n_gpu} 块 GPU，请传 0~{n_gpu-1}")
    else:
        idx = gpu_id
    name = torch.cuda.get_device_name(idx)
    mem  = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
    print(f"[Device] 使用 GPU {idx}: {name} ({mem:.1f} GB)")
    return torch.device(f"cuda:{idx}")


def load_data(cfg):
    d = cfg.data
    assert d.data_path, "data_path 未设置"
    if d.dataset == "solar":
        return load_solar_energy(d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size)
    elif d.dataset == "electricity":
        return load_electricity(d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size)
    elif d.dataset == "weather":
        return load_weather(d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size,
                            feature_idx=getattr(d, "weather_feature_idx", 0))
    else:
        raise ValueError(f"未知数据集: '{d.dataset}'")


def setup_logger(result_dir: str, dataset: str, timestamp: str) -> logging.Logger:
    logger = logging.getLogger("baselines")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, f"baselines_{dataset}_{timestamp}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    print(f"[Log] {log_path}")
    return logger


def _fmt_row(name, m_norm, m_raw):
    """格式化一行结果，同时显示归一化和原始量纲指标。"""
    crps_n  = m_norm.get("CRPS",  float("nan"))
    crps_r  = m_raw.get("CRPS",   float("nan"))
    picp_r  = m_raw.get("PICP",   float("nan"))
    pinaw_r = m_raw.get("PINAW",  float("nan"))
    return (
        f"{name:<12} | "
        f"{m_norm['MAE']:>8.4f} | {m_norm['RMSE']:>8.4f} | {crps_n:>8.4f} | "
        f"{m_raw['MAE']:>8.4f} | {m_raw['RMSE']:>8.4f} | {crps_r:>8.4f} | "
        f"{picp_r:>7.4f} | {pinaw_r:>7.4f}"
    )


# ---------------------------------------------------------------------------
# HA / VAR 的归一化域评估包装
# ---------------------------------------------------------------------------

def run_ha_both(train_loader, val_loader, test_loader, scaler, logger):
    """
    [Fix-3] HA 同时返回归一化域和反归一化域的指标。
    HA = Persistence forecast（最后观测值），与论文一致。
    """
    from baselines import HistoricalAverage
    model = HistoricalAverage().fit(train_loader)

    val_norm = model.evaluate(val_loader, scaler=None)
    val_raw  = model.evaluate(val_loader, scaler=scaler)
    logger.info(f"[HA] Val (norm) | MAE={val_norm['MAE']:.4f}  RMSE={val_norm['RMSE']:.4f}"
                f"  CRPS={val_norm['CRPS']:.4f}")
    logger.info(f"[HA] Val (raw)  | MAE={val_raw['MAE']:.4f}  RMSE={val_raw['RMSE']:.4f}"
                f"  CRPS={val_raw['CRPS']:.4f}")

    test_norm = model.evaluate(test_loader, scaler=None)
    test_raw  = model.evaluate(test_loader, scaler=scaler)
    logger.info(f"[HA] Test(norm) | MAE={test_norm['MAE']:.4f}  RMSE={test_norm['RMSE']:.4f}"
                f"  CRPS={test_norm['CRPS']:.4f}")
    logger.info(f"[HA] Test(raw)  | MAE={test_raw['MAE']:.4f}  RMSE={test_raw['RMSE']:.4f}"
                f"  CRPS={test_raw['CRPS']:.4f}")
    return test_norm, test_raw


def run_var_both(train_loader, val_loader, test_loader, scaler, max_lag, logger):
    """[Fix-3] VAR 同时返回归一化域和反归一化域的指标。"""
    from baselines import VARModel
    logger.info("[VAR] 开始拟合...")
    import time
    t0 = time.time()
    model = VARModel(max_lag=max_lag).fit(train_loader)
    logger.info(f"[VAR] 拟合完成 ({time.time()-t0:.1f}s)")

    val_norm  = model.evaluate(val_loader, scaler=None)
    val_raw   = model.evaluate(val_loader, scaler=scaler)
    logger.info(f"[VAR] Val (norm)| MAE={val_norm['MAE']:.4f}  RMSE={val_norm['RMSE']:.4f}"
                f"  CRPS={val_norm['CRPS']:.4f}")
    logger.info(f"[VAR] Val (raw) | MAE={val_raw['MAE']:.4f}  RMSE={val_raw['RMSE']:.4f}"
                f"  CRPS={val_raw['CRPS']:.4f}")

    test_norm = model.evaluate(test_loader, scaler=None)
    test_raw  = model.evaluate(test_loader, scaler=scaler)
    logger.info(f"[VAR] Test(norm)| MAE={test_norm['MAE']:.4f}  RMSE={test_norm['RMSE']:.4f}"
                f"  CRPS={test_norm['CRPS']:.4f}")
    logger.info(f"[VAR] Test(raw) | MAE={test_raw['MAE']:.4f}  RMSE={test_raw['RMSE']:.4f}"
                f"  CRPS={test_raw['CRPS']:.4f}")
    return test_norm, test_raw


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GridCFN Baselines Runner v2")
    parser.add_argument("--preset",  type=str, default="solar",
                        choices=["solar", "electricity", "weather"])
    parser.add_argument("--models",  type=str, nargs="+",
                        default=["ha", "var", "dcrnn", "stgcn", "mtgnn", "agcrn"],
                        help="要运行的 baseline 列表")
    parser.add_argument("--gpu_id",  type=int, default=-1,
                        help="GPU 编号，-1=自动选 cuda:0，>=0=指定 GPU")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--epochs",  type=int, default=None,
                        help="深度模型最大 epoch 数（默认使用 preset 配置）")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg       = get_config(args.preset)
    dataset   = cfg.data.dataset

    # [Fix-1] 结果目录：result/baselines/<dataset>/<timestamp>/
    result_dir = os.path.join("result", "baselines", dataset, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    logger = setup_logger(result_dir, dataset, timestamp)
    set_seed(args.seed)
    device = get_device(args.gpu_id)   # [Fix-2]

    logger.info(f"Dataset    : {dataset}")
    logger.info(f"Device     : {device}")
    logger.info(f"Result dir : {result_dir}")
    logger.info(f"Baselines  : {args.models}")

    # ── 加载数据 ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, adj, scaler, in_dim = load_data(cfg)
    adj_norm   = GridCFN.normalize_adj(adj)
    edge_index = GridCFN.adj_to_edge_index(adj)
    n_nodes    = adj.shape[0]
    T_in       = cfg.data.T_in
    out_dim    = cfg.model.out_dim

    logger.info(f"Nodes      : {n_nodes}, in_dim={in_dim}, out_dim={out_dim}, T_in={T_in}")
    logger.info(f"Scaler     : mean={scaler.mean:.4f}, std={scaler.std:.4f}")

    max_epochs = args.epochs if args.epochs is not None else cfg.train.max_epochs

    # all_results 存双域结果：
    # {model_name: {"norm": {...}, "raw": {...}}}
    all_results = {}

    # ── HA ──────────────────────────────────────────────────────────────
    if "ha" in args.models:
        logger.info("\n" + "="*50)
        logger.info("[HA] Historical Average")
        logger.info("="*50)
        test_norm, test_raw = run_ha_both(
            train_loader, val_loader, test_loader, scaler, logger)
        all_results["HA"] = {"norm": test_norm, "raw": test_raw}

    # ── VAR ─────────────────────────────────────────────────────────────
    if "var" in args.models:
        logger.info("\n" + "="*50)
        logger.info("[VAR] Vector AutoRegression")
        logger.info("="*50)
        test_norm, test_raw = run_var_both(
            train_loader, val_loader, test_loader, scaler,
            max_lag=12, logger=logger)
        all_results["VAR"] = {"norm": test_norm, "raw": test_raw}

    # ── 深度 baseline（DCRNN, STGCN, MTGNN, AGCRN）────────────────────
    deep_models = [m for m in args.models if m in ("dcrnn", "stgcn", "mtgnn", "agcrn")]
    for model_name in deep_models:
        logger.info("\n" + "="*50)
        logger.info(f"[{model_name.upper()}*] 深度概率模型")
        logger.info("="*50)

        model = build_baseline(
            name=model_name, in_dim=in_dim, out_dim=out_dim,
            n_nodes=n_nodes, T_in=T_in, hidden_dim=64,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  参数量: {n_params:,}")

        save_path = os.path.join(
            result_dir, f"best_{model_name}_{dataset}_{timestamp}.pt"
        )

        # [Fix-2] 将 adj/edge_index 移动到对应 device
        train_deep_baseline(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            adj_norm=adj_norm.to(device),
            edge_index=edge_index.to(device),
            device=device,
            max_epochs=max_epochs,
            patience=cfg.train.patience,
            lr=cfg.train.lr,
            grad_clip=cfg.train.grad_clip,
            weight_decay=cfg.train.weight_decay,
            save_path=save_path,
            logger=logger,
            model_name=model_name.upper(),
        )

        # [Fix-3] 同时评估归一化域和反归一化域
        test_norm = evaluate_deep_baseline(
            model=model, loader=test_loader,
            adj_norm=adj_norm.to(device), edge_index=edge_index.to(device),
            device=device, scaler=None, inverse_transform=False,
        )
        test_raw = evaluate_deep_baseline(
            model=model, loader=test_loader,
            adj_norm=adj_norm.to(device), edge_index=edge_index.to(device),
            device=device, scaler=scaler, inverse_transform=True,
        )

        logger.info(f"[{model_name.upper()}*] Test(norm) → " +
                    "  ".join(f"{k}={v:.4f}" for k, v in test_norm.items()))
        logger.info(f"[{model_name.upper()}*] Test(raw)  → " +
                    "  ".join(f"{k}={v:.4f}" for k, v in test_raw.items()))
        all_results[f"{model_name.upper()}*"] = {"norm": test_norm, "raw": test_raw}

    # ── 汇总打印（双域对比表）────────────────────────────────────────────
    sep = "=" * 110
    hdr = (f"{'Model':<12} | "
           f"{'[归一化域 Norm]':^30} | "
           f"{'[反归一化域 Raw]':^50}")
    sub = (f"{'':12} | "
           f"{'MAE':>8} {'RMSE':>8} {'CRPS':>8}  | "
           f"{'MAE':>8} {'RMSE':>8} {'CRPS':>8} {'PICP':>7} {'PINAW':>7}")
    logger.info("\n" + sep)
    logger.info(hdr)
    logger.info(sub)
    logger.info("-" * 110)
    for name, res in all_results.items():
        logger.info(_fmt_row(name, res["norm"], res["raw"]))
    logger.info(sep)
    logger.info("")
    logger.info("注意: 归一化域(Norm)对应论文 Table II 的数值尺度，")
    logger.info("      反归一化域(Raw)为实际物理量纲（Solar: MW, Electricity: kWh 等）。")

    # ── 保存结果 ─────────────────────────────────────────────────────────
    result_path = os.path.join(result_dir, f"baselines_{dataset}_{timestamp}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n所有结果已保存至: {result_path}")
    print(f"\n所有结果已保存至: {result_dir}")
    return all_results


if __name__ == "__main__":
    main()