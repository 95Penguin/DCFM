"""
run_baselines.py — 2025 年对比算法统一运行入口

用法示例:
  # 运行 TSFlow，Solar 数据集
  uv run baselines/run_baselines.py --model tsflow --preset solar

  # 运行 K2VAE，Electricity 数据集，T_out=24
  uv run baselines/run_baselines.py --model k2vae --preset electricity --T_out 24

  # 运行全部 2025 baselines，SDWPF 数据集
  uv run baselines/run_baselines.py --model all --preset sdwpf

接口设计:
  - 复用 GridCFN 的 config.py / dataset.py / train.py 工具函数
  - 各 baseline 使用与 GridCFN 相同的 DataLoader、Scaler、评估指标
  - 结果保存在 result/<dataset>/<timestamp>_<model>/ 下
"""

import argparse
import json
import logging
import os
import sys
import random
from datetime import datetime

import numpy as np
import torch

# 将项目根目录加入 path（baselines/ 在项目根下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from dataset import load_solar_energy, load_electricity, load_weather, load_sdwpf
from baselines.tsflow import run_tsflow
from baselines.k2vae import run_k2vae


# ---------------------------------------------------------------------------
# 工具函数（复用 main.py 逻辑）
# ---------------------------------------------------------------------------

def setup_logger(name: str, result_dir: str, dataset: str, timestamp: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, f"{name}_{dataset}_{timestamp}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id < 0:
        return torch.device("cuda:0")
    return torch.device(f"cuda:{gpu_id}")


def load_data(cfg):
    d = cfg.data
    if d.dataset == "solar":
        return load_solar_energy(d.data_path, d.T_in, d.T_out,
                                 d.adj_threshold, d.batch_size)
    elif d.dataset == "electricity":
        return load_electricity(d.data_path, d.T_in, d.T_out,
                                d.adj_threshold, d.batch_size)
    elif d.dataset == "weather":
        return load_weather(
            d.data_path, d.T_in, d.T_out,
            d.adj_threshold, d.batch_size,
            feature_idx=getattr(d, "weather_feature_idx", 0),
        )
    elif d.dataset == "sdwpf":
        return load_sdwpf(d.data_path, d.T_in, d.T_out,
                          d.adj_threshold, d.batch_size)
    else:
        raise ValueError(f"未知数据集: {d.dataset}")


# ---------------------------------------------------------------------------
# 运行单个 baseline
# ---------------------------------------------------------------------------

def run_one(model_name: str, cfg, timestamp: str):
    dataset = cfg.data.dataset
    result_dir = os.path.join("result", dataset,
                              f"{timestamp}_{model_name}")
    os.makedirs(result_dir, exist_ok=True)

    # save_path 覆盖：各 baseline 存各自目录
    cfg.train.save_path = os.path.join(
        result_dir, f"{model_name}_{dataset}_{timestamp}.pt"
    )

    logger = setup_logger(model_name, result_dir, dataset, timestamp)
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.gpu_id)

    logger.info(f"{'='*50}")
    logger.info(f"Model    : {model_name.upper()}")
    logger.info(f"Dataset  : {dataset}")
    logger.info(f"T_in={cfg.data.T_in}, T_out={cfg.data.T_out}")
    logger.info(f"Device   : {device}")
    logger.info(f"{'='*50}")

    train_loader, val_loader, test_loader, adj, scaler, in_dim = load_data(cfg)
    cfg.model.in_dim = in_dim
    logger.info(f"in_dim={in_dim}")

    if model_name == "tsflow":
        history = run_tsflow(
            train_loader, val_loader, test_loader,
            scaler, cfg, device,
            logger=logger,
        )
    elif model_name == "k2vae":
        history = run_k2vae(
            train_loader, val_loader, test_loader,
            scaler, cfg, device,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 保存 history
    history_path = os.path.join(
        result_dir, f"history_{model_name}_{dataset}_{timestamp}.json"
    )
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"History 已保存: {history_path}")
    logger.info(f"结果目录: {result_dir}")
    return history


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GridCFN 2025 Baselines Runner")
    parser.add_argument("--model", type=str, default="tsflow",
                        choices=["tsflow", "k2vae", "all"],
                        help="选择运行的 baseline")
    parser.add_argument("--preset", type=str, default="solar",
                        choices=["solar", "electricity", "weather", "sdwpf"],
                        help="数据集 preset")
    parser.add_argument("--T_out", type=int, default=None,
                        help="预测步长（覆盖 preset 默认值）")
    parser.add_argument("--gpu_id", type=int, default=-1)
    args = parser.parse_args()

    cfg = get_config(args.preset)
    if args.T_out is not None:
        cfg.data.T_out = args.T_out
    cfg.train.gpu_id = args.gpu_id

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    models_to_run = (
        ["tsflow", "k2vae"] if args.model == "all" else [args.model]
    )

    all_results = {}
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Running: {model_name.upper()} on {args.preset}")
        print(f"{'='*60}")
        history = run_one(model_name, cfg, timestamp)
        all_results[model_name] = history.get("test_metrics", {})

    # 打印汇总对比
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'CRPS':>8} "
              f"{'PICP':>8} {'PINAW':>8}")
        for name, m in all_results.items():
            if m:
                print(f"{name:<12} {m.get('MAE',0):>8.4f} "
                      f"{m.get('RMSE',0):>8.4f} {m.get('CRPS',0):>8.4f} "
                      f"{m.get('PICP',0):>8.4f} {m.get('PINAW',0):>8.4f}")


if __name__ == "__main__":
    main()
