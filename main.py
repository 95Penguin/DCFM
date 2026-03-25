"""
GridCFN – 主入口
================
所有参数在 config.py 里修改，这里不需要动。

快速开始：
  python main.py                        # config.py 默认配置
  python main.py --preset solar         # Solar-Energy 预设
  python main.py --preset debug         # 极小规模调试
  python main.py --preset electricity   # Electricity 预设

日志文件自动保存到 logs/ 目录，文件名含时间戳，例如：
  logs/gridcfn_electricity_20240315_143022.log
"""

import argparse
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

from config import Config, get_config
from model import GridCFN
from dataset import (load_solar_energy, load_electricity,
                     load_weather, make_synthetic_dataset)
from train import train


# ---------------------------------------------------------------------------
# 日志系统
# ---------------------------------------------------------------------------

def setup_logger(cfg_train) -> logging.Logger:
    """
    创建 logger，同时输出到控制台和文件。
    文件名格式：logs/gridcfn_<dataset>_<timestamp>.log
    """
    logger = logging.getLogger("gridcfn")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()          # 防止重复添加 handler

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler
    if cfg_train.log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # 文件 handler
    if cfg_train.log_dir is not None:
        os.makedirs(cfg_train.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file  = os.path.join(
            cfg_train.log_dir, f"gridcfn_{timestamp}.log"
        )
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        # 用 print 输出文件路径，因为 logger 还没完全初始化
        print(f"[Log] 日志文件: {log_file}")

    return logger


# ---------------------------------------------------------------------------
# GPU 选择
# ---------------------------------------------------------------------------

def get_device(gpu_id: int) -> torch.device:
    """
    gpu_id = -1 : 自动选择（有 GPU 用第 0 块，没有用 CPU）
    gpu_id >= 0 : 指定卡号（多卡时用）
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    n_gpu = torch.cuda.device_count()
    if gpu_id < 0:
        device = torch.device("cuda:0")
    elif gpu_id >= n_gpu:
        raise ValueError(
            f"gpu_id={gpu_id} 超出范围，当前只有 {n_gpu} 块 GPU（0~{n_gpu-1}）"
        )
    else:
        device = torch.device(f"cuda:{gpu_id}")

    return device


def gpu_info(device: torch.device) -> str:
    """返回 GPU 信息字符串，CPU 时返回 'CPU'。"""
    if device.type == "cpu":
        return "CPU（未检测到 GPU）"
    idx  = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    mem  = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
    return f"GPU {idx}: {name}  ({mem:.1f} GB)"


# ---------------------------------------------------------------------------
# 数据 / 模型构建
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(cfg: Config):
    d = cfg.data
    if d.dataset == "synthetic":
        return make_synthetic_dataset(
            T=d.synthetic_T, N=d.synthetic_N, F=d.synthetic_F,
            T_in=d.T_in, T_out=d.T_out,
            batch_size=d.batch_size, seed=cfg.train.seed,
        )
    elif d.dataset == "solar":
        assert d.data_path, "solar 需要设置 data.data_path"
        return load_solar_energy(d.data_path, d.T_in, d.T_out,
                                 d.adj_threshold, d.batch_size)
    elif d.dataset == "electricity":
        assert d.data_path, "electricity 需要设置 data.data_path"
        return load_electricity(d.data_path, d.T_in, d.T_out,
                                d.adj_threshold, d.batch_size)
    elif d.dataset == "weather":
        assert d.data_path, "weather 需要设置 data.data_path"
        return load_weather(d.data_path, d.T_in, d.T_out,
                            d.adj_threshold, d.batch_size)
    else:
        raise ValueError(f"未知数据集: {d.dataset}")


def build_model(cfg: Config) -> GridCFN:
    m = cfg.model
    return GridCFN(
        in_dim=m.in_dim, gcn_hidden=m.gcn_hidden, gcn_layers=m.gcn_layers,
        tcn_hidden=m.tcn_hidden, tcn_layers=m.tcn_layers,
        env_dim=m.env_dim, stoch_dim=m.stoch_dim, ms_out_dim=m.ms_out_dim,
        n_scg_layers=m.n_scg_layers, out_dim=m.out_dim, lambda_mi=m.lambda_mi,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(cfg: Config):
    # 1. Logger
    logger = setup_logger(cfg.train)

    # 2. 随机种子
    set_seed(cfg.train.seed)

    # 3. 设备
    device = get_device(cfg.train.gpu_id)

    # 4. 打印配置摘要
    logger.info(cfg.summary())
    logger.info(f"Device : {gpu_info(device)}")

    # 5. 数据
    train_loader, val_loader, test_loader, adj, scaler = load_data(cfg)

    # 6. 模型
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters : {n_params:,}\n")

    # 7. 训练（把 logger 传给 train）
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adj=adj,
        device=device,
        scaler=scaler,
        cfg_train=cfg.train,
        logger=logger,
    )
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridCFN Training")
    parser.add_argument(
        "--preset", type=str, default="default",
        choices=["default", "solar", "electricity", "weather", "debug"],
        help="预设配置名（在 config.py 中定义）",
    )
    args = parser.parse_args()
    cfg  = get_config(args.preset)
    main(cfg)

