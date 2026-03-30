"""
GridCFN-Improved – 主入口（接口与原版完全一致）

运行：
  python main.py                      # 合成数据快速验证
  python main.py --preset solar
  python main.py --preset electricity
  python main.py --preset debug       # 极小规模，5 epoch
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


def setup_logger(cfg_train):
    logger = logging.getLogger("gridcfn")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    if cfg_train.log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    if cfg_train.log_dir is not None:
        os.makedirs(cfg_train.log_dir, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh  = logging.FileHandler(
            os.path.join(cfg_train.log_dir, f"gridcfn_improved_{ts}.log"),
            encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_device(gpu_id):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if gpu_id < 0:
        return torch.device("cuda:0")
    if gpu_id >= n:
        raise ValueError(f"gpu_id={gpu_id} 超出范围（共 {n} 块）")
    return torch.device(f"cuda:{gpu_id}")


def gpu_info(device):
    if device.type == "cpu":
        return "CPU"
    idx  = device.index or 0
    name = torch.cuda.get_device_name(idx)
    mem  = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    return f"GPU {idx}: {name} ({mem:.1f} GB)"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(cfg):
    d = cfg.data
    if d.dataset == "synthetic":
        return make_synthetic_dataset(
            T=d.synthetic_T, N=d.synthetic_N, F=d.synthetic_F,
            T_in=d.T_in, T_out=d.T_out,
            batch_size=d.batch_size, seed=cfg.train.seed)
    elif d.dataset == "solar":
        return load_solar_energy(d.data_path, d.T_in, d.T_out,
                                 d.adj_threshold, d.batch_size)
    elif d.dataset == "electricity":
        return load_electricity(d.data_path, d.T_in, d.T_out,
                                d.adj_threshold, d.batch_size)
    elif d.dataset == "weather":
        return load_weather(d.data_path, d.T_in, d.T_out,
                            d.adj_threshold, d.batch_size)
    else:
        raise ValueError(f"未知数据集: {d.dataset}")


def build_model(cfg) -> GridCFN:
    m = cfg.model
    return GridCFN(
        in_dim=m.in_dim, gcn_hidden=m.gcn_hidden, gcn_layers=m.gcn_layers,
        tcn_hidden=m.tcn_hidden, tcn_layers=m.tcn_layers,
        env_dim=m.env_dim, stoch_dim=m.stoch_dim, ms_out_dim=m.ms_out_dim,
        n_scg_layers=m.n_scg_layers, out_dim=m.out_dim,
        n_codes=m.n_codes, K=m.K,
        beta_vq=m.beta_vq, beta_mi=m.beta_mi,
        n_heads=m.n_heads)


def main(cfg: Config):
    logger = setup_logger(cfg.train)
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.gpu_id)
    logger.info(cfg.summary())
    logger.info(f"Device : {gpu_info(device)}")

    train_loader, val_loader, test_loader, adj, scaler = load_data(cfg)
    model   = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters : {n_params:,}\n")

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adj=adj,
        device=device,
        scaler=scaler,
        cfg_train=cfg.train,
        logger=logger)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridCFN-Improved")
    parser.add_argument("--preset", type=str, default="default",
                        choices=["default","solar","electricity","weather","debug"])
    args = parser.parse_args()
    main(get_config(args.preset))
