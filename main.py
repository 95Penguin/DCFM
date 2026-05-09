"""
GridCFN – 主入口
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

from config import Config, get_config
from model import GridCFN
from dataset import load_solar_energy, load_electricity, load_weather, load_sdwpf
from train import train


def setup_logger(cfg_train, dataset_name="", timestamp="", result_dir=""):
    logger = logging.getLogger("gridcfn")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(fmt="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if cfg_train.log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    log_dirs = []
    if cfg_train.log_dir:
        log_dirs.append(cfg_train.log_dir)
    if result_dir:
        log_dirs.append(result_dir)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    for d in log_dirs:
        os.makedirs(d, exist_ok=True)
        log_file = os.path.join(d, f"gridcfn_{dataset_name}_{ts}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        if d == log_dirs[0]:
            print(f"[Log] 日志文件: {log_file}")
    return logger


def get_device(gpu_id):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    if gpu_id < 0:
        return torch.device("cuda:0")
    if gpu_id >= n_gpu:
        raise ValueError(f"gpu_id={gpu_id} 超出范围，共 {n_gpu} 块 GPU")
    return torch.device(f"cuda:{gpu_id}")


def gpu_info(device):
    if device.type == "cpu":
        return "CPU"
    idx  = device.index if device.index is not None else 0
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
    assert d.data_path, "data_path 未设置"
    if d.dataset == "solar":
        return load_solar_energy(d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size)
    elif d.dataset == "electricity":
        return load_electricity(d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size)
    elif d.dataset == "weather":
        return load_weather(d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size,
                            feature_idx=getattr(d, "weather_feature_idx", 0))
    elif d.dataset == "sdwpf":
        return load_sdwpf(d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size)
    else:
        raise ValueError(f"未知数据集: '{d.dataset}'，支持: solar, electricity, weather, sdwpf")


def build_model(cfg, in_dim=None):
    m = cfg.model
    return GridCFN(
        in_dim=in_dim if in_dim is not None else m.in_dim,
        gcn_hidden=m.gcn_hidden, gcn_layers=m.gcn_layers,
        tcn_hidden=m.tcn_hidden, tcn_layers=m.tcn_layers,
        env_dim=m.env_dim, stoch_dim=m.stoch_dim, ms_out_dim=m.ms_out_dim,
        n_scg_layers=m.n_scg_layers, out_dim=m.out_dim, lambda_mi=m.lambda_mi,
        cfm_hidden=getattr(m, "cfm_hidden", 128),
        cfm_time_emb_dim=getattr(m, "cfm_time_emb_dim", 16),
        chunk_size=getattr(m, "chunk_size", 16384),
        ms_dilations=getattr(m, "ms_dilations", (1, 7, 30)),
    )


def main(cfg):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset    = cfg.data.dataset
    result_dir = os.path.join("result", dataset, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    cfg.train.save_path = os.path.join(result_dir, f"gridcfn_{dataset}_{timestamp}.pt")

    logger = setup_logger(cfg.train, dataset, timestamp, result_dir)
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.gpu_id)

    logger.info(cfg.summary())
    logger.info(f"Device     : {gpu_info(device)}")
    logger.info(f"Result dir : {result_dir}")

    train_loader, val_loader, test_loader, adj, scaler, in_dim = load_data(cfg)
    logger.info(f"in_dim     : {in_dim}")

    adj_norm   = GridCFN.normalize_adj(adj)
    edge_index = GridCFN.adj_to_edge_index(adj)
    logger.info(f"Graph      : {adj.shape[0]} nodes, {edge_index.shape[1]} edges")

    model    = build_model(cfg, in_dim=in_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters : {n_params:,}\n")

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adj_norm=adj_norm,
        edge_index=edge_index,
        device=device,
        scaler=scaler,
        cfg_train=cfg.train,
        logger=logger,
    )

    history_path = os.path.join(result_dir, f"history_{dataset}_{timestamp}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"History 已保存: {history_path}")
    logger.info(f"训练完成，所有结果保存于: {result_dir}")
    return history, result_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridCFN Training")
    parser.add_argument("--preset", type=str, default="solar",
                        choices=["solar", "electricity", "weather", "sdwpf"])
    args = parser.parse_args()
    cfg  = get_config(args.preset)
    history, result_dir = main(cfg)
    print(f"\n所有结果已保存至: {result_dir}")