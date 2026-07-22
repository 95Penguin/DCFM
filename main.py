"""
DCFM – 主入口（多步预测版）

相对单步版的改动：
  - build_model 传入 T_out（来自 cfg.data.T_out）
  - 其余逻辑完全不变
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
from model import DCFM
from dataset import load_solar_energy, load_electricity, load_weather, load_sdwpf, load_pjm
from train import train


def setup_logger(cfg_train, dataset_name="", timestamp="", result_dir=""):
    logger = logging.getLogger("dcfm")
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
        log_file = os.path.join(d, f"dcfm_{dataset_name}_{ts}.log")
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
    elif d.dataset == "pjm":
        return load_pjm(d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size)
    else:
        raise ValueError(f"未知数据集: '{d.dataset}'")


def build_model(cfg, in_dim=None, n_nodes=None, wind_mask=None):
    m = cfg.model
    return DCFM(
        n_nodes=n_nodes,
        in_dim=in_dim if in_dim is not None else m.in_dim,
        gcn_hidden=m.gcn_hidden, gcn_layers=m.gcn_layers,
        tcn_hidden=m.tcn_hidden, tcn_layers=m.tcn_layers,
        env_dim=m.env_dim, stoch_dim=m.stoch_dim, ms_out_dim=m.ms_out_dim,
        n_scg_layers=m.n_scg_layers, out_dim=m.out_dim, lambda_mi=m.lambda_mi,
        cfm_hidden=getattr(m, "cfm_hidden", 256),
        cfm_time_emb_dim=getattr(m, "cfm_time_emb_dim", 16),
        chunk_size=getattr(m, "chunk_size", 16384),
        ms_dilations=getattr(m, "ms_dilations", (1, 7, 30)),
        T_out=cfg.data.T_out,
        T_in=cfg.data.T_in,
        # adap_dim 已移除：DCFM 从未实际使用该参数
        rank_r=getattr(m, "rank_r", 8),
        lambda_rank=getattr(m, "lambda_rank", 0.01),
        freq_candidates=getattr(m, "freq_candidates", (12, 24, 48, 96)),
        wind_mask=wind_mask,   # SDWPF 风向掩码由 main() 构造后传入
    )


def main(cfg):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset    = cfg.data.dataset
    result_dir = os.path.join("result", dataset, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    cfg.train.save_path = os.path.join(result_dir, f"dcfm_{dataset}_{timestamp}.pt")

    logger = setup_logger(cfg.train, dataset, timestamp, result_dir)
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.gpu_id)

    logger.info(cfg.summary())
    logger.info(f"Device     : {gpu_info(device)}")
    logger.info(f"Result dir : {result_dir}")

    train_loader, val_loader, test_loader, adj, scaler, in_dim = load_data(cfg)
    logger.info(f"in_dim     : {in_dim},  T_out={cfg.data.T_out}")

    adj_norm   = DCFM.normalize_adj(adj)
    edge_index = DCFM.adj_to_edge_index(adj)
    n_nodes    = adj.shape[0]
    logger.info(f"Graph      : {n_nodes} nodes, {edge_index.shape[1]} edges")

    # ── SDWPF 风向掩码：从坐标文件构造有向尾流掩码 ──────────────────────────
    # 若坐标文件不存在，wind_mask=None，SparseGCN 退化为无向自适应图（仍可训练）
    wind_mask = None
    if cfg.data.dataset == "sdwpf":
        from wind_mask_utils import build_wind_mask_from_csv
        coord_path = os.path.join(os.path.dirname(cfg.data.data_path), "sdwpf_turb_location.csv")
        if os.path.exists(coord_path):
            wind_mask = build_wind_mask_from_csv(coord_path, angle_tol=45.0)
            logger.info(f"WindMask   : 已加载，shape={tuple(wind_mask.shape)}，"
                        f"非零边={int(wind_mask.sum())} / {n_nodes * n_nodes}")
        else:
            logger.info(f"WindMask   : 坐标文件未找到（{coord_path}），跳过风向掩码")
    if wind_mask is not None and wind_mask.shape[0] != n_nodes:
        raise ValueError(
            f"wind_mask 台数 {wind_mask.shape[0]} 与数据节点数 n_nodes={n_nodes} 不一致，"
            f"请检查坐标文件与数据文件的风机数量是否匹配。"
        )

    model    = build_model(cfg, in_dim=in_dim, n_nodes=n_nodes, wind_mask=wind_mask).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters : {n_params:,}  (cfm_dim={model.cfm_dim})\n")

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
    parser = argparse.ArgumentParser(description="DCFM Multi-Step Training")
    parser.add_argument("--preset", type=str, default="solar",
                        choices=["solar", "electricity", "weather", "sdwpf", "pjm"])
    # 可选：命令行覆盖 T_out
    parser.add_argument("--T_out", type=int, default=None,
                        help="预测步长，覆盖 preset 默认值（如 --T_out 24）")
    args = parser.parse_args()
    cfg  = get_config(args.preset)
    if args.T_out is not None:
        cfg.data.T_out = args.T_out
    history, result_dir = main(cfg)
    print(f"\n所有结果已保存至: {result_dir}")
