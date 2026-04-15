"""
GridCFN + GMM v2 – 主入口
==========================

与原 main.py 的区别：
  1. import 来源改为 model_gmm_v2 / train_gmm / config_gmm_v2
  2. build_model 传入所有 v2 参数（字段名与 ModelConfig 严格对齐）
  3. 数据加载后、模型构建时传入 n_nodes，并调用 model.fit_grin(train_data)
     初始化 GRIN 统计量

快速开始：
  python main_gmm.py --preset solar
  python main_gmm.py --preset electricity
  python main_gmm.py --preset weather
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

from config_gmm import Config, get_config
from model_gmm  import GridCFN
from dataset       import load_solar_energy, load_electricity, load_weather
from train_gmm     import train


# ---------------------------------------------------------------------------
# 日志系统（不变）
# ---------------------------------------------------------------------------

def setup_logger(cfg_train, dataset_name: str = "",
                 timestamp: str = "", result_dir: str = "") -> logging.Logger:
    logger = logging.getLogger("gridcfn")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if cfg_train.log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    log_dirs = []
    if cfg_train.log_dir is not None:
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


# ---------------------------------------------------------------------------
# 设备选择（不变）
# ---------------------------------------------------------------------------

def get_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    if gpu_id < 0:
        return torch.device("cuda:0")
    if gpu_id >= n_gpu:
        raise ValueError(
            f"gpu_id={gpu_id} 超出范围，当前只有 {n_gpu} 块 GPU（0~{n_gpu-1}）"
        )
    return torch.device(f"cuda:{gpu_id}")


def gpu_info(device: torch.device) -> str:
    if device.type == "cpu":
        return "CPU（未检测到 GPU）"
    idx  = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    mem  = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
    return f"GPU {idx}: {name}  ({mem:.1f} GB)"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(cfg: Config):
    d = cfg.data
    assert d.data_path, "data_path 未设置，请在 config.py 或 --preset 中指定"

    if d.dataset == "solar":
        return load_solar_energy(
            d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size
        )
    elif d.dataset == "electricity":
        return load_electricity(
            d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size
        )
    elif d.dataset == "weather":
        return load_weather(
            d.data_path, d.T_in, d.T_out, d.adj_threshold, d.batch_size,
            feature_idx=getattr(d, "weather_feature_idx", 0),
        )
    else:
        raise ValueError(f"未知数据集: '{d.dataset}'，可选: solar, electricity, weather")


def build_model(cfg: Config, in_dim: int = None, n_nodes: int = 1) -> GridCFN:
    """
    构建模型。

    参数名与 ModelConfig 字段名、GridCFN.__init__ 参数名三者严格对齐。
    n_nodes 由 main() 从数据加载结果中提取后传入，用于 GRIN per-node 参数。
    """
    m             = cfg.model
    actual_in_dim = in_dim if in_dim is not None else m.in_dim
    return GridCFN(
        in_dim=actual_in_dim,
        gcn_hidden=m.gcn_hidden,
        gcn_layers=m.gcn_layers,
        tcn_hidden=m.tcn_hidden,
        tcn_layers=m.tcn_layers,
        env_dim=m.env_dim,
        stoch_dim=m.stoch_dim,
        ms_out_dim=m.ms_out_dim,
        n_scg_layers=m.n_scg_layers,
        out_dim=m.out_dim,
        lambda_mi=m.lambda_mi,
        # GMM v1
        n_components=m.n_components,
        lambda_mean=m.lambda_mean,
        lambda_weight=m.lambda_weight,
        # GMM v2
        proj_dim=m.proj_dim,
        sigma_min_heavy=m.sigma_min_heavy,
        huber_delta=m.huber_delta,
        use_grin=m.use_grin,
        n_nodes=n_nodes,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(cfg: Config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset   = cfg.data.dataset

    result_dir = os.path.join("result", dataset, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    model_save_path  = os.path.join(result_dir, f"gridcfn_{dataset}_{timestamp}.pt")
    cfg.train.save_path = model_save_path

    logger = setup_logger(cfg.train, dataset, timestamp, result_dir)
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.gpu_id)

    logger.info(cfg.summary())
    logger.info(f"Device     : {gpu_info(device)}")
    logger.info(f"Result dir : {result_dir}")

    # 1. 数据加载
    train_loader, val_loader, test_loader, adj, scaler, in_dim = load_data(cfg)
    logger.info(f"in_dim     : {in_dim}（数据集实际特征维度，已自动覆盖 config）")

    # 2. 预计算 adj_norm 和 edge_index
    adj_norm   = GridCFN.normalize_adj(adj)
    edge_index = GridCFN.adj_to_edge_index(adj)
    n_nodes    = adj.shape[0]
    logger.info(f"Graph      : {n_nodes} nodes, {edge_index.shape[1]} edges")

    # 3. 构建模型（传入 n_nodes 供 GRIN 使用）
    model = build_model(cfg, in_dim=in_dim, n_nodes=n_nodes).to(device)

    # 4. 初始化 GRIN（用训练集第一个 batch 的节点特征统计量）
    #    从 train_loader 里取出训练集数据，计算 per-node 均值/方差
    if cfg.model.use_grin:
        # 收集训练集所有数据（仅用于统计，不消耗大量内存：只取 [T, N] 切片）
        # dataset 的 data 属性已存储完整训练集
        train_ds = train_loader.dataset
        train_data_full = train_ds.data   # [T_train, N, F] tensor
        # 取第一个特征做统计，形状 [T_train, N]
        x_train_for_grin = train_data_full[..., 0]   # [T_train, N]
        model.fit_grin(x_train_for_grin)
        logger.info(f"GRIN       : 已初始化（use_grin=True, n_nodes={n_nodes}）")
    else:
        logger.info(f"GRIN       : 已禁用（use_grin=False）")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters : {n_params:,}\n")

    # 5. 训练
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
        use_grin=cfg.model.use_grin,
    )

    # 6. 保存 history
    history_path = os.path.join(result_dir, f"history_{dataset}_{timestamp}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"History 已保存: {history_path}")

    logger.info(f"训练完成，所有结果保存于: {result_dir}")
    return history, result_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridCFN + GMM v2 Training")
    parser.add_argument(
        "--preset", type=str, default="solar",
        choices=["solar", "electricity", "weather"],
        help="预设配置名",
    )
    args = parser.parse_args()
    cfg  = get_config(args.preset)
    history, result_dir = main(cfg)
    print(f"\n所有结果已保存至: {result_dir}")
