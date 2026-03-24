"""
GridCFN – 主入口
================
所有参数在 config.py 里修改，这里不需要动。

快速开始：
  python main.py                        # 用 config.py 里的默认配置
  python main.py --preset solar         # 切换到 Solar-Energy 预设
  python main.py --preset debug         # 极小规模调试
  python main.py --preset electricity   # Electricity 预设
"""

import argparse
import random
import numpy as np
import torch

from config import Config, get_config
from model import GridCFN
from dataset import (load_solar_energy, load_electricity,
                     load_weather, make_synthetic_dataset)
from train import train


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


def main(cfg: Config):
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(cfg.summary())
    print(f"\nDevice : {device}")

    train_loader, val_loader, test_loader, adj, scaler = load_data(cfg)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}\n")

    history = train(
        model=model, train_loader=train_loader,
        val_loader=val_loader, test_loader=test_loader,
        adj=adj, device=device, scaler=scaler,
        cfg_train=cfg.train,
    )
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridCFN Training")
    parser.add_argument(
        "--preset", type=str, default="default",
        choices=["default", "solar", "electricity", "weather", "debug"],
        help="预设配置名（在 config.py 中定义）"
    )
    args = parser.parse_args()
    cfg = get_config(args.preset)
    main(cfg)
