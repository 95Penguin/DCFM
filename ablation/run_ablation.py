# run_ablation.py
"""
GridCFN 消融实验主入口
────────────────────────────────────────────────────────────────────────
本文件位于项目根目录下的 ablation/ 子目录中：

    GridCFN/                  <- 项目根目录（model.py, config.py 等都在这里）
    ├── config.py
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── wind_mask_utils.py
    └── ablation/              <- 本文件所在目录
        ├── ablation_model.py
        ├── ablation_train.py
        └── run_ablation.py

由于 ablation_model.py / ablation_train.py / 本文件都需要 import 项目根目录
下的 model.py、config.py、dataset.py、train.py、wind_mask_utils.py，而 Python
默认不会把"脚本所在目录的上级目录"加入搜索路径，所以下面紧跟在 docstring 之后
的几行代码会先把项目根目录插入 sys.path —— 必须在所有 from model import ... /
from config import ... 之前执行，否则会报 ModuleNotFoundError。

用法（在项目根目录或 ablation/ 目录下均可执行，效果一致）：
  python ablation/run_ablation.py --preset solar
  cd ablation && python run_ablation.py --preset solar
  python ablation/run_ablation.py --preset sdwpf --epochs_override 60 --quick
  python ablation/run_ablation.py --preset solar --only full noclub noms
  uv run ablation/run_ablation.py --preset solar --only full noclub noms
  uv run ablation/run_ablation.py --preset solar --only noclub noms noscgmp nomsscgmp norank none

实验组合（默认跑全部）：
  full        完整模型（CLUB + MultiScaleContext + SCGMP + RankLoss，全开）
  noclub      关闭 CLUB 互信息解耦损失
  noms        关闭 MultiScaleContext（Env Context，多尺度环境上下文提炼）
  noscgmp     关闭 SCGMP（空间因果门控消息传递）
  norank      关闭 LowRankGCN 的低秩正则损失（rank_loss）
  nowind      关闭风向掩码（仅对 sdwpf 数据集有意义，其余数据集该开关无效果）
  none        以上能关的全部关闭（CLUB/MS/SCGMP/Rank 全部关闭，下界基线）

每组实验都从同一个随机种子、同一份数据划分、同一组其余超参数出发，
只改动 ablation 开关，确保差异只来自被消融的模块本身。

输出（固定在项目根目录下，不受执行时所在目录影响）：
  GridCFN/ablation_results/<dataset>/<timestamp>/<tag>/history_<tag>.json
  GridCFN/ablation_results/<dataset>/<timestamp>/summary.csv
  GridCFN/ablation_results/<dataset>/<timestamp>/summary.md
"""

import os
import sys

# ── 路径处理：把项目根目录插入 sys.path（必须在 import model/config 等之前） ──
# __file__ 是 .../GridCFN/ablation/run_ablation.py
# os.path.dirname(__file__) 是 .../GridCFN/ablation
# 再上一级 os.path.dirname(...) 就是 .../GridCFN，即项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import copy
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from config import get_config
from model import GridCFN
from dataset import load_solar_energy, load_electricity, load_weather, load_sdwpf, load_pjm

from ablation_model import AblationConfig, build_ablation_model
from ablation_train import ablation_train


# ---------------------------------------------------------------------------
# 实验组合定义
# ---------------------------------------------------------------------------

def get_ablation_configs(only=None):
    configs = {
        "full":    AblationConfig(use_club=True,  use_ms_context=True,  use_scgmp=True,  use_rank_loss=True,  use_wind_mask=True,  name="full"),
        "noclub":  AblationConfig(use_club=False, use_ms_context=True,  use_scgmp=True,  use_rank_loss=True,  use_wind_mask=True,  name="noclub"),
        "noms":    AblationConfig(use_club=True,  use_ms_context=False, use_scgmp=True,  use_rank_loss=True,  use_wind_mask=True,  name="noms"),
        "noscgmp": AblationConfig(use_club=True,  use_ms_context=True,  use_scgmp=False, use_rank_loss=True,  use_wind_mask=True,  name="noscgmp"),
        "norank":  AblationConfig(use_club=True,  use_ms_context=True,  use_scgmp=True,  use_rank_loss=False, use_wind_mask=True,  name="norank"),
        "nowind":  AblationConfig(use_club=True,  use_ms_context=True,  use_scgmp=True,  use_rank_loss=True,  use_wind_mask=False, name="nowind"),
        "nomsscgmp": AblationConfig(use_club=True, use_ms_context=False, use_scgmp=False, use_rank_loss=True,  use_wind_mask=True,  name="nomsscgmp"),
        "none":    AblationConfig(use_club=False, use_ms_context=False, use_scgmp=False, use_rank_loss=False, use_wind_mask=False, name="none"),
    }
    if only:
        missing = [k for k in only if k not in configs]
        if missing:
            raise ValueError(f"未知的实验名: {missing}，可选: {list(configs.keys())}")
        return {k: configs[k] for k in only}
    return configs


# ---------------------------------------------------------------------------
# 工具函数（与 main.py 基本一致，避免相互 import 造成耦合）
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    if gpu_id < 0:
        return torch.device("cuda:0")
    if gpu_id >= n_gpu:
        raise ValueError(f"gpu_id={gpu_id} 超出范围，共 {n_gpu} 块 GPU")
    return torch.device(f"cuda:{gpu_id}")


def _resolve_data_path(path: str) -> str:
    """
    config.py 里的 data_path 默认写成相对路径（如 "./data/solar_AL.txt"），
    这是假设脚本从项目根目录执行。现在 run_ablation.py 可能从 ablation/ 子
    目录下执行（如 `cd ablation && python run_ablation.py`），相对路径会先
    被解析成相对"当前工作目录"，找不到文件。

    这里做一个简单兜底：如果原始相对路径在当前工作目录下不存在，就尝试拼接
    项目根目录后再找一次；只要其中一种能找到文件就使用该路径，两种都找不到
    则保留原始路径（交给后续 load_xxx 函数报出原本的 FileNotFoundError，
    报错信息更直接，不在这里过度包装）。
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(candidate):
        return candidate
    return path


def load_data(cfg):
    d = cfg.data
    assert d.data_path, "data_path 未设置"
    d.data_path = _resolve_data_path(d.data_path)
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


def build_wind_mask_if_needed(cfg, n_nodes):
    """与 main.py::main() 中构造 wind_mask 的逻辑保持一致。"""
    if cfg.data.dataset != "sdwpf":
        return None
    from wind_mask_utils import build_wind_mask_from_csv
    coord_path = os.path.join(os.path.dirname(cfg.data.data_path), "sdwpf_turb_location.csv")
    if os.path.exists(coord_path):
        wind_mask = build_wind_mask_from_csv(coord_path, angle_tol=45.0)
        if wind_mask.shape[0] != n_nodes:
            raise ValueError(
                f"wind_mask 台数 {wind_mask.shape[0]} 与数据节点数 n_nodes={n_nodes} 不一致"
            )
        return wind_mask
    return None


# ---------------------------------------------------------------------------
# 单组消融实验
# ---------------------------------------------------------------------------

def run_single_ablation(cfg, ablation: AblationConfig, device,
                         train_loader, val_loader, test_loader,
                         adj_norm, edge_index, in_dim, n_nodes, wind_mask,
                         result_root: str, logger=None):
    """
    跑一组消融配置，返回包含 test 指标的 dict（已写入 history json）。
    每组实验都重新 set_seed，保证模型初始化、dropout mask 等随机性
    在不同消融组之间是同分布的（虽然由于 forward 结构不同，具体数值
    路径仍会不同，但至少初始化策略一致，不引入额外偏差）。
    """
    set_seed(cfg.train.seed)

    tag = ablation.tag()
    tag_dir = os.path.join(result_root, tag)
    os.makedirs(tag_dir, exist_ok=True)
    save_path = os.path.join(tag_dir, f"model_{tag}.pt")

    model = build_ablation_model(
        cfg, in_dim=in_dim, n_nodes=n_nodes, wind_mask=wind_mask, ablation=ablation
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if logger:
        logger.info(f"\n{'='*70}\n[实验: {tag}] 参数量(含未使用模块)={n_params:,}\n{'='*70}")

    history = ablation_train(
        model=model,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        adj_norm=adj_norm, edge_index=edge_index, device=device,
        cfg_train=cfg.train, scaler=cfg._scaler, logger=logger,
        save_path=save_path,
    )

    history_path = os.path.join(tag_dir, f"history_{tag}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 训练完即释放显存，避免多组实验连续跑导致显存累积
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return history


# ---------------------------------------------------------------------------
# 汇总输出
# ---------------------------------------------------------------------------

def summarize(results: dict, result_root: str):
    """
    results: {tag: history_dict}
    生成 summary.csv 和 summary.md，汇报指标取"反归一化域校准后"
    （history['test_metrics']，与 train.py 主报告口径一致，物理量纲可读）。
    """
    rows = []
    for tag, hist in results.items():
        tm = hist["test_metrics"]
        rows.append({
            "experiment":   tag,
            "MAE":          round(tm["MAE"], 4),
            "RMSE":         round(tm["RMSE"], 4),
            "MAPE":         round(tm["MAPE"], 4),
            "CRPS":         round(tm["CRPS"], 4),
            "PICP":         round(tm["PICP"], 4),
            "PINAW":        round(tm["PINAW"], 4),
            "best_val_CRPS": round(hist["best_val_crps"], 4),
            "epochs_run":   hist["n_epochs_run"],
            "best_T":       round(hist["best_temperature"], 3),
        })
    df = pd.DataFrame(rows).set_index("experiment")

    # 如果存在 full 基线，额外算一列"相对 full 的 CRPS 变化百分比"，
    # 正值表示该消融变体比完整模型更差（CRPS 越小越好）。
    if "full" in df.index:
        base_crps = df.loc["full", "CRPS"]
        df["CRPS_vs_full(%)"] = ((df["CRPS"] - base_crps) / base_crps * 100).round(2)

    csv_path = os.path.join(result_root, "summary.csv")
    md_path  = os.path.join(result_root, "summary.md")
    df.to_csv(csv_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 消融实验汇总（{datetime.now().strftime('%Y-%m-%d %H:%M')}）\n\n")
        f.write(df.to_markdown())
        f.write("\n\n注：以上指标为反归一化域、温度校准后的测试集结果，"
                "CRPS/MAE/RMSE/PINAW 越小越好，PICP 越接近 0.95 越好。\n")

    print(f"\n汇总表已保存:\n  CSV : {csv_path}\n  MD  : {md_path}\n")
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GridCFN 消融实验")
    parser.add_argument("--preset", type=str, default="solar",
                         choices=["solar", "electricity", "weather", "sdwpf", "pjm"])
    parser.add_argument("--only", type=str, nargs="+", default=None,
                         help="只跑指定的实验组合，例如 --only full noclub noms")
    parser.add_argument("--epochs_override", type=int, default=None,
                         help="覆盖 max_epochs，用于快速试跑")
    parser.add_argument("--patience_override", type=int, default=None,
                         help="覆盖 patience（早停轮数）")
    parser.add_argument("--quick", action="store_true",
                         help="快速模式：max_epochs=10, patience=5, cfm_n_samples_test=20，"
                              "用于验证脚本本身是否跑通，不代表真实结果")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    parser.add_argument("--gpu_id", type=int, default=None)
    args = parser.parse_args()

    cfg = get_config(args.preset)

    if args.quick:
        cfg.train.max_epochs = 10
        cfg.train.patience = 5
        cfg.train.cfm_n_samples_test = 20
        cfg.train.cfm_n_steps = 8
    if args.epochs_override is not None:
        cfg.train.max_epochs = args.epochs_override
    if args.patience_override is not None:
        cfg.train.patience = args.patience_override
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.gpu_id is not None:
        cfg.train.gpu_id = args.gpu_id

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = os.path.join(PROJECT_ROOT, "ablation_results", cfg.data.dataset, timestamp)
    os.makedirs(result_root, exist_ok=True)

    import logging
    logger = logging.getLogger("gridcfn.ablation.main")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(result_root, "ablation.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    logger.info(cfg.summary())
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.gpu_id)
    logger.info(f"Device: {device}")

    # ── 数据只加载一次，所有消融组共享同一份划分，保证公平对比 ──
    train_loader, val_loader, test_loader, adj, scaler, in_dim = load_data(cfg)
    cfg._scaler = scaler   # 挂在 cfg 上方便 run_single_ablation 取用，不污染 dataclass 字段

    adj_norm   = GridCFN.normalize_adj(adj)
    edge_index = GridCFN.adj_to_edge_index(adj)
    n_nodes    = adj.shape[0]
    logger.info(f"Graph: {n_nodes} nodes, {edge_index.shape[1]} edges, in_dim={in_dim}")

    wind_mask = build_wind_mask_if_needed(cfg, n_nodes)
    if wind_mask is not None:
        logger.info(f"WindMask: shape={tuple(wind_mask.shape)}, 非零边={int(wind_mask.sum())}")

    ablation_configs = get_ablation_configs(only=args.only)
    logger.info(f"将运行 {len(ablation_configs)} 组消融实验: {list(ablation_configs.keys())}")

    results = {}
    for name, ablation in ablation_configs.items():
        history = run_single_ablation(
            cfg, ablation, device,
            train_loader, val_loader, test_loader,
            adj_norm, edge_index, in_dim, n_nodes, wind_mask,
            result_root=result_root, logger=logger,
        )
        results[ablation.tag()] = history

    df = summarize(results, result_root)

    combined_path = os.path.join(result_root, "all_histories.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"完整历史已保存: {combined_path}")
    logger.info(f"所有消融实验结果保存于: {result_root}")


if __name__ == "__main__":
    main()