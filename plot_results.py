"""
GridCFN – 结果可视化
====================
从 history JSON 或直接从 main.py 的返回值生成全套图表，
保存到对应的 result/<dataset>/<timestamp>/ 目录。

使用方式：
  # 方式一：在 main.py 训练完后直接调用（推荐）
  from plot_results import plot_all
  plot_all(history, result_dir, dataset_name="solar")

  # 方式二：指定已有的 history JSON 文件
  python plot_results.py --history result/solar/20260403_111007/history_solar_20260403_111007.json

生成图表：
  1. convergence.png      Val CRPS 收敛曲线 + LR 下降节点
  2. loss_components.png  Train Loss / NLL / MI 三条曲线
  3. metrics_curve.png    Val MAE + Val RMSE 双轴曲线
  4. test_radar.png       测试集五维指标雷达图（vs 论文基线）
  5. baselines_bar.png    与论文各基线的柱状对比图（Solar 数据集论文数据）
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import numpy as np

# ---------------------------------------------------------------------------
# 中文字体注册（Linux 服务器环境）
# ---------------------------------------------------------------------------
def _setup_chinese_font():
    """
    尝试注册系统中文字体，优先级：
      1. Noto Sans CJK SC（Ubuntu/Debian 上通常预装）
      2. WenQuanYi Zen Hei（备选）
      3. 回退到英文标签（不报错，只降级）
    """
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            font_name = prop.get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    # 回退：全部用英文，不触发 Glyph 警告
    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans"

_FONT = _setup_chinese_font()

# ---------------------------------------------------------------------------
# 全局样式（字体在 _setup_chinese_font 里已设置）
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "legend.framealpha": 0.85,
    "axes.unicode_minus": False,
})

BLUE   = "#378ADD"
TEAL   = "#1D9E75"
AMBER  = "#EF9F27"
RED    = "#E24B4A"
PURPLE = "#7F77DD"
GRAY   = "#888780"

# 论文 Table II 基线数据（Solar-Energy 数据集，归一化尺度）
PAPER_BASELINES_SOLAR = {
    "DCRNN*":  {"MAE": 0.165, "RMSE": 0.295, "CRPS": 0.160},
    "STGCN*":  {"MAE": 0.160, "RMSE": 0.290, "CRPS": 0.155},
    "MTGNN*":  {"MAE": 0.152, "RMSE": 0.275, "CRPS": 0.145},
    "AGCRN*":  {"MAE": 0.155, "RMSE": 0.280, "CRPS": 0.148},
    "GridCFN\n(论文)": {"MAE": 0.143, "RMSE": 0.260, "CRPS": 0.135},
}

PAPER_BASELINES_ELECTRICITY = {
    "DCRNN*":  {"MAE": 0.098, "RMSE": 0.170, "CRPS": 0.092},
    "STGCN*":  {"MAE": 0.095, "RMSE": 0.165, "CRPS": 0.090},
    "MTGNN*":  {"MAE": 0.088, "RMSE": 0.155, "CRPS": 0.082},
    "AGCRN*":  {"MAE": 0.090, "RMSE": 0.160, "CRPS": 0.085},
    "GridCFN\n(论文)": {"MAE": 0.079, "RMSE": 0.145, "CRPS": 0.070},
}

PAPER_BASELINES_WEATHER = {
    "DCRNN*":  {"MAE": 1.95, "RMSE": 3.05, "CRPS": 1.90},
    "STGCN*":  {"MAE": 1.90, "RMSE": 3.00, "CRPS": 1.85},
    "MTGNN*":  {"MAE": 1.80, "RMSE": 2.85, "CRPS": 1.75},
    "AGCRN*":  {"MAE": 1.82, "RMSE": 2.90, "CRPS": 1.78},
    "GridCFN\n(论文)": {"MAE": 1.75, "RMSE": 2.68, "CRPS": 1.63},
}

BASELINES = {
    "solar":       PAPER_BASELINES_SOLAR,
    "electricity": PAPER_BASELINES_ELECTRICITY,
    "weather":     PAPER_BASELINES_WEATHER,
}


# ---------------------------------------------------------------------------
# 图1：Val CRPS 收敛曲线
# ---------------------------------------------------------------------------
def plot_convergence(history: Dict, result_dir: str, dataset: str):
    val_crps   = history["val_crps"]
    epochs     = list(range(1, len(val_crps) + 1))
    train_loss = history.get("train_loss", [])

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(epochs, val_crps, color=BLUE, lw=1.8, label="Val CRPS")

    # 标注 LR 下降节点（CRPS 出现阶梯状下降的位置）
    if len(val_crps) > 5:
        crps_arr = np.array(val_crps)
        # 简单启发式：相邻 5 个 epoch 均值降幅超过 3% 的位置
        for i in range(4, len(crps_arr) - 1):
            prev = crps_arr[max(0, i-4):i].mean()
            curr = crps_arr[i]
            if prev > 0 and (prev - curr) / prev > 0.05:
                ax.axvline(i + 1, color=AMBER, lw=0.8, ls="--", alpha=0.6)

    best_idx   = int(np.argmin(val_crps))
    best_crps  = val_crps[best_idx]
    ax.scatter([best_idx + 1], [best_crps], color=RED, zorder=5, s=60,
               label=f"最优 epoch {best_idx+1}  CRPS={best_crps:.4f}")

    # 论文基线参考线
    baselines = BASELINES.get(dataset, {})
    if "GridCFN\n(论文)" in baselines:
        paper_crps = baselines["GridCFN\n(论文)"]["CRPS"]
        ax.axhline(paper_crps, color=RED, lw=1, ls=":", alpha=0.7,
                   label=f"论文 GridCFN CRPS={paper_crps}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val CRPS")
    ax.set_title(f"收敛曲线 — {dataset}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(result_dir, "convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [图] convergence.png")


# ---------------------------------------------------------------------------
# 图2：Loss 分解曲线（Loss / NLL / MI）
# ---------------------------------------------------------------------------
def plot_loss_components(history: Dict, result_dir: str, dataset: str):
    train_loss = history.get("train_loss", [])
    train_nll  = history.get("train_nll",  [])
    train_mi   = history.get("train_mi",   [])
    epochs     = list(range(1, len(train_loss) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(epochs, train_loss, color=BLUE,  lw=1.5, label="Train Loss")
    if train_nll:
        ax1.plot(epochs, train_nll, color=TEAL, lw=1.2, ls="--", label="NLL")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"训练损失分解 — {dataset}")
    ax1.legend(fontsize=9)

    if train_mi:
        ax2.plot(epochs, train_mi, color=PURPLE, lw=1.5, label="MI 估计")
        ax2.axhline(0, color=GRAY, lw=0.8, ls="--")
        ax2.set_ylabel("MI 估计值")
        ax2.set_xlabel("Epoch")
        ax2.legend(fontsize=9)

        # 标注三个阶段
        mi_arr = np.array(train_mi)
        zero_cross = np.where(np.diff(np.sign(mi_arr)))[0]
        if len(zero_cross) >= 1:
            ax2.axvline(zero_cross[0] + 1, color=AMBER, lw=0.8, ls=":", alpha=0.7,
                        label="MI→0")
        # MI 重新激活：找从 0 附近跳到 >0.5 的点
        reactivate = np.where((mi_arr[:-1] < 0.1) & (mi_arr[1:] > 0.5))[0]
        if len(reactivate) >= 1:
            ax2.axvline(reactivate[0] + 2, color=TEAL, lw=0.8, ls=":", alpha=0.7,
                        label="MI 重激活")
        ax2.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(result_dir, "loss_components.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [图] loss_components.png")


# ---------------------------------------------------------------------------
# 图3：Val MAE + Val RMSE 双曲线
# ---------------------------------------------------------------------------
def plot_metrics_curve(history: Dict, result_dir: str, dataset: str):
    val_mae  = history.get("val_mae",  [])
    val_rmse = history.get("val_rmse", [])
    epochs   = list(range(1, len(val_mae) + 1))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, val_mae,  color=BLUE,  lw=1.5, label="Val MAE")
    if val_rmse:
        ax2 = ax.twinx()
        ax2.plot(epochs, val_rmse, color=AMBER, lw=1.2, ls="--", label="Val RMSE")
        ax2.set_ylabel("Val RMSE", color=AMBER)
        ax2.tick_params(axis="y", colors=AMBER)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(AMBER)
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MAE", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_title(f"验证集指标曲线 — {dataset}")
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    fig.tight_layout()
    path = os.path.join(result_dir, "metrics_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [图] metrics_curve.png")


# ---------------------------------------------------------------------------
# 图4：预测区间可视化（选 4 个典型节点，各画一段时间窗口）
# ---------------------------------------------------------------------------
def plot_prediction_intervals(history: Dict, result_dir: str, dataset: str,
                               n_nodes: int = 4, window: int = 120):
    """
    从测试集预测结果中选 n_nodes 个节点，各展示 window 个时间步的
    真实值 y、预测均值 μ 和 95% 预测区间 [μ-1.96σ, μ+1.96σ]。

    选节点策略：按方差从大到小排序，选方差最大的几个节点，
    这些节点的不确定性最高，区间可视化最有代表性。
    """
    if "test_mu" not in history:
        print("  [跳过] test_mu 不在 history 中，需要用新版 train.py 重新训练")
        return

    shape = history["test_shape"]          # [T_test, N, Fout]
    mu    = np.array(history["test_mu"   ]).reshape(shape)
    sigma = np.array(history["test_sigma"]).reshape(shape)
    y     = np.array(history["test_y"    ]).reshape(shape)

    T_test, N, Fout = shape
    # 只画第一个输出变量
    mu_f    = mu[..., 0]     # [T_test, N]
    sigma_f = sigma[..., 0]
    y_f     = y[..., 0]

    # 选方差最大的 n_nodes 个节点（从动态最丰富的节点取）
    node_var   = sigma_f.var(axis=0)                       # [N]
    top_nodes  = np.argsort(node_var)[::-1][:n_nodes]

    # 选一段有代表性的时间窗口：从测试集中段开始
    t_start = max(0, T_test // 3)
    t_end   = min(T_test, t_start + window)
    t_range = np.arange(t_start, t_end)

    fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 3 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]

    for ax, node_idx in zip(axes, top_nodes):
        y_seg     = y_f[t_start:t_end, node_idx]
        mu_seg    = mu_f[t_start:t_end, node_idx]
        sigma_seg = sigma_f[t_start:t_end, node_idx]
        upper     = mu_seg + 1.96 * sigma_seg
        lower     = mu_seg - 1.96 * sigma_seg
        steps     = np.arange(len(y_seg))

        ax.fill_between(steps, lower, upper,
                        color=BLUE, alpha=0.18, label="95% 预测区间")
        ax.plot(steps, y_seg,  color=GRAY, lw=1.2, label="真实值")
        ax.plot(steps, mu_seg, color=BLUE, lw=1.5, label="预测均值 μ")
        ax.set_ylabel("值（归一化）")
        ax.set_title(f"节点 {node_idx}  (σ 方差={node_var[node_idx]:.4f})")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("时间步（测试集）")
    fig.suptitle(f"预测区间可视化 — {dataset}  (95% CI)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(result_dir, "prediction_intervals.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [图] prediction_intervals.png")


# ---------------------------------------------------------------------------
# 图5：PICP 校准图（Reliability Diagram）
# ---------------------------------------------------------------------------
def plot_reliability_diagram(history: Dict, result_dir: str, dataset: str):
    """
    画不同置信度（10%~95%）下的实际覆盖率（PICP），
    理想情况下应落在对角线 y=x 上。
    偏上说明区间偏保守（过宽），偏下说明过于自信（过窄）。
    """
    if "test_mu" not in history:
        print("  [跳过] test_mu 不在 history 中，需要用新版 train.py 重新训练")
        return

    from scipy.stats import norm

    shape = history["test_shape"]
    mu    = np.array(history["test_mu"   ]).reshape(shape).flatten()
    sigma = np.array(history["test_sigma"]).reshape(shape).flatten()
    y     = np.array(history["test_y"    ]).reshape(shape).flatten()

    confidences  = np.arange(0.05, 1.00, 0.05)
    actual_picps = []
    for conf in confidences:
        z       = norm.ppf((1 + conf) / 2)
        covered = ((y >= mu - z * sigma) & (y <= mu + z * sigma))
        actual_picps.append(covered.mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="理想校准（y=x）")
    ax.plot(confidences, actual_picps, color=BLUE, lw=2,
            marker="o", markersize=4, label="实际覆盖率")
    ax.fill_between(confidences, confidences, actual_picps,
                    where=np.array(actual_picps) > confidences,
                    color=TEAL, alpha=0.15, label="保守（过宽）")
    ax.fill_between(confidences, confidences, actual_picps,
                    where=np.array(actual_picps) < confidences,
                    color=RED, alpha=0.15, label="自信（过窄）")

    ax.set_xlabel("置信度（名义覆盖率）")
    ax.set_ylabel("实际覆盖率（PICP）")
    ax.set_title(f"概率校准图（Reliability Diagram）— {dataset}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(result_dir, "reliability_diagram.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [图] reliability_diagram.png")


# ---------------------------------------------------------------------------
# 图6：误差分布图
# ---------------------------------------------------------------------------
def plot_error_distribution(history: Dict, result_dir: str, dataset: str):
    """
    左图：预测误差 (μ - y) 的直方图，检验误差是否以 0 为中心的正态分布。
    右图：按误差绝对值大小分桶，展示各区段 sigma 的平均值，
          验证模型不确定性是否和误差大小正相关（好的概率模型应该做到这点）。

    Solar 数据集额外说明：
      夜间时段真实值 y ≈ 0，误差接近 0；
      白天峰值时段误差更大，这在右图中会体现为高误差桶对应更大的 sigma。
    """
    if "test_mu" not in history:
        print("  [跳过] test_mu 不在 history 中，需要用新版 train.py 重新训练")
        return

    shape = history["test_shape"]
    mu    = np.array(history["test_mu"   ]).reshape(shape).flatten()
    sigma = np.array(history["test_sigma"]).reshape(shape).flatten()
    y     = np.array(history["test_y"    ]).reshape(shape).flatten()

    errors    = mu - y                        # 预测偏差
    abs_errors = np.abs(errors)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # 左图：误差直方图
    ax1.hist(errors, bins=60, color=BLUE, alpha=0.75, edgecolor="white", lw=0.3)
    ax1.axvline(0,            color=RED,  lw=1.5, ls="--", label="零误差线")
    ax1.axvline(errors.mean(), color=AMBER, lw=1.2, ls="--",
                label=f"均值={errors.mean():.4f}")
    ax1.set_xlabel("预测误差 (μ - y)")
    ax1.set_ylabel("频数")
    ax1.set_title("误差分布直方图")
    ax1.legend(fontsize=9)

    # 右图：误差绝对值 vs 平均不确定性（验证 sigma 的"有效性"）
    n_bins    = 10
    bins      = np.percentile(abs_errors, np.linspace(0, 100, n_bins + 1))
    bin_sigma = []
    bin_center = []
    for i in range(n_bins):
        mask = (abs_errors >= bins[i]) & (abs_errors < bins[i + 1])
        if mask.sum() > 0:
            bin_sigma.append(sigma[mask].mean())
            bin_center.append((bins[i] + bins[i + 1]) / 2)

    ax2.plot(bin_center, bin_sigma, color=TEAL, lw=2,
             marker="o", markersize=5, label="平均 σ")
    ax2.set_xlabel("|误差| 分桶中心（归一化）")
    ax2.set_ylabel("平均预测标准差 σ")
    ax2.set_title("误差大小 vs 预测不确定性\n（理想情况：σ 随误差增大而增大）")
    ax2.legend(fontsize=9)

    fig.suptitle(f"误差分析 — {dataset}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(result_dir, "error_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [图] error_distribution.png")


# ---------------------------------------------------------------------------
# 主入口：一键生成全部图表
# ---------------------------------------------------------------------------
def plot_all(history: Dict, result_dir: str, dataset: str = "solar"):
    """
    生成全部图表并保存到 result_dir。
    在 main.py 训练完后直接调用，或通过命令行指定 history JSON。

    图表清单：
      convergence.png          Val CRPS 收敛曲线
      loss_components.png      Train Loss / NLL / MI 分解
      metrics_curve.png        Val MAE + RMSE 双曲线
      baselines_bar.png        与论文基线柱状对比
      prediction_intervals.png 95% 预测区间可视化（需 test_mu）
      reliability_diagram.png  PICP 校准图（需 test_mu）
      error_distribution.png   误差分布 + 误差-不确定性相关图（需 test_mu）
    """
    print(f"\n[Plot] 开始生成图表，保存至 {result_dir}")
    os.makedirs(result_dir, exist_ok=True)

    test_metrics = history.get("test_metrics", {})
    has_preds    = "test_mu" in history

    # 训练过程图（只需 history，无需模型）
    plot_convergence(history,     result_dir, dataset)
    plot_loss_components(history, result_dir, dataset)
    plot_metrics_curve(history,   result_dir, dataset)

    # # 测试集汇总图
    # if test_metrics:
    #     plot_baselines_bar(test_metrics, result_dir, dataset)

    # 需要预测数组的图
    if has_preds:
        plot_prediction_intervals(history, result_dir, dataset)
        plot_reliability_diagram(history,  result_dir, dataset)
        plot_error_distribution(history,   result_dir, dataset)
    else:
        print("  [提示] history 中无 test_mu，跳过需要模型输出的三张图")
        print("         用新版 train.py 重新训练即可自动生成")

    print(f"[Plot] 全部图表已保存至 {result_dir}\n")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridCFN 结果可视化")
    parser.add_argument(
        "--history", type=str, required=True,
        help="history JSON 文件路径，例如 result/solar/20260403_111007/history_solar_20260403_111007.json",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="数据集名称（solar/electricity/weather）。不指定则从 JSON 路径自动推断。",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    result_dir   = str(history_path.parent)

    # 自动推断数据集名
    if args.dataset:
        dataset = args.dataset
    else:
        for ds in ("solar", "electricity", "weather"):
            if ds in history_path.stem:
                dataset = ds
                break
        else:
            dataset = "solar"

    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    plot_all(history, result_dir, dataset)
