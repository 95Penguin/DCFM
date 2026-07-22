"""Unified Matplotlib template for the thesis result figures.

Run with the local plotting environment:
    .venv-paper-plots/bin/python plot_paper_result_figures.py

The script uses only the values already reported in Tables 4-2, 4-3 and 4-6.
It creates two alternatives for Figure 3 (bar and dumbbell styles) and the
PICP--PINAW scatter plot for Figure 4.  Use only one Figure-3 alternative in
the manuscript; the dumbbell version is recommended for the main text.
"""

import os
from pathlib import Path

# Keep the Matplotlib font cache inside the project, making the figure script
# reproducible on this computer without modifying the user's home directory.
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
# Do not invoke the macOS GUI backend.  This script is only for file export,
# and Agg prevents a Python.app crash on the current Python 3.14 installation.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


OUTPUT_DIR = PROJECT_ROOT / "result" / "plot" / "paper_figures"
DATASETS = ["Solar-AL", "SDWPF", "Electricity", "PJM"]

# All three reported criteria are minimized. "Best baseline" is the lowest
# result among the non-DCFM models in the corresponding table entry.
METRICS = {
    "MAE": {"DCFM": [0.0794, 0.2423, 0.0154, 0.0294], "Best baseline": [0.0846, 0.2478, 0.0163, 0.0302]},
    "RMSE": {"DCFM": [0.2021, 0.4088, 0.1125, 0.0568], "Best baseline": [0.2187, 0.4053, 0.1169, 0.0603]},
    "CRPS": {"DCFM": [0.0562, 0.1812, 0.0118, 0.0217], "Best baseline": [0.0604, 0.1794, 0.0121, 0.0218]},
}

INTERVAL_METRICS = {
    "TSDiff": {"PICP": [0.9240, 0.9185, 0.9175, 0.8524], "PINAW": [0.0678, 0.3150, 0.0215, 0.0275]},
    "TSFlow": {"PICP": [0.9455, 0.9272, 0.9377, 0.8865], "PINAW": [0.0504, 0.3265, 0.0230, 0.0245]},
    "DCFM": {"PICP": [0.9491, 0.9347, 0.9484, 0.9564], "PINAW": [0.0399, 0.3796, 0.0160, 0.0216]},
}

BLUE = "#1F5FD0"
ORANGE = "#D97706"
RED = "#D83A3A"
GRAY = "#777777"
GRID = "#D9DDE3"


def setup_style() -> None:
    """Match the default sans-serif style used by plot_prediction.py (Fig. 4)."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"{name}.{extension}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def relative_improvement(values: dict[str, list[float]]) -> np.ndarray:
    baseline = np.asarray(values["Best baseline"])
    return (baseline - np.asarray(values["DCFM"])) / baseline * 100.0


def plot_relative_bar() -> None:
    """Figure 3 alternative A: signed gains, concise and table-complementary."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, layout="constrained")
    positions = np.arange(len(DATASETS))
    for ax, (metric, values) in zip(axes, METRICS.items()):
        gains = relative_improvement(values)
        ax.bar(positions, gains, width=0.64, color=[BLUE if gain >= 0 else ORANGE for gain in gains], edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="#404040", linewidth=0.8)
        ax.set_title(metric, fontweight="normal")
        ax.set_xticks(positions, DATASETS)
        ax.set_ylim(-10, 10)
        ax.set_yticks(np.arange(-10, 11, 5))
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for pos, gain in zip(positions, gains):
            ax.annotate(f"{gain:+.1f}%", (pos, gain), xytext=(0, 4 if gain >= 0 else -5), textcoords="offset points", ha="center", va="bottom" if gain >= 0 else "top", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Relative improvement over best baseline (%)")
    fig.legend(handles=[Patch(facecolor=BLUE, label="Improved"), Patch(facecolor=ORANGE, label="Decreased")], loc="outside lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
    save_figure(fig, "fig2_relative_performance_bar")


def plot_relative_dumbbell() -> None:
    """Figure 3 alternative B: recommended main-text comparison view.

    The vertical reference at 100% denotes the strongest non-DCFM baseline.
    A DCFM point left of it represents a smaller error/CRPS and hence a gain.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), sharey=True, layout="constrained")
    y = np.arange(len(DATASETS))
    for ax, (metric, values) in zip(axes, METRICS.items()):
        ratio = np.asarray(values["DCFM"]) / np.asarray(values["Best baseline"]) * 100.0
        ax.axvline(100, color="#555555", linewidth=1.0, linestyle="--", zorder=0)
        for y_pos, ratio_value in zip(y, ratio):
            color = BLUE if ratio_value <= 100 else ORANGE
            ax.hlines(y_pos, min(100, ratio_value), max(100, ratio_value), color=color, linewidth=2.2, zorder=1)
            ax.scatter(100, y_pos, s=42, color="#969696", edgecolor="white", linewidth=0.8, zorder=2)
            ax.scatter(ratio_value, y_pos, s=56, color=color, edgecolor="white", linewidth=0.9, zorder=3)
            ax.annotate(f"{ratio_value:.1f}", (ratio_value, y_pos), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.2, color=color, fontweight="bold")
        ax.set_title(metric, fontweight="normal")
        ax.set_xlim(88, 110)
        ax.set_xticks([90, 95, 100, 105, 110])
        ax.set_xlabel("DCFM / best baseline (%)")
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, DATASETS)
    axes[0].invert_yaxis()
    fig.legend(
        handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=7, label="DCFM"), Line2D([0], [0], marker="o", color="none", markerfacecolor="#969696", markeredgecolor="white", markersize=7, label="Best baseline")],
        loc="outside lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )
    save_figure(fig, "fig3b_relative_performance_dumbbell")


def plot_picp_pinaw() -> None:
    """Figure 4: coverage--sharpness scatter plot, suitable for main text."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.2), sharey=True, layout="constrained")
    styles = {"TSDiff": {"color": GRAY, "marker": "s", "size": 46}, "TSFlow": {"color": RED, "marker": "^", "size": 54}, "DCFM": {"color": BLUE, "marker": "o", "size": 60}}
    for index, (dataset, ax) in enumerate(zip(DATASETS, axes)):
        widths = [result["PINAW"][index] for result in INTERVAL_METRICS.values()]
        padding = max((max(widths) - min(widths)) * 0.30, 0.0035)
        ax.set_xlim(min(widths) - padding, max(widths) + padding)
        ax.set_ylim(0.84, 0.97)
        ax.axhline(0.95, color="#4A4A4A", linestyle="--", linewidth=0.9, zorder=0)
        ax.text(0.98, 0.953, "target = 0.95", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color="#555555")
        for method, results in INTERVAL_METRICS.items():
            style = styles[method]
            x, y = results["PINAW"][index], results["PICP"][index]
            ax.scatter(x, y, s=style["size"], color=style["color"], marker=style["marker"], edgecolor="white", linewidth=0.9, zorder=3)
            ax.annotate(method, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9.5, color=style["color"], fontweight="bold" if method == "DCFM" else "normal")
        ax.set_title(dataset, fontweight="normal")
        ax.set_xlabel("PINAW ↓")
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("PICP (target = 0.95)")
    save_figure(fig, "fig3_picp_pinaw_tradeoff")


if __name__ == "__main__":
    setup_style()
    plot_relative_bar()
    plot_relative_dumbbell()
    plot_picp_pinaw()
    print(f"Figures saved to: {OUTPUT_DIR}")
