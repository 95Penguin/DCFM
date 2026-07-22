"""Render the DCFM framework as an editable, publication-ready vector figure.

Outputs are written to ``result/plot/framework/`` in SVG, PDF, and PNG.
The SVG/PDF versions are recommended for the thesis manuscript.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "result" / "plot" / "framework"

# Palette: neutral module shells, with blue/green reserved for the two branches.
INK = "#20242A"
OUTLINE = "#2D3138"
SHELL = "#F7F8FA"
INPUT = "#E9EEF4"
HDRL = "#EEF1F4"
CRE = "#EAF2F1"
CFM = "#FBF4E5"
LONG_BAND = "#D9EAFB"
SHORT_BAND = "#E0F0DE"
LONG = "#51A7E8"
LONG_EDGE = "#267DBB"
SHORT = "#7DCB72"
SHORT_EDGE = "#4C9D50"
BLOCK = "#DEE5EC"
ACCENT = "#F7E7C2"


def box(ax, x, y, width, height, text="", *, face=BLOCK, edge=OUTLINE,
        lw=1.0, radius=0.12, dashed=False, fontsize=9.2, weight="normal",
        rotation=0, zorder=2):
    linestyle = (0, (3, 3)) if dashed else "solid"
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight, color=INK, rotation=rotation,
                zorder=zorder + 1)
    return patch


def arrow(ax, start, end, *, text=None, text_offset=(0, 0.12), lw=1.0,
          style="-|>", connection="arc3", zorder=4):
    line = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=11, linewidth=lw,
        color=INK, connectionstyle=connection, shrinkA=0, shrinkB=0, zorder=zorder,
    )
    ax.add_patch(line)
    if text:
        cx = (start[0] + end[0]) / 2 + text_offset[0]
        cy = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(cx, cy, text, fontsize=8.2, ha="center", va="center", color=INK,
                zorder=zorder + 1)
    return line


def feature(ax, x, y, *, color, edge, vertical=False, n=4, size=0.19):
    for index in range(n):
        dx = 0 if vertical else index * size
        dy = index * size if vertical else 0
        ax.add_patch(Rectangle((x + dx, y + dy), size, size, facecolor=color,
                               edgecolor=edge, linewidth=0.55, zorder=5))


def graph_icon(ax, x, y, *, scale=1.0):
    points = [(x, y + 0.34 * scale), (x + 0.46 * scale, y + 0.62 * scale),
              (x + 0.86 * scale, y + 0.23 * scale), (x + 0.39 * scale, y - 0.12 * scale),
              (x + 0.93 * scale, y - 0.34 * scale)]
    edges = [(0, 1), (1, 2), (0, 3), (1, 3), (2, 3), (3, 4)]
    for i, j in edges:
        ax.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]],
                color="#5B616A", lw=0.75, zorder=5)
    for px, py in points:
        ax.add_patch(Circle((px, py), 0.10 * scale, facecolor="white",
                            edgecolor=OUTLINE, linewidth=0.8, zorder=6))


def stacked_graph(ax, x, y, *, label, edge):
    for offset in (0.18, 0.09, 0):
        box(ax, x + offset, y + offset, 1.02, 1.15, face="white", edge=edge,
            lw=0.85, radius=0.12, zorder=4)
    graph_icon(ax, x + 0.20, y + 0.38, scale=0.72)
    ax.text(x + 0.60, y + 1.37, label, ha="center", va="bottom", fontsize=8.8,
            color=INK, zorder=6)


def setup_axis():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(18, 5.2))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 6)
    ax.axis("off")
    return fig, ax


def module_shell(ax, x, width, title, index, *, fill=SHELL, title_size=10):
    box(ax, x, 0.28, width, 5.42, face=fill, edge=OUTLINE, lw=1.2, radius=0.32,
        dashed=True, zorder=0)
    ax.text(x + width / 2, 5.48, f"({index}) {title}", ha="center", va="center",
            fontsize=title_size, weight="bold", color=INK, zorder=8, linespacing=1.05)


def draw_input(ax):
    module_shell(ax, 0.15, 2.5, "Input", 1, fill=INPUT)
    ax.text(1.40, 4.90, r"$\mathbf{X}$", ha="center", va="center", fontsize=12)
    for row in range(4):
        for col in range(4):
            ax.add_patch(Rectangle((0.78 + col * 0.26, 3.64 + row * 0.26), 0.22, 0.22,
                                   facecolor="white", edgecolor=OUTLINE, linewidth=0.65))
    ax.text(1.40, 2.98, r"$G=(\mathcal{V},\mathcal{E},\mathbf{A})$", ha="center",
            va="center", fontsize=10.5)
    graph_icon(ax, 0.53, 1.95, scale=1.25)


def draw_hdrl(ax):
    module_shell(ax, 2.98, 8.13,
                 "Heterogeneous Disentangled\nRepresentation Learning (HDRL)", 2,
                 fill=HDRL, title_size=8.5)
    # Branch bands only identify the two signal components.
    box(ax, 4.77, 3.03, 5.88, 2.18, face=LONG_BAND, edge="none", lw=0,
        radius=0.30, zorder=1)
    box(ax, 4.77, 0.55, 5.88, 2.18, face=SHORT_BAND, edge="none", lw=0,
        radius=0.30, zorder=1)

    box(ax, 3.18, 1.33, 1.55, 2.83, "Adaptive\nMulti-scale\nFrequency\nDecomposition",
        face="#D8ECEE", edge=OUTLINE, lw=1.0, radius=0.22, fontsize=8.7, zorder=3)
    arrow(ax, (2.65, 2.75), (3.18, 2.75))
    arrow(ax, (4.73, 2.98), (5.12, 4.12), connection="angle,angleA=0,angleB=90,rad=0")
    arrow(ax, (4.73, 2.41), (5.12, 1.70), connection="angle,angleA=0,angleB=-90,rad=0")
    ax.text(4.84, 4.39, r"$X_L$", fontsize=10)
    ax.text(4.84, 1.36, r"$X_S$", fontsize=10)

    stacked_graph(ax, 5.15, 3.26, label="LowRankGCN", edge="#647280")
    stacked_graph(ax, 5.15, 0.78, label="SparseGCN", edge="#647280")
    box(ax, 7.98, 3.72, 1.48, 0.75, "TCN", face=BLOCK, edge=OUTLINE, fontsize=9.5)
    box(ax, 7.98, 1.24, 1.48, 0.75, "TCN", face=BLOCK, edge=OUTLINE, fontsize=9.5)
    arrow(ax, (6.35, 4.04), (7.98, 4.04))
    arrow(ax, (6.35, 1.56), (7.98, 1.56))
    arrow(ax, (9.46, 4.04), (9.75, 4.04))
    arrow(ax, (9.46, 1.56), (9.75, 1.56))
    feature(ax, 9.76, 3.94, color="#A9D4F5", edge=LONG_EDGE, n=4)
    feature(ax, 9.76, 1.46, color="#B8E1B4", edge=SHORT_EDGE, n=4)
    ax.text(10.25, 4.40, r"$\mathbf{H}_L$", fontsize=10)
    ax.text(10.25, 1.92, r"$\mathbf{H}_S$", fontsize=10)
    box(ax, 9.25, 2.31, 1.58, 0.58, "Mutual Information\nConstraint", face=ACCENT,
        edge="#C19D58", lw=0.9, radius=0.10, dashed=True, fontsize=7.0, zorder=4)
    arrow(ax, (10.18, 3.93), (10.00, 2.89), style="-|>", lw=0.75)
    arrow(ax, (10.18, 1.66), (10.00, 2.31), style="-|>", lw=0.75)


def draw_cre(ax):
    module_shell(ax, 11.40, 5.64, "Conditional Representation\nEnhancement (CRE)", 3,
                 fill=CRE, title_size=8.5)
    box(ax, 11.67, 3.03, 5.12, 2.18, face=LONG_BAND, edge="none", lw=0,
        radius=0.30, zorder=1)
    box(ax, 11.67, 0.55, 5.12, 2.18, face=SHORT_BAND, edge="none", lw=0,
        radius=0.30, zorder=1)

    # ECA: a compact multi-scale temporal aggregation block.
    box(ax, 12.32, 3.57, 2.32, 1.34, "ECA", face="white", edge=OUTLINE,
        lw=1.0, radius=0.03, dashed=True, fontsize=10, zorder=3)
    for px in (12.65, 13.22, 14.38):
        feature(ax, px, 3.95, color="#A9D4F5", edge=LONG_EDGE, vertical=True, n=4)
    for px in (13.00, 13.57):
        ax.add_patch(Circle((px, 4.30), 0.11, facecolor="white", edgecolor="#6D7884", lw=0.65, zorder=5))
        ax.text(px, 4.30, "+", ha="center", va="center", fontsize=8, zorder=6)
    arrow(ax, (10.50, 4.04), (12.32, 4.04), connection="angle,angleA=0,angleB=90,rad=0")
    arrow(ax, (14.64, 4.23), (15.07, 4.23))
    feature(ax, 15.10, 3.95, color=LONG, edge=LONG_EDGE, vertical=True, n=4)
    ax.text(15.51, 4.00, r"$\mathbf{C}_L$", fontsize=10)

    # ESE: long-term context gates short-term spatial messages.
    box(ax, 11.88, 0.74, 3.98, 1.78, "ESE", face="white", edge=OUTLINE,
        lw=1.0, radius=0.03, dashed=True, fontsize=10, zorder=3)
    for y, color, edge, label in ((2.10, "#A9D4F5", LONG_EDGE, r"$\mathbf{H}_L^i,\mathbf{H}_L^j$"),
                                  (1.17, "#B8E1B4", SHORT_EDGE, r"$\mathbf{H}_S^i,\mathbf{H}_S^j$")):
        box(ax, 12.08, y - 0.12, 1.22, 0.50, face="#F7F9FA", edge="#647280",
            lw=0.75, radius=0.10, dashed=True, zorder=4)
        feature(ax, 12.43, y, color=color, edge=edge, n=4)
        ax.text(12.12, y + 0.13, label, fontsize=7.0, ha="left", va="center")
    box(ax, 13.65, 1.01, 0.66, 1.25, "Gating\nUnit", face=BLOCK, edge=OUTLINE,
        lw=0.85, radius=0.03, fontsize=7.3, rotation=270)
    box(ax, 14.78, 1.16, 0.26, 0.92, "Aggregation", face=BLOCK, edge=OUTLINE,
        lw=0.85, radius=0.03, fontsize=6.5, rotation=270)
    arrow(ax, (10.50, 4.04), (11.88, 2.13), connection="angle,angleA=0,angleB=-90,rad=0")
    arrow(ax, (10.50, 1.56), (11.88, 1.42), connection="angle,angleA=0,angleB=90,rad=0")
    arrow(ax, (13.30, 2.13), (13.65, 2.13))
    arrow(ax, (13.30, 1.42), (13.65, 1.42))
    arrow(ax, (14.31, 1.64), (14.78, 1.64))
    arrow(ax, (15.04, 1.64), (15.43, 1.64))
    feature(ax, 15.45, 1.30, color=SHORT, edge=SHORT_EDGE, vertical=True, n=4)
    ax.text(15.88, 1.59, r"$\mathbf{C}_S$", fontsize=10)


def draw_cfm(ax):
    module_shell(ax, 17.32, 4.50, "Conditional Flow Matching\nPrediction (CFM)", 4,
                 fill=CFM, title_size=8.5)
    box(ax, 18.12, 3.90, 2.73, 0.74, "Conditional Vector Field", face="#FFF9F0",
        edge=OUTLINE, lw=0.95, radius=0.02, fontsize=9.1)
    arrow(ax, (16.04, 4.23), (18.12, 4.23), text="Bias", text_offset=(0, 0.16),
          connection="angle,angleA=0,angleB=90,rad=0")
    arrow(ax, (16.16, 1.64), (18.12, 4.10), text="Scale", text_offset=(-0.13, 0.05),
          connection="angle,angleA=0,angleB=90,rad=0")
    ax.text(20.87, 5.12, "Noise", fontsize=9.5, ha="center")
    arrow(ax, (20.50, 5.08), (20.50, 4.64), connection="angle,angleA=180,angleB=90,rad=0")
    arrow(ax, (19.49, 3.90), (19.49, 3.36))
    box(ax, 18.12, 2.47, 2.73, 0.90, "ODE Solver", face="#FFF9F0", edge=OUTLINE,
        lw=0.95, radius=0.02, fontsize=9.5)
    for index, (px, py) in enumerate(((18.56, 2.72), (18.95, 2.95), (19.34, 3.13), (19.74, 3.24), (20.24, 3.05))):
        ax.add_patch(Circle((px, py), 0.10, facecolor="#FFE5B4" if index < 4 else "#F7C37B",
                            edgecolor="#C88F38", lw=0.65, zorder=5))
        if index:
            ax.plot([prev[0], px], [prev[1], py], color="#D3A457", lw=0.65, ls=(0, (2, 2)), zorder=4)
        prev = (px, py)
    arrow(ax, (19.49, 2.47), (19.49, 1.96))
    box(ax, 18.12, 1.57, 2.73, 0.39, "Prediction Samples", face="#F3F5F7", edge="#78828E",
        lw=0.8, radius=0.02, fontsize=8.8)
    arrow(ax, (19.49, 1.57), (19.49, 1.14))
    box(ax, 18.12, 0.65, 2.73, 0.49, "Probabilistic Forecast", face="#E8F4E6", edge="#74A96C",
        lw=0.9, radius=0.02, fontsize=9.0)


def main():
    fig, ax = setup_axis()
    draw_input(ax)
    draw_hdrl(ax)
    draw_cre(ax)
    draw_cfm(ax)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"dcfm_framework.{suffix}", dpi=400,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved framework figure to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
