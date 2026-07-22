"""
命令eg:
python3 plot_prediction.py result/baselines/solar/20260709_174923 \
  --models DCFM MTGNN TSFlow PatchTST \
  --zoom 60 100 \
  --out result/plot/PredictionComparison.png
"""


import argparse
import glob
import os
from typing import List, Optional, Tuple

# This script exports figures only.  Use the non-interactive backend so it
# remains stable on macOS environments without a GUI-capable Python backend.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _normalize_name(name: str, style_map: dict) -> str:
    """Return the style-map key that matches 'name' case-insensitively."""
    aliases = {"gridcfn": "DCFM", "dcfm": "DCFM"}
    alias = aliases.get(name.lower())
    if alias is not None:
        return alias
    if name in style_map:
        return name
    for key in style_map:
        if key.lower() == name.lower():
            return key
    return name


def load_prediction_arrays(root_dir: str,
                           models: Optional[List[str]] = None):
    root_dir = os.path.abspath(root_dir)
    gt_path = os.path.join(root_dir, "ground_truth.npy")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing ground truth file: {gt_path}")

    pred_files = sorted(glob.glob(os.path.join(root_dir, "*_prediction.npy")))
    if not pred_files:
        raise FileNotFoundError(f"No '*_prediction.npy' files found in {root_dir}")

    data = {"ground_truth": np.load(gt_path)}
    style_map = _build_style_map()
    for path in pred_files:
        name = os.path.basename(path).replace("_prediction.npy", "")
        # normalize to match the style-map key (case-insensitive)
        normalized = _normalize_name(name, style_map)
        if models is not None:
            if not any(m.lower() in {name.lower(), normalized.lower()} for m in models):
                continue
        data[normalized] = np.load(path)

    if "DCFM" not in data:
        print("Warning: DCFM prediction file not found in the directory.")
    return data


def _build_style_map() -> dict:
    return {
        "DCFM":     {"color": "#1F5FD0", "linewidth": 2.8, "linestyle": "-"},
        "MTGNN":    {"color": "#26924A", "linewidth": 2.2, "linestyle": "--"},
        "TSFlow":   {"color": "#D83A3A", "linewidth": 2.2, "linestyle": "-."},
        "PatchTST": {"color": "#8B5FBF", "linewidth": 1.8, "linestyle": ":"},
    }


def _plot_zoom(ax, x: np.ndarray, y_gt: np.ndarray, pred_series: list,
               zoom_range: Tuple[int, int]):
    # inset axes
    axins = inset_axes(ax, width="40%", height="30%", loc="lower left",
                       bbox_to_anchor=(0.55, 0.12, 0.4, 0.4), bbox_transform=ax.transAxes)
    # clip zoom_range to valid indices
    z0 = max(0, zoom_range[0])
    z1 = min(len(x), zoom_range[1])
    if z0 >= z1:
        z0 = max(0, len(x) - 1 - 10)
        z1 = len(x)
        print(f"Warning: zoom_range out of bounds; clipped to [{z0}, {z1})")
    x_zoom = x[z0:z1]
    axins.plot(x_zoom, y_gt[z0:z1], color="black", linewidth=2.5,
               linestyle=(0, (5, 1.5)))
    for name, y_pred, style in pred_series:
        axins.plot(x_zoom, y_pred[z0:z1], label=name, **style)
    axins.grid(alpha=0.18)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("Zoomed view", fontsize=9)

    # draw rectangle on main axes to indicate zoom area
    x0 = x[z0]
    x1 = x[z1-1]
    yrange = ax.get_ylim()
    rect = Rectangle((x0, yrange[0]), width=(x1 - x0), height=(yrange[1] - yrange[0]),
                     linewidth=1.0, edgecolor='gray', facecolor='none', linestyle='-')
    ax.add_patch(rect)

    # connection lines between inset and rectangle
    bbox = axins.get_position()
    con1 = ConnectionPatch(xyA=(0.05, 0.95), coordsA=axins.transAxes,
                           xyB=(x0, yrange[1]), coordsB=ax.transData, color='gray')
    con2 = ConnectionPatch(xyA=(0.95, 0.95), coordsA=axins.transAxes,
                           xyB=(x1, yrange[1]), coordsB=ax.transData, color='gray')
    axins.add_artist(con1)
    axins.add_artist(con2)


def _reconstruct_from_windows(arr: np.ndarray, node: int, feature: int,
                              stride: int, merge_method: str):
    # arr: (total, T_out, N, F)
    total, T_out, N, F = arr.shape
    out_len = (total - 1) * stride + T_out
    if merge_method == 'median':
        buckets = [[] for _ in range(out_len)]
        for s in range(total):
            for t in range(T_out):
                idx = s * stride + t
                val = float(arr[s, t, node, feature])
                buckets[idx].append(val)
        out = np.zeros((out_len,), dtype=float)
        for i in range(out_len):
            if not buckets[i]:
                out[i] = 0.0
            else:
                out[i] = float(np.median(np.array(buckets[i])))
        return out

    sum_arr = np.zeros((out_len,), dtype=float)
    count = np.zeros((out_len,), dtype=int)
    last_arr = np.zeros((out_len,), dtype=float)
    for s in range(total):
        for t in range(T_out):
            idx = s * stride + t
            val = arr[s, t, node, feature]
            sum_arr[idx] += val
            count[idx] += 1
            last_arr[idx] = val

    if merge_method == 'last':
        return last_arr

    # default avg
    out = np.zeros_like(sum_arr)
    mask = count > 0
    out[mask] = sum_arr[mask] / count[mask]
    return out


def plot_sample(root_dir: str,
                sample: int = 0,
                node: int = 0,
                feature: int = 0,
                models: Optional[List[str]] = None,
                zoom_range: Optional[Tuple[int, int]] = None,
                save_path: str = "result/plot/PredictionComparison.png",
                concat: bool = False,
                concat_samples: Optional[Tuple[int, int]] = None,
                merge_method: str = 'none',
                stride: Optional[int] = None,
                horizon: Optional[int] = None,
                start: int = 0,
                length: Optional[int] = None):
    data = load_prediction_arrays(root_dir, models=models)
    gt = data.pop("ground_truth")

    # prepare figure with larger size and nicer spines
    # fig, ax = plt.subplots(figsize=(14, 5))
    fig, ax = plt.subplots(figsize=(10, 4))
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # Fixed-horizon mode creates a valid continuous timeline: each point is
    # the same forecast horizon from consecutive, overlapping test windows.
    # This must not be confused with directly concatenating full windows.
    if horizon is not None:
        if concat:
            raise ValueError("--horizon cannot be used together with --concat")
        if not 0 <= horizon < gt.shape[1]:
            raise ValueError(f"horizon must be in [0, {gt.shape[1] - 1}]")
        total = gt.shape[0]
        stop = total if length is None else min(total, start + length)
        if not 0 <= start < stop:
            raise ValueError(f"Invalid slice: start={start}, stop={stop}")
        # Use local consecutive time steps in the figure. The absolute window
        # index is an implementation detail and is not a physical timestamp.
        x = np.arange(1, stop - start + 1)
        y_gt = gt[start:stop, horizon, node, feature]
    # If concat=True, reconstruct long timeline according to merge/stride
    elif concat:
        total, T_out, N, F = gt.shape
        # select sample subset for concatenation if provided
        if concat_samples is not None:
            s0, s1 = concat_samples
            gt = gt[s0:s1]
            for k in list(data.keys()):
                data[k] = data[k][s0:s1]
            total = gt.shape[0]

        if stride is None:
            stride = T_out

        if merge_method == 'none':
            gt_long = gt.reshape(total * T_out, N, F)
            x = np.arange(gt_long.shape[0])
            y_gt = gt_long[:, node, feature]
        else:
            y_gt = _reconstruct_from_windows(gt, node, feature, stride, merge_method)
            x = np.arange(y_gt.shape[0])
    else:
        x = np.arange(gt.shape[1])
        y_gt = gt[sample, :, node, feature]

    ax.plot(x, y_gt, label="Ground Truth", color="#111111", linewidth=2.8,
            linestyle=(0, (5, 1.5)))

    style_map = _build_style_map()
    pred_series = []
    for name, arr in sorted(data.items()):
        if horizon is not None:
            y_pred = arr[start:stop, horizon, node, feature]
        elif concat:
            if merge_method == 'none':
                arr_long = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
                y_pred = arr_long[:, node, feature]
            else:
                y_pred = _reconstruct_from_windows(arr, node, feature, stride, merge_method)
        else:
            y_pred = arr[sample, :, node, feature]
        style = style_map.get(name, {"linewidth": 1.8, "linestyle": "-"})
        # draw slightly transparent fill for visual thickness
        ax.plot(x, y_pred, label=name, **style, alpha=0.92)
        pred_series.append((name, y_pred, style))

    ax.set_xlabel("Time Step", fontsize=14)
    ax.set_ylabel("Electricity Load (raw scale)", fontsize=14)
    if horizon is not None:
        ax.set_title(f"{horizon + 1}-step-ahead Forecasting Case", fontsize=16)
    else:
        ax.set_title("Prediction Visualization", fontsize=16)
    ax.grid(color="#D9DEE7", alpha=0.65, linestyle='-')
    ax.legend(fontsize=11, loc="upper left", frameon=True, fancybox=True, framealpha=0.9)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(axis="both", labelsize=11)

    # inset/zoom box (enabled when --zoom is given)
    if zoom_range is not None:
        _plot_zoom(ax, x, y_gt, pred_series, zoom_range)

    # use subplots_adjust to avoid tight_layout issues with inset artists
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.10)
    save_plot(fig, save_path)


def save_plot(plt_obj, save_path: str):
    save_path = os.path.abspath(save_path)
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    base, ext = os.path.splitext(save_path)
    if ext == "":
        save_path = save_path + ".png"
        ext = ".png"

    if ext.lower() not in {".png"}:
        save_path = base + ".png"
        ext = ".png"

    # plt_obj may be a Figure or the pyplot module
    try:
        # Figure-like
        plt_obj.savefig(save_path, dpi=600, bbox_inches="tight")
        # close figure via pyplot
        import matplotlib.pyplot as _plt
        _plt.close(plt_obj)
    except Exception:
        # fallback: assume pyplot module
        plt_obj.savefig(save_path, dpi=600, bbox_inches="tight")
        plt_obj.close()
    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot comparison of ground truth and model predictions from saved npy arrays.")
    parser.add_argument("root_dir", type=str,
                        help="Directory containing ground_truth.npy and *_prediction.npy files")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample index to plot")
    parser.add_argument("--node", type=int, default=0,
                        help="Node index to plot")
    parser.add_argument("--feature", type=int, default=0,
                        help="Feature index to plot")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Optional list of model names to plot (e.g. DCFM MTGNN TSFlow)")
    parser.add_argument("--zoom", nargs=2, type=int, default=None,
                        help="Optional zoom range as two ints: start end")
    parser.add_argument("--concat", action="store_true",
                        help="Concatenate all samples' T_out windows into a long timeseries before plotting")
    parser.add_argument("--concat-samples", nargs=2, type=int, default=None,
                        help="Optional sample range [start end) to use when --concat is set")
    parser.add_argument("--merge-method", choices=["none", "avg", "last", "median"], default="none",
                        help="When --concat is used, merge overlapping windows using 'avg' or 'last' (default none)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride between consecutive windows when reconstructing a timeline (default=T_out)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Plot one fixed forecast horizon across consecutive windows (0-based).")
    parser.add_argument("--start", type=int, default=0,
                        help="Start sample index for --horizon mode.")
    parser.add_argument("--length", type=int, default=None,
                        help="Number of consecutive samples for --horizon mode.")
    parser.add_argument("--out", type=str, default="result/plot/PredictionComparison.png",
                        help="Output PNG file path")
    args = parser.parse_args()

    zoom_range = tuple(args.zoom) if args.zoom is not None else None
    concat_samples = tuple(args.concat_samples) if args.concat_samples is not None else None
    plot_sample(args.root_dir,
                sample=args.sample,
                node=args.node,
                feature=args.feature,
                models=args.models,
                zoom_range=zoom_range,
                save_path=args.out,
                concat=args.concat,
                concat_samples=concat_samples,
                merge_method=args.merge_method,
                stride=args.stride,
                horizon=args.horizon,
                start=args.start,
                length=args.length)
