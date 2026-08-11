#!/usr/bin/env python3
"""Paired bootstrap tests using matched test windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import canonical_samples, canonical_target, load_array, metrics_per_window, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcfm", required=True, help="DCFM prediction .npy")
    parser.add_argument("--baseline", action="append", required=True,
                        help="LABEL=prediction.npy; may be repeated")
    parser.add_argument("--target", required=True, help="ground_truth.npy")
    parser.add_argument("--metrics", nargs="+", default=["MAE", "RMSE", "CRPS"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="paper_extra_results/bootstrap")
    return parser.parse_args()


def main():
    args = parse_args()
    y_raw = load_array(args.target)
    y = canonical_target(y_raw)
    dcfm = canonical_samples(load_array(args.dcfm), y)
    dcfm_window = metrics_per_window(dcfm, y_raw)
    rng = np.random.default_rng(args.seed)
    rows = []
    details = {}
    for item in args.baseline:
        if "=" not in item:
            raise ValueError("--baseline must be LABEL=PATH")
        label, path = item.split("=", 1)
        baseline = canonical_samples(load_array(path), y)
        base_window = metrics_per_window(baseline, y_raw)
        details[label] = {}
        for metric in args.metrics:
            # Positive improvement means DCFM has lower error.
            diff = base_window[metric] - dcfm_window[metric]
            n = len(diff)
            boot = np.empty(args.bootstrap, dtype=np.float64)
            for start in range(0, args.bootstrap, 500):
                count = min(500, args.bootstrap - start)
                indices = rng.integers(0, n, size=(count, n))
                boot[start:start + count] = diff[indices].mean(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            p_two_sided = min(1.0, 2.0 * min(
                float((boot <= 0).mean()), float((boot >= 0).mean())))
            row = {
                "baseline": label,
                "metric": metric,
                "dcfm": float(dcfm_window[metric].mean()),
                "baseline_value": float(base_window[metric].mean()),
                "improvement": float(diff.mean()),
                "relative_improvement_percent": float(
                    100 * diff.mean() / (base_window[metric].mean() + 1e-12)),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "p_value_two_sided": p_two_sided,
                "n_windows": n,
            }
            rows.append(row)
            details[label][metric] = row

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "paired_bootstrap.json", {
        "bootstrap_repetitions": args.bootstrap,
        "seed": args.seed,
        "interpretation": "positive improvement favors DCFM",
        "comparisons": details,
    })
    with (out / "paired_bootstrap.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out / 'paired_bootstrap.csv'}")


if __name__ == "__main__":
    main()
