#!/usr/bin/env python3
"""Evaluate sensitivity to the number of generated trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import aggregate_metrics, canonical_samples, canonical_target, load_array, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True,
                        help="Full probabilistic samples, not a mean-only prediction")
    parser.add_argument("--target", required=True)
    parser.add_argument("--sample-counts", nargs="+", type=int,
                        default=[5, 10, 20, 50, 100])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="paper_extra_results/sample_count")
    return parser.parse_args()


def main():
    args = parse_args()
    y_raw = load_array(args.target)
    y = canonical_target(y_raw)
    samples = canonical_samples(load_array(args.prediction), y)
    available = samples.shape[0]
    invalid = [count for count in args.sample_counts if count > available]
    if invalid:
        raise ValueError(f"Requested {invalid}, but file has only {available} samples")
    if available == 1:
        raise ValueError("This is a mean-only prediction; rerun NFE sweep with --save-samples")

    rng = np.random.default_rng(args.seed)
    rows = []
    raw = {}
    for count in args.sample_counts:
        trials = []
        for _ in range(args.repeats):
            index = rng.choice(available, size=count, replace=False)
            trials.append(aggregate_metrics(samples[index], y_raw))
        raw[str(count)] = trials
        for metric in trials[0]:
            values = np.asarray([trial[metric] for trial in trials])
            rows.append({
                "samples": count,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "repeats": args.repeats,
            })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "sample_count_sensitivity.json", {
        "available_samples": available, "seed": args.seed, "trials": raw})
    with (out / "sample_count_sensitivity.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out / 'sample_count_sensitivity.csv'}")


if __name__ == "__main__":
    main()
