#!/usr/bin/env python3
"""Sweep DCFM ODE steps without retraining and report accuracy/cost."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.utils import compute_prob_metrics  # noqa: E402
from config import get_config  # noqa: E402
from main import build_model, load_data  # noqa: E402
from model import DCFM  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True,
                        choices=["solar", "sdwpf", "electricity", "pjm"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[3, 5, 10, 15, 20])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=20,
                        help="0 evaluates the full test set")
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-samples", action="store_true",
                        help="save full samples for sample-count analysis")
    parser.add_argument("--output-dir", default="paper_extra_results/nfe")
    return parser.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(args.seed)
    cfg = get_config(args.preset)
    _, _, original_test, adj, _, in_dim = load_data(cfg)
    loader = DataLoader(original_test.dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, drop_last=False)
    batches = []
    limit = args.max_batches if args.max_batches > 0 else len(loader)
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        batches.append(batch)
    if not batches:
        raise RuntimeError("No test batches loaded")

    n_nodes = int(adj.shape[0])
    model = build_model(cfg, in_dim=in_dim, n_nodes=n_nodes, wind_mask=None).to(device)
    load_state(model, args.checkpoint, device)
    model.eval()
    adj_norm = DCFM.normalize_adj(adj.to(device))
    edge_index = DCFM.adj_to_edge_index(adj.to(device))

    out = Path(args.output_dir) / args.preset
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for steps in args.steps:
        seed_everything(args.seed)  # same initial noise across the sweep
        samples_parts, target_parts = [], []
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        measured_seconds = 0.0
        measured_windows = 0
        with torch.inference_mode():
            for batch_index, (x, y) in enumerate(batches):
                x = x.to(device)
                sync(device)
                batch_start = time.perf_counter()
                he, hs, *_ = model(x, adj_norm, edge_index)
                samples = model.sample(
                    he, hs, n_samples=args.n_samples, n_steps=steps)
                sync(device)
                if batch_index >= args.warmup_batches:
                    measured_seconds += time.perf_counter() - batch_start
                    measured_windows += int(x.shape[0])
                samples_parts.append(samples.cpu().numpy())
                target_parts.append(y.permute(0, 2, 1, 3).numpy())
                if batch_index + 1 == args.warmup_batches:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
        samples_all = np.concatenate(samples_parts, axis=1)
        target_all = np.concatenate(target_parts, axis=0)
        metrics = compute_prob_metrics(samples_all, target_all)
        row = {
            "steps": steps,
            "nfe": 2 * steps,
            "n_samples": args.n_samples,
            "windows": int(target_all.shape[0]),
            "measured_windows": measured_windows,
            "seconds": measured_seconds,
            "milliseconds_per_window": (
                1000 * measured_seconds / max(1, measured_windows)),
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda" else None),
            **{key: value for key, value in metrics.items()
               if key in {"MAE", "RMSE", "CRPS", "PICP", "PINAW"}},
        }
        rows.append(row)
        if args.save_samples:
            np.save(out / f"samples_steps_{steps}.npy",
                    samples_all.astype(np.float32))
            np.save(out / "ground_truth.npy",
                    target_all.transpose(0, 2, 1, 3).astype(np.float32))
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del samples_all, samples_parts
        gc.collect()

    with (out / "nfe_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (out / "nfe_sensitivity.json").open("w", encoding="utf-8") as f:
        json.dump({"arguments": vars(args), "results": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"Saved {out / 'nfe_sensitivity.csv'}")


if __name__ == "__main__":
    main()
