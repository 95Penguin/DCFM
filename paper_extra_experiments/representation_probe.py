#!/usr/bin/env python3
"""Lightweight diagnostic evidence for the two learned representations."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_config  # noqa: E402
from main import build_model, load_data  # noqa: E402
from model import DCFM  # noqa: E402
from paper_extra_experiments.common import save_json  # noqa: E402


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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="paper_extra_results/representation")
    return parser.parse_args()


def cosine_abs(a, b):
    dimension = min(a.shape[-1], b.shape[-1])
    a = a[..., :dimension].reshape(-1, dimension)
    b = b[..., :dimension].reshape(-1, dimension)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return float(np.mean(np.abs(numerator / denominator)))


def norm_correlation(a, b):
    x = np.linalg.norm(a, axis=-1).reshape(-1)
    y = np.linalg.norm(b, axis=-1).reshape(-1)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    args = parse_args()
    device = torch.device(args.device)
    seed_everything(args.seed)
    cfg = get_config(args.preset)
    _, _, test_loader, adj, _, in_dim = load_data(cfg)
    model = build_model(cfg, in_dim=in_dim, n_nodes=int(adj.shape[0]),
                        wind_mask=None).to(device)
    load_state(model, args.checkpoint, device)
    model.eval()
    adj_norm = DCFM.normalize_adj(adj.to(device))

    fields = {"He": [], "Hs": [], "low": [], "high": []}
    club_e, club_s = [], []
    with torch.inference_mode():
        for index, (x, _) in enumerate(test_loader):
            if args.max_batches > 0 and index >= args.max_batches:
                break
            _, he, hs, low, high = model.backbone(x.to(device), adj_norm)
            fields["He"].append(he.cpu().numpy())
            fields["Hs"].append(hs.cpu().numpy())
            fields["low"].append(low.cpu().numpy())
            fields["high"].append(high.cpu().numpy())
            club_e.append(float(model.club_e(he, high).cpu()))
            club_s.append(float(model.club_s(hs, low).cpu()))
    values = {key: np.concatenate(parts, axis=0) for key, parts in fields.items()}
    result = {
        "preset": args.preset,
        "batches": len(club_e),
        "branch_cosine_abs": cosine_abs(values["He"], values["Hs"]),
        "environment_to_low_norm_correlation":
            norm_correlation(values["He"], values["low"]),
        "environment_to_high_norm_correlation":
            norm_correlation(values["He"], values["high"]),
        "short_term_to_high_norm_correlation":
            norm_correlation(values["Hs"], values["high"]),
        "short_term_to_low_norm_correlation":
            norm_correlation(values["Hs"], values["low"]),
        "cross_club_environment_high": float(np.mean(club_e)),
        "cross_club_short_term_low": float(np.mean(club_s)),
    }
    out = Path(args.output_dir) / args.preset
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "representation_probe.json", result)
    with (out / "representation_probe.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
