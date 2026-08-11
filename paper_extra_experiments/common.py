"""Shared array handling and probabilistic metrics for paper experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_array(path: str | Path) -> np.ndarray:
    return np.asarray(np.load(path, mmap_mode="r"))


def canonical_target(y: np.ndarray) -> np.ndarray:
    """Return project NPY layout [window,horizon,node,feature] as canonical."""
    y = np.asarray(y)
    if y.ndim == 3:
        y = y[..., None]
    if y.ndim != 4:
        raise ValueError(f"Target must have 3/4 dimensions, got {y.shape}")
    # All result arrays written by train.py/run_baselines.py use this layout.
    return y.transpose(0, 2, 1, 3)


def canonical_samples(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Align predictions to an already-canonical target array."""
    pred = np.asarray(pred)
    if pred.ndim == 4:
        pred = canonical_target(pred)[None]
    elif pred.ndim == 5:
        candidates = [pred]
        # Common alternative: [window, sample, ...].
        candidates.append(pred.transpose(1, 0, 2, 3, 4))
        valid = []
        for value in candidates:
            tail = value.shape[1:]
            if tail == y.shape:
                valid.append(value)
            elif (
                tail[0] == y.shape[0]
                and tail[1] == y.shape[2]
                and tail[2] == y.shape[1]
                and tail[3] == y.shape[3]
            ):
                valid.append(value.transpose(0, 1, 3, 2, 4))
        if not valid:
            raise ValueError(
                f"Cannot align prediction {pred.shape} with target {y.shape}")
        pred = valid[0]
    else:
        raise ValueError(f"Prediction must have 4/5 dimensions, got {pred.shape}")
    if pred.shape[1:] != y.shape:
        raise ValueError(f"Prediction {pred.shape} and target {y.shape} mismatch")
    return pred


def _crps_per_window(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact empirical CRPS, averaged over non-window dimensions."""
    first = np.abs(samples - y[None]).mean(axis=(0, 2, 3, 4))
    ordered = np.sort(samples, axis=0)
    s = ordered.shape[0]
    if s == 1:
        return first
    weights = (2 * np.arange(1, s + 1) - s - 1).reshape(
        s, *([1] * (ordered.ndim - 1)))
    pair = (ordered * weights).sum(axis=0) / (s * s)
    return first - pair.mean(axis=(1, 2, 3))


def metrics_per_window(samples: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    y = canonical_target(y)
    samples = canonical_samples(samples, y)
    mean = samples.mean(axis=0)
    error = mean - y
    lo = np.quantile(samples, 0.025, axis=0)
    hi = np.quantile(samples, 0.975, axis=0)
    data_range = float(y.max() - y.min()) + 1e-8
    axes = (1, 2, 3)
    return {
        "MAE": np.abs(error).mean(axis=axes),
        "RMSE": np.sqrt(np.square(error).mean(axis=axes)),
        "CRPS": _crps_per_window(samples, y),
        "PICP": ((y >= lo) & (y <= hi)).mean(axis=axes),
        "PINAW": ((hi - lo) / data_range).mean(axis=axes),
    }


def aggregate_metrics(samples: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {key: float(value.mean())
            for key, value in metrics_per_window(samples, y).items()}


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
