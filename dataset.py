"""
Dataset utilities for GridCFN.
Supports the three datasets used in the paper:
  - Solar-Energy  (137 PV plants, 10-min, 2007)
  - Electricity   (UCI, 321 clients, 1-hour, 2012-2014)
  - Weather2k     (1866 stations, 1-hour, 2017-2021)

All datasets follow the same interface:
  - Download / load raw CSV
  - Z-score normalise
  - Build sliding-window samples  (T_in=168, T_out=1 as per paper)
  - Build adjacency matrix (correlation-based if no physical topology given)
  - Split 70/10/20
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Adjacency matrix helpers
# ---------------------------------------------------------------------------

def build_correlation_adj(data: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """
    Build adjacency from Pearson correlation on training data.
    data : [T, N]  (single feature or mean across features)
    Returns A : [N, N]  binary (1 if |corr| >= threshold)
    """
    corr = np.corrcoef(data.T)           # [N, N]
    adj = (np.abs(corr) >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)             # no self-loops (added later in GCN)
    return adj


def build_distance_adj(coords: np.ndarray, sigma: float = 0.1,
                       threshold: float = 0.1) -> np.ndarray:
    """
    Build adjacency from geographic coordinates using Gaussian kernel.
    coords : [N, 2]  (lat, lon)
    """
    from scipy.spatial.distance import cdist
    dists = cdist(coords, coords, metric='euclidean')
    weights = np.exp(-dists ** 2 / sigma)
    adj = (weights >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return adj


# ---------------------------------------------------------------------------
# Core sliding-window dataset
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """
    Generic multi-variate time-series dataset with sliding windows.

    data  : [T, N, F]  normalised
    adj   : [N, N]
    T_in  : input window length   (paper: 168)
    T_out : prediction horizon    (paper: 1)
    """
    def __init__(self, data: np.ndarray, adj: np.ndarray,
                 T_in: int = 168, T_out: int = 1):
        super().__init__()
        self.data  = torch.tensor(data,  dtype=torch.float32)  # [T, N, F]
        self.adj   = torch.tensor(adj,   dtype=torch.float32)  # [N, N]
        self.T_in  = T_in
        self.T_out = T_out
        self.n_samples = len(data) - T_in - T_out + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.T_in]                         # [T_in, N, F]
        y = self.data[idx + self.T_in : idx + self.T_in + self.T_out]# [T_out, N, F]
        return x, y.squeeze(0)   # x:[T_in,N,F],  y:[N,F]  (T_out=1 → squeeze)


# ---------------------------------------------------------------------------
# Z-score normalisation
# ---------------------------------------------------------------------------

class Scaler:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, data: np.ndarray):
        """data: any shape, compute stats over all elements."""
        self.mean = data.mean()
        self.std  = data.std() + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """Works with numpy arrays or torch tensors."""
        return data * self.std + self.mean


# ---------------------------------------------------------------------------
# Dataset loaders  (one per dataset used in the paper)
# ---------------------------------------------------------------------------

def load_solar_energy(data_path: str, T_in: int = 168, T_out: int = 1,
                      adj_threshold: float = 0.7, batch_size: int = 32):
    """
    Solar-Energy: 137 PV plants in Alabama, 2007, 10-min resolution.
    Expected file: solar_AL.txt  (space-separated, [T, N])
    Download: LSTNet repo  https://github.com/laiguokun/multivariate-time-series-data
    """
    # raw = np.loadtxt(data_path)          # [T, N=137]
    raw = np.loadtxt(data_path, delimiter=',')
    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Solar')


def load_electricity(data_path: str, T_in: int = 168, T_out: int = 1,
                     adj_threshold: float = 0.7, batch_size: int = 32):
    """
    Electricity (UCI): 321 clients, hourly, 2012-2014.
    Expected file: electricity.txt  (space-separated, [T, N])
    Download: same LSTNet repo
    """
    raw = np.loadtxt(data_path)          # [T, N=321]
    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Electricity')


def load_weather(data_path: str, T_in: int = 168, T_out: int = 1,
                 adj_threshold: float = 0.7, batch_size: int = 32):
    """
    Weather2k: 1866 stations, hourly, 2017-2021.
    Expected file: weather2k.npy  shape [T, N, F] or [T, N]
    """
    raw = np.load(data_path)
    if raw.ndim == 2:
        raw = raw[:, :, None]            # add feature dim
    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Weather')


def _build_loaders(raw: np.ndarray, T_in: int, T_out: int,
                   adj_threshold: float, batch_size: int,
                   name: str = ''):
    """
    Shared logic:
      1. Expand to [T, N, F] if needed
      2. 70/10/20 split
      3. Z-score normalise (fit on train only)
      4. Build adjacency from training data
      5. Return (train_loader, val_loader, test_loader, adj, scaler)
    """
    if raw.ndim == 2:
        raw = raw[:, :, None]            # [T, N, 1]

    T, N, F = raw.shape
    n_train = int(T * 0.7)
    n_val   = int(T * 0.1)
    n_test  = T - n_train - n_val

    train_raw = raw[:n_train]
    val_raw   = raw[n_train : n_train + n_val]
    test_raw  = raw[n_train + n_val :]

    # Normalise
    scaler = Scaler().fit(train_raw)
    train_data = scaler.transform(train_raw)
    val_data   = scaler.transform(val_raw)
    test_data  = scaler.transform(test_raw)

    # Adjacency (correlation on training data, first feature)
    adj = build_correlation_adj(train_data[:, :, 0], threshold=adj_threshold)

    print(f"[{name}] T={T}, N={N}, F={F} | "
          f"train={n_train}, val={n_val}, test={n_test} | "
          f"adj density={adj.mean():.3f}")

    # Datasets
    train_ds = SlidingWindowDataset(train_data, adj, T_in, T_out)
    val_ds   = SlidingWindowDataset(val_data,   adj, T_in, T_out)
    test_ds  = SlidingWindowDataset(test_data,  adj, T_in, T_out)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    return train_loader, val_loader, test_loader, adj_tensor, scaler


# ---------------------------------------------------------------------------
# Synthetic dataset for quick testing (no download required)
# ---------------------------------------------------------------------------

def make_synthetic_dataset(T: int = 2000, N: int = 20, F: int = 3,
                            T_in: int = 12, T_out: int = 1,
                            batch_size: int = 32, seed: int = 42):
    """
    Generates a synthetic spatio-temporal dataset for unit testing.
    Note: paper uses T_in=168, T_out=1. Here we use T_in=12 for speed.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 4 * np.pi, T)

    # Each node has a mix of sinusoidal trends + noise
    data = np.zeros((T, N, F), dtype=np.float32)
    for n in range(N):
        freq = 0.5 + rng.random() * 2
        phase = rng.random() * np.pi
        for f in range(F):
            data[:, n, f] = (np.sin(freq * t + phase + f)
                             + 0.3 * rng.standard_normal(T))

    # Random sparse adjacency
    adj = (rng.random((N, N)) > 0.7).astype(np.float32)
    adj = np.maximum(adj, adj.T)          # symmetrise
    np.fill_diagonal(adj, 0)

    # Split + normalise
    n_train = int(T * 0.7)
    n_val   = int(T * 0.1)
    scaler  = Scaler().fit(data[:n_train])
    train_data = scaler.transform(data[:n_train])
    val_data   = scaler.transform(data[n_train : n_train + n_val])
    test_data  = scaler.transform(data[n_train + n_val :])

    train_ds = SlidingWindowDataset(train_data, adj, T_in, T_out)
    val_ds   = SlidingWindowDataset(val_data,   adj, T_in, T_out)
    test_ds  = SlidingWindowDataset(test_data,  adj, T_in, T_out)

    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    return (make_loader(train_ds, True),
            make_loader(val_ds,   False),
            make_loader(test_ds,  False),
            adj_tensor, scaler)
