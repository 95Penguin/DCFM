"""
GridCFN – 数据集工具
支持论文使用的三个数据集：
  - Solar-Energy  (137 PV plants, 10-min, 2007)
  - Electricity   (UCI, 321 clients, 1-hour, 2012-2014)
  - Weather2k     (1866 stations, 1-hour, 2017-2021)

数据文件获取（与论文一致）：
  Solar / Electricity → LSTNet repo:
    https://github.com/laiguokun/multivariate-time-series-data
  Weather2k → 原始论文仓库或 Kaggle 搜索 Weather2k

所有数据集遵循相同接口：
  1. 加载原始 CSV / npy
  2. Z-score 归一化（均值方差在训练集上计算）
  3. 滑动窗口切片（T_in=168, T_out=1，对应论文设置）
  4. 构建相关性邻接矩阵
  5. 70/10/20 划分
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 邻接矩阵构建
# ---------------------------------------------------------------------------

def build_correlation_adj(data: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """
    基于训练集皮尔逊相关系数构建二值邻接矩阵。
    data : [T, N]（单特征或多特征均值）
    返回 A : [N, N]（|corr| >= threshold 置 1，对角线置 0）
    """
    corr = np.corrcoef(data.T)                          # [N, N]
    adj  = (np.abs(corr) >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)                            # 无自环（GCN 内部添加）
    return adj


# ---------------------------------------------------------------------------
# 滑动窗口数据集
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """
    通用多变量时序滑动窗口数据集。

    data  : [T, N, F]  已归一化
    adj   : [N, N]
    T_in  : 输入窗口长度（论文: 168）
    T_out : 预测步长（论文: 1）
    """
    def __init__(self, data: np.ndarray, adj: np.ndarray,
                 T_in: int = 168, T_out: int = 1):
        super().__init__()
        self.data     = torch.tensor(data, dtype=torch.float32)  # [T, N, F]
        self.adj      = torch.tensor(adj,  dtype=torch.float32)  # [N, N]
        self.T_in     = T_in
        self.T_out    = T_out
        self.n_samples = len(data) - T_in - T_out + 1
        assert self.n_samples > 0, (
            f"数据长度 {len(data)} 不足以构造窗口 T_in={T_in} + T_out={T_out}"
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.T_in]                           # [T_in, N, F]
        y = self.data[idx + self.T_in : idx + self.T_in + self.T_out] # [T_out, N, F]
        # T_out=1 时 squeeze 掉时间维，输出 y:[N,F]
        return x, y.squeeze(0)


# ---------------------------------------------------------------------------
# Z-score 归一化
# ---------------------------------------------------------------------------

class Scaler:
    """全局 Z-score 归一化（在训练集上 fit）。"""
    def __init__(self):
        self.mean: float = 0.0
        self.std:  float = 1.0

    def fit(self, data: np.ndarray) -> "Scaler":
        self.mean = float(data.mean())
        self.std  = float(data.std()) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """支持 numpy array 或 torch tensor。"""
        return data * self.std + self.mean


# ---------------------------------------------------------------------------
# 数据集加载器（三个真实数据集）
# ---------------------------------------------------------------------------

def load_solar_energy(data_path: str, T_in: int = 168, T_out: int = 1,
                      adj_threshold: float = 0.95, batch_size: int = 32):
    """
    Solar-Energy: 137 个 PV 电站，2007 年，10 分钟分辨率。
    文件格式: solar_AL.txt（逗号分隔，[T=52560, N=137]）
    """
    raw = np.loadtxt(data_path, delimiter=',')          # [T, N=137]
    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Solar')


def load_electricity(data_path: str, T_in: int = 168, T_out: int = 1,
                     adj_threshold: float = 0.7, batch_size: int = 32):
    """
    Electricity (UCI): 321 个用电客户，小时级，2012-2014。
    文件格式: electricity.txt（逗号分隔，[T=26304, N=321]）
    """
    raw = np.loadtxt(data_path, delimiter=',')          # [T, N=321]
    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Electricity')


def load_weather(data_path: str, T_in: int = 168, T_out: int = 1,
                 adj_threshold: float = 0.6, batch_size: int = 32):
    """
    Weather2k: 1866 个气象站，小时级，2017-2021。
    文件格式: weather2k.npy（shape [T, N] 或 [T, N, F]）
    """
    raw = np.load(data_path)
    if raw.ndim == 2:
        raw = raw[:, :, None]                           # 补充特征维 → [T, N, 1]
    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Weather')


def _build_loaders(raw: np.ndarray,
                   T_in: int, T_out: int,
                   adj_threshold: float,
                   batch_size: int,
                   name: str = '') -> tuple:
    """
    共享加载逻辑：
      1. 扩展到 [T, N, F]
      2. 70/10/20 划分
      3. Z-score 归一化（fit on train）
      4. 构建相关性邻接矩阵（fit on train）
      5. 返回 (train_loader, val_loader, test_loader, adj_tensor, scaler)
    """
    if raw.ndim == 2:
        raw = raw[:, :, None]                           # [T, N, 1]

    T, N, F = raw.shape
    n_train  = int(T * 0.7)
    n_val    = int(T * 0.1)
    # n_test  = T - n_train - n_val（剩余部分）

    train_raw = raw[:n_train]
    val_raw   = raw[n_train : n_train + n_val]
    test_raw  = raw[n_train + n_val :]

    # 归一化：只在训练集上 fit
    scaler     = Scaler().fit(train_raw)
    train_data = scaler.transform(train_raw)
    val_data   = scaler.transform(val_raw)
    test_data  = scaler.transform(test_raw)

    # 邻接矩阵：基于训练集第一特征的皮尔逊相关
    adj = build_correlation_adj(train_data[:, :, 0], threshold=adj_threshold)

    print(f"[{name}] T={T}, N={N}, F={F} | "
          f"train={n_train}, val={n_val}, test={T - n_train - n_val} | "
          f"edges={int(adj.sum())}, adj_density={adj.mean():.3f}")

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
