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

    健壮性处理：
      - Electricity 数据集中部分客户用电量长期为 0（缺失/停业），其相关系数
        为 NaN（零向量无法计算皮尔逊相关）。nan_to_num 将 NaN 替换为 0，
        避免后续矩阵运算中 NaN 传播。
      - 阈值过高时可能产生孤立节点（度为 0），GCN normalize_adj 会因除以 0
        出错。检测并给孤立节点补一条自环边。
    """
    corr = np.corrcoef(data.T)                                   # [N, N]
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0) # 修复 NaN
    adj  = (np.abs(corr) >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)                                     # 先去掉自环

    # 孤立节点保护：度为 0 的节点补自环，防止 normalize_adj 除以 0
    isolated = (adj.sum(axis=1) == 0)
    if isolated.any():
        print(f"  [adj] 检测到 {isolated.sum()} 个孤立节点，已补自环")
        adj[isolated, isolated] = 1.0

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
    """
    全局 Z-score 归一化（在训练集上 fit）。

    log_transform=True 时，inverse_transform 会额外做 expm1 反变换，
    用于 Electricity 等重尾数据集（load_electricity 中已预先做了 log1p）。
    训练和评估都在 log 域进行，指标也在 log 域报告，与论文口径一致。
    """
    def __init__(self, log_transform: bool = False):
        self.mean: float = 0.0
        self.std:  float = 1.0
        self.log_transform: bool = log_transform

    def fit(self, data: np.ndarray) -> "Scaler":
        self.mean = float(data.mean())
        self.std  = float(data.std()) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        支持 numpy array 或 torch tensor。
        log_transform=True 时：先反 Z-score，再 expm1（对应 log1p 的逆）。
        """
        out = data * self.std + self.mean
        if self.log_transform:
            if isinstance(out, np.ndarray):
                out = np.expm1(out)
            else:
                out = out.exp() - 1.0
        return out


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

    [Fix-Elec] log1p 变换：
      Electricity 原始数据是重尾分布（工业用户 vs 居民用户量级差异 100x+）。
      直接 Z-score 后大量数据点压缩到接近 0，导致：
        · sigma 极易压到最小值（NLL 持续为负）
        · Hs 数值范围极小，CLUB 变分网络无法区分正负样本（MI≈0）

      log1p(x) = log(1 + x) 变换将重尾分布压缩为近似正态，
      Z-score 后方差分布更均匀，sigma 学习难度大幅降低。

      注意：
        · 原始数据必须 ≥ 0（用电量天然满足）
        · 训练/验证/测试指标均在 log 域报告，与论文口径一致
        · Scaler(log_transform=True) 的 inverse_transform 包含 expm1 反变换，
          供需要原始量纲的场景使用
    """
    raw = np.loadtxt(data_path, delimiter=',')   # [T, N=321]

    # [Fix-Elec] log1p 变换：将重尾用电量分布压缩为近似正态
    # clip(0) 防止极少数负值（数据噪声）导致 log 出现 NaN
    raw = np.log1p(np.clip(raw, 0, None))

    print(f"  [Electricity] log1p 变换后：mean={raw.mean():.3f}, "
          f"std={raw.std():.3f}, min={raw.min():.3f}, max={raw.max():.3f}")

    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size,
                          name='Electricity', log_transform=True)


def load_weather(data_path: str, T_in: int = 168, T_out: int = 1,
                 adj_threshold: float = 0.6, batch_size: int = 32,
                 feature_idx: int = 0):
    """
    Weather2k: 1866 个气象站，小时级，2017-2021。
    文件格式: weather2k.npy

    Weather2k 的 npy 文件有两种常见 shape，代码自动识别：

    格式 A（标准时序格式）: shape = (T, N, F)
      T = 时间步数，N = 站点数，F = 特征数
      直接使用，无需转置。

    格式 B（Weather2k 论文原始格式）: shape = (N, C, L)
      N = 1866 站点，C = 特征数（如 13），L = 时间步数（如 13632）
      需要 transpose 成 (L, N, C) = (T, N, F)。
      识别条件：shape[0] == 1866（论文站点数）且 ndim == 3。

    参数：
      feature_idx : 选取哪个特征（0 = 第一个，通常是温度）。
                    设为 -1 则使用全部特征。
    """
    raw = np.load(data_path)
    print(f"  [Weather2k] 原始 shape: {raw.shape}, dtype: {raw.dtype}")

    if raw.ndim == 2:
        # (T, N) → (T, N, 1)
        raw = raw[:, :, None]
        print(f"  [Weather2k] 2D → 3D: {raw.shape}")

    elif raw.ndim == 3:
        T0, D1, D2 = raw.shape
        # 识别格式 B：第 0 维是站点数（论文 1866），第 2 维是最长的时间轴
        # 判断条件：shape[0] 远小于 shape[2]，且 shape[0] 与论文站点数接近
        if D2 > D1 and D2 > T0 and T0 < 5000:
            # 格式 B: (N=stations, F=features, T=time) → transpose → (T, N, F)
            raw = raw.transpose(2, 0, 1)   # (T, N, F)
            print(f"  [Weather2k] 检测到格式 B (N,F,T)，已转置为 (T,N,F): {raw.shape}")
        else:
            print(f"  [Weather2k] 检测到格式 A (T,N,F): {raw.shape}")

    T, N, F = raw.shape
    print(f"  [Weather2k] 解析结果: T={T} 时间步, N={N} 站点, F={F} 特征")

    if F > 1 and feature_idx >= 0:
        print(f"  [Weather2k] 选取特征索引 {feature_idx}（共 {F} 个特征）")
        raw = raw[:, :, feature_idx : feature_idx + 1]   # (T, N, 1)
    elif feature_idx < 0:
        print(f"  [Weather2k] 使用全部 {F} 个特征（config.in_dim 需设为 {F}）")

    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Weather')


def _build_loaders(raw: np.ndarray,
                   T_in: int, T_out: int,
                   adj_threshold: float,
                   batch_size: int,
                   name: str = '',
                   log_transform: bool = False) -> tuple:
    """
    共享加载逻辑：
      1. 扩展到 [T, N, F]
      2. 70/10/20 划分
      3. Z-score 归一化（fit on train）
      4. 构建相关性邻接矩阵（fit on train，基于第一个特征）
      5. 返回 (train_loader, val_loader, test_loader, adj_tensor, scaler, in_dim)
         in_dim = F，供 main.py 自动设置模型的 in_dim，无需手动修改 config
    """
    if raw.ndim == 2:
        raw = raw[:, :, None]                           # [T, N, 1]

    T, N, F = raw.shape
    n_train  = int(T * 0.7)
    n_val    = int(T * 0.1)

    train_raw = raw[:n_train]
    val_raw   = raw[n_train : n_train + n_val]
    test_raw  = raw[n_train + n_val :]

    # 归一化：只在训练集上 fit
    scaler     = Scaler(log_transform=log_transform).fit(train_raw)
    train_data = scaler.transform(train_raw)
    val_data   = scaler.transform(val_raw)
    test_data  = scaler.transform(test_raw)

    # 邻接矩阵：基于训练集第一个特征的皮尔逊相关
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
    # 返回 in_dim=F，供 main.py 自动覆盖 config.model.in_dim
    return train_loader, val_loader, test_loader, adj_tensor, scaler, F