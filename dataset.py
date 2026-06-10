"""
GridCFN – 数据集工具（多特征支持版，已集成 PJM 负荷时空数据集）
"""

import os
import platform
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


# ---------------------------------------------------------------------------
# 邻接矩阵
# ---------------------------------------------------------------------------

def build_correlation_adj(data: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    corr = np.corrcoef(data.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    adj  = (np.abs(corr) >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)

    isolated = (adj.sum(axis=1) == 0)
    if isolated.any():
        print(f"  [adj] 检测到 {isolated.sum()} 个孤立节点，已补自环")
        adj[isolated, isolated] = 1.0

    return adj


# ---------------------------------------------------------------------------
# 滑动窗口数据集
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, adj: np.ndarray,
                 T_in: int = 168, T_out: int = 1):
        super().__init__()
        self.data      = torch.tensor(data, dtype=torch.float32)
        self.adj       = torch.tensor(adj,  dtype=torch.float32)
        self.T_in      = T_in
        self.T_out     = T_out
        self.n_samples = len(data) - T_in - T_out + 1
        assert self.n_samples > 0, (
            f"数据长度 {len(data)} 不足以构造窗口 T_in={T_in} + T_out={T_out}"
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.T_in]
        # 对标签切片 [:, :1]，保证输入是 4 维，预测目标永远是 index=0 的物理功率 (F_out=1)
        y = self.data[idx + self.T_in : idx + self.T_in + self.T_out, :, :1]
        return x, y


# ---------------------------------------------------------------------------
# Z-score 归一化
# ---------------------------------------------------------------------------

class Scaler:
    def __init__(self, axis=(0, 1), log_transform: bool = False):
        self.axis = axis
        self.mean = None
        self.std  = None
        self.log_transform: bool = log_transform

    def fit(self, data: np.ndarray) -> "Scaler":
        self.mean = data.mean(axis=self.axis, keepdims=True)
        self.std  = data.std(axis=self.axis, keepdims=True) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        is_torch = torch.is_tensor(data)
        device = data.device if is_torch else None
        dtype = data.dtype if is_torch else None

        mean_val = self.mean
        std_val  = self.std

        # ─── [终极防错] 如果输入是展平的一维，自动按 Baseline 的时空轴顺序进行安全重组 ───
        original_shape = None
        if len(data.shape) == 1:
            original_shape = data.shape
            V = data.shape[0]
            F = self.mean.shape[-1]
            if self.axis == (0, 1):  # 全局归一化
                if is_torch:
                    data = data.view(-1, F)
                else:
                    data = data.reshape(-1, F)
            else:  # 逐节点归一化下，一维数组根据 Baseline 默认的 [B, T_out, N, F] 顺序进行重建
                N = self.mean.shape[1]
                t_out_found = 1
                for t_candidate in [12, 24, 1]:
                    if (V % (N * t_candidate * F)) == 0:
                        t_out_found = t_candidate
                        break
                B = V // (N * t_out_found * F)
                # 重组为 Baseline 格式
                if is_torch:
                    data = data.view(B, t_out_found, N, F)
                else:
                    data = data.reshape(B, t_out_found, N, F)
        # ──────────────────────────────────────────────────────────────────

        F_data = data.shape[-1]
        mean_val = self.mean[..., :F_data]
        std_val  = self.std[..., :F_data]

        if is_torch:
            m = torch.tensor(mean_val, dtype=dtype, device=device)
            s = torch.tensor(std_val, dtype=dtype, device=device)
        else:
            m = mean_val
            s = std_val

        # 全局尺度：[1, 1, F]
        if self.axis == (0, 1):
            out = data * s + m
        else:
            # 逐节点尺度
            ndims = len(data.shape)
            if ndims == 3:  # [T, N, F]
                m_aligned, s_aligned = m, s
            elif ndims == 4:  
                # ─── [自适应对齐核心] 动态检测 N 在第 1 维 (GridCFN) 还是第 2 维 (Baselines) ───
                N_size = m.shape[1]
                F_size = m.shape[2]
                if data.shape[1] == N_size:
                    # [B, N, T_out, F] (GridCFN 格式)
                    if is_torch:
                        m_aligned = m.view(1, N_size, 1, F_size)
                        s_aligned = s.view(1, N_size, 1, F_size)
                    else:
                        m_aligned = m.reshape(1, N_size, 1, F_size)
                        s_aligned = s.reshape(1, N_size, 1, F_size)
                elif data.shape[2] == N_size:
                    # [B, T_out, N, F] (Baselines 格式)
                    if is_torch:
                        m_aligned = m.view(1, 1, N_size, F_size)
                        s_aligned = s.view(1, 1, N_size, F_size)
                    else:
                        m_aligned = m.reshape(1, 1, N_size, F_size)
                        s_aligned = s.reshape(1, 1, N_size, F_size)
                else:
                    m_aligned, s_aligned = m, s
            elif ndims == 5:  # [S, B, N, T_out, F]
                if is_torch:
                    m_aligned = m.view(1, 1, m.shape[1], 1, m.shape[2])
                    s_aligned = s.view(1, 1, s.shape[1], 1, s.shape[2])
                else:
                    m_aligned = m.reshape(1, 1, m.shape[1], 1, m.shape[2])
                    s_aligned = s.reshape(1, 1, s.shape[1], 1, s.shape[2])
            else:
                m_aligned, s_aligned = m, s

            out = data * s_aligned + m_aligned

        if original_shape is not None:
            if is_torch:
                out = out.view(original_shape)
            else:
                out = out.reshape(original_shape)
        return out


# ---------------------------------------------------------------------------
# 数据集加载器
# ---------------------------------------------------------------------------

def load_solar_energy(data_path: str, T_in: int = 168, T_out: int = 1,
                      adj_threshold: float = 0.95, batch_size: int = 32):
    raw = np.loadtxt(data_path, delimiter=',')
    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Solar')


def load_electricity(data_path: str, T_in: int = 168, T_out: int = 1,
                     adj_threshold: float = 0.7, batch_size: int = 32):
    raw = np.loadtxt(data_path, delimiter=',')
    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Electricity')


def load_weather(data_path: str, T_in: int = 168, T_out: int = 1,
                 adj_threshold: float = 0.6, batch_size: int = 32,
                 feature_idx: int = 0):
    raw = np.load(data_path)
    print(f"  [Weather2k] 原始 shape: {raw.shape}, dtype: {raw.dtype}")

    if raw.ndim == 2:
        raw = raw[:, :, None]
    elif raw.ndim == 3:
        T0, D1, D2 = raw.shape
        is_format_b = (D2 > T0) and (D2 > D1) and (T0 <= D1)
        if is_format_b:
            raw = raw.transpose(2, 0, 1)

    T, N, F = raw.shape
    if F > 1 and feature_idx >= 0:
        raw = raw[:, :, feature_idx : feature_idx + 1]

    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Weather')


def load_sdwpf(data_path: str, T_in: int = 168, T_out: int = 1,
               adj_threshold: float = 0.88, batch_size: int = 32):
    raw = _load_sdwpf_csv(data_path)
    return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='SDWPF')


def load_pjm(data_path: str, T_in: int = 168, T_out: int = 1,
             adj_threshold: float = 0.7, batch_size: int = 32):
    """
    PJM 区域小时负荷数据集载入器（终极纯净抗噪版）
    """
    print(f"  [PJM] 正在扫描并合并各分区的负荷数据: {data_path}")
    if not os.path.isdir(data_path):
        raise ValueError(f"PJM 文件夹路径不存在: {data_path}")
    
    # 扫描目录下所有的 csv 文件
    all_files = [f for f in os.listdir(data_path) if f.endswith('_hourly.csv')]
    
    # 剔除无用的总量估计或非分区专属文件
    exclude_files = ['pjm_hourly_est.csv', 'PJM_Load_hourly.csv']
    valid_files = [f for f in all_files if f not in exclude_files]
    
    if not valid_files:
        raise ValueError(f"在 {data_path} 下未找到有效的 PJM 分区 CSV 负荷文件")
        
    dfs = []
    for f in sorted(valid_files):
        col_name = f.replace('_hourly.csv', '')
        filepath = os.path.join(data_path, f)
        
        # 1. 显式读取
        df = pd.read_csv(filepath)
        
        # 兼容大小写 Datetime 命名
        time_col = None
        for c in ['Datetime', 'datetime', 'Date', 'date']:
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            continue
            
        # 2. 强力转换为无时区统一时间戳
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        df[time_col] = df[time_col].dt.tz_localize(None)
        
        # 3. 去重并设为索引
        df = df.drop_duplicates(subset=[time_col])
        df = df.set_index(time_col)
        
        # 4. 只提取负荷物理列
        load_col = None
        for c in df.columns:
            if 'MW' in c or 'load' in c.lower():
                load_col = c
                break
        if load_col is None:
            load_col = df.columns[0]
            
        df = df[[load_col]].copy()
        df.columns = [col_name]
        dfs.append(df)
        
    if not dfs:
        raise ValueError("没有成功解析任何电网负荷文件，请检查数据文件夹内容。")
        
    # 5. 外连接合并
    print("  [PJM] 正在进行时空索引对齐合并...")
    merged_df = pd.concat(dfs, axis=1, join='outer')
    merged_df = merged_df.sort_index()
    
    # 6. 裁剪通用时段 (2013-2018)
    merged_df = merged_df.loc['2013-01-01':'2018-01-01']
    
    # ─── 防火墙一：剔除由于时段不重合导致在裁剪区间内缺失严重的异常节点 ───
    missing_ratios = merged_df.isna().mean()
    bad_cols = missing_ratios[missing_ratios > 0.3].index  # 剔除缺失率大于 30% 的列
    if len(bad_cols) > 0:
        print(f"  [PJM] 警告: 发现缺失率过高的异常列 {list(bad_cols)}，已自动剔除！")
        merged_df = merged_df.drop(columns=bad_cols)
        
    # 7. 线性插值填补中途少量的细微缺失，再用前后向填充防漏
    merged_df = merged_df.interpolate(method='linear', axis=0).ffill().bfill()
    
    raw_data = merged_df.values.astype(np.float32)
    raw_data = raw_data[:, :, None]
    
    # ─── 防火墙二：终极无死角 NaN 物理兜底，确保送入网络的数据 100% 纯净 ───
    nan_count = np.isnan(raw_data).sum()
    if nan_count > 0:
        print(f"  [PJM] 警告: 最终矩阵中残留 {nan_count} 个 NaN 坏点！正在进行物理均值填充...")
        col_mean = np.nanmean(raw_data, axis=0, keepdims=True)
        col_mean = np.nan_to_num(col_mean, nan=0.0)  # 防止整列为空时均值也是 NaN
        nan_mask = np.isnan(raw_data)
        raw_data[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
        
    print(f"  [PJM] 最终对齐完成。最终 shape={raw_data.shape}, 共有 {raw_data.shape[1]} 个电网节点")
    return _build_loaders(raw_data, T_in, T_out, adj_threshold, batch_size, name='PJM')

def _load_sdwpf_csv(data_path: str) -> np.ndarray:
    print(f"  [SDWPF] 正在读取多特征物理数据: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = [c.split('(')[0].strip() for c in df.columns]
    df['time_step'] = _parse_time_step(df)
    df = _clean_sdwpf(df)

    expected_turbs = sorted(df['TurbID'].unique())
    feature_cols = ['Patv', 'Wspd', 'Etmp', 'Pab1']
    feature_arrays = []
    
    for col in feature_cols:
        pivot = df.pivot_table(
            index='time_step', columns='TurbID', values=col, aggfunc='mean'
        )
        pivot = pivot.reindex(columns=expected_turbs)
        t_min, t_max = pivot.index.min(), pivot.index.max()
        full_index   = np.arange(t_min, t_max + 1)
        pivot        = pivot.reindex(full_index)

        raw_feat = pivot.values.astype(np.float32)
        df_feat = pd.DataFrame(raw_feat)
        df_feat = df_feat.interpolate(method='linear', axis=0, limit_direction='both')
        raw_feat = df_feat.values.astype(np.float32)

        remaining_nan = np.isnan(raw_feat).sum()
        if remaining_nan > 0:
            col_means = np.nanmean(raw_feat, axis=0)
            nan_cols = np.isnan(col_means)
            if nan_cols.any():
                global_mean = np.nanmean(raw_feat) if not np.all(nan_cols) else 0.0
                col_means[nan_cols] = global_mean
            nan_mask = np.isnan(raw_feat)
            raw_feat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
            
        feature_arrays.append(raw_feat[..., None])

    raw_multifeats = np.concatenate(feature_arrays, axis=-1)
    print(f"  [SDWPF] 多特征整合成功: shape={raw_multifeats.shape} (功率, 风速, 温度, 桨距角)")
    return raw_multifeats


def _parse_time_step(df: pd.DataFrame) -> pd.Series:
    def tmstamp_to_step(ts: str) -> int:
        try:
            h, m = map(int, str(ts).strip().split(':'))
            return h * 6 + m // 10
        except Exception:
            return 0
    intra_step = df['Tmstamp'].apply(tmstamp_to_step)
    day_0based = (df['Day'] - df['Day'].min())
    return (day_0based * 144 + intra_step).astype(int)


def _clean_sdwpf(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df['Patv'] < 0, 'Patv'] = 0.0
    if 'Wspd' in df.columns:
        mask_stop = (df['Patv'] <= 0) & (df['Wspd'] > 2.5)
        df.loc[mask_stop, 'Patv'] = np.nan
    pab_cols = [c for c in ['Pab1', 'Pab2', 'Pab3'] if c in df.columns]
    if pab_cols:
        mask_pab = (df[pab_cols] > 89.0).any(axis=1)
        df.loc[mask_pab, 'Patv'] = np.nan
    if 'Wspd' in df.columns:
        df.loc[df['Wspd'] < 0, 'Patv'] = np.nan
    for col in ['Etmp', 'Itmp']:
        if col in df.columns:
            mask_temp = (df[col] < -40) | (df[col] > 80)
            df.loc[mask_temp, 'Patv'] = np.nan
    return df


def _build_loaders(raw: np.ndarray,
                   T_in: int, T_out: int,
                   adj_threshold: float,
                   batch_size: int,
                   name: str = '',
                   log_transform: bool = False) -> tuple:
    if raw.ndim == 2:
        raw = raw[:, :, None]

    T, N, F = raw.shape
    n_train  = int(T * 0.7)
    n_val    = int(T * 0.1)

    train_raw = raw[:n_train]
    val_raw   = raw[n_train : n_train + n_val]
    test_raw  = raw[n_train + n_val :]

    # 全局归一化：所有数据集统一使用 (0, 1) 轴，保证与 baseline 公平对比
    scale_axis = (0, 1)
    print(f"  [{name}] 启用全局标准化(Global Scaling)...")

    scaler     = Scaler(axis=scale_axis, log_transform=log_transform).fit(train_raw)
    train_data = scaler.transform(train_raw)
    val_data   = scaler.transform(val_raw)
    test_data  = scaler.transform(test_raw)

    adj = build_correlation_adj(train_data[:, :, 0], threshold=adj_threshold)

    print(f"[{name}] T={T}, N={N}, F={F} | scale_axis={scale_axis} | "
          f"train={n_train}, val={n_val}, test={T - n_train - n_val} | "
          f"edges={int(adj.sum())}, adj_density={adj.mean():.3f}")

    train_ds = SlidingWindowDataset(train_data, adj, T_in, T_out)
    val_ds   = SlidingWindowDataset(val_data,   adj, T_in, T_out)
    test_ds  = SlidingWindowDataset(test_data,  adj, T_in, T_out)

    is_windows = platform.system() == 'Windows'
    nw_train = 0 if (is_windows or batch_size <= 4 or N > 500) else 4
    nw_eval  = 0 if (is_windows or batch_size <= 4 or N > 500) else 2
    pin_train = (nw_train > 0)
    pin_eval  = (nw_eval  > 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=nw_train,
                              pin_memory=pin_train,
                              persistent_workers=(nw_train > 0))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=nw_eval,
                              pin_memory=pin_eval,
                              persistent_workers=(nw_eval > 0))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=nw_eval,
                              pin_memory=pin_eval,
                              persistent_workers=(nw_eval > 0))

    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    return train_loader, val_loader, test_loader, adj_tensor, scaler, F