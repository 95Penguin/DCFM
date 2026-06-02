# """
# GridCFN – 数据集工具

# 支持四个数据集：
#   - Solar-Energy  (137 PV plants, 10-min, 2007)
#   - Electricity   (UCI, 321 clients, 1-hour, 2012-2014)
#   - Weather2k     (1866 stations, 1-hour, 2017-2021)
#   - SDWPF         (134 wind turbines, 10-min, 2020-2021, KDD Cup 2022)

# 数据文件获取：
#   Solar / Electricity → https://github.com/laiguokun/multivariate-time-series-data
#   Weather2k           → 原始论文仓库或 Kaggle
#   SDWPF               → https://figshare.com/articles/dataset/SDWPF_dataset/24798654
#                         （Nature Scientific Data 2024，龙源电力 SCADA 实采数据）
# """

# import os
# import platform
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from typing import Optional


# # ---------------------------------------------------------------------------
# # 邻接矩阵
# # ---------------------------------------------------------------------------

# def build_correlation_adj(data: np.ndarray, threshold: float = 0.7) -> np.ndarray:
#     """
#     基于皮尔逊相关系数构建二值邻接矩阵。

#     data : [T, N]
#     返回 A : [N, N]，|corr| >= threshold 置 1，对角线置 0

#     孤立节点保护：度为 0 的节点补自环，防止 normalize_adj 除以 0。
#     部分数据集（如 Electricity）有全零列（缺失节点），其相关系数为 NaN，
#     nan_to_num 将其替换为 0。
#     """
#     corr = np.corrcoef(data.T)
#     corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
#     adj  = (np.abs(corr) >= threshold).astype(np.float32)
#     np.fill_diagonal(adj, 0)

#     isolated = (adj.sum(axis=1) == 0)
#     if isolated.any():
#         print(f"  [adj] 检测到 {isolated.sum()} 个孤立节点，已补自环")
#         adj[isolated, isolated] = 1.0

#     return adj


# # ---------------------------------------------------------------------------
# # 滑动窗口数据集
# # ---------------------------------------------------------------------------

# class SlidingWindowDataset(Dataset):
#     """
#     多变量时序滑动窗口数据集。

#     data  : [T, N, F]  已归一化
#     adj   : [N, N]
#     T_in  : 输入窗口长度
#     T_out : 预测步长
#     """
#     def __init__(self, data: np.ndarray, adj: np.ndarray,
#                  T_in: int = 168, T_out: int = 1):
#         super().__init__()
#         self.data      = torch.tensor(data, dtype=torch.float32)
#         self.adj       = torch.tensor(adj,  dtype=torch.float32)
#         self.T_in      = T_in
#         self.T_out     = T_out
#         self.n_samples = len(data) - T_in - T_out + 1
#         assert self.n_samples > 0, (
#             f"数据长度 {len(data)} 不足以构造窗口 T_in={T_in} + T_out={T_out}"
#         )

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         x = self.data[idx : idx + self.T_in]
#         y = self.data[idx + self.T_in : idx + self.T_in + self.T_out]
#         return x, y   # shape: [T_in, N, F], [T_out, N, F]  —— 保持时间维一致


# # ---------------------------------------------------------------------------
# # Z-score 归一化
# # ---------------------------------------------------------------------------

# class Scaler:
#     """
#     全局 Z-score 归一化，在训练集上 fit。

#     log_transform=True 时，inverse_transform 额外执行 expm1 反变换，
#     用于预先做过 log1p 的数据集（Electricity 重尾场景）。
#     """
#     def __init__(self, log_transform: bool = False):
#         self.mean: float = 0.0
#         self.std:  float = 1.0
#         self.log_transform: bool = log_transform

#     def fit(self, data: np.ndarray) -> "Scaler":
#         self.mean = float(data.mean())
#         self.std  = float(data.std()) + 1e-8
#         return self

#     def transform(self, data: np.ndarray) -> np.ndarray:
#         return (data - self.mean) / self.std

#     def inverse_transform(self, data):
#         """支持 numpy array 或 torch tensor。"""
#         out = data * self.std + self.mean
#         if self.log_transform:
#             if isinstance(out, np.ndarray):
#                 out = np.expm1(out)
#             else:
#                 out = out.exp() - 1.0
#         return out


# # ---------------------------------------------------------------------------
# # 数据集加载器
# # ---------------------------------------------------------------------------

# def load_solar_energy(data_path: str, T_in: int = 168, T_out: int = 1,
#                       adj_threshold: float = 0.95, batch_size: int = 32):
#     """
#     Solar-Energy: 137 个 PV 电站，2007 年，10 分钟分辨率。
#     文件格式: solar_AL.txt（逗号分隔，[T=52560, N=137]）
#     """
#     raw = np.loadtxt(data_path, delimiter=',')
#     return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Solar')


# def load_electricity(data_path: str, T_in: int = 168, T_out: int = 1,
#                      adj_threshold: float = 0.7, batch_size: int = 32):
#     """
#     Electricity (UCI): 321 个用电客户，小时级，2012-2014。
#     文件格式: electricity.txt（逗号分隔，[T=26304, N=321]）

#     注：Electricity 是重尾分布（工业 vs 居民用电量差异 100x+）。
#     若直接 Z-score 导致 sigma 学习困难，可在此处开启 log1p 预处理：
#       raw = np.log1p(np.clip(raw, 0, None))
#     并将 _build_loaders 的 log_transform 参数改为 True。
#     """
#     # raw = np.loadtxt(data_path, delimiter=',')
#     # raw = np.log1p(np.clip(raw, 0, None))

#     # print(f"  [Electricity] log1p 变换后：mean={raw.mean():.3f}, "
#     #       f"std={raw.std():.3f}, min={raw.min():.3f}, max={raw.max():.3f}")

#     # return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size,
#     #                       name='Electricity', log_transform=True)
    
#     raw = np.loadtxt(data_path, delimiter=',')
#     return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Electricity')



# def load_weather(data_path: str, T_in: int = 168, T_out: int = 1,
#                  adj_threshold: float = 0.6, batch_size: int = 32,
#                  feature_idx: int = 0):
#     """
#     Weather2k: 1866 个气象站，小时级，2017-2021。
#     文件格式: weather2k.npy

#     npy 文件有两种常见 shape，代码自动识别：
#       格式 A (T, N, F)：直接使用。
#       格式 B (N, F, T)：Weather2k 论文原始格式，第 0 维为站点数（1866），
#                         第 2 维为时间步（远大于 N），自动 transpose 为 (T, N, F)。

#     feature_idx : 选取哪个特征（0=第一个，通常为温度；-1=全部特征）。
#     """
#     raw = np.load(data_path)
#     print(f"  [Weather2k] 原始 shape: {raw.shape}, dtype: {raw.dtype}")

#     if raw.ndim == 2:
#         raw = raw[:, :, None]
#         print(f"  [Weather2k] 2D → 3D: {raw.shape}")

#     elif raw.ndim == 3:
#         T0, D1, D2 = raw.shape
#         # 格式 B (N, F, T)：第0维是站点数（通常 < 5000），第2维是时间步（远大于站点数）。
#         # 判据：T0 < D2（时间步比站点数大得多）且 T0 < D1（站点数 > 特征数不成立时也安全）。
#         # 相比原来硬编码 T0<5000，改为 T0 < D2 且 D2 > D1 的相对判据，
#         # 避免在格式A且站点数恰好<5000时误判。
#         is_format_b = (D2 > T0) and (D2 > D1) and (T0 <= D1)
#         if is_format_b:
#             raw = raw.transpose(2, 0, 1)
#             print(f"  [Weather2k] 检测到格式 B (N,F,T)，已转置为 (T,N,F): {raw.shape}")
#         else:
#             print(f"  [Weather2k] 检测到格式 A (T,N,F): {raw.shape}")

#     T, N, F = raw.shape
#     print(f"  [Weather2k] 解析结果: T={T} 时间步, N={N} 站点, F={F} 特征")

#     if F > 1 and feature_idx >= 0:
#         print(f"  [Weather2k] 选取特征索引 {feature_idx}（共 {F} 个特征）")
#         raw = raw[:, :, feature_idx : feature_idx + 1]
#     elif feature_idx < 0:
#         print(f"  [Weather2k] 使用全部 {F} 个特征（config.in_dim 需设为 {F}）")

#     return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='Weather')


# def load_sdwpf(data_path: str, T_in: int = 168, T_out: int = 1,
#                adj_threshold: float = 0.88, batch_size: int = 32):
#     """
#     SDWPF：134 台风机，10 分钟分辨率，2020-2021年。
#     来源：龙源电力 SCADA 系统实采数据，KDD Cup 2022 / Nature Scientific Data 2024。
#     下载：https://figshare.com/articles/dataset/SDWPF_dataset/24798654

#     文件格式：sdwpf_245days_v1.csv（KDD版，约245天）
#               或 sdwpf_full/ 目录下的完整版（约两年）
#     推荐使用 KDD 版（245天，约52,560个时间步，与 Solar 规模相近）。

#     CSV 列：TurbID, Day, Tmstamp, Wspd, Wdir, Etmp, Itmp, Ndir,
#             Pab1, Pab2, Pab3, Prtv, Patv
#     目标列：Patv（有功功率，kW）

#     数据清洗规则（来自官方论文）：
#       1. Patv < 0 → 置 0（传感器底噪，风机未发电）
#       2. Patv <= 0 且 Wspd > 2.5 → 标记为 NaN（风机停机，功率未知）
#       3. Pab1/2/3 > 89° → 标记为 NaN（风机静止，桨叶顺桨）
#       4. 任意列存在异常值（Wspd<0, Etmp/Itmp 超出合理范围等）→ 标记为 NaN
#       最终 NaN 用该风机列的线性插值填充，首尾 NaN 用前向/后向填充兜底。

#     adj_threshold 建议 0.88：风机聚集在同一风场，相关性普遍较高（>0.9），
#     0.88 可保留适量边，避免图过于稠密导致过平滑。
#     """
#     raw = _load_sdwpf_csv(data_path)
#     return _build_loaders(raw, T_in, T_out, adj_threshold, batch_size, name='SDWPF')


# def _load_sdwpf_csv(data_path: str) -> np.ndarray:
#     """
#     将 SDWPF CSV 解析为 [T, N=134] 的有功功率矩阵。

#     处理流程：
#       1. 读取 CSV，统一列名
#       2. 构造时间索引（Day + Tmstamp → 全局时间步编号）
#       3. pivot：行=时间步，列=TurbID，值=Patv
#       4. 数据清洗（异常置 NaN → 插值填充）
#       5. 返回 [T, N] float32 数组
#     """
#     print(f"  [SDWPF] 正在读取: {data_path}")
#     df = pd.read_csv(data_path)

#     # ── 列名标准化 ────────────────────────────────────────────────────────
#     # 原始列名可能带单位后缀（如 "Wspd (m/s)"），统一去掉括号内容
#     df.columns = [c.split('(')[0].strip() for c in df.columns]
#     # 确保关键列存在
#     required = {'TurbID', 'Day', 'Tmstamp', 'Patv'}
#     missing  = required - set(df.columns)
#     assert not missing, f"CSV 缺少列: {missing}，现有列: {list(df.columns)}"

#     print(f"  [SDWPF] 原始行数: {len(df):,}，风机数: {df['TurbID'].nunique()}")

#     # ── 构造全局时间步编号 ────────────────────────────────────────────────
#     # Tmstamp 格式为 "HH:MM"（每10分钟一步，每天144步）
#     # 全局时间步 = (Day - 1) * 144 + 步内编号
#     df['time_step'] = _parse_time_step(df)

#     # ── 数据清洗：异常置 NaN ──────────────────────────────────────────────
#     df = _clean_sdwpf(df)

#     # ── Pivot：[时间步, 风机] ─────────────────────────────────────────────
#     pivot = df.pivot_table(
#         index='time_step', columns='TurbID', values='Patv', aggfunc='mean'
#     )
#     # 确保风机列完整有序（1~134）
#     expected_turbs = sorted(df['TurbID'].unique())
#     pivot = pivot.reindex(columns=expected_turbs)

#     # 确保时间步连续（补全缺失时间步为 NaN）
#     t_min, t_max = pivot.index.min(), pivot.index.max()
#     full_index   = np.arange(t_min, t_max + 1)
#     pivot        = pivot.reindex(full_index)

#     T, N = pivot.shape
#     print(f"  [SDWPF] Pivot 完成: T={T} 时间步（{T/144:.1f} 天），N={N} 风机")

#     # ── 插值填充 NaN ──────────────────────────────────────────────────────
#     raw = pivot.values.astype(np.float32)   # [T, N]
#     nan_ratio = np.isnan(raw).mean()
#     print(f"  [SDWPF] 清洗后 NaN 比例: {nan_ratio:.3%}")

#     # 逐列线性插值（时间轴），首尾用前向/后向填充兜底
#     df_raw = pd.DataFrame(raw)
#     df_raw = df_raw.interpolate(method='linear', axis=0, limit_direction='both')
#     raw    = df_raw.values.astype(np.float32)

#     remaining_nan = np.isnan(raw).sum()
#     if remaining_nan > 0:
#         # 用各列（各风机）非 NaN 均值填充，而非固定 0。
#         # 0 会被模型误学为停机状态，引入系统性偏差；
#         # 列均值是最保守的中性填充，不引入额外分布偏移。
#         col_means = np.nanmean(raw, axis=0)          # [N]
#         # 若某列全为 NaN，np.nanmean 返回 NaN，fallback 到全局均值或 0
#         nan_cols = np.isnan(col_means)
#         if nan_cols.any():
#             global_mean = np.nanmean(raw) if not np.all(nan_cols) else 0.0
#             col_means[nan_cols] = global_mean
#             print(f"  [SDWPF] {nan_cols.sum()} 台风机整列无有效数据，已用全局均值 "
#                   f"({global_mean:.2f} kW) 填充")
#         nan_mask  = np.isnan(raw)
#         raw[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
#         print(f"  [SDWPF] 插值后仍有 {remaining_nan} 个 NaN，已用各风机列均值填充")

#     print(f"  [SDWPF] 最终数据: shape={raw.shape}, "
#           f"min={raw.min():.2f}, max={raw.max():.2f}, mean={raw.mean():.2f} kW")
#     return raw   # [T, N=134]


# def _parse_time_step(df: pd.DataFrame) -> pd.Series:
#     """
#     将 Day + Tmstamp（"HH:MM"）转为全局时间步编号（从0开始）。
#     每天 144 步（24h × 6步/h）。
#     """
#     def tmstamp_to_step(ts: str) -> int:
#         """"HH:MM" → 步内编号（0~143）"""
#         try:
#             h, m = map(int, str(ts).strip().split(':'))
#             return h * 6 + m // 10
#         except Exception:
#             return 0

#     intra_step = df['Tmstamp'].apply(tmstamp_to_step)
#     # Day 从1开始，转为0-based
#     day_0based = (df['Day'] - df['Day'].min())
#     return (day_0based * 144 + intra_step).astype(int)


# def _clean_sdwpf(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     官方论文定义的数据清洗规则，将无效功率值置为 NaN。

#     规则1: Patv < 0 → 置 0（传感器底噪，非停机）
#     规则2: Patv <= 0 且 Wspd > 2.5 → NaN（风速足够但功率为零，风机停机）
#     规则3: Pab1/2/3 > 89° → NaN（桨叶顺桨，风机静止）
#     规则4: Wspd < 0 → NaN（风速传感器异常）
#     规则5: Etmp/Itmp 超出合理范围（< -40 或 > 80°C）→ NaN
#     """
#     # 规则1：负功率底噪置0
#     df.loc[df['Patv'] < 0, 'Patv'] = 0.0

#     # 规则2：停机标记（Patv<=0 且 Wspd>2.5）
#     if 'Wspd' in df.columns:
#         mask_stop = (df['Patv'] <= 0) & (df['Wspd'] > 2.5)
#         df.loc[mask_stop, 'Patv'] = np.nan

#     # 规则3：桨叶顺桨（任一桨距角 > 89°）
#     pab_cols = [c for c in ['Pab1', 'Pab2', 'Pab3'] if c in df.columns]
#     if pab_cols:
#         mask_pab = (df[pab_cols] > 89.0).any(axis=1)
#         df.loc[mask_pab, 'Patv'] = np.nan

#     # 规则4：风速传感器异常
#     if 'Wspd' in df.columns:
#         df.loc[df['Wspd'] < 0, 'Patv'] = np.nan

#     # 规则5：温度传感器异常
#     for col in ['Etmp', 'Itmp']:
#         if col in df.columns:
#             mask_temp = (df[col] < -40) | (df[col] > 80)
#             df.loc[mask_temp, 'Patv'] = np.nan

#     return df


# def _build_loaders(raw: np.ndarray,
#                    T_in: int, T_out: int,
#                    adj_threshold: float,
#                    batch_size: int,
#                    name: str = '',
#                    log_transform: bool = False) -> tuple:
#     """
#     共享加载逻辑：
#       1. 扩展到 [T, N, F]
#       2. 70/10/20 划分
#       3. Z-score 归一化（fit on train）
#       4. 皮尔逊相关邻接矩阵（fit on train，基于第一个特征）
#       5. 返回 (train_loader, val_loader, test_loader, adj_tensor, scaler, in_dim)
#          in_dim=F 供 main.py 自动设置 config.model.in_dim
#     """
#     if raw.ndim == 2:
#         raw = raw[:, :, None]

#     T, N, F = raw.shape
#     n_train  = int(T * 0.7)
#     n_val    = int(T * 0.1)

#     train_raw = raw[:n_train]
#     val_raw   = raw[n_train : n_train + n_val]
#     test_raw  = raw[n_train + n_val :]

#     scaler     = Scaler(log_transform=log_transform).fit(train_raw)
#     train_data = scaler.transform(train_raw)
#     val_data   = scaler.transform(val_raw)
#     test_data  = scaler.transform(test_raw)

#     adj = build_correlation_adj(train_data[:, :, 0], threshold=adj_threshold)

#     print(f"[{name}] T={T}, N={N}, F={F} | "
#           f"train={n_train}, val={n_val}, test={T - n_train - n_val} | "
#           f"edges={int(adj.sum())}, adj_density={adj.mean():.3f}")

#     train_ds = SlidingWindowDataset(train_data, adj, T_in, T_out)
#     val_ds   = SlidingWindowDataset(val_data,   adj, T_in, T_out)
#     test_ds  = SlidingWindowDataset(test_data,  adj, T_in, T_out)

#     # num_workers 根据节点数自适应：Weather(1866节点) 等超大图用 0 避免 /dev/shm OOM，
#     # 其余数据集用多进程加速（4/2）。batch_size<=4 说明节点极多，保守用 0。
#     # Windows 下 DataLoader 多进程需要 if __name__=='__main__' 保护，直接禁用以避免死锁。
#     is_windows = platform.system() == 'Windows'
#     nw_train = 0 if (is_windows or batch_size <= 4 or N > 500) else 4
#     nw_eval  = 0 if (is_windows or batch_size <= 4 or N > 500) else 2
#     # pin_memory 只在对应 loader 使用多进程时开启，避免 shm 问题
#     pin_train = (nw_train > 0)
#     pin_eval  = (nw_eval  > 0)

#     train_loader = DataLoader(train_ds, batch_size=batch_size,
#                               shuffle=True,  num_workers=nw_train,
#                               pin_memory=pin_train,
#                               persistent_workers=(nw_train > 0))
#     val_loader   = DataLoader(val_ds,   batch_size=batch_size,
#                               shuffle=False, num_workers=nw_eval,
#                               pin_memory=pin_eval,
#                               persistent_workers=(nw_eval > 0))
#     test_loader  = DataLoader(test_ds,  batch_size=batch_size,
#                               shuffle=False, num_workers=nw_eval,
#                               pin_memory=pin_eval,
#                               persistent_workers=(nw_eval > 0))

#     adj_tensor = torch.tensor(adj, dtype=torch.float32)
#     return train_loader, val_loader, test_loader, adj_tensor, scaler, F






"""
GridCFN – 数据集工具

支持四个数据集：
  - Solar-Energy  (137 PV plants, 10-min, 2007)
  - Electricity   (UCI, 321 clients, 1-hour, 2012-2014)
  - Weather2k     (1866 stations, 1-hour, 2017-2021)
  - SDWPF         (134 wind turbines, 10-min, 2020-2021, KDD Cup 2022)
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
    """
    基于皮尔逊相关系数构建二值邻接矩阵。
    """
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
        y = self.data[idx + self.T_in : idx + self.T_in + self.T_out]
        return x, y   # shape: [T_in, N, F], [T_out, N, F]


# ---------------------------------------------------------------------------
# Z-score 归一化 (优化版)
# ---------------------------------------------------------------------------

class Scaler:
    """
    自适应 Z-score 归一化。
    通过 self.axis 参数自适应选择：
      - 全局归一化 (axis=(0, 1))：计算所有时间步和所有节点的均值/标准差。
      - 逐节点归一化 (axis=(0,))：独立计算每个节点的时间序列均值/标准差，适应高异质性场景。
    """
    def __init__(self, axis=(0, 1), log_transform: bool = False):
        self.axis = axis
        self.mean = None
        self.std  = None
        self.log_transform: bool = log_transform

    def fit(self, data: np.ndarray) -> "Scaler":
        # data: [T, N, F]
        self.mean = data.mean(axis=self.axis, keepdims=True)
        self.std  = data.std(axis=self.axis, keepdims=True) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """支持 numpy array 或 torch tensor，实现灵活的高维广播。"""
        is_torch = torch.is_tensor(data)
        device = data.device if is_torch else None
        dtype = data.dtype if is_torch else None

        mean_val = self.mean
        std_val  = self.std

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
            # 逐节点尺度：m, s 为 [1, N, F]
            # 对齐高维数据的 N 维度，避免 flatten 重组时无法广播
            ndims = len(data.shape)
            if ndims == 3:  # [T, N, F]
                m_aligned, s_aligned = m, s
            elif ndims == 4:  # [B, N, T_out, F]
                if is_torch:
                    m_aligned = m.view(1, m.shape[1], 1, m.shape[2])
                    s_aligned = s.view(1, s.shape[1], 1, s.shape[2])
                else:
                    m_aligned = m.reshape(1, m.shape[1], 1, m.shape[2])
                    s_aligned = s.reshape(1, s.shape[1], 1, s.shape[2])
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

        if self.log_transform:
            if is_torch:
                out = torch.expm1(out)
            else:
                out = np.expm1(out)
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


def _load_sdwpf_csv(data_path: str) -> np.ndarray:
    print(f"  [SDWPF] 正在读取: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = [c.split('(')[0].strip() for c in df.columns]
    df['time_step'] = _parse_time_step(df)
    df = _clean_sdwpf(df)

    pivot = df.pivot_table(
        index='time_step', columns='TurbID', values='Patv', aggfunc='mean'
    )
    expected_turbs = sorted(df['TurbID'].unique())
    pivot = pivot.reindex(columns=expected_turbs)

    t_min, t_max = pivot.index.min(), pivot.index.max()
    full_index   = np.arange(t_min, t_max + 1)
    pivot        = pivot.reindex(full_index)

    raw = pivot.values.astype(np.float32)
    df_raw = pd.DataFrame(raw)
    df_raw = df_raw.interpolate(method='linear', axis=0, limit_direction='both')
    raw    = df_raw.values.astype(np.float32)

    remaining_nan = np.isnan(raw).sum()
    if remaining_nan > 0:
        col_means = np.nanmean(raw, axis=0)
        nan_cols = np.isnan(col_means)
        if nan_cols.any():
            global_mean = np.nanmean(raw) if not np.all(nan_cols) else 0.0
            col_means[nan_cols] = global_mean
        nan_mask  = np.isnan(raw)
        raw[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return raw


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

    # ─── [修改] 重点：对节点异质性高的数据集开启逐节点归一化 ───
    scale_axis = (0,1)

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