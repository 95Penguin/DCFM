"""
wind_mask_utils.py
──────────────────
为 SDWPF 风电场构造有向尾流掩码（wind_mask），用于 SparseGCN 的 _adj() 方法。

掩码含义：
  wind_mask[i, j] = 1  表示风机 j 在风机 i 的下风向扇区内，
                        即 i 的尾流会影响 j，允许 i→j 的消息传递。

使用方式：
  from wind_mask_utils import build_wind_mask_from_csv
  wind_mask = build_wind_mask_from_csv("sdwpf_turb_location.csv", angle_tol=45.0)
  # wind_mask: torch.FloatTensor, shape [N, N]

坐标文件格式（sdwpf_turb_location.csv，来自 Kaggle SDWPF 数据集）：
  TurbID,x,y     （或 TurbID,X,Y，不区分大小写）
  1,100.0,200.0
  2,150.0,210.0
  ...

风向约定：
  - 主风向由坐标文件对应的 SDWPF 原始气象数据统计得到，
    默认用数据中 Wdir 字段众数（如无气象数据则使用 dominant_dir 参数手动指定）。
  - angle_tol（度）控制扇区半角，45° 表示 ±45° 共 90° 的下风向扇区。
"""

import math
import os
from typing import Optional

import numpy as np
import torch


def _dominant_wind_dir_from_data(data_path: str) -> Optional[float]:
    """
    从 SDWPF 原始 CSV 数据文件中统计主风向（度，气象约定：0°=北，顺时针）。
    返回主风向角度（float），若读取失败返回 None。
    """
    try:
        import pandas as pd
        df = pd.read_csv(data_path, nrows=50000, usecols=lambda c: c.strip().lower() in ("wdir", "wind_dir"))
        col = df.columns[0]
        vals = df[col].dropna().values
        # 转为弧度再做圆形均值，比直接平均角度更鲁棒
        rad = np.deg2rad(vals)
        mean_dir = float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)
        return mean_dir
    except Exception:
        return None


def build_wind_mask_from_csv(
    coord_csv: str,
    data_csv: Optional[str] = None,
    dominant_dir: Optional[float] = None,
    angle_tol: float = 45.0,
    min_dist: float = 0.0,
) -> torch.Tensor:
    """
    从坐标 CSV 文件构造有向尾流掩码。

    参数
    ────
    coord_csv     : 风机坐标文件路径（含 TurbID, x/X, y/Y 列）
    data_csv      : 原始气象数据路径（含 Wdir 列），用于自动统计主风向；
                    若为 None，则尝试在 coord_csv 同目录下寻找 *.csv 中含 Wdir 列的文件
    dominant_dir  : 手动指定主风向（度，气象约定 0°=北顺时针）；
                    若为 None 则自动从 data_csv 统计，若仍失败则默认 270°（西风）
    angle_tol     : 下风向扇区半角（度），默认 45°（即 ±45°，共 90° 扇区）
    min_dist      : 最小距离过滤（与自身距离=0，自动排除自环），单位与坐标相同

    返回
    ────
    wind_mask : torch.FloatTensor, shape [N, N]
        wind_mask[i, j] = 1 表示允许 i → j 消息传递（j 在 i 下风向）
    """
    import pandas as pd

    # ── 1. 读取坐标 ──────────────────────────────────────────────────────────
    df_coord = pd.read_csv(coord_csv)
    df_coord.columns = [c.strip().lower() for c in df_coord.columns]

    # 兼容列名：turbid/turb_id/id, x/lon/longitude, y/lat/latitude
    id_col = next((c for c in df_coord.columns if "turb" in c or c == "id"), df_coord.columns[0])
    x_col  = next((c for c in df_coord.columns if c in ("x", "lon", "longitude")), None)
    y_col  = next((c for c in df_coord.columns if c in ("y", "lat", "latitude")),  None)
    if x_col is None or y_col is None:
        # 退而求其次：取第2、3列
        x_col, y_col = df_coord.columns[1], df_coord.columns[2]

    df_coord = df_coord.sort_values(id_col).reset_index(drop=True)
    coords   = df_coord[[x_col, y_col]].values.astype(np.float32)   # [N, 2]
    N        = len(coords)

    # ── 2. 确定主风向（气象角：0°=北，顺时针） ──────────────────────────────
    if dominant_dir is None:
        # 优先尝试 data_csv，其次在同目录搜索
        if data_csv is not None and os.path.exists(data_csv):
            dominant_dir = _dominant_wind_dir_from_data(data_csv)
        if dominant_dir is None:
            data_dir = os.path.dirname(coord_csv)
            for fn in os.listdir(data_dir):
                if fn.endswith(".csv") and fn != os.path.basename(coord_csv):
                    candidate = _dominant_wind_dir_from_data(os.path.join(data_dir, fn))
                    if candidate is not None:
                        dominant_dir = candidate
                        break
        if dominant_dir is None:
            dominant_dir = 270.0   # 默认西风（华北风电场常见主风向）
            print(f"[wind_mask] 未找到气象数据，使用默认主风向 {dominant_dir}°（西风）")
        else:
            print(f"[wind_mask] 自动统计主风向: {dominant_dir:.1f}°")
    else:
        print(f"[wind_mask] 手动指定主风向: {dominant_dir:.1f}°")

    # 气象角 → 数学角（从 x 轴正方向逆时针）
    # 气象角 θ_met：0=北(+y)，顺时针
    # 数学角 θ_math = 90 - θ_met
    # 风从 θ_met 方向吹来，尾流指向 θ_met + 180° 方向（下风向）
    wake_dir_met  = (dominant_dir + 180.0) % 360.0   # 下风向方向（气象角）
    wake_dir_math = math.radians(90.0 - wake_dir_met)  # 转数学弧度

    # ── 3. 构造掩码 ──────────────────────────────────────────────────────────
    # dx[i,j] = coords[j,0] - coords[i,0]：从 i 指向 j 的向量
    dx = coords[:, 0][np.newaxis, :] - coords[:, 0][:, np.newaxis]   # [N, N]
    dy = coords[:, 1][np.newaxis, :] - coords[:, 1][:, np.newaxis]   # [N, N]
    dist = np.sqrt(dx ** 2 + dy ** 2)                                  # [N, N]

    # i→j 向量与下风向方向的夹角
    angle_ij   = np.arctan2(dy, dx)                  # 数学角，弧度
    angle_diff = angle_ij - wake_dir_math            # 与下风向的偏差
    # 归一化到 [-π, π]
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
    angle_diff_deg = np.abs(np.degrees(angle_diff))  # [N, N]，绝对角偏差（度）

    tol_rad = angle_tol   # 度
    in_sector = angle_diff_deg < tol_rad             # j 在 i 下风向扇区内
    not_self  = dist > min_dist                      # 排除自环（dist=0）

    mask = (in_sector & not_self).astype(np.float32)  # [N, N]

    # ── 4. 安全检查：若某行全零（无下风向邻居），允许全连接（退化为无向图） ──
    zero_rows = mask.sum(axis=1) == 0
    if zero_rows.any():
        n_zero = int(zero_rows.sum())
        print(f"[wind_mask] {n_zero} 台风机无下风向邻居，对应行改为全1（退化无向）")
        mask[zero_rows, :] = 1.0
        np.fill_diagonal(mask, 0.0)   # 仍排除自环

    density = mask.sum() / (N * (N - 1)) * 100
    print(f"[wind_mask] N={N}, 非零边={int(mask.sum())}, 密度={density:.1f}%")

    return torch.from_numpy(mask)


# ── 单元测试（直接运行此文件时执行） ──────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, csv

    # 构造一个 5 台风机的虚拟坐标文件
    coords_fake = [
        (1, 0.0,   0.0),
        (2, 100.0, 0.0),
        (3, 200.0, 0.0),
        (4, 0.0,   100.0),
        (5, 100.0, 100.0),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TurbID", "x", "y"])
        writer.writerows(coords_fake)
        tmp_path = f.name

    # 西风（270°），下风向为东（0°/360°）
    mask = build_wind_mask_from_csv(tmp_path, dominant_dir=270.0, angle_tol=45.0)
    print("\n风向掩码（西风，下风向向东，容差±45°）:")
    print(mask)
    # 期望：风机1的下风方向是风机2、3（x更大），风机4→风机5
    os.unlink(tmp_path)