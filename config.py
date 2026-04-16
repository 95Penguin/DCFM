"""
GridCFN Configuration（CFM 版）
================================
相比 Gaussian 版，TrainConfig 新增两个 CFM 推断参数：
  cfm_n_samples : 验证/推断时的采样粒子数（越大越准，但越慢）
  cfm_n_steps   : ODE 欧拉积分步数（越大越精确）

ModelConfig 新增两个 CFM 向量场参数：
  cfm_hidden      : 向量场 MLP 隐藏层维度
  cfm_time_emb_dim: 时间傅里叶编码维度

使用方法不变：
  python main.py --preset solar
  python main.py --preset electricity
  python main.py --preset weather
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


# ===========================================================================
# 数据集配置（不变）
# ===========================================================================
@dataclass
class DataConfig:
    # 数据集名称: "solar" | "electricity" | "weather"
    dataset: str = "solar"

    # 真实数据集的文件路径
    data_path: Optional[str] = "./data/solar_AL.txt"

    # 输入窗口长度（论文 Table I: 168）
    T_in: int = 168

    # 预测步长（论文: 1）
    T_out: int = 1

    # 构建邻接矩阵时的相关性阈值
    adj_threshold: float = 0.95

    # DataLoader batch size（论文: 32）
    batch_size: int = 32


# ===========================================================================
# 模型结构配置
# ===========================================================================
@dataclass
class ModelConfig:
    # 输入特征维度 F
    in_dim: int = 1

    # GCN 隐藏层维度
    gcn_hidden: int = 64

    # GCN 层数
    gcn_layers: int = 2

    # TCN 输出维度 D（论文 d_hidden=64）
    tcn_hidden: int = 64

    # TCN 层数（膨胀系数 1,2,4,...,2^(tcn_layers-1)）
    tcn_layers: int = 4

    # 环境上下文维度 De
    env_dim: int = 32

    # 随机实体维度 Ds
    stoch_dim: int = 32

    # 多尺度上下文输出维度 De'
    ms_out_dim: int = 32

    # SCG-MP 层数 L_SCG（论文: 3）
    n_scg_layers: int = 3

    # 输出变量数 Fout（单步单变量预测为 1）
    out_dim: int = 1

    # MI 正则化权重 λ（论文: 0.5）
    lambda_mi: float = 0.5

    # [CFM 新增] 向量场网络参数
    cfm_hidden: int = 128       # 向量场 MLP 隐藏层维度
    cfm_time_emb_dim: int = 8   # 时间傅里叶编码维度（偶数，n_freqs = dim//2）


# ===========================================================================
# 训练配置
# ===========================================================================
@dataclass
class TrainConfig:
    # 初始学习率（论文: 0.001）
    lr: float = 1e-3

    # 最大训练轮数（论文: 200）
    max_epochs: int = 200

    # 早停耐心值（验证集 CRPS 不再提升的容忍轮数）
    patience: int = 20

    # 学习率衰减因子（ReduceLROnPlateau）
    lr_decay_factor: float = 0.5

    # 学习率衰减耐心值
    lr_decay_patience: int = 10

    # 梯度裁剪阈值
    grad_clip: float = 1.0

    # 权重衰减（L2 正则）
    weight_decay: float = 1e-5

    # 最佳模型保存路径
    save_path: str = "best_model.pt"

    # 随机种子
    seed: int = 42

    # 热身 epoch 数：前 warmup_epochs 个 epoch MI 正则权重为 0
    warmup_epochs: int = 5

    # GPU 配置（-1 自动选择，>=0 指定卡号）
    gpu_id: int = -1

    # 日志配置
    log_dir: Optional[str] = "logs"
    log_to_console: bool = True

    # [CFM 新增] 推断采样参数
    # cfm_n_samples: 验证时 50（速度优先），测试时自动 ×4（精度优先）
    # 推荐范围：验证 20~100，最终测试 100~500
    cfm_n_samples: int = 50

    # cfm_n_steps: ODE 欧拉步数，20 步通常已足够（Linear Flow Matching 路径简单）
    # 推荐范围：10~50
    cfm_n_steps: int = 20


# ===========================================================================
# 顶层配置
# ===========================================================================
@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def summary(self) -> str:
        lines = ["=" * 52, "GridCFN Configuration (CFM)", "=" * 52]
        for section_name, section in [("Data",  self.data),
                                       ("Model", self.model),
                                       ("Train", self.train)]:
            lines.append(f"\n[{section_name}]")
            for k, v in asdict(section).items():
                lines.append(f"  {k:<22} = {v}")
        lines.append("=" * 52)
        return "\n".join(lines)


# ===========================================================================
# 预设配置
# ===========================================================================

def get_config(preset: str = "solar") -> Config:
    """
    快速获取预设配置。
    preset: "solar" | "electricity" | "weather"
    """
    if preset == "solar":
        return Config(
            data=DataConfig(
                dataset="solar",
                data_path="./data/solar_AL.txt",
                T_in=168, T_out=1,
                adj_threshold=0.95,
                batch_size=32,
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1, lambda_mi=0.5,
                cfm_hidden=128, cfm_time_emb_dim=8,
            ),
            train=TrainConfig(
                lr=5e-4, max_epochs=200, patience=20,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=50, cfm_n_steps=20,
            ),
        )

    elif preset == "electricity":
        return Config(
            data=DataConfig(
                dataset="electricity",
                data_path="./data/electricity.txt",
                T_in=168, T_out=1,
                adj_threshold=0.7,
                batch_size=32,
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1, lambda_mi=0.5,
                cfm_hidden=128, cfm_time_emb_dim=8,
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200, patience=20,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=50, cfm_n_steps=20,
            ),
        )

    elif preset == "weather":
        return Config(
            data=DataConfig(
                dataset="weather",
                data_path="./data/weather2k.npy",
                T_in=168, T_out=1,
                adj_threshold=0.8,
                batch_size=4,
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1, lambda_mi=0.5,
                cfm_hidden=128, cfm_time_emb_dim=8,
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200, patience=20,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=50, cfm_n_steps=20,
            ),
        )

    else:
        raise ValueError(
            f"Unknown preset: '{preset}'. "
            "Choose from: solar, electricity, weather"
        )
