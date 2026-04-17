"""
GridCFN Configuration（CFM 版 v2）
====================================
相比 v1，新增参数：
  TrainConfig:
    cfm_n_samples    : 验证集采样粒子数（默认 10，快速；测试时自动 ×10）
    cfm_n_steps      : ODE 欧拉积分步数（默认 20）
    cfm_n_t_samples  : 每 batch 采 t 的次数（默认 4，降梯度方差）

  ModelConfig:
    cfm_hidden       : 向量场 MLP 隐藏层维度（默认 128）
    cfm_time_emb_dim : 时间编码维度（默认 16，必须是偶数）

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
    dataset:       str            = "solar"
    data_path:     Optional[str]  = "./data/solar_AL.txt"
    T_in:          int            = 168
    T_out:         int            = 1
    adj_threshold: float          = 0.95
    batch_size:    int            = 32


# ===========================================================================
# 模型结构配置
# ===========================================================================
@dataclass
class ModelConfig:
    in_dim:          int   = 1
    gcn_hidden:      int   = 64
    gcn_layers:      int   = 2
    tcn_hidden:      int   = 64
    tcn_layers:      int   = 4
    env_dim:         int   = 32
    stoch_dim:       int   = 32
    ms_out_dim:      int   = 32
    n_scg_layers:    int   = 3
    out_dim:         int   = 1
    lambda_mi:       float = 0.5

    # CFM 向量场参数
    cfm_hidden:       int = 128   # MLP 隐藏层维度
    cfm_time_emb_dim: int = 16    # 时间编码维度（必须为偶数）


# ===========================================================================
# 训练配置
# ===========================================================================
@dataclass
class TrainConfig:
    lr:                float        = 1e-3
    max_epochs:        int          = 200
    patience:          int          = 20
    lr_decay_factor:   float        = 0.5
    lr_decay_patience: int          = 10
    grad_clip:         float        = 1.0
    weight_decay:      float        = 1e-5
    save_path:         str          = "best_model.pt"
    seed:              int          = 42
    warmup_epochs:     int          = 5
    gpu_id:            int          = -1
    log_dir:           Optional[str] = "logs"
    log_to_console:    bool         = True

    # CFM 推断参数
    # n_samples 控制验证速度（验证时小，测试时 ×10）
    # 10 粒子的 CRPS 估计误差 < 1%，可以放心用小值加速验证
    cfm_n_samples:    int = 10    # 验证集粒子数（测试自动 ×10）
    cfm_n_steps:      int = 20    # ODE 欧拉步数
    cfm_n_t_samples:  int = 4     # 每 batch 采 t 次数（降梯度方差）


# ===========================================================================
# 顶层配置
# ===========================================================================
@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def summary(self) -> str:
        lines = ["=" * 52, "GridCFN Configuration (CFM v2)", "=" * 52]
        for section_name, section in [("Data",  self.data),
                                       ("Model", self.model),
                                       ("Train", self.train)]:
            lines.append(f"\n[{section_name}]")
            for k, v in asdict(section).items():
                lines.append(f"  {k:<24} = {v}")
        lines.append("=" * 52)
        return "\n".join(lines)


# ===========================================================================
# 预设配置
# ===========================================================================

def get_config(preset: str = "solar") -> Config:
    """preset: "solar" | "electricity" | "weather" """

    # 公共 ModelConfig（三个数据集共享相同模型结构）
    _model = ModelConfig(
        in_dim=1, gcn_hidden=64, gcn_layers=2,
        tcn_hidden=64, tcn_layers=4,
        env_dim=32, stoch_dim=32, ms_out_dim=32,
        n_scg_layers=3, out_dim=1, lambda_mi=0.5,
        cfm_hidden=128, cfm_time_emb_dim=16,
    )

    if preset == "solar":
        return Config(
            data=DataConfig(
                dataset="solar",
                data_path="./data/solar_AL.txt",
                T_in=168, T_out=1,
                adj_threshold=0.95,
                batch_size=32,
            ),
            model=_model,
            train=TrainConfig(
                lr=5e-4, max_epochs=200, patience=20,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=10, cfm_n_steps=20, cfm_n_t_samples=4,
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
            model=_model,
            train=TrainConfig(
                lr=1e-3, max_epochs=200, patience=20,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=10, cfm_n_steps=20, cfm_n_t_samples=4,
            ),
        )

    elif preset == "weather":
        # Weather: 1866 节点，batch=4，验证采样用最小值 10
        return Config(
            data=DataConfig(
                dataset="weather",
                data_path="./data/weather2k.npy",
                T_in=168, T_out=1,
                adj_threshold=0.8,
                batch_size=4,
            ),
            model=_model,
            train=TrainConfig(
                lr=1e-3, max_epochs=200, patience=20,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                # Weather 节点多，验证采样用最小值避免 OOM/超时
                cfm_n_samples=10, cfm_n_steps=20, cfm_n_t_samples=4,
            ),
        )

    else:
        raise ValueError(
            f"Unknown preset: '{preset}'. "
            "Choose from: solar, electricity, weather"
        )
