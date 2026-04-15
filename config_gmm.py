"""
GridCFN + GMM Configuration v2
================================

字段名与 model_gmm_v2.py 中 GridCFN.__init__ 的参数名严格一一对应，
main.py 的 build_model 按名称传参，不会出现 AttributeError。

ModelConfig 中所有字段：
  基础（不变）：in_dim / gcn_hidden / gcn_layers / tcn_hidden / tcn_layers /
               env_dim / stoch_dim / ms_out_dim / n_scg_layers / out_dim / lambda_mi
  GMM v1：n_components / lambda_mean / lambda_weight
  GMM v2（新增）：proj_dim / sigma_min_heavy / huber_delta / use_grin
  注：n_nodes 由 main.py 在数据加载后自动设置，不在 config 中定义。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


# ===========================================================================
# 数据集配置（不变）
# ===========================================================================
@dataclass
class DataConfig:
    dataset:       str           = "solar"
    data_path:     Optional[str] = "./data/solar_AL.txt"
    T_in:          int           = 168
    T_out:         int           = 1
    adj_threshold: float         = 0.95
    batch_size:    int           = 32


# ===========================================================================
# 模型结构配置
# ===========================================================================
@dataclass
class ModelConfig:
    # ── 基础参数 ──────────────────────────────────────────────────────────
    in_dim:       int   = 1
    gcn_hidden:   int   = 64
    gcn_layers:   int   = 2
    tcn_hidden:   int   = 64
    tcn_layers:   int   = 4
    env_dim:      int   = 32
    stoch_dim:    int   = 32
    ms_out_dim:   int   = 32
    n_scg_layers: int   = 3
    out_dim:      int   = 1
    lambda_mi:    float = 0.5

    # ── GMM v1 参数 ────────────────────────────────────────────────────────
    n_components:  int   = 3
    lambda_mean:   float = 0.1
    lambda_weight: float = 0.01

    # ── GMM v2 新增参数（字段名与 GridCFN.__init__ 严格对齐）──────────────
    proj_dim:        int   = 64     # CLUB projection MLP 维度
    sigma_min_heavy: float = 0.5    # GMMHead 重尾分量最小 sigma
    huber_delta:     float = 1.0    # Huber loss delta
    use_grin:        bool  = True   # 是否启用 GRIN denorm


# ===========================================================================
# 训练配置（不变）
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
    log_dir:           Optional[str]= "logs"
    log_to_console:    bool         = True


# ===========================================================================
# 顶层配置
# ===========================================================================
@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def summary(self) -> str:
        lines = ["=" * 52, "GridCFN + GMM v2 Configuration", "=" * 52]
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
                n_components=3, lambda_mean=0.1, lambda_weight=0.01,
                proj_dim=64,
                sigma_min_heavy=0.05,   # Solar 无重尾，与 sigma_min 相同
                huber_delta=1.0,
                use_grin=False, #True,
            ),
            train=TrainConfig(
                lr=5e-4, max_epochs=200, patience=20,
                seed=42, grad_clip=1.0, warmup_epochs=5,
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
                n_components=5, lambda_mean=0.1, lambda_weight=0.01,
                proj_dim=64,
                sigma_min_heavy=0.5,    # 重尾分量捕捉突变用户
                huber_delta=0.5,        # 更激进地抑制大误差梯度
                use_grin=False,  #True,
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200, patience=20,
                seed=42, grad_clip=1.0, warmup_epochs=5,
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
                n_components=3, lambda_mean=0.1, lambda_weight=0.01,
                proj_dim=64,
                sigma_min_heavy=0.1,    # 轻度重尾保护
                huber_delta=1.0,
                use_grin=True,
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200, patience=20,
                seed=42, grad_clip=1.0, warmup_epochs=5,
            ),
        )

    else:
        raise ValueError(
            f"Unknown preset: '{preset}'. "
            "Choose from: solar, electricity, weather"
        )
