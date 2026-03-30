"""
GridCFN-Improved Configuration
================================
新增超参：n_codes, K, beta_vq, beta_mi, n_heads
移除：warmup_epochs, grl_alpha, lambda_mi（不再使用）
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DataConfig:
    dataset:       str           = "synthetic"
    data_path:     Optional[str] = None
    T_in:          int           = 12
    T_out:         int           = 1
    adj_threshold: float         = 0.7
    batch_size:    int           = 32
    synthetic_T:   int           = 2000
    synthetic_N:   int           = 20
    synthetic_F:   int           = 1


@dataclass
class ModelConfig:
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
    # ★ CaST 相关
    n_codes:      int   = 64
    n_heads:      int   = 4     # 需整除 env_dim
    # ★ GMM 相关
    K:            int   = 3
    # ★ 损失权重
    beta_vq:      float = 0.25
    beta_mi:      float = 0.1
    # 兼容旧接口
    lambda_mi:    float = 0.5


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
    gpu_id:            int          = -1
    log_dir:           Optional[str] = "logs"
    log_to_console:    bool         = True


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def summary(self) -> str:
        lines = ["=" * 52, "GridCFN-Improved Configuration", "=" * 52]
        for name, sec in [("Data", self.data),
                          ("Model", self.model),
                          ("Train", self.train)]:
            lines.append(f"\n[{name}]")
            for k, v in asdict(sec).items():
                lines.append(f"  {k:<22} = {v}")
        lines.append("=" * 52)
        return "\n".join(lines)


def get_config(preset: str = "default") -> Config:
    if preset == "default":
        return Config()

    elif preset == "solar":
        return Config(
            data=DataConfig(
                dataset="solar", data_path="./data/solar_AL.txt",
                T_in=168, T_out=1, adj_threshold=0.95, batch_size=32),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                n_codes=64, n_heads=4, K=3, beta_vq=0.25, beta_mi=0.1),
            train=TrainConfig(
                lr=5e-4, max_epochs=200, patience=20, seed=42, grad_clip=1.0))

    elif preset == "electricity":
        return Config(
            data=DataConfig(
                dataset="electricity", data_path="./data/electricity.txt",
                T_in=168, T_out=1, adj_threshold=0.7, batch_size=32),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                n_codes=64, n_heads=4, K=3, beta_vq=0.25, beta_mi=0.1),
            train=TrainConfig(
                lr=1e-3, max_epochs=200, patience=20, seed=42))

    elif preset == "weather":
        return Config(
            data=DataConfig(
                dataset="weather", data_path="./data/weather2k.npy",
                T_in=168, T_out=1, adj_threshold=0.6, batch_size=32),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                n_codes=64, n_heads=4, K=3, beta_vq=0.25, beta_mi=0.1),
            train=TrainConfig(
                lr=1e-3, max_epochs=200, patience=20, seed=42))

    elif preset == "debug":
        return Config(
            data=DataConfig(
                dataset="synthetic", T_in=12, T_out=1,
                batch_size=8, synthetic_T=300,
                synthetic_N=5, synthetic_F=1),
            model=ModelConfig(
                in_dim=1, gcn_hidden=16, gcn_layers=2,
                tcn_hidden=16, tcn_layers=2,
                env_dim=8, stoch_dim=8, ms_out_dim=8,
                n_scg_layers=2, out_dim=1,
                # env_dim=8, n_heads=2 → 8/2=4 能整除，无问题
                n_codes=16, n_heads=2, K=2, beta_vq=0.25, beta_mi=0.1),
            train=TrainConfig(
                lr=1e-3, max_epochs=5, patience=5, seed=0))

    else:
        raise ValueError(
            f"Unknown preset '{preset}'. "
            "Choices: default, solar, electricity, weather, debug")
