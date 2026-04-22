"""
GridCFN Configuration（CFM 版 v4，全量修复）

[v4 改动]
  全局：
    · build_model 中新增 chunk_size 参数（Weather 传 4096，Solar/Electricity 传 16384）
      防止 SCGMessagePassingLayer 对大图一次性创建 [B,E,dim] 张量导致 OOM

  Electricity：
    · lambda_mi: 0.1 → 0.05（CLUB 修复后 MI 不再塌缩，降低权重防止过强正则）
    · adj_threshold: 0.7 → 0.6（减少边数，降低 SCG-MP 过平滑风险）
    · warmup_epochs: 5 → 8（延长让主网络先学好表征再引入 CLUB）

  Solar：基本不变（问题不大）

  Weather：
    · chunk_size: 4096（N=1866 节点，边数可能达数十万，需要分块处理）
    · cfm_n_samples: 20（节点多，并行采样显存压力大）
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DataConfig:
    dataset:       str           = "solar"
    data_path:     Optional[str] = "./data/solar_AL.txt"
    T_in:          int           = 168
    T_out:         int           = 1
    adj_threshold: float         = 0.95
    batch_size:    int           = 32


@dataclass
class ModelConfig:
    in_dim:           int   = 1
    gcn_hidden:       int   = 64
    gcn_layers:       int   = 2
    tcn_hidden:       int   = 64
    tcn_layers:       int   = 4
    env_dim:          int   = 32
    stoch_dim:        int   = 32
    ms_out_dim:       int   = 32
    n_scg_layers:     int   = 3
    out_dim:          int   = 1
    lambda_mi:        float = 0.5
    cfm_hidden:       int   = 128
    cfm_time_emb_dim: int   = 16
    chunk_size:       int   = 16384   # SCGMessagePassingLayer 边分块大小


@dataclass
class TrainConfig:
    lr:                  float        = 1e-3
    max_epochs:          int          = 200
    patience:            int          = 20
    lr_decay_factor:     float        = 0.5
    lr_decay_patience:   int          = 10
    grad_clip:           float        = 1.0
    weight_decay:        float        = 1e-5
    save_path:           str          = "best_model.pt"
    seed:                int          = 42
    warmup_epochs:       int          = 5
    gpu_id:              int          = -1
    log_dir:             Optional[str] = "logs"
    log_to_console:      bool         = True

    cfm_n_samples:       int = 50
    cfm_n_samples_test:  int = 200
    cfm_n_steps:         int = 20
    cfm_n_t_samples:     int = 4


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def summary(self) -> str:
        lines = ["=" * 52, "GridCFN Configuration (CFM v4)", "=" * 52]
        for section_name, section in [("Data",  self.data),
                                       ("Model", self.model),
                                       ("Train", self.train)]:
            lines.append(f"\n[{section_name}]")
            for k, v in asdict(section).items():
                lines.append(f"  {k:<24} = {v}")
        lines.append("=" * 52)
        return "\n".join(lines)


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
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.5,
                cfm_hidden=128, cfm_time_emb_dim=16,
                chunk_size=16384,   # Solar E≈34k，不需要分块，16384 等效不分块
            ),
            train=TrainConfig(
                lr=5e-4, max_epochs=200,
                patience=30,
                lr_decay_factor=0.5,
                lr_decay_patience=15,
                seed=42, grad_clip=1.0,
                warmup_epochs=5,
                cfm_n_samples=50,
                cfm_n_samples_test=200,
                cfm_n_steps=20,
                cfm_n_t_samples=4,
            ),
        )

    elif preset == "electricity":
        return Config(
            data=DataConfig(
                dataset="electricity",
                data_path="./data/electricity.txt",
                T_in=168, T_out=1,
                adj_threshold=0.6,   # [v4] 0.7 → 0.6，减少边数降低过平滑
                batch_size=32,
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.05,      # [v4] 0.1 → 0.05，CLUB 修复后降低权重
                cfm_hidden=128, cfm_time_emb_dim=16,
                chunk_size=16384,    # Electricity E≈20k~34k，不需要分块
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200,
                patience=25,
                lr_decay_factor=0.5,
                lr_decay_patience=12,
                seed=42, grad_clip=1.0,
                warmup_epochs=8,     # [v4] 5 → 8，延长 warmup
                cfm_n_samples=50,
                cfm_n_samples_test=200,
                cfm_n_steps=20,
                cfm_n_t_samples=4,
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
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.5,
                cfm_hidden=128, cfm_time_emb_dim=16,
                chunk_size=4096,    # [v4] Weather E可能达数十万，分块处理防 OOM
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200,
                patience=25,
                lr_decay_factor=0.5,
                lr_decay_patience=12,
                seed=42, grad_clip=1.0,
                warmup_epochs=5,
                cfm_n_samples=20,        # Weather 节点多，并行采样显存压力大
                cfm_n_samples_test=100,
                cfm_n_steps=20,
                cfm_n_t_samples=4,
            ),
        )

    else:
        raise ValueError(f"Unknown preset: '{preset}'. Choose: solar, electricity, weather")