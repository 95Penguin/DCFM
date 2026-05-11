from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DataConfig:
    dataset:       str           = "solar"
    data_path:     Optional[str] = "./data/solar_AL.txt"
    T_in:          int           = 168
    T_out:         int           = 12     # ← 多步版默认 12，可设 12~48
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
    out_dim:          int   = 1           # 每步每节点的特征维度，通常保持 1
    lambda_mi:        float = 0.5
    # CFM hidden 在多步时适当加宽，因为输出维度 = T_out * out_dim
    cfm_hidden:       int   = 256         # ← 多步版加宽（原 128）
    cfm_time_emb_dim: int   = 16
    chunk_size:       int   = 16384
    ms_dilations:     tuple = (1, 7, 30)


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
    warmup_epochs:       int          = 3
    gpu_id:              int          = -1
    log_dir:             Optional[str] = "logs"
    log_to_console:      bool         = True

    cfm_n_samples:       int   = 50
    cfm_n_samples_test:  int   = 200
    cfm_n_steps:         int   = 20
    cfm_n_t_samples:     int   = 4
    cfm_sigma_min:       float = 0.01
    cfm_x0_scale:        float = 1.0


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def summary(self) -> str:
        lines = ["=" * 52, "GridCFN Configuration (Multi-Step)", "=" * 52]
        for section_name, section in [("Data",  self.data),
                                       ("Model", self.model),
                                       ("Train", self.train)]:
            lines.append(f"\n[{section_name}]")
            for k, v in asdict(section).items():
                lines.append(f"  {k:<24} = {v}")
        lines.append("=" * 52)
        return "\n".join(lines)


def get_config(preset: str = "solar") -> Config:
    """
    多步预测 preset。

    T_out 建议值（中期预测）：
      Solar / SDWPF (10min) : T_out=12 → 2h；T_out=24 → 4h；T_out=48 → 8h
      Electricity / Weather (1h): T_out=12 → 12h；T_out=24 → 1day；T_out=48 → 2days

    cfm_hidden 随 T_out 增大应适当加宽（T_out*feat_dim 增大，需要更大容量）：
      T_out=12  → cfm_hidden=256
      T_out=24  → cfm_hidden=384
      T_out=48  → cfm_hidden=512
    """

    if preset == "solar":
        T_out = 12   # Solar 10min，T_out=12 → 预测未来 2h
        return Config(
            data=DataConfig(
                dataset="solar",
                data_path="./data/solar_AL.txt",
                T_in=168, T_out=T_out,
                adj_threshold=0.95,
                batch_size=32,
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.5,
                cfm_hidden=256, cfm_time_emb_dim=16,
                chunk_size=16384,
                ms_dilations=(1, 24, 84),
            ),
            train=TrainConfig(
                lr=5e-4, max_epochs=200,
                patience=30, lr_decay_factor=0.5, lr_decay_patience=15,
                seed=42, grad_clip=1.0, warmup_epochs=3,
                cfm_n_samples=50, cfm_n_samples_test=200,
                cfm_n_steps=20, cfm_n_t_samples=4,
            ),
        )

    elif preset == "electricity":
        T_out = 24   # Electricity 1h，T_out=24 → 预测未来 1 天
        return Config(
            data=DataConfig(
                dataset="electricity",
                data_path="./data/electricity.txt",
                T_in=168, T_out=T_out,
                adj_threshold=0.6,
                batch_size=16,   # 节点多 + T_out 大，适当减小 batch
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.05,
                cfm_hidden=384, cfm_time_emb_dim=16,
                chunk_size=16384,
                ms_dilations=(1, 12, 84),
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200,
                patience=25, lr_decay_factor=0.5, lr_decay_patience=12,
                seed=42, grad_clip=1.0, warmup_epochs=3,
                cfm_n_samples=50, cfm_n_samples_test=200,
                cfm_n_steps=20, cfm_n_t_samples=4,
            ),
        )

    elif preset == "weather":
        T_out = 24   # Weather 1h，T_out=24 → 预测未来 1 天
        return Config(
            data=DataConfig(
                dataset="weather",
                data_path="./data/weather2k.npy",
                T_in=168, T_out=T_out,
                adj_threshold=0.8,
                batch_size=2,    # Weather 节点极多，减小 batch
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.5,
                cfm_hidden=384, cfm_time_emb_dim=16,
                chunk_size=4096,
                ms_dilations=(1, 12, 84),
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200,
                patience=25, lr_decay_factor=0.5, lr_decay_patience=12,
                seed=42, grad_clip=1.0, warmup_epochs=3,
                cfm_n_samples=10,        # Weather 节点极多，严格限制并行采样数
                cfm_n_samples_test=50,
                cfm_n_steps=20, cfm_n_t_samples=4,
            ),
        )

    elif preset == "sdwpf":
        T_out = 12   # SDWPF 10min，T_out=12 → 预测未来 2h
        return Config(
            data=DataConfig(
                dataset="sdwpf",
                data_path="./data/sdwpf_245days_v1.csv",
                T_in=168, T_out=T_out,
                adj_threshold=0.88,
                batch_size=32,
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.05,  #0.5->0.05
                cfm_hidden=256, cfm_time_emb_dim=16,
                chunk_size=16384,
                ms_dilations=(1, 24, 84),
            ),
            train=TrainConfig(
                lr=5e-4, max_epochs=200,
                patience=30, lr_decay_factor=0.5, lr_decay_patience=15,
                seed=42, grad_clip=1.0,
                warmup_epochs=10, #3->10
                cfm_n_samples=50, cfm_n_samples_test=200,
                cfm_n_steps=20, cfm_n_t_samples=4,
                cfm_sigma_min=0.01,
                cfm_x0_scale=1.0,
            ),
        )

    else:
        raise ValueError(
            f"Unknown preset: '{preset}'. Choose: solar, electricity, weather, sdwpf"
        )