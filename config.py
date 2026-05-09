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
    # SCGMessagePassingLayer 边分块大小，防止大图一次性创建 [B,E,dim] 张量 OOM
    # Weather(E~200k) 用 4096，Solar/Electricity/SDWPF(E~10k-34k) 用 16384（等效不分块）
    chunk_size:       int   = 16384
    # MultiScaleContext dilation 按数据集时间分辨率配置：
    #   Solar/SDWPF (10min): (1, 24, 84)  → 10min / 4h / 14h
    #   Elec/Weather (1h):   (1, 12, 84)  → 1h / 12h / 3.5d
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
    warmup_epochs:       int          = 5   # warmup 期间跳过 CLUB，让主网络先稳定
    gpu_id:              int          = -1
    log_dir:             Optional[str] = "logs"
    log_to_console:      bool         = True

    cfm_n_samples:       int   = 50     # 验证时并行采样粒子数
    cfm_n_samples_test:  int   = 200    # 测试时并行采样粒子数
    cfm_n_steps:         int   = 20     # Euler ODE 积分步数
    cfm_n_t_samples:     int   = 4      # 每 batch 随机采样的时间点数
    cfm_sigma_min:       float = 0.01   # OT-CFM sigma_min，防止条件分布退化为 delta
    cfm_x0_scale:        float = 1.0    # 初始噪声幅值缩放，PICP 偏低时可调大


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def summary(self) -> str:
        lines = ["=" * 52, "GridCFN Configuration", "=" * 52]
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
                chunk_size=16384,
                ms_dilations=(1, 24, 84),   # Solar 10min分辨率
            ),
            train=TrainConfig(
                lr=5e-4, max_epochs=200,
                patience=30, lr_decay_factor=0.5, lr_decay_patience=15,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=50, cfm_n_samples_test=200,
                cfm_n_steps=20, cfm_n_t_samples=4,
            ),
        )

    elif preset == "electricity":
        return Config(
            data=DataConfig(
                dataset="electricity",
                data_path="./data/electricity.txt",
                T_in=168, T_out=1,
                adj_threshold=0.6,   # 降低阈值减少边数，缓解 GCN 过平滑
                batch_size=32,
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.05,      # Electricity 分布较规范，降低 MI 正则权重
                cfm_hidden=128, cfm_time_emb_dim=16,
                chunk_size=16384,
                ms_dilations=(1, 12, 84),   # Electricity 1h分辨率
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200,
                patience=25, lr_decay_factor=0.5, lr_decay_patience=12,
                seed=42, grad_clip=1.0, warmup_epochs=8,
                cfm_n_samples=50, cfm_n_samples_test=200,
                cfm_n_steps=20, cfm_n_t_samples=4,
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
                chunk_size=4096,    # Weather E~200k，分块处理防 OOM
                ms_dilations=(1, 12, 84),   # Weather 1h分辨率
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200,
                patience=25, lr_decay_factor=0.5, lr_decay_patience=12,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=20,        # Weather 节点多，限制并行采样数
                cfm_n_samples_test=100,
                cfm_n_steps=20, cfm_n_t_samples=4,
            ),
        )

    elif preset == "sdwpf":
        return Config(
            data=DataConfig(
                dataset="sdwpf",
                # KDD版（约245天，~35280时间步）或完整版均可
                # 推荐路径：./data/sdwpf_245days_v1.csv
                data_path="./data/sdwpf_245days_v1.csv",
                T_in=168,        # 与 Solar 一致（168步×10min = 28小时历史窗口）
                T_out=1,
                adj_threshold=0.88,  # 同场风机相关性高，0.88 保留适量边防过平滑
                batch_size=32,
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.5,
                cfm_hidden=128, cfm_time_emb_dim=16,
                chunk_size=16384,   # N=134，边数 ~8k-15k，不需要分块
                # 与 Solar 完全相同的 dilation（同为 10min 分辨率）
                ms_dilations=(1, 24, 84),   # 10min / 4h / 14h
            ),
            train=TrainConfig(
                lr=5e-4, max_epochs=200,
                patience=30, lr_decay_factor=0.5, lr_decay_patience=15,
                seed=42, grad_clip=1.0,
                warmup_epochs=5,
                cfm_n_samples=50, cfm_n_samples_test=200,
                cfm_n_steps=20, cfm_n_t_samples=4,
                # SDWPF 风电功率有较强非平稳性（夜间低、白天高、季节差异大），
                # sigma_min 和 x0_scale 与 Solar 保持一致先跑，若 PICP 偏低再调大
                cfm_sigma_min=0.01,
                cfm_x0_scale=1.0,
            ),
        )

    else:
        raise ValueError(
            f"Unknown preset: '{preset}'. Choose: solar, electricity, weather, sdwpf"
        )