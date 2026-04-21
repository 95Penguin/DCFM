"""
GridCFN Configuration（CFM 版 v3）

[v3 改动]
  · cfm_n_samples 默认从 10 → 50（修复验证 CRPS 噪声大、PICP 低估问题）
  · cfm_n_samples_test 新增（测试时用更多粒子，默认 200）
  · Solar patience: 20 → 30（防止在噪声平台上过早停止）
  · Electricity lambda_mi: 0.5 → 0.1（MI 出现负值说明 0.5 偏强）
  · 各数据集 lr_decay_patience 微调

关于 cfm_n_samples=50 的速度影响：
  · Solar (137 节点, batch=32): 验证约 30s/epoch，可接受
  · Electricity (321 节点, batch=32): 验证约 70s/epoch，可接受
  · Weather (1866 节点, batch=4): 验证约 120s/epoch，如果太慢改回 20
    （Weather 节点多但 batch 小，并行度低，显存压力也大）

使用方法不变：
  python main.py --preset solar
  python main.py --preset electricity
  python main.py --preset weather

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

    # CFM 推断参数（v3 更新）
    cfm_n_samples:       int = 50    # 验证时采样粒子数（v3: 50，解决 PICP 低估）
    cfm_n_samples_test:  int = 200   # 测试时采样粒子数（更精确）
    cfm_n_steps:         int = 20    # ODE 步数
    cfm_n_t_samples:     int = 4     # 训练时每 batch 采 t 次数


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def summary(self) -> str:
        lines = ["=" * 52, "GridCFN Configuration (CFM v3)", "=" * 52]
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
                lambda_mi=0.5,              # Solar MI 正常，保持 0.5
                cfm_hidden=128, cfm_time_emb_dim=16,
            ),
            train=TrainConfig(
                lr=5e-4, max_epochs=200,
                patience=30,               # v3: Solar 延长到 30，防止噪声早停
                lr_decay_factor=0.5,
                lr_decay_patience=15,      # v3: 跟着 patience 调大
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=50,          # v3: 50 粒子验证
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
                adj_threshold=0.7,
                batch_size=32,
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.1,             # v3: 从 0.5 降到 0.1
                                           # 原因：Electricity 的 MI 频繁出现负值，
                                           # 说明 0.5 过强干扰了 CFM 主损失
                cfm_hidden=128, cfm_time_emb_dim=16,
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200,
                patience=25,               # v3: 适当延长
                lr_decay_factor=0.5,
                lr_decay_patience=12,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=50,
                cfm_n_samples_test=200,
                cfm_n_steps=20,
                cfm_n_t_samples=4,
            ),
        )

    elif preset == "weather":
        # Weather: 1866 节点，batch=4，显存压力大
        # 并行采样时 B*S*N = 4*50*1866 = 373200，可能 OOM
        # 如果 OOM 把 cfm_n_samples 降到 20
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
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200,
                patience=25,
                lr_decay_factor=0.5,
                lr_decay_patience=12,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                # Weather 节点多，并行采样显存占用大，保守用 20
                # 如果 OOM 降到 10 并在 model.sample() 里改回串行
                cfm_n_samples=20,
                cfm_n_samples_test=100,
                cfm_n_steps=20,
                cfm_n_t_samples=4,
            ),
        )

    else:
        raise ValueError(f"Unknown preset: '{preset}'. Choose: solar, electricity, weather")