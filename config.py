"""
config.py
GridCFN 时空网络参数配置文件
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DataConfig:
    dataset:       str           = "solar"
    data_path:     Optional[str] = "./data/solar_AL.txt"
    T_in:          int           = 168
    T_out:         int           = 12     # 多步版默认12步
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
    out_dim:          int   = 1           # 每步预测变量数，预测功率为 1
    lambda_mi:        float = 0.5         # 互信息CLUB损失权重
    cfm_hidden:       int   = 256         # 连续流匹配隐层维度
    cfm_time_emb_dim: int   = 16
    chunk_size:       int   = 16384
    ms_dilations:     tuple = (1, 7, 30)
    rank_r:           int   = 8           # 低秩矩阵隐秩
    lambda_rank:      float = 0.1         
    freq_candidates:  tuple = (12, 24, 48, 96)  
    dropout:          float = 0.1         # 默认正则化 Dropout 


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
    # Monte-Carlo empirical interval sampling (for baselines)
    mc_samples_test:     int   = 200
    mc_chunk_size:       int   = 0
    


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
    if preset == "solar":
        T_out = 12   
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
                lambda_mi=0.01,            
                cfm_hidden=256, cfm_time_emb_dim=16,
                chunk_size=16384,
                ms_dilations=(1, 24, 84),
                rank_r=6,                 
                lambda_rank=0.01,          
                freq_candidates=(12, 24, 48, 96),   
                dropout=0.1,
            ),
            train=TrainConfig(
                lr=5e-4, max_epochs=200,
                patience=30, lr_decay_factor=0.5, lr_decay_patience=15,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=100,  
                cfm_n_samples_test=100, 
                cfm_n_steps=15,  
                cfm_n_t_samples=4,
            ),
        )

    elif preset == "electricity":
        T_out = 12   
        return Config(
            data=DataConfig(
                dataset="electricity",
                data_path="./data/electricity.txt",
                T_in=168, T_out=T_out,
                adj_threshold=0.6,
                batch_size=16,   
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.01,           
                cfm_hidden=384, cfm_time_emb_dim=16,
                chunk_size=16384,
                ms_dilations=(1, 12, 84),
                rank_r=12,                
                lambda_rank=0.01,          
                freq_candidates=(6, 12, 24, 48),    
                dropout=0.1,
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200,
                patience=25, lr_decay_factor=0.5, lr_decay_patience=12,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=100,  
                cfm_n_samples_test=100, 
                cfm_n_steps=15,      
                cfm_n_t_samples=4,
            ),
        )

    elif preset == "weather":
        T_out = 12   
        return Config(
            data=DataConfig(
                dataset="weather",
                data_path="./data/weather2k.npy",
                T_in=168, T_out=T_out,
                adj_threshold=0.8,
                batch_size=2,    
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
                dropout=0.1,
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200,
                patience=25, lr_decay_factor=0.5, lr_decay_patience=12,
                seed=42, grad_clip=1.0, warmup_epochs=3,
                cfm_n_samples=10,        
                cfm_n_samples_test=50,
                cfm_n_steps=20, cfm_n_t_samples=4,
            ),
        )

    # elif preset == "sdwpf":
    #     T_out = 12   
    #     return Config(
    #         data=DataConfig(
    #             dataset="sdwpf",
    #             data_path="./data/sdwpf_245days_v1.csv",
    #             T_in=168, T_out=T_out,
    #             adj_threshold=0.94,       
    #             batch_size=16,            
    #         ),
    #         model=ModelConfig(
    #             in_dim=4,                 
    #             gcn_hidden=96,            
    #             gcn_layers=2,
    #             tcn_hidden=96,            
    #             tcn_layers=4,
    #             env_dim=48,               
    #             stoch_dim=48,             
    #             ms_out_dim=48,
    #             n_scg_layers=3, 
    #             out_dim=1,                
    #             lambda_mi=0.001,          
    #             cfm_hidden=320,           
    #             cfm_time_emb_dim=16,
    #             chunk_size=16384,
    #             ms_dilations=(1, 6, 24),  
    #             rank_r=8,                 
    #             lambda_rank=0.01,          
    #             freq_candidates=(12, 24, 48, 96),   
    #             dropout=0.20,             
    #         ),
    #         train=TrainConfig(
    #             lr=3e-4, max_epochs=200,  
    #             patience=30, lr_decay_factor=0.5, lr_decay_patience=15,
    #             seed=42, grad_clip=1.0,
    #             weight_decay=2e-4,        
    #             warmup_epochs=15,         
    #             cfm_n_samples=100,  
    #             cfm_n_samples_test=100, 
    #             cfm_n_steps=15,      
    #             cfm_n_t_samples=4,
    #             cfm_sigma_min=0.01,
    #             cfm_x0_scale=1.0,
    #         ),
    #     )

    elif preset == "sdwpf":
        T_out = 12   
        return Config(
            data=DataConfig(
                dataset="sdwpf",
                data_path="./data/sdwpf_245days_v1.csv",
                T_in=168, T_out=T_out,
                adj_threshold=0.88,       # 稍微放宽阈值，引入更多空间风机关联
                batch_size=16,            
            ),
            model=ModelConfig(
                in_dim=4,                 
                gcn_hidden=64,            # 降低隐藏层维度（96->64），通过剪枝减少模型容量，强力防过拟合
                gcn_layers=2,
                tcn_hidden=64,            # 降低隐藏层维度（96->64）
                tcn_layers=4,
                env_dim=32,               
                stoch_dim=32,             
                ms_out_dim=32,
                n_scg_layers=3, 
                out_dim=1,                
                lambda_mi=0.01,           
                cfm_hidden=256,           
                cfm_time_emb_dim=16,
                chunk_size=16384,
                ms_dilations=(1, 12, 48),  # 10min粒度: 1步/12步(2h)/48步(8h)
                                           # 感受野=3×48=144 < T_in=168，全部有效
                                           # 原(1,24,84)中84对应感受野252>168，末层被截断
                rank_r=6,                 
                lambda_rank=0.01,          
                freq_candidates=(12, 24, 48, 96),   
                dropout=0.25,             # 提升至 0.25，全面阻断噪声过拟合
            ),
            train=TrainConfig(
                lr=1.5e-4, max_epochs=200, # 降低学习率，平滑更新
                patience=30, lr_decay_factor=0.5, lr_decay_patience=15,
                seed=42, grad_clip=1.0,
                weight_decay=1e-3,        # 强力 L2 正则化约束 (1e-5 -> 1e-3)
                warmup_epochs=5,         
                cfm_n_samples=100,  
                cfm_n_samples_test=100, 
                cfm_n_steps=15,      
                cfm_n_t_samples=4,
                cfm_sigma_min=0.01,
                cfm_x0_scale=1.0,
                tsdiff_val_max_batches=2,
                tsdiff_test_max_batches=10,
            ),
        )

    elif preset == "pjm":
        T_out = 12   # PJM 1h 粒度下，默认预测未来24小时
        return Config(
            data=DataConfig(
                dataset="pjm",
                data_path="./data/archive",  # 请确保您的 Kaggle 解压 CSV 存放在该路径
                T_in=168, T_out=T_out,
                adj_threshold=0.6,          # 稍微放宽以确保图连接通畅
                batch_size=32,
            ),
            model=ModelConfig(
                in_dim=1, gcn_hidden=64, gcn_layers=2,
                tcn_hidden=64, tcn_layers=4,
                env_dim=32, stoch_dim=32, ms_out_dim=32,
                n_scg_layers=3, out_dim=1,
                lambda_mi=0.01,            
                cfm_hidden=256, cfm_time_emb_dim=16,
                chunk_size=16384,
                ms_dilations=(1, 12, 24),  # 小时级数据更关注12h、24h的周期特征
                rank_r=4,                  # PJM 分区约有11个独立节点，设秩r=4完全足够
                lambda_rank=0.01,          
                freq_candidates=(12, 24, 48),  # 对应小时级多尺度周期（半天、一天、两天）
                dropout=0.1,
            ),
            train=TrainConfig(
                lr=1e-3, max_epochs=200,
                patience=30, lr_decay_factor=0.5, lr_decay_patience=12,
                seed=42, grad_clip=1.0, warmup_epochs=5,
                cfm_n_samples=100,  
                cfm_n_samples_test=100, 
                cfm_n_steps=15,  
                cfm_n_t_samples=4,
            ),
        )
    else:
        raise ValueError(
            f"Unknown preset: '{preset}'. Choose: solar, electricity, weather, sdwpf, pjm"
        )