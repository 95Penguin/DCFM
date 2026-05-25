# GridCFN v2

**Grid-based Causal Flow Network** — 面向电力系统时空序列的概率预测框架。

融合双轨因果解耦（Dual-track Causal Disentanglement）与条件流匹配（Conditional Flow Matching, CFM），在分布层面建模未来不确定性，输出预测均值、方差和校准后的置信区间。支持多步预测。

---

## 文件结构

```
.
├── main.py              # 主入口，命令行解析，训练调度
├── model.py             # 模型定义（FrequencyDecomposer / LowRankGCN / SparseGCN /
│                        #          DualTrackBackbone / CLUBEstimator /
│                        #          MultiScaleContext / SCGMP / CFMVectorField / GridCFN）
├── train.py             # 训练循环，评估指标（MAE/RMSE/CRPS/PICP/PINAW），早停，温度校准
├── dataset.py           # 数据加载与预处理（Solar / Electricity / Weather2k / SDWPF）
├── config.py            # 所有超参数集中管理
├── plot_results.py      # 7 种出版级可视化图表
├── baselines/           # 所有对比算法（经典 + 现代概率基线统一放置）
│   ├── __init__.py
│   ├── run_baselines.py # 统一入口，--model 指定算法，--dataset 指定数据集
│   ├── utils.py         # 共享工具函数
│   ├── ha.py            # Historical Average
│   ├── var_model.py     # Vector AutoRegression
│   ├── dcrnn.py         # DCRNN
│   ├── stgcn.py         # STGCN
│   ├── mtgnn.py         # MTGNN
│   ├── agcrn.py         # AGCRN
│   ├── stid.py          # STID
│   ├── csdi.py          # CSDI
│   ├── patchtst.py      # PatchTST
│   ├── tsdiff.py        # TSDiff
│   ├── tsflow.py        # TSFlow (ICLR 2025)
│   └── k2vae.py         # K2VAE (ICML 2025 Spotlight)
├── data/                # 数据文件（需自行下载）
├── logs/                # 训练日志（自动创建）
└── result/              # 实验结果（自动创建）
```

---

## 环境依赖

```bash
# 1. 创建虚拟环境（Python 3.9-3.11 推荐）
uv venv

# 2. 激活虚拟环境（Windows）
.\.venv\Scripts\Activate
# (Mac/Linux)
source .venv/bin/activate

# 3. 安装 PyTorch（CUDA 12.6）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. 安装其余依赖
uv pip install numpy scipy pandas matplotlib

# 5. 验证 GPU
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 数据准备

| 数据集 | 文件 | 节点数 | 分辨率 | 来源 |
|---|---|---|---|---|
| Solar-Energy | `solar_AL.txt` | 137 PV 电站 | 10 min | [LSTNet repo](https://github.com/laiguokun/multivariate-time-series-data) |
| Electricity | `electricity.txt` | 321 用电客户 | 1 hour | 同上 |
| Weather2k | `weather2k.npy` | 1866 气象站 | 1 hour | Weather2k 论文仓库 |
| SDWPF | `sdwpf_245days_v1.csv` | 134 风机 | 10 min | [Figshare](https://figshare.com/articles/dataset/SDWPF_dataset/24798654) |

下载后放入 `./data/` 目录。

---

## 快速开始

```bash
# Solar-Energy（预测未来 2 小时）
uv run main.py --preset solar

# Electricity（预测未来 1 天）
uv run main.py --preset electricity

# Weather2k（预测未来 1 天）
uv run main.py --preset weather

# SDWPF 风功率（预测未来 2 小时）
uv run main.py --preset sdwpf

# 自定义预测步长
uv run main.py --preset solar --T_out 24
```

训练日志保存到 `result/<dataset>/<timestamp>/`，包含日志文件和最佳模型权重。

---

## 模型架构（DMSD 版）

本版本采用 **双轨多尺度解耦（Dual-track Multi-Scale Disentanglement, DMSD）** 架构，将因果解耦从后端线性层（原 CausalDisentangler）提前到前端输入，以更显式的方式分离趋势与残差信息。

```
Input  X : [B, T_in, N, F]
  │
  ├─ FrequencyDecomposer ──────────────────────────────────
  │   可学习多尺度移动平均，权重由全局均值动态生成
  │   X_low  = Σ_k w_k · MA_k(X)        （趋势支路）
  │   X_high = X − X_low                 （残差支路）
  │
  ├─ 环境支路（低频趋势 → 因果/全局表征）─────────────────
  │   X_low → LowRankGCN（A = softmax(U·Uᵀ/√r)，rank-r 邻接）
  │          → TCN_e（4 层因果膨胀卷积）
  │          → proj_tcn_e
  │          → He_seq : [B, T, N, env_dim]
  │          → AttentionPool → He : [B, N, env_dim]
  │
  ├─ 因果支路（高频残差 → 随机/局部表征）─────────────────
  │   X_high → SparseGCN（可微稀疏化，软阈值 sigmoid）
  │           → TCN_s（4 层因果膨胀卷积）
  │           → proj_tcn_s
  │           → AttentionPool → Hs : [B, N, stoch_dim]
  │
  ├─ CLUB（双实例互信息约束）──────────────────────────────
  │   club_e: MI(He, X_low_pooled)   — 环境表征不含残差信息
  │   club_s: MI(Hs, X_high_pooled)  — 随机表征不含趋势信息
  │   minimax 训练迫使 He ⊥ X_high，Hs ⊥ X_low
  │
  ├─ MultiScaleContext ────────────────────────────────────
  │   He_seq → 3 路膨胀因果卷积（dilations 按数据集配置）
  │          → He_prime : [B, N, ms_out_dim]
  │
  ├─ SCGMP（Spatial Causal Gating Message Passing）────────
  │   Hs, He → 3 层因果门控图消息传递 → Hs_prime : [B, N, stoch_dim]
  │            门控 = σ(MLP(Hs_dst, Hs_src, He_dst, He_src))
  │
  └─ CFMVectorField ───────────────────────────────────────
      ├─ 时序位置编码（Temporal PE）→ 感知多步位置
      ├─ 时间嵌入（正弦余弦）→ time_proj
      ├─ 双流 AdaLN：He_prime 控制 shift，Hs_prime 控制 scale
      └─ 3 层 MLP + 残差 → v_θ(x_t, t | He_prime, Hs_prime) : [B, N, T_out·F]
```

### 与原版的关键架构差异

| 模块 | 原版（单轨） | DMSD 版（双轨） |
|---|---|---|
| 输入分解 | 无 | FrequencyDecomposer：可学习 MA 分解趋势/残差 |
| 图卷积 | AdaptiveGCN（固定皮尔逊图 + 可学习邻接叠加） | 双轨：LowRankGCN（rank-r 邻接）+ SparseGCN（可微稀疏化） |
| 解耦位置 | 后端 CausalDisentangler（线性层 + 注意力池化） | 前端输入分流，解耦更显式 |
| CLUB 约束 | 单实例 MI(He, Hs) | 双实例：MI(He, X_low) + MI(Hs, X_high) |
| 秩正则 | 无 | L_rank = −log det(UᵀU + εI)，防止低秩退化 |

### 训练流程

- **DMSD CLUB minimax**：每个 batch 两步 — Step1 对 `club_e` / `club_s` 内循环梯度上升；Step2 对主网络梯度下降最小化 CFM + λ_rank·L_rank + λ_club·clamp(MI, 0, +∞)
- **Warmup**：前 N 个 epoch 跳过 CLUB，之后 MI 权重线性升温（3 epoch ramp）
- **CFM 损失**：OT-CFM，分层随机 t 采样（n_t_samples 个区间各取一点）
- **早停**：基于验证集 CRPS，patience 可配置

### 推理

- **Heun 二阶 ODE 求解器**：预测-校正两步，同等步数下误差低于 Euler
- **Temperature Calibration**：验证集上网格搜索 [0.5, 5.0] 最优温度，校准预测区间

---

## 评估指标

| 指标 | 含义 | 方向 |
|---|---|---|
| MAE | 预测均值与真实值的平均绝对误差 | 越小越好 |
| RMSE | 预测均值与真实值的均方根误差 | 越小越好 |
| CRPS | 连续排名概率得分，衡量整个预测分布的质量 | 越小越好 |
| PICP | 95% 预测区间覆盖率（越接近 0.95 越好） | → 0.95 |
| PINAW | 归一化平均区间宽度 | 越小越好 |

评估在反归一化域（真实量纲）进行，同时汇报平均指标和各预测步 (h1, h2, ...) 的逐步指标。

---

## 超参数说明

所有超参数在 `config.py` 中集中管理，按数据集聚类为 preset。

### DataConfig

| 参数 | 默认值 | 说明 |
|---|---|---|
| `T_in` | 168 | 输入历史步数 |
| `T_out` | 12 | 预测步数（12=2h @10min, 24=1day @1h） |
| `adj_threshold` | 0.6-0.95 | 皮尔逊相关阈值，按数据集调整 |
| `batch_size` | 2-32 | 按节点数自适应调整 |

### ModelConfig（DMSD 新增参数）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `gcn_hidden` / `tcn_hidden` | 64 | 双轨 Backbone 隐藏维度 |
| `env_dim` / `stoch_dim` | 32 | He / Hs 表征维度 |
| `ms_out_dim` | 32 | 多尺度上下文输出维度 |
| `n_scg_layers` | 3 | SCG-MP 消息传递层数 |
| `lambda_mi` | 0.05-0.5 | CLUB MI 约束权重（`lambda_club`，节点多时适当降低） |
| `cfm_hidden` | 256-512 | CFM 向量场隐藏维度（随 T_out 增大加宽） |
| `ms_dilations` | (1,24,84) 等 | 多尺度卷积膨胀率，按数据集粒度配置 |
| `rank_r` | 6-12 | LowRankGCN 的秩（Solar/SDWPF=6，Electricity=12） |
| `lambda_rank` | 0.01 | 秩正则权重 L_rank |
| `freq_candidates` | (12,24,48,96) | FrequencyDecomposer 候选 MA 窗口（按数据粒度配置） |

### TrainConfig

| 参数 | 默认值 | 说明 |
|---|---|---|
| `lr` | 1e-3 或 5e-4 | 主网络学习率（CLUB lr = 0.5×） |
| `max_epochs` | 200 | 最大训练轮数 |
| `patience` | 20-30 | 早停耐心值 |
| `warmup_epochs` | 3-5 | CLUB 热身期 |
| `grad_clip` | 1.0 | 梯度裁剪 |
| `cfm_n_samples` | 100 | 训练/验证时采样粒子数 |
| `cfm_n_samples_test` | 100 | 测试时采样粒子数 |
| `cfm_n_steps` | 15 | ODE 求解步数 |
| `cfm_n_t_samples` | 4 | CFM 分层 t 采样数 |

---

## 结果可视化

```python
from plot_results import plot_all
plot_all(history, result_dir, dataset_name="solar")
```

生成 7 张图表：

| 图 | 内容 |
|---|---|
| `convergence.png` | Val CRPS 收敛曲线 + 最佳 epoch 标注 |
| `loss_components.png` | Train Loss / CFM / MI / Rank 分解曲线 |
| `metrics_curve.png` | Val MAE + Val RMSE 双轴曲线 |
| `prediction_intervals.png` | 4 个高不确定性节点的 95% 预测区间 |
| `reliability_diagram.png` | PICP 校准图（实际覆盖率 vs 名义置信度） |
| `error_distribution.png` | 误差分布直方图 + 误差-不确定性相关图 |

---

## 基线

所有对比算法统一放在 `baselines/` 目录，通过 `run_baselines.py` 的 `--model` 参数选择。

| 类别 | 算法 | 文件 |
|---|---|---|
| 统计基线 | Historical Average | `ha.py` |
| 统计基线 | Vector AutoRegression | `var_model.py` |
| 图神经网络 | DCRNN | `dcrnn.py` |
| 图神经网络 | STGCN | `stgcn.py` |
| 图神经网络 | MTGNN | `mtgnn.py` |
| 图神经网络 | AGCRN | `agcrn.py` |
| Transformer | STID | `stid.py` |
| Transformer | PatchTST | `patchtst.py` |
| 概率生成 | CSDI | `csdi.py` |
| 概率生成 | TSDiff | `tsdiff.py` |
| 概率生成 | TSFlow (ICLR 2025) | `tsflow.py` |
| 概率生成 | K2VAE (ICML 2025 Spotlight) | `k2vae.py` |

```bash
# 运行任意基线，--model 指定算法名，--dataset 指定数据集
uv run baselines/run_baselines.py --model ha --dataset solar
uv run baselines/run_baselines.py --model mtgnn --dataset electricity
uv run baselines/run_baselines.py --model tsflow --dataset solar
uv run baselines/run_baselines.py --model k2vae --dataset sdwpf
```

---

## 主要改进

| 模块 | 改进 | 说明 |
|---|---|---|
| 输入层 | FrequencyDecomposer | 可学习多尺度移动平均，自适应分解趋势与残差 |
| 图卷积（环境支路） | LowRankGCN | A = softmax(U·Uᵀ/√r)，rank-r 参数化，只学习全局同步集群 |
| 图卷积（因果支路） | SparseGCN | 可微稀疏化（软阈值 sigmoid）替代硬 Top-K，梯度稳定 |
| 解耦架构 | DualTrackBackbone | 解耦职责从后端 Disentangler 前移至输入分流，语义更清晰 |
| CLUB 约束 | 双实例 | club_e + club_s 分别约束两路表征，解耦目标更明确 |
| 秩正则 | L_rank | −log det(UᵀU + εI)，防止 LowRankGCN 退化为秩-1 |
| CFM 采样 | Heun 二阶求解器 | 替换 Euler，相同步数误差更低 |
| CFM 训练 | 分层 t 采样 | [0,1] 均分区间各取一点，覆盖更均匀 |
| 训练稳定性 | MI 升温 + 截断 | MI 权重线性升温，负 MI 不奖励 |