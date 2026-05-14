# GridCFN v2

**Grid-based Causal Flow Network** — 面向电力系统时空序列的概率预测框架。

融合因果解耦（Causal Disentanglement）与条件流匹配（Conditional Flow Matching, CFM），在分布层面建模未来不确定性，输出预测均值、方差和校准后的置信区间。支持多步预测。

---

## 文件结构

```
.
├── main.py              # 主入口，命令行解析，训练调度
├── model.py             # 模型定义（AdaptiveGCN / TCN / Disentangler / CLUB / MultiScaleContext / SCGMP / CFMVectorField）
├── train.py             # 训练循环，评估指标（MAE/RMSE/CRPS/PICP/PINAW），早停，温度校准
├── dataset.py           # 数据加载与预处理（Solar / Electricity / Weather2k / SDWPF）
├── config.py            # 所有超参数集中管理
├── plot_results.py      # 7 种出版级可视化图表
├── baselines/           # 现代概率基线（TSFlow / K2VAE）
├── baselines0/          # 经典基线（HA / VAR / DCRNN / STGCN / MTGNN / AGCRN / STID / CSDI）
├── data/                # 数据文件（需自行下载）
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
python main.py --preset solar

# Electricity（预测未来 1 天）
python main.py --preset electricity

# Weather2k（预测未来 1 天）
python main.py --preset weather

# SDWPF 风功率（预测未来 2 小时）
python main.py --preset sdwpf

# 自定义预测步长
python main.py --preset solar --T_out 24
```

训练日志保存到 `result/<dataset>/<timestamp>/`，包含日志文件和最佳模型权重。

---

## 模型架构

```
Input  X : [B, T_in, N, F]
  │
  ├─ Backbone ─────────────────────────────────────
  │   ├─ AdaptiveGCN（自适应图卷积，逐时间步空间建模）
  │   │   └─ A_total = α·softmax(ReLU(E1·E2^T)) + (1-α)·A_fixed
  │   └─ TCN（因果膨胀卷积，时序建模）
  │       └─ 4 层 CausalConv1d + GroupNorm + 残差
  │   → H : [B, T, N, D=64]
  │
  ├─ CausalDisentangler ──────────────────────────
  │   ├─ env_proj  + 注意力池化 → He : [B, N, De=32] （环境/因果表征）
  │   └─ stoch_proj + 注意力池化 → Hs : [B, N, Ds=32] （随机/噪声表征）
  │   → He_seq : [B, T, N, De=32]（保留全序列，供多尺度上下文）
  │
  ├─ CLUB Mutual Information Estimator ────────────
  │   └─ 估计 I(He; Hs)，minimax 训练迫使 He ⊥ Hs
  │
  ├─ MultiScaleContext ────────────────────────────
  │   └─ 3 路膨胀因果卷积（dilations 按数据集配置）→ He' : [B, N, 32]
  │
  ├─ SCGMP（Spatial Causal Gating Message Passing）─
  │   └─ 3 层因果门控图消息传递 → Hs' : [B, N, 32]
  │       门控 = σ(MLP(Hs_dst, Hs_src, He_dst, He_src))
  │
  └─ CFMVectorField ───────────────────────────────
      ├─ 时序位置编码（Temporal PE）→ 感知多步位置
      ├─ 时间嵌入（正弦余弦）→ time_proj
      ├─ 双流 AdaLN：He' 控制 shift，Hs' 控制 scale
      └─ 3 层 MLP + 残差 → v_θ(x_t, t | He', Hs') : [B, N, T_out·F]
```

### 训练

- **CLUB minimax**：每个 batch 两步 — Step1 对 CLUB 梯度上升最大化 MI 估计；Step2 对主网络梯度下降最小化 CFM Loss + λ·clamp(MI, 0, +∞)
- **Warmup**：前 N 个 epoch 跳过 CLUB，之后 MI 权重线性升温
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

### ModelConfig

| 参数 | 默认值 | 说明 |
|---|---|---|
| `gcn_hidden` / `tcn_hidden` | 64 | Backbone 隐藏维度 |
| `env_dim` / `stoch_dim` | 32 | He / Hs 表征维度 |
| `ms_out_dim` | 32 | 多尺度上下文输出维度 |
| `n_scg_layers` | 3 | SCG-MP 消息传递层数 |
| `lambda_mi` | 0.05-0.5 | MI 正则化权重（节点多的数据集适当降低） |
| `cfm_hidden` | 256-512 | CFM 向量场隐藏维度（随 T_out 增大加宽） |
| `ms_dilations` | (1, 7, 30) 或 (1, 12/24, 84) | 多尺度卷积膨胀率 |

### TrainConfig

| 参数 | 默认值 | 说明 |
|---|---|---|
| `lr` | 1e-3 或 5e-4 | 主网络学习率（CLUB lr = 0.5×） |
| `max_epochs` | 200 | 最大训练轮数 |
| `patience` | 20-30 | 早停耐心值 |
| `warmup_epochs` | 3-10 | CLUB 热身期 |
| `grad_clip` | 1.0 | 梯度裁剪 |
| `cfm_n_samples` | 50 | 训练/验证时采样粒子数 |
| `cfm_n_samples_test` | 200 | 测试时采样粒子数 |
| `cfm_n_steps` | 20 | ODE 求解步数 |
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
| `loss_components.png` | Train Loss / CFM / MI / VarLoss 分解曲线 |
| `metrics_curve.png` | Val MAE + Val RMSE 双轴曲线 |
| `prediction_intervals.png` | 4 个高不确定性节点的 95% 预测区间 |
| `reliability_diagram.png` | PICP 校准图（实际覆盖率 vs 名义置信度） |
| `error_distribution.png` | 误差分布直方图 + 误差-不确定性相关图 |

---

## 基线

### 经典基线（`baselines0/`）

HA, VAR, DCRNN, STGCN, MTGNN, AGCRN, STID, CSDI

### 现代概率基线（`baselines/`）

- **TSFlow** (ICLR 2025) — Flow Matching with Gaussian Process Priors
- **K2VAE** (ICML 2025 Spotlight) — Koopman-Kalman Enhanced VAE

```bash
# 运行经典基线
python baselines0/run_baselines.py --model mtgnn --dataset solar

# 运行现代基线
python baselines/run_baselines.py --model tsflow --dataset solar
```

---

## 与论文原版的主要改进

本实现基于 GridCFN 核心思想重构，相比原版的主要变化：

| 模块 | 改进 | 说明 |
|---|---|---|
| Backbone | AdaptiveGCN | 在固定皮尔逊图上叠加可学习自适应邻接矩阵，图结构随训练更新 |
| CausalDisentangler | 注意力时间池化 | He/Hs 通过可学习注意力聚合全序列，不再只取最后一步 |
| CFM | 时序位置编码 + 多步输出 | 向量场输入加入可学习 Temporal PE，直接输出 T_out 步 |
| CFM 采样 | Heun 二阶求解器 | 替换 Euler，相同步数误差更低 |
| CFM 训练 | 分层 t 采样 | [0,1] 均分区间各取一点，覆盖更均匀 |
| CLUB | Tanh + clamp log_var | 直接 clamp(-6, 4) 替代 softplus 嵌套，梯度路径更清晰 |
| 训练稳定性 | MI 升温 + 截断 | MI 权重线性升温，负 MI 不奖励 |
| 图构建 | 预计算一次 | 邻接矩阵归一化在数据加载时完成，不每次 forward 重算 |
