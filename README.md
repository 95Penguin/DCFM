# GridCFN 复现代码

论文：**GridCFN: A Causal Spatio-Temporal Framework for Power Flow Uncertainty Prediction**
（IC2ECS 2025，Zhao et al., Beijing University of Posts and Telecommunications）

---

## 文件结构

```
.
├── model.py      # 模型定义（Backbone / Disentangler / MINE / SCG-MP / Predictor）
├── train.py      # 训练循环、评估指标、早停
├── dataset.py    # 数据加载与预处理（三个真实数据集）
├── config.py     # 所有超参数（修改这里，不需要动其他文件）
├── main.py       # 主入口
├── data/         # 放数据文件（需自行下载）
│   ├── solar_AL.txt
│   ├── electricity.txt
│   └── weather2k.npy
└── logs/         # 训练日志（自动创建）
```

---

## 环境依赖

```bash
# 1. 创建虚拟环境 (Python 3.9-3.11 推荐，PyTorch 兼容性最好)
uv venv

# 2. 激活虚拟环境 (Windows)
.\.venv\Scripts\Activate
# (Mac/Linux)
source .venv/bin/activate

# 3. 安装 PyTorch（CUDA 12.6 对应 cu124，这是目前 PyTorch 官方支持的最近版本）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. 安装其余依赖
uv pip install numpy scipy

# 5. 验证 GPU 可用
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Python >= 3.8，PyTorch >= 1.12。

---

## 数据准备

| 数据集 | 文件名 | 来源 |
|---|---|---|
| Solar-Energy | `solar_AL.txt` | [LSTNet repo](https://github.com/laiguokun/multivariate-time-series-data) |
| Electricity (UCI) | `electricity.txt` | 同上 |
| Weather2k | `weather2k.npy` | Weather2k 论文仓库 |

下载后放入 `./data/` 目录，或在 `config.py` 里修改 `data_path`。

---

## 快速开始

```bash
# Solar-Energy 数据集
python main.py --preset solar

# Electricity 数据集
python main.py --preset electricity

# Weather2k 数据集
python main.py --preset weather
```

训练日志自动保存到 `logs/gridcfn_<dataset>_<timestamp>.log`。

---

## 超参数说明

所有超参数在 `config.py` 里集中管理，主要参数如下：

**数据（DataConfig）**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `T_in` | 168 | 输入历史步数 |
| `T_out` | 1 | 预测步数 |
| `adj_threshold` | 0.7–0.95 | 构建邻接矩阵的相关系数阈值 |
| `batch_size` | 32 | |

**模型（ModelConfig）**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `gcn_hidden` / `tcn_hidden` | 64 | 隐藏层维度 d_hidden |
| `env_dim` / `stoch_dim` | 32 | He 和 Hs 的维度 De / Ds |
| `ms_out_dim` | 32 | 多尺度上下文输出维度 |
| `n_scg_layers` | 3 | SCG-MP 层数 L_SCG |
| `lambda_mi` | 0.5 | MI 正则化权重 λ |

**训练（TrainConfig）**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `lr` | 1e-3 | 主网络学习率（MINE lr = 2×） |
| `max_epochs` | 200 | |
| `patience` | 20 | 早停耐心值（基于 val CRPS） |
| `warmup_epochs` | 5 | 热身期（MI 正则权重为 0） |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |

---

## 模型架构

```
Input X [B, T, N, F]
  └─ Backbone: GCN（逐帧空间建模）+ TCN（因果膨胀卷积时序建模）
       └─ H [B, T, N, D=64]
            └─ CausalDisentangler
                 ├─ env_proj  → He_seq [B, T, N, 32]，He [B, N, 32]
                 └─ stoch_proj（只对最后帧）→ Hs [B, N, 32]
                      │
                      ├─ MINE(He, Hs) → mi_loss（MI 正则化）
                      │
                      ├─ MultiScaleContext(He_seq)
                      │    └─ 4路膨胀卷积（dilation=1,2,4,8）→ H'e [B, N, 32]
                      │
                      └─ SCGMP(Hs, H'e, edge_index)  ×3层
                           └─ 因果门控消息传递 → H's [B, N, 32]
                                └─ Concat([H'e, H's]) → H_final [B, N, 64]
                                     └─ ProbabilisticPredictor
                                          └─ (μ, σ) [B, N, 1]
```

**MINE 的 minimax 训练**：每个 batch 分两步：
- Step1：`mine_optimizer` 对 MINE 参数梯度上升（最大化 MI 估计）
- Step2：`optimizer` 对主网络梯度下降（最小化 NLL + λ·MI），`mi_loss.detach()` 确保梯度不回传到 MINE

---

## 评估指标

| 指标 | 含义 | 越小越好 |
|---|---|---|
| MAE | 平均绝对误差 | ✓ |
| RMSE | 均方根误差 | ✓ |
| CRPS | 连续排名概率分 | ✓ |
| PICP | 95% 预测区间覆盖概率 | 越接近 0.95 越好 |
| PINAW | 归一化平均区间宽度 | ✓ |

论文报告的基准结果（Electricity 数据集）：

| 模型 | MAE | RMSE | CRPS |
|---|---|---|---|
| MTGNN* | 0.088 | 0.155 | 0.082 |
| **GridCFN** | **0.079** | **0.145** | **0.070** |

---

## 相比论文原代码的修复

| 编号 | 位置 | 问题 | 修复 |
|---|---|---|---|
| Fix-A/B | train.py | MINE detach 导致 mine_optimizer 完全无效 | 单次 forward + `mi_loss.detach()` 隔离两步优化 |
| Fix-E | model.py | Hs_seq 全序列投影但只用最后帧，Weather 下浪费 T 倍显存 | stoch_proj 只对最后帧运算 |
| Fix-G | model.py | 自行添加的 degree normalization 与论文不符 | 删除，改用 LayerNorm |
| Fix-H | model.py | TCNBlock 的 permute+LayerNorm+permute 冗余 | 改用 GroupNorm |
| Fix-I | model.py / main.py | normalize_adj 每次 forward 重算 | main.py 预计算一次，forward 直接接收 |
