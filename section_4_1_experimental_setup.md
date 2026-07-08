# 4.1 实验设置

## 4.1.1 数据集

实验使用四个公开的真实时空数据集：

- **Solar-AL**（Lai et al., 2018）：阿拉巴马州 137 个光伏电站的功率数据，10 分钟采样频率，共 52,560 个时间步。
- **Electricity**（Lai et al., 2018）：321 个用户的每小时用电量数据，共 26,304 个时间步。
- **SDWPF**（Zhou et al., 2022）：百度 KDD Cup 2022 风功率数据集，包含 134 台风机、245 天的 10 分钟级运行数据。输入特征为实际功率、风速、环境温度和桨距角。
- **PJM Hourly Load**：美国 PJM 电力市场 11 个定价分区的逐时负荷数据，时间跨度为 2013 年 1 月至 2018 年 1 月。

所有数据集按 7:1:2 的时间顺序划分为训练集、验证集和测试集。采用全局 Z-score 归一化，统计量来自训练集，指标在反归一化后的原始量纲上计算。

## 4.1.2 评估指标

采用六个指标全面评估点预测精度和概率预测质量：

- **MAE** 和 **RMSE**：衡量预测均值与真实值之间的误差。
- **MAPE**：百分比误差，提供无量纲的精度度量。
- **CRPS**（Continuous Ranked Probability Score）：评估整个预测分布与真实值之间的差异，通过 Monte Carlo 采样计算经验 CRPS。值越低表示分布预测越好。
- **PICP**（Prediction Interval Coverage Probability）：95% 预测区间的经验覆盖率，理想值为 0.95。
- **PINAW**（Prediction Interval Normalized Average Width）：预测区间的归一化平均宽度，在覆盖率相当的前提下越窄越好。

对于确定性基线（HA、VAR、DCRNN、STGCN、MTGNN、AGCRN），仅报告 MAE、RMSE 和 MAPE；概率性基线（STID、CSDI、TSDiff、TSFlow、K2VAE）及 GridCFN 报告全部六项指标。

## 4.1.3 基线方法

选取 12 种基线方法，覆盖统计方法、图神经网络、Transformer/MLP 和概率生成模型四大类：

**统计方法：**
- **HA**（Historical Average）：以输入窗口均值作为所有预测步的预测值。
- **VAR**（Vector AutoRegression）（Lütkepohl, 2005）：逐节点拟合线性自回归模型。

**图神经网络：**
- **DCRNN**（Li et al., 2018）：双向扩散图卷积 + GRU 循环的时空预测模型。
- **STGCN**（Yu et al., 2018）：Chebyshev 图卷积与门控时序卷积结合的经典模型。
- **MTGNN**（Wu et al., 2020）：联合学习隐式图结构的多变量时序 GNN。
- **AGCRN**（Bai et al., 2020）：利用节点嵌入自适应学习节点特异模式的 GCN-RNN 模型。

**Transformer/MLP：**
- **STID**（Shao et al., 2022）：基于可学习时空嵌入的简单 MLP 模型，无需显式图结构。
- **PatchTST**（Nie et al., 2023）：将时序切分成 patch 后送入 Transformer，采用通道独立策略。

**概率生成模型：**
- **CSDI**（Tashiro et al., 2021）：基于分数扩散的条件概率时序预测模型。
- **TSDiff**（Rasul et al., 2023）：无条件扩散训练 + 替换引导推理的概率预测模型。
- **TSFlow**（Kollovieh et al., 2025, ICLR）：结合 OU 高斯过程先验的连续流匹配模型。
- **K2VAE**（Wu et al., 2025, ICML Spotlight）：融合 Koopman 算子与 VAE 的概率预测框架。

所有基线共享相同的数据划分和输入输出长度。确定性基线用 L1 损失训练，概率性基线用各自的原生目标函数训练。

## 4.1.4 实验细节

对所有数据集，输入长度设置为 $T_{\text{in}} = 168$ 步，预测步长为 $T_{\text{out}} = 12$ 步（对应 10 分钟数据约 2 小时，小时级数据约 12 小时）。GCN 和 TCN 的隐藏维度均为 64，环境表征 $\text{dim}_{\text{env}}$ 和随机表征 $\text{dim}_{\text{stoch}}$ 均为 32。CFM 向量场使用 3 层 MLP，隐藏单元 256，采用 Heun 二阶 ODE 求解器，积分步数为 15 步，测试时采样 100 个样本。

所有模型使用 Adam 优化器训练，初始学习率 $5 \times 10^{-4}$（Solar-AL）或 $1 \times 10^{-3}$（其余数据集），采用 ReduceLROnPlateau 学习率调度（衰减因子 0.5）。梯度裁剪阈值为 1.0，最大训练轮数 200，基于验证集 CRPS 进行早停（耐心值为 25–30）。CLUB 预热 5 个 epoch 后线性升温 3 个 epoch 至目标权重 $\lambda_{\text{club}} = 0.01$。温度校准在验证集上通过网格搜索完成，校准区间为 $[0.5, 2.0]$。
