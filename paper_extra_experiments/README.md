# 论文补充实验

这个目录只增加评估脚本，不修改训练代码和已有结果。建议按下面的优先级运行。

## 哪些实验需要新代码

1. **配对 Bootstrap 显著性检验**：需要，已实现。现有均值预测可直接检验 MAE/RMSE。
2. **生成轨迹数敏感性**：需要，已实现。必须读取完整的概率轨迹，均值预测文件不能用于该实验。
3. **NFE–精度–速度权衡**：需要，已实现。读取已有 checkpoint，只重新推理，不训练。
4. **双分支表示诊断**：需要，已实现。读取 checkpoint，报告分支相似性、分支与高低频分量的关联及交叉 CLUB 估计。
5. **多随机种子**：不需要新代码，项目根目录已有 `run_multiseed.py`。
6. **DiffSTG 断点续跑/导出 NPY**：不需要新代码，现有 baseline 已支持缓存与续跑。

所有命令均在项目根目录执行：

```bash
cd GridCFN-v3-infra
```

## 1. 配对 Bootstrap（可立即运行）

现有 `*_prediction.npy` 多数是均值预测，所以当前只检验 MAE 和 RMSE：

```bash
uv run python paper_extra_experiments/bootstrap_significance.py \
  --dcfm result/plot/solar/GridCFN_prediction.npy \
  --baseline TSFlow=result/plot/solar/tsflow_prediction.npy \
  --baseline MTGNN=result/plot/solar/mtgnn_prediction.npy \
  --target result/plot/solar/ground_truth.npy \
  --metrics MAE RMSE \
  --bootstrap 10000 \
  --output-dir paper_extra_results/bootstrap/solar
```

输出中 `improvement > 0` 表示 DCFM 更好；`ci95_low > 0` 表示差异在
95% Bootstrap 区间下稳定。不要用均值预测声称做了概率 CRPS 显著性检验。

## 2. NFE–精度–速度（优先补 Solar）

以下命令不重新训练。`--max-batches 20` 适合先确认趋势；用于论文时改为
`0` 跑完整测试集。Heun 求解器每步调用两次向量场，所以 NFE = 2 × steps。

```bash
uv run python paper_extra_experiments/nfe_sensitivity.py \
  --preset solar \
  --checkpoint result/solar/20260722_081956/dcfm_solar_20260722_081956.pt \
  --steps 3 5 10 15 20 \
  --n-samples 100 \
  --batch-size 8 \
  --max-batches 20 \
  --warmup-batches 2 \
  --device cuda:0 \
  --save-samples \
  --output-dir paper_extra_results/nfe
```

建议先跑 20 个 batch，确认脚本和显存正常，再将 `--max-batches 0`。结果同时
包含 MAE、RMSE、CRPS、PICP、PINAW、耗时和峰值显存。`--save-samples`
会保存完整轨迹，磁盘占用较大；只需在最终采用的 steps（通常 15）保留一次。

## 3. 生成轨迹数敏感性

先完成上一步并保存完整轨迹，再执行：

```bash
uv run python paper_extra_experiments/sample_count_sensitivity.py \
  --prediction paper_extra_results/nfe/solar/samples_steps_15.npy \
  --target paper_extra_results/nfe/solar/ground_truth.npy \
  --sample-counts 5 10 20 50 100 \
  --repeats 10 \
  --output-dir paper_extra_results/sample_count/solar
```

如果 50 条以后 CRPS/PICP 基本稳定，正文可说明 100 条轨迹足够；不需要把
这张表放主文，修稿时可放附录或补充材料。

## 4. 双分支表示诊断

```bash
uv run python paper_extra_experiments/representation_probe.py \
  --preset solar \
  --checkpoint result/solar/20260722_081956/dcfm_solar_20260722_081956.pt \
  --max-batches 20 \
  --device cuda:0 \
  --output-dir paper_extra_results/representation
```

解释时重点看：

- `branch_cosine_abs`：越低表示两个表示方向越不相似；
- environment 分支与 low 的相关性是否高于其与 high 的相关性；
- short-term 分支与 high 的相关性是否高于其与 low 的相关性。

这些是诊断性证据，不应写成“证明完全解耦”。若趋势不符合预期，不放论文，
保留现有消融实验即可。

## 推荐执行顺序

1. Solar 的 NFE 小规模测试；
2. Solar 完整 NFE，并保存 steps=15 的完整轨迹；
3. 样本数敏感性；
4. 四个数据集的 MAE/RMSE Bootstrap；
5. 表示诊断；
6. 只有审稿人明确要求时，再将 NFE/表示诊断扩展到四个数据集。

