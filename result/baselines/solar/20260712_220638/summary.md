# Baseline 汇总（solar, 20260712_220638）

| model    | status | MAE    | RMSE   | MAPE          | CRPS   | MAE_norm | RMSE_norm | MAPE_norm | CRPS_norm | error                                                   |
| -------- | ------ | ------ | ------ | ------------- | ------ | -------- | --------- | --------- | --------- | ------------------------------------------------------- |
| tsflow   | OK     | 0.9299 | 2.3425 | 30114380.6560 | 0.6282 | 0.0892   | 0.2247    | 70.7537   | 0.0603    |                                                         |
| patchtst | ERROR  |        |        |               |        |          |           |           |           | 'TrainConfig' object has no attribute 'mc_samples_test' |
| stid     | ERROR  |        |        |               |        |          |           |           |           | 'TrainConfig' object has no attribute 'mc_samples_test' |

注：无后缀指标为反归一化域；`*_norm` 指标为归一化域。ERROR 行表示该模型运行失败，但其他模型结果已保留。
