# Baseline 汇总（solar, 20260709_174923）

| model   | status | MAE    | RMSE   | MAPE           | CRPS   | MAE_norm | RMSE_norm | MAPE_norm | CRPS_norm | error                                                                                         |
| ------- | ------ | ------ | ------ | -------------- | ------ | -------- | --------- | --------- | --------- | --------------------------------------------------------------------------------------------- |
| diffstg | ERROR  |        |        |                |        |          |           |           |           | The size of tensor a (137) must match the size of tensor b (180) at non-singleton dimension 3 |
| tsdiff  | OK     | 3.0031 | 5.5761 | 329613845.0035 | 1.7696 | 0.2881   | 0.5350    | 103.6682  | 0.1698    |                                                                                               |

注：无后缀指标为反归一化域；`*_norm` 指标为归一化域。ERROR 行表示该模型运行失败，但其他模型结果已保留。
