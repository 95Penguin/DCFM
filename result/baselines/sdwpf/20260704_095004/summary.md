# Baseline 汇总（sdwpf, 20260704_095004）

| model    | status | MAE      | RMSE     | MAPE              | CRPS     | MAE_norm | RMSE_norm | MAPE_norm | CRPS_norm | error                                          |
| -------- | ------ | -------- | -------- | ----------------- | -------- | -------- | --------- | --------- | --------- | ---------------------------------------------- |
| tsflow   | ERROR  |          |          |                   |          |          |           |           |           | tsflow exceeded 10800s                         |
| patchtst | OK     | 98.4669  | 166.6904 | 39749670400.0000  | 98.3288  | 0.2423   | 0.4102    | 178.7658  | 0.1812    |                                                |
| k2vae    | OK     | 321.3689 | 424.1467 | 581704679424.0000 | 302.9250 | 0.7908   | 1.0437    | 165.4342  | 0.7454    |                                                |
| tsdiff   | ERROR  |          |          |                   |          |          |           |           |           | tsdiff exceeded 10800s                         |
| csdi     | ERROR  |          |          |                   |          |          |           |           |           | csdi exceeded 10800s                           |
| diffstg  | ERROR  |          |          |                   |          |          |           |           |           | f-string: expecting '}' (diffstg.py, line 304) |

注：无后缀指标为反归一化域；`*_norm` 指标为归一化域。ERROR 行表示该模型运行失败，但其他模型结果已保留。
