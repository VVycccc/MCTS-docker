# DirecTune-MCTS 28题加速比汇总

来源：`/home/wangyichen/work-log.md` 中 “2026-07-18 DirecTune-MCTS 全量对比实验（MCTS 侧单侧完成）” 的 28 题表。

## L1 18题

| 题目 | vs seed | vs PyTorch |
|---|---:|---:|
| 01_matmul | 5.42× | 2.60× |
| 17_matmul_tb | 1.19× | 1.71× |
| 19_relu | 1.98× | 1.00× |
| 88_mingptnewgelu | 1.27× | 8.98× |
| 23_softmax | 2.29× | 1.30× |
| 24_logsoftmax | 2.27× | 1.30× |
| 33_batchnorm | 1.46× | 1.03× |
| 40_layernorm | 57.33× | 5.09× |
| 89_cumsum | 1.80× | 1.26× |
| 90_cumprod | 26.95× | 1.26× |
| 50_conv2d | 8.63× | 1.15× |
| 57_convT2d | 1.00× | 0.08× |
| 42_maxpool | 2.47× | 2.17× |
| 45_avgpool | — | — |
| 98_kldivloss | 1.00× | 1.00× |
| 16_matmul_tA | 2.29× | 1.29× |
| 97_sdpa | 1.00× | 0.17× |
| 94_mseloss | 1.00× | 1.00× |

## L2 10题

| 题目 | vs seed | vs PyTorch |
|---|---:|---:|
| 9_Matmul | 15.44× | 3.24× |
| 76_Gemm | 4.61× | 3.35× |
| 30_Gemm | 7.64× | 7.64× |
| 1_Conv2D | 18.34× | 1.15× |
| 85_Conv | — | — |
| 5_ConvT | 1.00× | 1.00× |
| 15_ConvT3d | 1.24× | 1.24× |
| 18_Matmul | 209.09× | 147.99× |
| 99_Matmul | 1.64× | 1.64× |
| 62_Matmul | 3.42× | 3.42× |
