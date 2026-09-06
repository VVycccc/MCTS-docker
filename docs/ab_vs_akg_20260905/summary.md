# 等资源对照：DirecTune-MCTS vs AKG（best-of-N）

| 题目 | mcts passed | mcts ms | mcts speedup | akg passed | akg ms | akg speedup | 胜者 |
|---|---|---|---|---|---|---|---|
| 01_square_matrix_multiplication | 1.000 | 2.356 | 2.40 | 1.000 | 4.284 | 1.32 | mcts |
| 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp | 1.000 | 0.041 | 166.77 | 1.000 | 0.362 | 18.80 | mcts |
| 30_Gemm_GroupNorm_Hardtanh | 1.000 | 4.805 | 1.48 | 1.000 | 4.205 | 1.69 | akg |
| 40_layernorm | 1.000 | 1.502 | 4.74 | 1.000 | 3.499 | 2.03 | mcts |
| 42_max_pooling_2d | 1.000 | 8.379 | 2.17 | 1.000 | 6.116 | 3.01 | akg |
| 50_conv_standard_2d_square_input_square_kernel | 1.000 | 4.451 | 1.04 | 1.000 | 9.823 | 0.48 | mcts |
| 62_Matmul_GroupNorm_LeakyReLU_Sum | 1.000 | 2.346 | 3.23 | 1.000 | 4.075 | 1.85 | mcts |
| 76_Gemm_Add_ReLU | 1.000 | 2.197 | 3.15 | 1.000 | 4.144 | 1.67 | mcts |
| 88_mingptnewgelu | 1.000 | 0.637 | 8.98 | 1.000 | 0.637 | 8.97 | mcts |
| 9_Matmul_Subtract_Multiply_ReLU | 1.000 | 2.393 | 2.90 | 1.000 | 4.089 | 1.71 | mcts |

## 汇总

- **mcts**: 完成 10，通过 10（100%），geomean speedup_vs_pytorch = 4.194，calls mean = 46.5，tokens mean = 638.8K
- **akg**: 完成 10，通过 10（100%），geomean speedup_vs_pytorch = 2.381，calls mean = 43.5，tokens mean = 512.6K

## 资源对齐审计

AKG 臂预算来自同题 MCTS 臂实际消耗；比值 ≈1 为对齐。

| 题目 | mcts calls | akg calls | 比值 | mcts tokens | akg tokens | 比值 |
|---|---|---|---|---|---|---|
| 01_square_matrix_multiplication | 46 | 52 | 1.13 | 738973 | 457158 | 0.62 |
| 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp | 53 | 37 | 0.70 | 467696 | 495596 | 1.06 |
| 30_Gemm_GroupNorm_Hardtanh | 44 | 41 | 0.93 | 554573 | 568507 | 1.03 |
| 40_layernorm | 41 | 41 | 1.00 | 759555 | 561167 | 0.74 |
| 42_max_pooling_2d | 31 | 33 | 1.06 | 602457 | 403610 | 0.67 |
| 50_conv_standard_2d_square_input_square_kernel | 29 | 29 | 1.00 | 611229 | 391558 | 0.64 |
| 62_Matmul_GroupNorm_LeakyReLU_Sum | 45 | 47 | 1.04 | 506445 | 513672 | 1.01 |
| 76_Gemm_Add_ReLU | 70 | 48 | 0.69 | 713801 | 720899 | 1.01 |
| 88_mingptnewgelu | 66 | 67 | 1.02 | 727749 | 474666 | 0.65 |
| 9_Matmul_Subtract_Multiply_ReLU | 40 | 40 | 1.00 | 705628 | 538816 | 0.76 |
