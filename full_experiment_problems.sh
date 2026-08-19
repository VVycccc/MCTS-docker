#!/bin/bash
# ============================================================================
# DirecTune-MCTS 全量对比实验 — 题集（两侧共用）
# 20 L1（10 op_type × 2 道）+ 10 L2 融合题 = 30 题
# 题名已订正为 problems/kb_levelN/ 实际文件名
# ============================================================================

# ---- L1: 20 道 ----
# 顺序：先小后大（reduction/attention 盲区放最后）
L1_PROBLEMS=(
  "01_square_matrix_multiplication"                 # gemm
  "17_matmul_with_transposed_b"                      # gemm
  "19_relu"                                          # elementwise
  "88_mingptnewgelu"                                 # elementwise
  "23_softmax"                                       # softmax
  "24_logsoftmax"                                    # softmax
  "33_batchnorm"                                     # normalization
  "40_layernorm"                                     # normalization
  "89_cumsum"                                        # scan
  "90_cumprod"                                       # scan
  "50_conv_standard_2d_square_input_square_kernel"   # conv2d
  "57_conv_transposed_2d_square_input_square_kernel" # conv2d
  "42_max_pooling_2d"                                # pooling
  "45_average_pooling_2d"                            # pooling
  "98_kldivloss"                                     # loss
  "16_matmul_with_transposed_a"                      # gemm(代 loss 第二道)
  "97_scaleddotproductattention"                     # attention (盲区)
  "94_mseloss"                                       # reduction (盲区)
)

# ---- L2: 10 道融合题（含 .pt 权重）----
L2_PROBLEMS=(
  "9_Matmul_Subtract_Multiply_ReLU"
  "76_Gemm_Add_ReLU"
  "30_Gemm_GroupNorm_Hardtanh"
  "1_Conv2D_ReLU_BiasAdd"
  "85_Conv2d_GroupNorm_Scale_MaxPool_Clamp"
  "5_ConvTranspose2d_Subtract_Tanh"
  "15_ConvTranspose3d_BatchNorm_Subtract"
  "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp"
  "99_Matmul_GELU_Softmax"
  "62_Matmul_GroupNorm_LeakyReLU_Sum"
)
