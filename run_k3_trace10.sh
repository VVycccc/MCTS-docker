#!/bin/bash
# ============================================================================
# Kimi-K3 10 题 trace 批跑 — DirecTune-MCTS（trace 10 题集）
# 题集：同 ds4flash_trace10（L1×5 + L2×5）
#   L1: 01_matmul / 40_layernorm / 50_conv2d / 88_mingptnewgelu / 42_maxpool
#   L2: 9_Matmul / 18_Matmul / 30_Gemm / 62_Matmul / 76_Gemm
# 配置：config_k3_trace10.yaml（kimi-k3 @ autodl.art，
#       rollout_depth=2, max_depth=4, adaptive, time_budget=3600）
# timeout 10800 (3h)：kimi-k3 是推理模型，LLM 调用可能更慢
# 断点续跑：final_results.json 存在则跳过
# 产物：output/k3_trace10/<problem>/final_results.json
# ============================================================================
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /home/wangyichen/DirecTune-MCTS
source /home/wangyichen/miniconda3/etc/profile.d/conda.sh
conda activate forge

LOGROOT="output/k3_trace10"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.log"

run_one() {
  local level=$1 prob=$2
  local OUT="$LOGROOT/${prob}"
  if [ -f "$OUT/final_results.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP ${level}/${prob} (final_results.json exists)" | tee -a "$SUMMARY"
    return
  fi
  mkdir -p "$OUT"
  local PROB="problems/kb_${level}/${prob}.json"
  local INIT="problems/kb_${level}/${prob}_initial.py"
  if [ ! -f "$PROB" ] || [ ! -f "$INIT" ]; then
    echo "[$(date +%H:%M:%S)] MISSING files for ${level}/${prob} — skip" | tee -a "$SUMMARY"
    return
  fi
  echo "[$(date +%H:%M:%S)] START ${level}/${prob}" | tee -a "$SUMMARY"
  timeout 10800 python main.py --config config_k3_trace10.yaml \
      --problem "$PROB" --initial "$INIT" \
      --output "$OUT" --rounds 12 --breadth 4 --num-samples 1 \
      > "$OUT/run.log" 2>&1
  local RC=$?
  local CHAMP=""
  if [ -f "$OUT/final_results.json" ]; then
    CHAMP=$(python -c "
import json
d=json.load(open('$OUT/final_results.json'))
fc=d.get('final_candidates',[{}])[0]
lat=fc.get('latency_ms')
code=fc.get('code','')
triton='@triton.jit' in code
print(f'latency={lat}ms triton={triton}')
" 2>/dev/null)
  fi
  echo "[$(date +%H:%M:%S)] DONE ${level}/${prob} (rc=$RC) $CHAMP" | tee -a "$SUMMARY"
}

echo "[$(date +%H:%M:%S)] ===== KIMI-K3 TRACE 10-EXPERIMENT START =====" | tee -a "$SUMMARY"

echo "--- L1 (5) ---" | tee -a "$SUMMARY"
run_one level1 "01_square_matrix_multiplication"
run_one level1 "40_layernorm"
run_one level1 "50_conv_standard_2d_square_input_square_kernel"
run_one level1 "88_mingptnewgelu"
run_one level1 "42_max_pooling_2d"

echo "--- L2 (5) ---" | tee -a "$SUMMARY"
run_one level2 "9_Matmul_Subtract_Multiply_ReLU"
run_one level2 "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp"
run_one level2 "30_Gemm_GroupNorm_Hardtanh"
run_one level2 "62_Matmul_GroupNorm_LeakyReLU_Sum"
run_one level2 "76_Gemm_Add_ReLU"

echo "[$(date +%H:%M:%S)] ===== KIMI-K3 TRACE 10-EXPERIMENT DONE =====" | tee -a "$SUMMARY"