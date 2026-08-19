#!/bin/bash
# AutoDL gpt-5.6-luna MCTS trace10 批跑。
# 题集与 config_k3_trace10.yaml 对齐，输出独立保存到 output/gpt56_luna_trace10。
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /home/wangyichen/DirecTune-MCTS
source /home/wangyichen/miniconda3/etc/profile.d/conda.sh
conda activate forge

LOGROOT="output/gpt56_luna_trace10"
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
    echo "[$(date +%H:%M:%S)] MISSING ${level}/${prob}" | tee -a "$SUMMARY"
    return
  fi
  echo "[$(date +%H:%M:%S)] START ${level}/${prob}" | tee -a "$SUMMARY"
  timeout 10800 /home/wangyichen/miniconda3/envs/forge/bin/python3 main.py --config config_gpt56_luna_trace10.yaml \
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
print('latency=%sms triton=%s' % (fc.get('latency_ms'), '@triton.jit' in fc.get('code','')))
" 2>/dev/null)
  fi
  echo "[$(date +%H:%M:%S)] DONE ${level}/${prob} (rc=$RC) $CHAMP" | tee -a "$SUMMARY"
}

echo "[$(date +%H:%M:%S)] ===== GPT-5.6-LUNA TRACE10 START =====" | tee -a "$SUMMARY"
run_one level1 "01_square_matrix_multiplication"
run_one level1 "40_layernorm"
run_one level1 "50_conv_standard_2d_square_input_square_kernel"
run_one level1 "88_mingptnewgelu"
run_one level1 "42_max_pooling_2d"
run_one level2 "9_Matmul_Subtract_Multiply_ReLU"
run_one level2 "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp"
run_one level2 "30_Gemm_GroupNorm_Hardtanh"
run_one level2 "62_Matmul_GroupNorm_LeakyReLU_Sum"
run_one level2 "76_Gemm_Add_ReLU"
echo "[$(date +%H:%M:%S)] ===== GPT-5.6-LUNA TRACE10 DONE =====" | tee -a "$SUMMARY"
