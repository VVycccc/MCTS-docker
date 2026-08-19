#!/bin/bash
# ============================================================================
# DS4-flash 换模型对比批跑 — DirecTune-MCTS（trace + 换模型共用 5 题集）
# 题集：40_layernorm(L1) / 50_conv2d(L1) / 18_Matmul(L2) / 76_Gemm(L2) / 9_Matmul(L2)
#       与 2026-07-29 GLM-5.2 trace 题完全一致，便于直接对比。
# 配置：config_ds4flash_trace.yaml（deepseek-v4-flash @ api.deepseek.com/v1，
#       rollout_depth=2, max_depth=4, adaptive, time_budget=3600）
# 跑法：CUDA_VISIBLE_DEVICES=0（仅用 RTX 3090 #0），forge env
# timeout 9000：search_time_budget(3600) + naive gen/方向分类 + _expand overrun + 写盘余量。
#   注意 DS4-flash 无 429/限流，无 GLM 2min/call 的慢推理，实际单题通常远早于 3600s 完成；
#   但 unified_editor 一次 _expand 仍可能多 patch 串行，留足余量。
# 断点续跑：final_results.json 存在则跳过
# 产物：output/ds4flash_trace/<problem>/final_results.json
# ============================================================================
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /home/wangyichen/DirecTune-MCTS
source /home/wangyichen/miniconda3/etc/profile.d/conda.sh
conda activate forge

LOGROOT="output/ds4flash_trace"
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
  timeout 9000 python main.py --config config_ds4flash_trace.yaml \
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

echo "[$(date +%H:%M:%S)] ===== DS4-FLASH TRACE EXPERIMENT START =====" | tee -a "$SUMMARY"

echo "--- L1 ---" | tee -a "$SUMMARY"
run_one level1 "40_layernorm"
run_one level1 "50_conv_standard_2d_square_input_square_kernel"

echo "--- L2 ---" | tee -a "$SUMMARY"
run_one level2 "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp"
run_one level2 "76_Gemm_Add_ReLU"
run_one level2 "9_Matmul_Subtract_Multiply_ReLU"

echo "[$(date +%H:%M:%S)] ===== DS4-FLASH TRACE EXPERIMENT DONE =====" | tee -a "$SUMMARY"
