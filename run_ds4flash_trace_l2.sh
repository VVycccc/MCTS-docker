#!/bin/bash
# L2 5 题补跑 — ds4flash_trace10
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /home/wangyichen/DirecTune-MCTS
source /home/wangyichen/miniconda3/etc/profile.d/conda.sh
conda activate forge

LOGROOT="output/ds4flash_trace10"
SUMMARY="$LOGROOT/summary.log"

run_one() {
  local level=$1 prob=$2
  local OUT="$LOGROOT/${prob}"
  if [ -f "$OUT/final_results.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP ${prob} (final_results.json exists)" | tee -a "$SUMMARY"
    return
  fi
  mkdir -p "$OUT"
  local PROB="problems/kb_${level}/${prob}.json"
  local INIT="problems/kb_${level}/${prob}_initial.py"
  echo "[$(date +%H:%M:%S)] START ${prob}" | tee -a "$SUMMARY"
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
  echo "[$(date +%H:%M:%S)] DONE ${prob} (rc=$RC) $CHAMP" | tee -a "$SUMMARY"
}

echo "--- L2 (5) ---" | tee -a "$SUMMARY"
run_one level2 "9_Matmul_Subtract_Multiply_ReLU"
run_one level2 "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp"
run_one level2 "30_Gemm_GroupNorm_Hardtanh"
run_one level2 "62_Matmul_GroupNorm_LeakyReLU_Sum"
run_one level2 "76_Gemm_Add_ReLU"

echo "[$(date +%H:%M:%S)] ===== DS4-FLASH TRACE10 L2 DONE =====" | tee -a "$SUMMARY"