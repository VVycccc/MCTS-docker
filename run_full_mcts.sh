#!/bin/bash
# ============================================================================
# MCTS 侧全量实验批跑 — DirecTune-MCTS
# 题集：full_experiment_problems.sh（20 L1 + 10 L2）
# 配置：config_full.yaml（search_mode=mcts, rollout_depth=2, time_budget=3600）
# 跑法：CUDA_VISIBLE_DEVICES=0（仅用 RTX 3090 #0），forge env
# timeout 5400 = search_time_budget(3600) + naive gen/方向分类(~5min, 发生在 search_start 前)
#   + 单次 _expand overrun(unified_editor 5 patch ×~2min LLM 不可中断, ~12min) + 写盘余量。
#   原 3900 与 search deadline 只差 30s，rollout 中途的扩展 overrun 会撞 shell SIGTERM →
#   final_results.json 来不及写（01 题 rc=124 实例）。5400 留 ~13min 安全余量。
# 断点续跑：final_results.json 存在则跳过
# 产物：output/full_mcts/<problem>/final_results.json
# ============================================================================
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /home/wangyichen/DirecTune-MCTS
source /home/wangyichen/miniconda3/etc/profile.d/conda.sh
conda activate forge

source ./full_experiment_problems.sh

LOGROOT="output/full_mcts"
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
  timeout 9000 python main.py --config config_full.yaml \
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

echo "[$(date +%H:%M:%S)] ===== FULL MCTS EXPERIMENT START =====" | tee -a "$SUMMARY"
echo "  L1: ${#L1_PROBLEMS[@]} problems, L2: ${#L2_PROBLEMS[@]} problems" | tee -a "$SUMMARY"

echo "--- L1 ---" | tee -a "$SUMMARY"
for P in "${L1_PROBLEMS[@]}"; do run_one level1 "$P"; done

echo "--- L2 ---" | tee -a "$SUMMARY"
for P in "${L2_PROBLEMS[@]}"; do run_one level2 "$P"; done

echo "[$(date +%H:%M:%S)] ===== FULL MCTS EXPERIMENT DONE =====" | tee -a "$SUMMARY"
