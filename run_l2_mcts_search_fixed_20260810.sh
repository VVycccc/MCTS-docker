#!/usr/bin/env bash
# Re-run the selected 30 L2 problems with independent PyTorch/seed/champion metrics.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0

ROOT=/home/wangyichen/DirecTune-MCTS
cd "$ROOT" || exit 1
source /home/wangyichen/miniconda3/etc/profile.d/conda.sh
conda activate forge

CONFIG=config_full_l2_naive_triton.yaml
OUTROOT=output/full_mcts_l2_search_fixed_20260810
SEEDDIR=output/naive_seed_l2_30
SELECTION=l2_selected_30_20260726.txt
SUMMARY="$OUTROOT/summary.log"
mkdir -p "$OUTROOT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
assert cfg.get('gen_mode') == 'naive'
assert cfg.get('search_mode') == 'mcts'
assert cfg.get('skill_mode') == 'off'
assert cfg.get('mcts_rollout_depth') == 2
assert cfg.get('search_time_budget') == 3600
PY

is_complete() {
  local fp="$1"
  [ -f "$fp" ] || return 1
  python - "$fp" <<'PY' >/dev/null 2>&1
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    fc = (d.get('final_candidates') or [{}])[0]
    code = fc.get('code', '') or ''
    required = ('baseline_latency_ms', 'seed_latency_ms', 'champion_latency_ms',
                'speedup_vs_pytorch', 'speedup_vs_seed')
    raise SystemExit(0 if '@triton.jit' in code and all(k in d for k in required) else 1)
except Exception:
    raise SystemExit(1)
PY
}

mapfile -t PROBS < <(grep -v '^#' "$SELECTION" | grep -v '^$')

echo "[$(ts)] ===== FIXED L2 MCTS START =====" | tee -a "$SUMMARY"
echo "[$(ts)] total=${#PROBS[@]} output=$OUTROOT seed_source=$SEEDDIR" | tee -a "$SUMMARY"
echo "[$(ts)] independent baseline/reference + seed profiling; no seed regeneration" | tee -a "$SUMMARY"

for P in "${PROBS[@]}"; do
  OUT="$OUTROOT/$P"
  FINAL="$OUT/final_results.json"
  if is_complete "$FINAL"; then
    echo "[$(ts)] SKIP $P complete" | tee -a "$SUMMARY"
    continue
  fi
  mkdir -p "$OUT"
  PROB="problems/kb_level2/$P.json"
  SEED="$SEEDDIR/${P}_seed.py"
  if [ ! -f "$PROB" ] || [ ! -f "$SEED" ]; then
    echo "[$(ts)] MISSING_INPUT $P problem=$([ -f "$PROB" ] && echo yes || echo no) seed=$([ -f "$SEED" ] && echo yes || echo no)" | tee -a "$SUMMARY"
    continue
  fi
  echo "[$(ts)] START $P" | tee -a "$SUMMARY"
  timeout 7200 python main.py --config "$CONFIG" \
    --problem "$PROB" --initial "$SEED" \
    --output "$OUT" --rounds 12 --breadth 4 --num-samples 1 \
    > "$OUT/run.log" 2>&1
  RC=$?
  if is_complete "$FINAL"; then
    python - "$FINAL" <<'PY' | sed 's/^/[metrics] /' | tee -a "$SUMMARY"
import json,sys
d=json.load(open(sys.argv[1]))
print(json.dumps({k:d.get(k) for k in ('baseline_latency_ms','seed_latency_ms','champion_latency_ms','speedup_vs_pytorch','speedup_vs_seed','strict_triton')}, sort_keys=True))
PY
    echo "[$(ts)] DONE $P rc=$RC" | tee -a "$SUMMARY"
  else
    echo "[$(ts)] INCOMPLETE $P rc=$RC checkpoint=$(find "$OUT" -maxdepth 1 -name 'checkpoint*.json' | wc -l)" | tee -a "$SUMMARY"
  fi
  sleep 5
done

echo "[$(ts)] ===== FIXED L2 MCTS END =====" | tee -a "$SUMMARY"
python summarize_l2_30.py --root "$OUTROOT" --output "$OUTROOT/l2_30_results.json" | tee -a "$SUMMARY"
