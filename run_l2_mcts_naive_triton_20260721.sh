#!/usr/bin/env bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0

ROOT=/home/wangyichen/DirecTune-MCTS
cd "$ROOT" || exit 1
source /home/wangyichen/miniconda3/etc/profile.d/conda.sh
conda activate forge

CONFIG=config_full_l2_naive_triton.yaml
OUTROOT=output/full_mcts_l2_naive_triton
SUMMARY="$OUTROOT/summary.log"
mkdir -p "$OUTROOT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
assert cfg.get('gen_mode') == 'naive', f"expected gen_mode=naive, got {cfg.get('gen_mode')!r}"
assert cfg.get('search_mode') == 'mcts', f"expected search_mode=mcts, got {cfg.get('search_mode')!r}"
assert cfg.get('skill_mode') == 'off', f"expected skill_mode=off, got {cfg.get('skill_mode')!r}"
assert cfg.get('mcts_rollout_depth') == 2, f"expected mcts_rollout_depth=2, got {cfg.get('mcts_rollout_depth')!r}"
PY

is_valid_triton_final() {
  local fp="$1"
  [ -f "$fp" ] || return 1
  python - "$fp" <<'PY' >/dev/null 2>&1
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    fc = (d.get('final_candidates') or [{}])[0]
    code = fc.get('code', '') or ''
    raise SystemExit(0 if '@triton.jit' in code else 1)
except Exception:
    raise SystemExit(1)
PY
}

summarize_final() {
  local fp="$1"
  python - "$fp" <<'PY'
import json, sys
fp = sys.argv[1]
d = json.load(open(fp))
fc = (d.get('final_candidates') or [{}])[0]
code = fc.get('code', '') or ''
champ = fc.get('latency_ms')
seed = d.get('seed_latency_ms') or d.get('initial_latency_ms')

def find_base(obj):
    if isinstance(obj, dict):
        for k in ('baseline_latency_ms', 'base_latency_ms'):
            if isinstance(obj.get(k), (int, float)):
                return obj[k]
        for k in ('baseline', 'baseline_result'):
            v = obj.get(k)
            if isinstance(v, dict) and isinstance(v.get('latency_ms'), (int, float)):
                return v.get('latency_ms')
        for v in obj.values():
            r = find_base(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_base(v)
            if r is not None:
                return r
    return None

base = find_base(d)
vs_pt = (base / champ) if isinstance(base, (int, float)) and isinstance(champ, (int, float)) and champ else None
vs_seed = (seed / champ) if isinstance(seed, (int, float)) and isinstance(champ, (int, float)) and champ else None
print('base_ms={} seed_ms={} champ_ms={} vs_pytorch={} vs_seed={} triton={}'.format(
    'NA' if base is None else f'{base:.6g}',
    'NA' if seed is None else f'{seed:.6g}',
    'NA' if champ is None else f'{champ:.6g}',
    'NA' if vs_pt is None else f'{vs_pt:.4f}x',
    'NA' if vs_seed is None else f'{vs_seed:.4f}x',
    '@triton.jit' in code,
))
PY
}

mapfile -t PROBS < <(find problems/kb_level2 -maxdepth 1 -name '*.json' -printf '%f\n' | sed 's/\.json$//' | sort)

valid=0; invalid=0; missing=0
for p in "${PROBS[@]}"; do
  fp="$OUTROOT/$p/final_results.json"
  if is_valid_triton_final "$fp"; then
    valid=$((valid + 1))
  elif [ -f "$fp" ]; then
    invalid=$((invalid + 1))
  else
    missing=$((missing + 1))
  fi
done

printf '[%s] ===== L2 MCTS NAIVE TRITON-STRICT START =====\n' "$(ts)" | tee -a "$SUMMARY"
printf '[%s] config=%s outroot=%s total=%d valid_triton=%d invalid_nontriton=%d missing=%d todo=%d\n' \
  "$(ts)" "$CONFIG" "$OUTROOT" "${#PROBS[@]}" "$valid" "$invalid" "$missing" "$((invalid + missing))" | tee -a "$SUMMARY"
printf '[%s] settings: gen_mode=naive search_mode=mcts mcts_rollout_depth=2 skill_mode=off CUDA_VISIBLE_DEVICES=%s\n' \
  "$(ts)" "${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$SUMMARY"

for P in "${PROBS[@]}"; do
  OUT="$OUTROOT/$P"
  FINAL="$OUT/final_results.json"

  if is_valid_triton_final "$FINAL"; then
    echo "[$(ts)] SKIP $P valid_triton $(summarize_final "$FINAL")" | tee -a "$SUMMARY"
    continue
  fi

  mkdir -p "$OUT"
  if [ -f "$FINAL" ]; then
    bak="$OUT/final_results.nontriton_invalid.$(date +%Y%m%d_%H%M%S).json"
    mv "$FINAL" "$bak"
    echo "[$(ts)] INVALID_RENAMED $P -> $(basename "$bak")" | tee -a "$SUMMARY"
  fi

  PROB="problems/kb_level2/$P.json"
  INIT="problems/kb_level2/${P}_initial.py"
  if [ ! -f "$PROB" ] || [ ! -f "$INIT" ]; then
    echo "[$(ts)] MISSING_FILES $P" | tee -a "$SUMMARY"
    continue
  fi

  echo "[$(ts)] START $P" | tee -a "$SUMMARY"
  timeout 9000 python main.py --config "$CONFIG" \
    --problem "$PROB" --initial "$INIT" \
    --output "$OUT" --rounds 12 --breadth 4 --num-samples 1 \
    > "$OUT/run.log" 2>&1
  RC=$?

  if grep -q 'gen_mode=akg\|\[Generator AKG\]' "$OUT/run.log" 2>/dev/null; then
    echo "[$(ts)] ERROR_AKG_PATH_DETECTED $P rc=$RC -- invalid for naive experiment" | tee -a "$SUMMARY"
  fi

  if is_valid_triton_final "$FINAL"; then
    echo "[$(ts)] DONE $P rc=$RC $(summarize_final "$FINAL")" | tee -a "$SUMMARY"
  elif [ -f "$FINAL" ]; then
    echo "[$(ts)] DONE_INVALID $P rc=$RC $(summarize_final "$FINAL")" | tee -a "$SUMMARY"
    bak="$OUT/final_results.nontriton_invalid.$(date +%Y%m%d_%H%M%S).json"
    mv "$FINAL" "$bak"
    echo "[$(ts)] INVALID_RENAMED $P -> $(basename "$bak")" | tee -a "$SUMMARY"
  else
    echo "[$(ts)] FAILED_NO_TRITON_FINAL $P rc=$RC" | tee -a "$SUMMARY"
  fi
  sleep 5
done

valid=0; invalid=0; missing=0
for p in "${PROBS[@]}"; do
  fp="$OUTROOT/$p/final_results.json"
  if is_valid_triton_final "$fp"; then
    valid=$((valid + 1))
  elif [ -f "$fp" ]; then
    invalid=$((invalid + 1))
  else
    missing=$((missing + 1))
  fi
done
printf '[%s] ===== L2 MCTS NAIVE TRITON-STRICT DONE valid_triton=%d invalid_nontriton=%d missing=%d =====\n' \
  "$(ts)" "$valid" "$invalid" "$missing" | tee -a "$SUMMARY"
