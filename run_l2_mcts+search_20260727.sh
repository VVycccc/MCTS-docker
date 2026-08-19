#!/usr/bin/env bash
# run_l2_mcts_search_20260727.sh — 30 道 L2 用已生成 naive seed 直接进 MCTS search。
# seed 来自 output/naive_seed_l2_30/{name}_seed.py（0 AKG，naiveness 1.00）。
# main.py 检测到 initial 含 @triton.jit 会跳过生成，直接进 search。
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0

ROOT=/home/wangyichen/DirecTune-MCTS
cd "$ROOT" || exit 1
source /home/wangyichen/miniconda3/etc/profile.d/conda.sh
conda activate forge

CONFIG=config_full_l2_naive_triton.yaml
OUTROOT=output/full_mcts_l2_search_20260727
SEEDDIR=output/naive_seed_l2_30
SELECTION=l2_selected_30_20260726.txt
SUMMARY="$OUTROOT/summary.log"
mkdir -p "$OUTROOT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# 校验 config
python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
assert cfg.get('gen_mode') == 'naive', f"expected gen_mode=naive, got {cfg.get('gen_mode')!r}"
assert cfg.get('search_mode') == 'mcts', f"expected search_mode=mcts, got {cfg.get('search_mode')!r}"
assert cfg.get('skill_mode') == 'off', f"expected skill_mode=off, got {cfg.get('skill_mode')!r}"
assert cfg.get('mcts_rollout_depth') == 2, f"expected mcts_rollout_depth=2, got {cfg.get('mcts_rollout_depth')!r}"
assert cfg.get('search_time_budget') == 3600, f"expected search_time_budget=3600, got {cfg.get('search_time_budget')!r}"
PY

# 判定 final_results.json champion 是否含 @triton.jit
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

# 汇总 final_results 的 base/seed/champion
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

# 读 selection
mapfile -t PROBS < <(grep -v '^#' "$SELECTION" | grep -v '^$' | sort)

# 统计已有有效结果
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

printf '[%s] ===== L2 MCTS SEARCH (naive seed) START =====\n' "$(ts)" | tee -a "$SUMMARY"
printf '[%s] config=%s outroot=%s total=%d valid_triton=%d invalid_nontriton=%d missing=%d todo=%d\n' \
  "$(ts)" "$CONFIG" "$OUTROOT" "${#PROBS[@]}" "$valid" "$invalid" "$missing" "$((invalid + missing))" | tee -a "$SUMMARY"
printf '[%s] settings: gen_mode=naive search_mode=mcts mcts_rollout_depth=2 search_time_budget=3600s skill_mode=off CUDA_VISIBLE_DEVICES=%s\n!s\n' \
  "$(ts)" "${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$SUMMARY"
printf '[%s] seed source: %s/{name}_seed.py (pre-generated, 0 AKG, naiveness 1.00)\n' "$(ts)" "$SEEDDIR" | tee -a "$SUMMARY"

for P in "${PROBS[@]}"; do
  OUT="$OUTROOT/$P"
  FINAL="$OUT/final_results.json"

  if is_valid_triton_final "$FINAL"; then
    echo "[$(ts)] SKIP $P valid_triton $(summarize_final "$FINAL")" | tee -a "$SUMMARY"
    continue
  fi

  mkdir -p "$OUT"
  if [ -f "$FINAL" ]; then
    bak="$OUT/final_results.invalid.$(date +%Y%m%d_%H%M%S).json"
    mv "$FINAL" "$bak"
    echo "[$(ts)] INVALID_RENAMED $P -> $(basename "$bak")" | tee -a "$SUMMARY"
  fi

  PROB="problems/kb_level2/$P.json"
  SEED="$SEEDDIR/${P}_seed.py"
  if [ ! -f "$PROB" ]; then
    echo "[$(ts)] MISSING_PROBLEM $P" | tee -a "$SUMMARY"
    continue
  fi
  if [ ! -f "$SEED" ]; then
    echo "[$(ts)] MISSING_SEED $P (expected $SEED) — skipping" | tee -a "$SUMMARY"
    continue
  fi

  echo "[$(ts)] START $P (seed=$SEED)" | tee -a "$SUMMARY"
  # search_time_budget=3600s + 一个 rollout 余量 (~30min) → timeout 5400s.
  # mcts 在 3600s 早停后，当前 _expand 要跑完才能写 final_results，余量 1800s 保 final 不丢。
  timeout 5400 python main.py --config "$CONFIG" \
    --problem "$PROB" --initial "$SEED" \
    --output "$OUT" --rounds 12 --breadth 4 --num-samples 1 \
    > "$OUT/run.log" 2>&1
  RC=$?

  # 校验：seed 已是 triton，不应触发 AKG 路径
  if grep -q 'gen_mode=akg\|\[Generator AKG\]' "$OUT/run.log" 2>/dev/null; then
    echo "[$(ts)] ERROR_AKG_PATH_DETECTED $P rc=$RC -- invalid for naive experiment" | tee -a "$SUMMARY"
  fi
  # 确认走了 skip-seed 分支
  if ! grep -q 'already Triton — skipping seed generation' "$OUT/run.log" 2>/dev/null; then
    echo "[$(ts)] WARN_SEED_NOT_SKIPPED $P rc=$RC (initial not detected as triton?)" | tee -a "$SUMMARY"
  fi

  if is_valid_triton_final "$FINAL"; then
    echo "[$(ts)] DONE $P rc=$RC $(summarize_final "$FINAL")" | tee -a "$SUMMARY"
  elif [ -f "$FINAL" ]; then
    echo "[$(ts)] DONE_INVALID $P rc=$RC $(summarize_final "$FINAL")" | tee -a "$SUMMARY"
    bak="$OUT/final_results.invalid.$(date +%Y%m%d_%H%M%S).json"
    mv "$FINAL" "$bak"
    echo "[$(ts)] INVALID_RENAMED $P -> $(basename "$bak")" | tee -a "$SUMMARY"
  else
    echo "[$(ts)] FAILED_NO_FINAL $P rc=$RC" | tee -a "$SUMMARY"
  fi
  sleep 5
done

# 终态汇总
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
printf '[%s] ===== L2 MCTS SEARCH DONE valid_triton=%d invalid_nontriton=%d missing=%d =====\n' \
  "$(ts)" "$valid" "$invalid" "$missing" | tee -a "$SUMMARY"
