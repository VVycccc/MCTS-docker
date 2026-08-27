#!/bin/bash
# A/B: thinking 全开(旧) vs patch关(现状) vs 全关(激进) — 3 题 × 3 臂
# 断点可续跑：已有 final_results.json 的 run 跳过。产物 output/ab_thinking/<arm>/<problem>/
cd /home/wangyichen/DirecTune-MCTS || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/home/wangyichen/miniconda3/envs/forge/bin/python3

PROBLEMS=(
  "kb_level1/01_square_matrix_multiplication"
  "kb_level1/40_layernorm"
  "kb_level2/30_Gemm_GroupNorm_Hardtanh"
)
ARMS=(on patch alloff)

for arm in "${ARMS[@]}"; do
  for p in "${PROBLEMS[@]}"; do
    name=$(basename "$p")
    out="output/ab_thinking/${arm}/${name}"
    if [ -f "${out}/final_results.json" ]; then
      echo "[skip] ${arm}/${name} already done"
      continue
    fi
    echo "=== [$(date +%H:%M:%S)] arm=${arm} problem=${name} ==="
    mkdir -p "$out"
    timeout 2700 $PY main.py --config "config_ab_${arm}.yaml" \
      --problem "problems/${p}.json" \
      --initial "problems/${p}_initial.py" \
      --output "$out" --rounds 8 > "${out}/run.log" 2>&1
    echo "--- [$(date +%H:%M:%S)] exit=$? tokens: $(grep -h 'prompt=' ${out}/run.log | tail -1)"
  done
done
echo "ALL DONE [$(date +%H:%M:%S)]"
