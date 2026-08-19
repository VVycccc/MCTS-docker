#!/bin/bash
# DirecTune-MCTS TLE 后端运行示例（forge_tle = FlagTree triton 3.6 + TLE，启用方向 ⑥ tle_async_smem）
# 用法：./run_tle.sh [problem_name]   默认跑 01_square_matrix_multiplication（GEMM，方向6适用）
cd /home/wangyichen/DirecTune-MCTS
export CUDA_VISIBLE_DEVICES=0
source /home/wangyichen/miniconda3/etc/profile.d/conda.sh
conda activate forge_tle

P="${1:-01_square_matrix_multiplication}"
LOGDIR=output/tle; mkdir -p "$LOGDIR"
OUT="$LOGDIR/$P"; mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] START $P (forge_tle backend, direction ⑥ enabled)"
timeout 1800 python main.py --config config_tle.yaml \
    --problem "problems/kb_level1/$P.json" \
    --initial "problems/kb_level1/${P}_initial.py" \
    --output "$OUT" --rounds 3 --breadth 2 --num-samples 1 > "$OUT/run.log" 2>&1
echo "[$(date +%H:%M:%S)] DONE $P (rc=$?)" | tee -a "$LOGDIR/summary.log"
