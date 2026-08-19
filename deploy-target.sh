#!/usr/bin/env bash
# 目标机器上的部署清单 — DirecTune-MCTS
# 前提：Linux + NVIDIA 驱动(>=525, nvidia-smi 显示 CUDA >= 12.6) + Docker + NVIDIA Container Toolkit
set -e

# 1. 导入镜像（directune-mcts.tar.gz 约 7.5GB，导入后约 14.5GB）
docker load < directune-mcts.tar.gz
docker images directune-mcts   # 确认存在

# 2. GPU 可用性检查（应输出 True）
docker run --rm --gpus all directune-mcts:latest \
    python -c "import torch; print('cuda:', torch.cuda.is_available())"

# 3. 建工作目录 + 写真实配置
mkdir -p ~/directune-run/output && cd ~/directune-run
docker run --rm directune-mcts:latest cat /app/config.example.yaml > config.yaml
#    编辑 config.yaml，填 model.url / model.model / model.api_key（OpenAI 兼容端点）

# 4. 跑一道 L1 验证（output 落在宿主机 ~/directune-run/output）
docker run --gpus all -it --rm \
    -v $PWD/config.yaml:/app/config.yaml \
    -v $PWD/output:/app/output \
    directune-mcts:latest \
    python main.py --config config.yaml \
        --problem problems/kb_level1/01_square_matrix_multiplication.json \
        --initial problems/kb_level1/01_square_matrix_multiplication_initial.py \
        --rounds 3 --breadth 2 --num-samples 1

# 5.（可选）跑 L2：挂载含 .pt 权重的 kb_level2 目录（镜像默认不含 24GB 权重）
#   -v /path/to/kb_level2:/app/problems/kb_level2
