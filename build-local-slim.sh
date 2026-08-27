#!/usr/bin/env bash
# build-local-slim.sh — 从本地已有的 directune-mcts 镜像抽取 slim 传输版。
#
# 背景：Docker Hub/镜像源在本机拉不动大层（EOF），但本地已有完整镜像。
# slim 版 = nvidia/cuda:12.2.0-base-ubuntu22.04（239MB，本地缓存）
#         + /opt/conda（torch/triton/依赖，剥掉 AKG fallback 栈后 ~5.3GB）
#         + /app（项目源码）
# 剥掉：CUDA toolkit 4.9GB（triton 用 wheel 自带 ptxas，不需要系统 nvcc）、
#       /opt/nvidia 1.2GB（nsight，仅 NCU profiling 需要——noop profiler 用不到）。
#
# 用法：
#   ./build-local-slim.sh              # directune-mcts:slim（含 STRIP）
#   STRIP=0 ./build-local-slim.sh      # 不剥 AKG 栈（要 gen_mode=akg fallback 时）
#   SRC=directune-mcts:latest ./build-local-slim.sh
set -euo pipefail
cd "$(dirname "$0")"

SRC="${SRC:-directune-mcts:latest}"
TAG="${TAG:-directune-mcts:slim}"
STRIP="${STRIP:-1}"

if ! docker image inspect "$SRC" > /dev/null 2>&1; then
  echo "ERROR: source image $SRC not found — run ./build.sh first" >&2
  exit 1
fi
if ! docker image inspect nvidia/cuda:12.2.0-base-ubuntu22.04 > /dev/null 2>&1; then
  echo "ERROR: nvidia/cuda:12.2.0-base-ubuntu22.04 not cached locally" >&2
  exit 1
fi

docker build -t "$TAG" -f - . <<EOF
# syntax=docker/dockerfile:1
FROM $SRC AS src
USER root
RUN if [ "$STRIP" = "1" ]; then \
      pip uninstall -y --quiet \
        langchain langchain-community langchain-openai langgraph \
        transformers tokenizers safetensors huggingface-hub \
        tiktoken networkx tenacity matplotlib scipy || true ; \
      rm -rf /opt/conda/pkgs /root/.cache /home/dtuser/.cache || true ; \
    fi

FROM nvidia/cuda:12.2.0-base-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini rsync ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# conda env is self-contained at the same path; CUDA runtime libs ride inside
# the env (torch bundles cudart/cublas/cudnn), the driver comes from the host
# via nvidia-container-toolkit. Triton compiles with its wheel-bundled ptxas.
COPY --from=src /opt/conda /opt/conda
COPY --from=src /app /app

ENV PATH=/opt/conda/bin:\$PATH \\
    LD_LIBRARY_PATH=/opt/conda/lib

ARG UID=1000
ARG GID=1000
RUN useradd -m -u \${UID} -g \${GID} dtuser 2>/dev/null || useradd -m -u \${UID} dtuser
USER dtuser
WORKDIR /app

# smoke: main pipeline import graph must survive the strip (AKG lazily degrades)
RUN python -c "import main, agents, mcts, generator, search; print('imports ok (slim)')"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
EOF

echo "Built $TAG"
docker images "$TAG" --format '{{.Repository}}:{{.Tag}}  {{.Size}}'
