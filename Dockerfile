# DirecTune-MCTS — deployable image
# Build:  ./build.sh                (stages a self-contained context in .build/)
# Run:    docker run --gpus all -it --rm \
#           -v $PWD/config.yaml:/app/config.yaml \
#           -v $PWD/output:/app/output \
#           directune-mcts:latest \
#           python main.py --config config.yaml \
#             --problem problems/kb_level1/01_square_matrix_multiplication.json \
#             --initial problems/kb_level1/01_square_matrix_multiplication_initial.py
#
# Requirements on the HOST: NVIDIA driver matching the base image's CUDA
# (>= 525 for CUDA 12.6). Nothing else — no CUDA toolkit, no conda.
#
# NOTE on versions: the dev environment uses torch 2.12 + triton 3.7 (CUDA 13).
# This image builds on the locally cached pytorch 2.6 / CUDA 12.6 base and
# upgrades triton to 3.7 — the pipeline only needs triton >= 3 and an
# OpenAI-compatible API endpoint. If registry access allows, prefer
# pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime for exact version parity.

FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# tini: proper signal handling so `docker stop` interrupts long searches cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini rsync ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (layer cache: source edits don't re-trigger pip install)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project source (akg_frontend is dereferenced into the context by build.sh;
# problems/ ships L1 by default, L2 weights are large: mount or copy as needed)
COPY . .

# Non-secret default config; real config.yaml is mounted over it at runtime.
# config.yaml (with real API keys) never enters the image.
COPY config.example.yaml config.yaml

# Write results as this uid/gid (matches host user to avoid root-owned output/)
ARG UID=1000
ARG GID=1000
RUN useradd -m -u ${UID} -g ${GID} dtuser 2>/dev/null || useradd -m -u ${UID} dtuser \
    && chown -R dtuser /app
USER dtuser

# Smoke-check the import graph at build time
RUN python -c "import main, agents, mcts, generator, search; print('imports ok')"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
