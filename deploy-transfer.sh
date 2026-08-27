#!/usr/bin/env bash
# deploy-transfer.sh — 把 directune-mcts 镜像传到 JumpServer 管理的目标机。
#
# 三种通道（按优先级试）：
#   stream  docker save | zstd | ssh 直接灌进目标 docker（无中间文件，最快）
#   file    落盘压缩分卷（默认 1GB/卷），适合 JumpServer 网页上传或有大小限制的中转
#   src     只传源码上下文（~20MB）+ 在目标机上拉基础镜像重建（目标机能连镜像源时最优）
#
# 用法：
#   ./deploy-transfer.sh stream user@target-host
#   ./deploy-transfer.sh file                         # 产出 directune-mcts.tar.zstd.*
#   ./deploy-transfer.sh src user@target-host
#   IMG=directune-mcts:slim ./deploy-transfer.sh ...  # 默认 slim
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-file}"
IMG="${IMG:-directune-mcts:slim}"
DEST="${2:-}"
CHUNK="${CHUNK:-1G}"           # file 模式分卷大小（JumpServer 网页上传限制通常 2-4GB）
LEVEL="${LEVEL:-19}"           # zstd 压缩级别（19 慢但小；赶时间用 3）

case "$MODE" in
  stream)
    [ -n "$DEST" ] || { echo "usage: $0 stream user@host" >&2; exit 1; }
    echo "streaming $IMG -> $DEST (docker load) ..."
    docker save "$IMG" | zstd -T0 -"$LEVEL" \
      | ssh "$DEST" 'zstd -d | docker load'
    echo "done. verify on target: docker images $IMG"
    ;;
  file)
    OUT="directune-mcts.tar.zstd"
    echo "saving + compressing $IMG -> $OUT.$LEVEL 分卷 ($CHUNK) ..."
    docker save "$IMG" | zstd -T0 -"$LEVEL" -o "$OUT"
    zstd -d -c "$OUT" | wc -c > /dev/null   # integrity check
    split -b "$CHUNK" "$OUT" "$OUT." && rm -f "$OUT"
    echo "chunks:"
    ls -lh "$OUT".* | awk '{print "  " $NF "  " $5}'
    cat <<'EOF'
目标机恢复（合并 -> 解压 -> load）：
  cat directune-mcts.tar.zstd.* > directune-mcts.tar.zstd
  zstd -d directune-mcts.tar.zstd | docker load
EOF
    ;;
  src)
    [ -n "$DEST" ] || { echo "usage: $0 src user@host" >&2; exit 1; }
    STAGE=.build
    rm -rf "$STAGE" && mkdir -p "$STAGE"
    AKG_REAL="$(readlink -f akg_frontend)"
    rsync -a --copy-links --exclude output --exclude .git --exclude __pycache__ \
          --exclude figures --exclude docs --exclude experiments \
          --exclude 'config*.yaml' --exclude .akg \
          --exclude 'problems/kb_level2/*.pt' --exclude problems/kb_level3 \
          --exclude .build ./ "$STAGE/"
    cp -r "$AKG_REAL" "$STAGE/akg_frontend"
    cp requirements.txt requirements-slim.txt Dockerfile "$STAGE/"
    echo "transferring context ($(du -sh $STAGE | cut -f1)) -> $DEST ..."
    rsync -az "$STAGE"/ "$DEST":directune-mctx-src/
    cat <<EOF
在目标机上构建（基础镜像走国内源前缀，二选一）：
  ssh $DEST
  cd directune-mctx-src
  docker pull docker.m.daocloud.io/pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime \\
    && docker tag docker.m.daocloud.io/pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime \\
       pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
  docker build -t directune-mcts:slim .
  # slim 依赖版：先 cp requirements-slim.txt requirements.txt 再 build
EOF
    ;;
  *)
    echo "usage: $0 stream|file|src [user@host]" >&2; exit 1 ;;
esac
