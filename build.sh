#!/usr/bin/env bash
# build.sh — build the DirecTune-MCTS docker image.
#
# akg_frontend is a symlink to ../DirecTune/akg_frontend on this machine;
# docker build does not follow symlinks outside the context, so we stage a
# self-contained build context in .build/ first.
#
# Usage:
#   ./build.sh                     # image tag: directune-mcts:latest
#   ./build.sh mytag               # image tag: directune-mcts:mytag
#   UID=1000 GID=1000 ./build.sh   # match host user for bind-mounted output/
set -euo pipefail
cd "$(dirname "$0")"

TAG="${1:-latest}"
AKG_REAL="$(readlink -f akg_frontend)"
if [ ! -d "$AKG_REAL/akg_agents" ]; then
    echo "ERROR: akg_frontend symlink broken: $AKG_REAL" >&2
    exit 1
fi

STAGE=.build
rm -rf "$STAGE"
mkdir -p "$STAGE"

# 1) copy project (dockerignore-style excludes enforced by rsync flags).
# --copy-links: dereference symlinks (problems/ and akg_frontend/ point at
# sibling repos on this machine — the image must be self-contained).
rsync -a --copy-links --exclude output --exclude .git --exclude __pycache__ \
      --exclude .backup_pre_deakg --exclude figures --exclude docs \
      --exclude experiments --exclude 'config*.yaml' \
      --exclude 'config.example.yaml' --exclude .akg/settings.json \
      --exclude akg_frontend --exclude direction_stats.json \
      --exclude 'problems/kb_level2/*.pt' --exclude problems/kb_level3 \
      --exclude .build \
      ./ "$STAGE/"

# 2) dereference akg_frontend into the staged context
cp -r "$AKG_REAL" "$STAGE/akg_frontend"
find "$STAGE" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# 3) restore the example config (it is the in-image default config.yaml)
cp config.example.yaml "$STAGE/config.example.yaml"

docker build -f Dockerfile -t "directune-mcts:$TAG" "$STAGE"

echo "Built directune-mcts:$TAG (staged context left in .build/ — delete at will)"
