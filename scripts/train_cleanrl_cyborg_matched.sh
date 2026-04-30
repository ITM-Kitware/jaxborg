#!/usr/bin/env bash
# Pure CybORG CleanRL PPO — uses the cyborg_matched recipe (gamma=0.85, etc).
set -euo pipefail

WORKDIR=/home/local/KHQ/paul.elliott/src/cyber/jaxborg/main
EXP_DIR=/home/local/KHQ/paul.elliott/src/cyber/jaxborg-exp

cd "$WORKDIR"

echo "=== CleanRL PPO on pure CybORG (cyborg_matched recipe) ==="
echo "Start: $(date)"

JAXBORG_EXP_DIR="$EXP_DIR" \
uv run python scripts/train/algorithms/ippo_cyborg.py \
  --recipe cyborg_matched --seed 42 --tag cyborg_matched

echo "=== CleanRL training finished: $(date) ==="
