#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

WANDB_MODE=offline JAXBORG_EXP_DIR=./exp \
  uv run python scripts/train/algorithms/ippo_jax.py \
    --recipe cotraining \
    --seed 0 \
    --num-envs 4 \
    --total-timesteps 4000 \
    --tag cotrain-smoke
