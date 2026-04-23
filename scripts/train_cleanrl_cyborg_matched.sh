#!/usr/bin/env bash
# Pure CybORG CleanRL PPO — aligned to JAXborg CYBORG_MATCHED hyperparams
set -euo pipefail

WORKDIR=/home/local/KHQ/paul.elliott/src/cyber/jaxborg/main
EXP_DIR=/home/local/KHQ/paul.elliott/src/cyber/jaxborg-exp

cd "$WORKDIR"

echo "=== CleanRL PPO on pure CybORG (CYBORG_MATCHED hyperparams) ==="
echo "Start: $(date)"

JAXBORG_EXP_DIR="$EXP_DIR" \
uv run python scripts/train/ppo_cleanrl_cyborg.py \
  --total-timesteps 20000000 \
  --num-envs 48 \
  --rollout-length 500 \
  --lr 3e-4 \
  --gamma 0.85 \
  --gae-lambda 0.95 \
  --num-epochs 4 \
  --num-minibatches 16 \
  --clip-coef 0.2 \
  --ent-coef 0.01 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --norm-rewards \
  --no-anneal-lr \
  --seed 42 \
  --tag cyborg_matched \
  --checkpoint-every 5000000

echo "=== CleanRL training finished: $(date) ==="
