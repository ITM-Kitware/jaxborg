#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=community
#SBATCH --job-name=jaxborg_cyborg_matched_20M
#SBATCH --output=/home/local/KHQ/paul.elliott/src/cyber/jaxborg-exp/jaxborg_cyborg_matched_20M_%j.log

set -euo pipefail

WORKDIR=/home/local/KHQ/paul.elliott/src/cyber/jaxborg/main
EXP_DIR=/home/local/KHQ/paul.elliott/src/cyber/jaxborg-exp

cd "$WORKDIR"

echo "=== Training JAXborg CYBORG_MATCHED policy (20M steps) ==="
echo "Start: $(date)"

JAXBORG_EXP_DIR="$EXP_DIR" \
uv run python scripts/train/ippo_jax.py \
  TOPOLOGY_MODE=cyborg_bank +TOPOLOGY_BANK_SIZE=32 \
  TOTAL_TIMESTEPS=20000000 \
  NUM_ENVS=1024 UPDATE_EPOCHS=4 NUM_MINIBATCHES=16 \
  ENT_COEF=0.01 LR=3e-4 ANNEAL_LR=false \
  GAMMA=0.85 \
  SEED=42 MLFLOW_ENABLED=false \
  CHECKPOINT_EVERY_UPDATES=10 \
  +NORM_REWARDS=true NETWORK_TYPE=shared BUSY_MASKING=false GRAD_CLIP_MODE=global

echo "Training finished: $(date)"

# Find the most recent checkpoint directory (the one we just created)
CKPT_DIR=$(ls -td "$EXP_DIR"/ippo_cc4_* 2>/dev/null | head -1)
CKPT_FILE="$CKPT_DIR/checkpoint_final.pkl"

if [ ! -f "$CKPT_FILE" ]; then
  echo "ERROR: checkpoint_final.pkl not found in $CKPT_DIR"
  exit 1
fi

echo "=== Evaluating checkpoint: $CKPT_FILE ==="

JAXBORG_EXP_DIR="$EXP_DIR" \
uv run python scripts/eval/transfer.py \
  --checkpoint "$CKPT_FILE" \
  --episodes 10 \
  --baselines \
  --plot

echo "=== Evaluation complete: $(date) ==="
