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

echo "=== Training JAXborg cyborg_matched recipe (20M steps) ==="
echo "Start: $(date)"

JAXBORG_EXP_DIR="$EXP_DIR" \
uv run python scripts/train/algorithms/ippo_jax.py \
  --recipe cyborg_matched --seed 42 --tag cyborg_matched

echo "Training finished: $(date)"

CKPT_FILE="$EXP_DIR/ippo_jax/cyborg_matched/model_cyborg_matched.pkl"

if [ ! -f "$CKPT_FILE" ]; then
  echo "ERROR: model_cyborg_matched.pkl not found at $CKPT_FILE"
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
