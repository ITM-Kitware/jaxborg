#!/usr/bin/env bash
# Evaluate every CEC Phase 1 checkpoint on all 3 testbeds.
#
# Usage:
#   bash scripts/eval/cec_phase1_eval_all.sh <train_root>
#
# Reads checkpoints under <train_root>/{arm}/seed{N}/checkpoint_final.pkl and
# writes evaluation JSON files alongside them as eval_phase1.json.

set -euo pipefail

TRAIN_ROOT="${1:-${JAXBORG_EXP_DIR:-./jaxborg-exp}/cec_phase1}"
EPISODES="${EPISODES:-30}"
ARMS=(gen-fixed gen-base gen-router)
SEEDS=(0 1 2)

if [[ ! -d "$TRAIN_ROOT" ]]; then
    echo "Train root not found: $TRAIN_ROOT" >&2
    exit 1
fi

for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ckpt="$TRAIN_ROOT/$arm/seed$seed/checkpoint_final.pkl"
        out="$TRAIN_ROOT/$arm/seed$seed/eval_phase1.json"
        if [[ ! -f "$ckpt" ]]; then
            echo "  [MISS] $ckpt"
            continue
        fi
        if [[ -f "$out" ]]; then
            echo "  [SKIP] $out exists"
            continue
        fi
        echo "=== eval $arm/seed$seed ==="
        srun --gres=gpu:1 --mem=64G -- \
            uv run python scripts/eval/cec_phase1_eval.py \
                --checkpoint "$ckpt" \
                --arm "$arm" \
                --episodes "$EPISODES" \
                --output "$out"
    done
done

echo
echo "All evals complete. Results under $TRAIN_ROOT"
ls "$TRAIN_ROOT"/*/seed*/eval_phase1.json 2>/dev/null
