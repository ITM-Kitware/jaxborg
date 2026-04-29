#!/usr/bin/env bash
# Phase 3 eval: 6 checkpoints × 5 testbeds.
#
# Locates each Phase 3 checkpoint under jaxborg-exp/cec_phase3_20M, runs
# cec_phase1_eval.py per ckpt with the standard testbed list, and dumps
# eval_phase3.json next to each ckpt.

set -euo pipefail

ROOT="${1:-jaxborg-exp}"
EPISODES="${EPISODES:-30}"
TESTBEDS="${TESTBEDS:-train,heldout_fsm,per_mission,heldout_unseen,heldout_thinner}"

OUT_ROOT="$(cd "$(dirname "$ROOT")" && pwd)/$(basename "$ROOT")/cec_phase3_eval"
mkdir -p "$OUT_ROOT"

ARMS=(gen-mission-10x-hidden gen-mission-10x-visible)
SEEDS=(1 2 3)

for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ckpt=$(find "$ROOT" -path "*cec_phase3_20M/$arm/seed$seed/checkpoint_final.pkl" 2>/dev/null | head -1)
        if [[ -z "$ckpt" ]]; then
            echo "[MISS] $arm/seed$seed — no checkpoint"
            continue
        fi
        out_dir="$OUT_ROOT/$arm/seed$seed"
        mkdir -p "$out_dir"
        out_json="$out_dir/eval_phase3.json"
        if [[ -f "$out_json" ]]; then
            echo "[SKIP] $arm/seed$seed — eval already done"
            continue
        fi
        log="$out_dir/eval.log"
        echo "[QUEUE] $arm/seed$seed → $out_json"
        sbatch \
            --partition=community \
            --cpus-per-task=2 \
            --mem=24G \
            --job-name="ev3_${arm}_s${seed}" \
            --output="$log" \
            --wrap="cd $PWD && CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run python scripts/eval/cec_phase1_eval.py \
                --checkpoint $ckpt \
                --arm $arm \
                --episodes $EPISODES \
                --testbeds $TESTBEDS \
                --output $out_json"
    done
done

echo
echo "Submitted.  Watch with: squeue -u \$USER"
echo "Eval JSONs land at: $OUT_ROOT/<arm>/seed<N>/eval_phase3.json"
