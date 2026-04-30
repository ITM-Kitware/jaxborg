#!/usr/bin/env bash
# Phase 4 ZSC eval: 6 Phase 3 checkpoints x 3 red agents.
#
# Held-out reds (per research-guide feedback): the CEC DV is generalization to
# new partners.  CC4 ships its own held-out FSM-red variants:
#   - fsm        FiniteStateRedAgent (training distribution)
#   - discovery  DiscoveryFSRed      (different priority + transition matrix)
#   - random     RandomSelectRedAgent (uniform over valid actions; floor)
#
# Output: <root>/cec_phase4_zsc/<arm>/seed<N>/<red>.json

set -euo pipefail

ROOT="${1:-jaxborg-exp}"
EPISODES="${EPISODES:-30}"

OUT_ROOT="$(cd "$(dirname "$ROOT")" && pwd)/$(basename "$ROOT")/cec_phase4_zsc"
mkdir -p "$OUT_ROOT"

ARMS=(gen-mission-10x-hidden gen-mission-10x-visible)
SEEDS=(1 2 3)
REDS=(fsm discovery random)

for arm in "${ARMS[@]}"; do
    # Hidden arm trained with mission slots = 0; visible arm with multipliers.
    # For ZSC eval use the default mission profile (1,1,1) for visible — it
    # was sampled equally with the others at training and gives a single
    # "neutral mission" eval point.
    if [[ "$arm" == *"-visible" ]]; then
        MULT="1,1,1"
    else
        MULT="0,0,0"
    fi
    for seed in "${SEEDS[@]}"; do
        ckpt=$(find "$ROOT" -path "*cec_phase3_20M/$arm/seed$seed/checkpoint_final.pkl" 2>/dev/null | head -1)
        if [[ -z "$ckpt" ]]; then
            echo "[MISS] $arm/seed$seed — no checkpoint"
            continue
        fi
        for red in "${REDS[@]}"; do
            out_dir="$OUT_ROOT/$arm/seed$seed"
            mkdir -p "$out_dir"
            out_json="$out_dir/${red}.json"
            log="$out_dir/${red}.log"
            if [[ -f "$out_json" ]]; then
                echo "[SKIP] $arm/seed$seed/$red — already done"
                continue
            fi
            echo "[QUEUE] $arm/seed$seed/$red → $out_json"
            sbatch \
                --partition=community \
                --cpus-per-task=2 \
                --mem=24G \
                --job-name="p4_${red}_${arm}_s${seed}" \
                --output="$log" \
                --wrap="cd $PWD && CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run python scripts/eval/cyborg.py \
                    --checkpoint $ckpt \
                    --episodes $EPISODES \
                    --seed 4000 \
                    --red-agent $red \
                    --mission-multipliers $MULT \
                    --output $out_json"
        done
    done
done

echo
echo "Submitted.  Watch with: squeue -u \$USER"
echo "JSONs land at: $OUT_ROOT/<arm>/seed<N>/<red>.json"
