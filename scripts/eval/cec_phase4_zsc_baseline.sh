#!/usr/bin/env bash
# Phase 4 ZSC eval: Phase 2 baseline arms against the same 3 held-out reds.
#
# Adds the missing "diversity OFF" anchor to compare against the Phase 3
# diverse arms.  Phase 2 ckpts live under timestamped subdirs of
# jaxborg-exp/2026-04-27/.../cec_phase2_20M/<arm>/seed<N>/.
#
# Output: <root>/cec_phase4_zsc/<arm>/seed<N>/<red>.json (same layout
# as the Phase 3 launcher; aggregator picks them up by arm name).

set -euo pipefail

ROOT="${1:-jaxborg-exp}"
EPISODES="${EPISODES:-30}"

OUT_ROOT="$(cd "$(dirname "$ROOT")" && pwd)/$(basename "$ROOT")/cec_phase4_zsc"
mkdir -p "$OUT_ROOT"

# Phase 2 arms we care about (no-msg only, to match Phase 3).
ARMS=(gen-fixed-nomsg gen-mission-nomsg)
SEEDS=(1 2 3)
REDS=(fsm discovery random)
# Phase 2 ckpts have 210-dim obs — no padding needed.  --mission-multipliers
# is parsed but ignored when policy input dim == 210.
MULT="0,0,0"

for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ckpt=$(find "$ROOT" -path "*cec_phase2_20M/$arm/seed$seed/checkpoint_final.pkl" 2>/dev/null | head -1)
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
                --job-name="p4b_${red}_${arm}_s${seed}" \
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
