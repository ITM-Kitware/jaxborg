#!/usr/bin/env bash
# Phase 3 action-distribution probe: 6 checkpoints × 120 episodes.
#
# Re-uses the Phase 2 probe script (cec_phase2_action_probe.py) — no code
# change needed, just a different checkpoint set and bumped episode count
# (Phase 2 used 30, which left only ~3 CI-heavy episodes per probe; 120
# gives ~30 per profile so per-class spread variance is tight).
#
# Output: <root>/cec_phase3_probe/<arm>-seed<N>.log per checkpoint.

set -euo pipefail

ROOT="${1:-jaxborg-exp}"
EPISODES="${EPISODES:-120}"

OUT_ROOT="$(cd "$(dirname "$ROOT")" && pwd)/$(basename "$ROOT")/cec_phase3_probe"
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
        log="$OUT_ROOT/${arm}-seed${seed}.log"
        if [[ -f "$log" ]] && grep -q "total L1 spread" "$log" 2>/dev/null; then
            echo "[SKIP] $arm/seed$seed — probe already done"
            continue
        fi
        echo "[QUEUE] $arm/seed$seed → $log"
        sbatch \
            --partition=community \
            --cpus-per-task=2 \
            --mem=24G \
            --job-name="pr3_${arm}_s${seed}" \
            --output="$log" \
            --wrap="cd $PWD && CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run python scripts/eval/cec_phase2_action_probe.py \
                --checkpoint $ckpt \
                --label $arm/seed$seed \
                --episodes $EPISODES"
    done
done

echo
echo "Submitted.  Watch with: squeue -u \$USER"
echo "Logs land at: $OUT_ROOT/<arm>-seed<N>.log"
