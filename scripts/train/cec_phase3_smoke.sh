#!/usr/bin/env bash
# CEC Phase 3 SMOKE: 2 arms × 1 seed × 1.5M steps to validate wiring before
# the 6-run matrix.
#
# Arms:
#   * gen-mission-10x-hidden  — 10x bank, OBS_MISSION_GOAL=false (control)
#   * gen-mission-10x-visible — 10x bank, OBS_MISSION_GOAL=true  (treatment)
#
# Both arms drop comms (Phase 2 found message head collapses at 20M).
#
# Usage: bash scripts/train/cec_phase3_smoke.sh [output_root]

set -euo pipefail

OUT_ROOT="${1:-${JAXBORG_EXP_DIR:-./jaxborg-exp}/cec_phase3_smoke}"
OUT_ROOT=$(cd "$(dirname "$OUT_ROOT")" && pwd)/$(basename "$OUT_ROOT")
TIMESTEPS="${TIMESTEPS:-1500000}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEED="${SEED:-1}"
ARMS=(gen-mission-10x-hidden gen-mission-10x-visible)

mkdir -p "$OUT_ROOT"

arm_overrides() {
    case "$1" in
        gen-mission-10x-hidden)
            echo "TOPOLOGY_FIXED_KEY=null VARY_ROUTER_LINKS=false VARY_PHASE_REWARDS=false VARY_MISSION_PROFILE=true BLUE_COMMS=false OBS_MISSION_GOAL=false"
            ;;
        gen-mission-10x-visible)
            echo "TOPOLOGY_FIXED_KEY=null VARY_ROUTER_LINKS=false VARY_PHASE_REWARDS=false VARY_MISSION_PROFILE=true BLUE_COMMS=false OBS_MISSION_GOAL=true"
            ;;
        *) echo "unknown arm: $1" >&2; exit 1 ;;
    esac
}

for arm in "${ARMS[@]}"; do
    save_dir="$OUT_ROOT/$arm/seed$SEED"
    mkdir -p "$save_dir"
    overrides=$(arm_overrides "$arm")
    logfile="$save_dir/train.log"
    echo "[smoke] $arm → $save_dir"
    # shellcheck disable=SC2086
    sbatch \
        --partition=community \
        --gres=gpu:1 \
        --mem=64G \
        --job-name="smoke3_${arm}" \
        --output="$logfile" \
        --wrap="cd $PWD && uv run python scripts/train/ippo_jax.py \
            SEED=$SEED \
            TOTAL_TIMESTEPS=$TIMESTEPS \
            NUM_ENVS=$NUM_ENVS \
            MLFLOW_ENABLED=false \
            SAVE_DIR=$save_dir \
            CHECKPOINT_EVERY_UPDATES=999999 \
            +TAG=cec_phase3_smoke_${arm} \
            $overrides"
done

echo
echo "Queued.  Monitor with: squeue -u \$USER  |  tail -f $OUT_ROOT/*/seed$SEED/train.log"
