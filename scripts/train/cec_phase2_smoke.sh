#!/usr/bin/env bash
# CEC Phase 2 SMOKE: 4 arms × 1 seed × 200k steps to validate wiring before
# the 12-run matrix.
#
# Each arm runs as its own srun job on community partition.  Logs land
# under $OUT_ROOT/<arm>/seed1/train.log.
#
# Usage: bash scripts/train/cec_phase2_smoke.sh [output_root]

set -euo pipefail

OUT_ROOT="${1:-${JAXBORG_EXP_DIR:-./jaxborg-exp}/cec_phase2_smoke}"
OUT_ROOT=$(cd "$(dirname "$OUT_ROOT")" && pwd)/$(basename "$OUT_ROOT")
# Default 1.5M = ~3 PPO updates at NUM_ENVS=1024 / NUM_STEPS=500.  200k from
# the plan would round down to 0 updates because of the 500*1024=512k samples-
# per-update floor; 1.5M gives enough updates to verify training runs.
TIMESTEPS="${TIMESTEPS:-1500000}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEED="${SEED:-1}"
ARMS=(gen-fixed-nomsg gen-fixed-msg gen-mission-nomsg gen-mission-msg)

mkdir -p "$OUT_ROOT"

arm_overrides() {
    case "$1" in
        gen-fixed-nomsg)
            echo "TOPOLOGY_FIXED_KEY=0 VARY_ROUTER_LINKS=false VARY_PHASE_REWARDS=false VARY_MISSION_PROFILE=false BLUE_COMMS=false"
            ;;
        gen-fixed-msg)
            echo "TOPOLOGY_FIXED_KEY=0 VARY_ROUTER_LINKS=false VARY_PHASE_REWARDS=false VARY_MISSION_PROFILE=false BLUE_COMMS=true"
            ;;
        gen-mission-nomsg)
            echo "TOPOLOGY_FIXED_KEY=null VARY_ROUTER_LINKS=false VARY_PHASE_REWARDS=false VARY_MISSION_PROFILE=true BLUE_COMMS=false"
            ;;
        gen-mission-msg)
            echo "TOPOLOGY_FIXED_KEY=null VARY_ROUTER_LINKS=false VARY_PHASE_REWARDS=false VARY_MISSION_PROFILE=true BLUE_COMMS=true"
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
        --job-name="smoke_${arm}" \
        --output="$logfile" \
        --wrap="cd $PWD && uv run python scripts/train/ippo_jax.py \
            SEED=$SEED \
            TOTAL_TIMESTEPS=$TIMESTEPS \
            NUM_ENVS=$NUM_ENVS \
            MLFLOW_ENABLED=false \
            SAVE_DIR=$save_dir \
            CHECKPOINT_EVERY_UPDATES=999999 \
            +TAG=cec_phase2_smoke_${arm} \
            $overrides"
done

echo
echo "Queued.  Monitor with: squeue -u \$USER  |  tail -f $OUT_ROOT/*/seed$SEED/train.log"
