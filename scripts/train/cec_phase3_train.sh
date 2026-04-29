#!/usr/bin/env bash
# CEC Phase 3 training grid: 2 arms × 3 seeds = 6 runs.
#
# Tests whether the agent develops goal-conditional behavior when the security
# goal is both observable (10x amplifier in obs) AND amplified enough to create
# meaningful gradient pressure (10x multiplier vs Phase 2's 3x).
#
# Arms (only one axis varies — visibility):
#   * gen-mission-10x-hidden  — 10x bank, OBS_MISSION_GOAL=false  (control)
#   * gen-mission-10x-visible — 10x bank, OBS_MISSION_GOAL=true   (treatment)
#
# 20M timesteps per run.  At ~2.4k sps × 1024 envs ≈ 90 min/run.
# 4-way A6000 concurrency → ~3h total wall.
#
# Usage:
#   bash scripts/train/cec_phase3_train.sh [output_root]
#
# Output structure:
#   <root>/{gen-mission-10x-hidden,gen-mission-10x-visible}/
#       seed{1,2,3}/checkpoint_final.pkl

set -euo pipefail

OUT_ROOT="${1:-${JAXBORG_EXP_DIR:-./jaxborg-exp}/cec_phase3_20M}"
OUT_ROOT=$(cd "$(dirname "$OUT_ROOT")" && pwd)/$(basename "$OUT_ROOT")
TIMESTEPS="${TIMESTEPS:-20000000}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEEDS=(1 2 3)
ARMS=(gen-mission-10x-hidden gen-mission-10x-visible)

mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"
echo "Timesteps per run: $TIMESTEPS"
echo "Num envs: $NUM_ENVS"
echo "Arms: ${ARMS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $((${#ARMS[@]} * ${#SEEDS[@]}))"
echo

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

launch_one() {
    local arm="$1" seed="$2"
    local save_dir="$OUT_ROOT/$arm/seed$seed"
    if [[ -e "$save_dir/checkpoint_final.pkl" ]]; then
        echo "  [SKIP] $arm/seed$seed already has checkpoint_final.pkl"
        return 0
    fi
    mkdir -p "$save_dir"
    local overrides
    overrides=$(arm_overrides "$arm")
    local logfile="$save_dir/train.log"
    echo "  [QUEUE] $arm/seed$seed → $save_dir"
    # shellcheck disable=SC2086
    sbatch \
        --partition=community \
        --gres=gpu:1 \
        --mem=64G \
        --job-name="cec3_${arm}_s${seed}" \
        --output="$logfile" \
        --wrap="cd $PWD && uv run python scripts/train/ippo_jax.py \
            SEED=$seed \
            TOTAL_TIMESTEPS=$TIMESTEPS \
            NUM_ENVS=$NUM_ENVS \
            MLFLOW_ENABLED=false \
            SAVE_DIR=$save_dir \
            CHECKPOINT_EVERY_UPDATES=999999 \
            +TAG=cec_phase3_${arm}_s${seed} \
            $overrides"
}

for seed in "${SEEDS[@]}"; do
    echo "=== seed=$seed ==="
    for arm in "${ARMS[@]}"; do
        launch_one "$arm" "$seed"
    done
done

echo
echo "Submitted.  Watch with: squeue -u \$USER"
echo "Checkpoints will land at: $OUT_ROOT"
