#!/usr/bin/env bash
# CEC Phase 2 training grid: 4 arms × 3 seeds = 12 runs.
#
# 2×2 factorial of (env diversity: gen-fixed vs gen-mission) × (comms: msg vs nomsg).
# All 12 runs train fresh on the new architecture (with message head).  Phase 1
# checkpoints are reference-only and not part of this matrix.
#
# Default: 20M timesteps per run (matches Phase 1 throughput).  3 waves × 4 GPUs
# ≈ 12h wall on 4× A6000.
#
# Usage:
#   bash scripts/train/cec_phase2_train.sh [output_root]
#
# Output structure:
#   <root>/{gen-fixed-nomsg,gen-fixed-msg,gen-mission-nomsg,gen-mission-msg}/
#       seed{1,2,3}/checkpoint_final.pkl

set -euo pipefail

OUT_ROOT="${1:-${JAXBORG_EXP_DIR:-./jaxborg-exp}/cec_phase2_20M}"
# Absolutize so Hydra's chdir=True doesn't move the save dir into the
# timestamped run folder.
OUT_ROOT=$(cd "$(dirname "$OUT_ROOT")" && pwd)/$(basename "$OUT_ROOT")
TIMESTEPS="${TIMESTEPS:-20000000}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEEDS=(1 2 3)
ARMS=(gen-fixed-nomsg gen-fixed-msg gen-mission-nomsg gen-mission-msg)

mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"
echo "Timesteps per run: $TIMESTEPS"
echo "Num envs: $NUM_ENVS"
echo "Arms: ${ARMS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $((${#ARMS[@]} * ${#SEEDS[@]}))"
echo

arm_overrides() {
    # Phase 2 arms differ along two boolean axes:
    #   * VARY_MISSION_PROFILE  — env-diversity (mission family)
    #   * BLUE_COMMS            — comms channel
    # The fixed arms also pin TOPOLOGY_FIXED_KEY=0 to collapse topology.
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
        --job-name="cec2_${arm}_s${seed}" \
        --output="$logfile" \
        --wrap="cd $PWD && uv run python scripts/train/ippo_jax.py \
            SEED=$seed \
            TOTAL_TIMESTEPS=$TIMESTEPS \
            NUM_ENVS=$NUM_ENVS \
            MLFLOW_ENABLED=false \
            SAVE_DIR=$save_dir \
            CHECKPOINT_EVERY_UPDATES=999999 \
            +TAG=cec_phase2_${arm}_s${seed} \
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
