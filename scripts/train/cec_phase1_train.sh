#!/usr/bin/env bash
# CEC Phase 1 (axis B) training grid: 3 arms × 3 seeds.
#
# Runs 3 jobs in parallel via Slurm (1 GPU each), waits for them, then moves
# to next batch. Default: 2M timesteps per run for signal-pilot scale; override
# via TIMESTEPS env var.
#
# Usage:
#   bash scripts/train/cec_phase1_train.sh [output_root]
#
# Output structure:
#   <root>/cec_phase1/{gen-fixed,gen-base,gen-router}/seed{0,1,2}/checkpoint_final.pkl

set -euo pipefail

OUT_ROOT="${1:-${JAXBORG_EXP_DIR:-./jaxborg-exp}/cec_phase1}"
# Defaults: NUM_ENVS=1024 + TIMESTEPS=20M ≈ 39 PPO updates / run.
# Matches matched_v2's update count exactly. With UPDATE_EPOCHS=4 (vs
# matched_v2's 10) per-update wallclock on uncontended A6000 ~3.4 min, so a
# 39-update run ~2.2h. 12 runs / 4 GPUs = 3 batches × 2.2h ≈ ~7h training
# + ~30min eval ≈ ~7.5h end-to-end.
TIMESTEPS="${TIMESTEPS:-20000000}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEEDS=(0 1 2)
ARMS=(gen-fixed gen-base gen-router gen-router-rewards)

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
        gen-fixed)          echo "TOPOLOGY_FIXED_KEY=0 VARY_ROUTER_LINKS=false VARY_PHASE_REWARDS=false" ;;
        gen-base)           echo "TOPOLOGY_FIXED_KEY=null VARY_ROUTER_LINKS=false VARY_PHASE_REWARDS=false" ;;
        gen-router)         echo "TOPOLOGY_FIXED_KEY=null VARY_ROUTER_LINKS=true VARY_PHASE_REWARDS=false" ;;
        gen-router-rewards) echo "TOPOLOGY_FIXED_KEY=null VARY_ROUTER_LINKS=true VARY_PHASE_REWARDS=true" ;;
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
    echo "  [START] $arm/seed$seed → $save_dir (log: $logfile)"
    # shellcheck disable=SC2086
    srun --gres=gpu:1 --mem=32G -- \
        uv run python scripts/train/ippo_jax.py \
            SEED=$seed \
            TOTAL_TIMESTEPS=$TIMESTEPS \
            NUM_ENVS=$NUM_ENVS \
            MLFLOW_ENABLED=false \
            SAVE_DIR=$save_dir \
            CHECKPOINT_EVERY_UPDATES=999999 \
            $overrides \
            >"$logfile" 2>&1 &
    echo $! >>"$OUT_ROOT/.pids"
}

rm -f "$OUT_ROOT/.pids"

for seed in "${SEEDS[@]}"; do
    echo "=== seed=$seed (parallel batch) ==="
    for arm in "${ARMS[@]}"; do
        launch_one "$arm" "$seed"
    done
    echo "  waiting for batch..."
    wait
    echo "  batch complete"
    echo
done

echo "All training runs complete."
echo "Checkpoints under: $OUT_ROOT"
ls -la "$OUT_ROOT"/*/seed*/checkpoint_final.pkl 2>/dev/null | head
