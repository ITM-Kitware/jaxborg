#!/usr/bin/env bash
# Phase 6 Option B + cocktail ablation launcher.
#
# Submits 18 training jobs (6 arms × 3 seeds × 10M timesteps) chained via
# slurm --dependency=afterany so they execute SERIALLY on one GPU slot.
# This avoids crowding GPUs that other users may need.
#
# Arms (each 10M timesteps):
#   C00_10M     — canonical control (no banks)
#   C11_10M     — full env-diversity cocktail (all 4 banks)
#   topo_10M    — ablation: topology bank only
#   miss_10M    — ablation: mission bank only
#   pbound_10M  — ablation: phase-boundary bank only
#   cjewel_10M  — ablation: phase-rewards (crown-jewel) bank only
#
# Estimated wall: ~1.5 hr per run × 18 runs = ~27 hr on 1 GPU.
#
# Usage:
#   ./scripts/train/cec_phase6_optionb_ablation.sh             # submit all
#   ./scripts/train/cec_phase6_optionb_ablation.sh --dry-run   # print only
#   ./scripts/train/cec_phase6_optionb_ablation.sh --arm C00_10M  # one arm
#   ./scripts/train/cec_phase6_optionb_ablation.sh --no-dep    # parallel (no chain)

set -euo pipefail

WORKDIR=$(git rev-parse --show-toplevel)
EXP_DIR="${JAXBORG_EXP_DIR:-$WORKDIR/../../jaxborg-exp}"
SLURM_LOG_DIR="$EXP_DIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

ARMS=("C00_10M" "C11_10M" "topo_10M" "miss_10M" "pbound_10M" "cjewel_10M")
SEEDS=(42 142 242)

DRY=0
ARM_FILTER=""
SEED_FILTER=""
NO_DEP=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --no-dep)  NO_DEP=1; shift ;;
    --arm)     ARM_FILTER="$2"; shift 2 ;;
    --seed)    SEED_FILTER="$2"; shift 2 ;;
    *)         echo "unrecognized arg: $1" >&2; exit 2 ;;
  esac
done

PREV_JID=""
submit_one() {
  local arm="$1"
  local seed="$2"
  local recipe="cec_phase6_${arm}"
  local tag="${recipe}_seed${seed}"
  local dep_args=()
  if [ "$NO_DEP" -eq 0 ] && [ -n "$PREV_JID" ]; then
    dep_args=(--dependency=afterany:"$PREV_JID")
  fi
  local cmd=(
    sbatch
    --parsable
    "${dep_args[@]}"
    --gres=gpu:1
    --mem=64G
    --time=14:00:00
    --partition=community
    --job-name="${tag}"
    --output="${SLURM_LOG_DIR}/${tag}_%j.log"
    --wrap "set -eu
cd ${WORKDIR}
unset JAX_PLATFORMS
JAXBORG_EXP_DIR=${EXP_DIR} uv run --extra cuda python scripts/train/algorithms/ippo_jax.py --recipe ${recipe} --seed ${seed} --tag ${tag}"
  )
  if [ "$DRY" -eq 1 ]; then
    printf '%q ' "${cmd[@]}"; echo
    PREV_JID="DRY"
  else
    local jid
    jid=$("${cmd[@]}")
    echo "submitted $tag → job $jid${PREV_JID:+ (depends on $PREV_JID)}"
    PREV_JID="$jid"
  fi
}

for arm in "${ARMS[@]}"; do
  if [ -n "$ARM_FILTER" ] && [ "$arm" != "$ARM_FILTER" ]; then continue; fi
  for seed in "${SEEDS[@]}"; do
    if [ -n "$SEED_FILTER" ] && [ "$seed" != "$SEED_FILTER" ]; then continue; fi
    submit_one "$arm" "$seed"
  done
done

if [ "$DRY" -eq 0 ]; then
  echo
  echo "Watch with: squeue -u \$USER"
  echo "Logs: $SLURM_LOG_DIR/cec_phase6_*_10M_seed*.log"
fi
