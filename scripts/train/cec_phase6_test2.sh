#!/usr/bin/env bash
# Phase 6 Test 2 launcher — submits 6 sbatch jobs:
#   2 arms (C00 control, C11 full env-diversity cocktail) × 3 seeds (42, 142, 242)
#
# Each job runs ippo_jax with --extra cuda so JAX uses the allocated A6000.
# Total wall: ~1.5 hr per run × 6 runs / 3 GPUs = ~3 hr (parallel) / ~9 hr (serial).
#
# Usage:
#   ./scripts/train/cec_phase6_test2.sh                # submit all 6
#   ./scripts/train/cec_phase6_test2.sh --dry-run      # print sbatch commands without submitting
#   ./scripts/train/cec_phase6_test2.sh C00 42         # submit one specific run
#   ./scripts/train/cec_phase6_test2.sh C11            # submit only C11 × all seeds

set -euo pipefail

WORKDIR=$(git rev-parse --show-toplevel)
EXP_DIR="${JAXBORG_EXP_DIR:-$WORKDIR/../../jaxborg-exp}"
SLURM_LOG_DIR="$EXP_DIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

ARMS=("C00" "C11")
SEEDS=(42 142 242)

DRY=0
ARM_FILTER=""
SEED_FILTER=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    C00|C11)   ARM_FILTER="$arg" ;;
    [0-9]*)    SEED_FILTER="$arg" ;;
    *)         echo "unrecognized arg: $arg" >&2; exit 2 ;;
  esac
done

submit_one() {
  local arm="$1"
  local seed="$2"
  local recipe="cec_phase6_${arm}"
  local tag="${recipe}_seed${seed}"
  local cmd=(
    sbatch
    --gres=gpu:1
    --mem=64G
    --time=04:00:00
    --partition=community
    --job-name="${tag}"
    --output="${SLURM_LOG_DIR}/${tag}_%j.log"
    --wrap "set -euo pipefail; cd ${WORKDIR}; unset JAX_PLATFORMS; JAXBORG_EXP_DIR=${EXP_DIR} uv run --extra cuda python scripts/train/algorithms/ippo_jax.py --recipe ${recipe} --seed ${seed} --tag ${tag}"
  )
  if [ "$DRY" -eq 1 ]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
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
  echo "Submitted. Watch with: squeue -u \$USER"
  echo "Logs: $SLURM_LOG_DIR/cec_phase6_*_*.log"
fi
