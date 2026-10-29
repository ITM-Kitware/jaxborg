#!/usr/bin/env bash
# Phase 6 Test 2 eval orchestrator — runs JAX-native held-out red sweep on CPU.
#
# Per checkpoint × held-out red, runs scripts/eval/cec_phase6_eval_jax.py
# at --episodes 90 (plan default for stat power; --episodes 30 for a smoke
# pass). Each (ckpt, red) job is independent and embarrassingly parallel —
# we submit them as individual CPU sbatch jobs so they fan out across the
# cluster without competing with training/diagnostic GPU jobs.
#
# Total: 6 ckpts × 5 reds = 30 jobs. CPU JAX is mostly bottlenecked on
# JIT compile (~3 min) plus a sub-minute rollout — ~5 min wall per cell,
# ~30 min total if 5+ jobs run concurrently on the cluster.
#
# Usage:
#   ./scripts/eval/cec_phase6_eval_test2.sh                       # full sweep
#   ./scripts/eval/cec_phase6_eval_test2.sh --episodes 30         # smoke pass
#   ./scripts/eval/cec_phase6_eval_test2.sh --dry-run             # print commands
#   ./scripts/eval/cec_phase6_eval_test2.sh --arm C11 --red cia_c # one cell

set -euo pipefail

WORKDIR=$(git rev-parse --show-toplevel)
EXP_DIR="${JAXBORG_EXP_DIR:-$WORKDIR/../../jaxborg-exp}"
SLURM_LOG_DIR="$EXP_DIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

if [ -n "${PHASE6_ARMS:-}" ]; then
  # shellcheck disable=SC2206
  ARMS=($PHASE6_ARMS)
else
  ARMS=("C00" "C11")
fi
SEEDS=(42 142 242)
# RandomSelectRedAgent is a CybORG-side construct; the JAX red-selector
# REGISTRY only has fsm + the CIA-biased reds, so the JAX-native eval skips
# "random" here. To get a random-red noise-floor row, run eval_recipe.py
# (CybORG-side) with --eval-red random instead.
REDS=("fsm" "cia_c" "cia_i" "cia_a")
EPISODES=90
EVAL_SEED=1000
DRY=0
ARM_FILTER=""
RED_FILTER=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --episodes)  EPISODES="$2"; shift 2 ;;
    --seed)      EVAL_SEED="$2"; shift 2 ;;
    --arm)       ARM_FILTER="$2"; shift 2 ;;
    --red)       RED_FILTER="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    *)           echo "unrecognized arg: $1" >&2; exit 2 ;;
  esac
done

submit_one() {
  local tag="$1"
  local red="$2"
  local model="$EXP_DIR/ippo_jax/${tag}/model_${tag}.safetensors"
  if [ ! -f "$model" ]; then
    echo "SKIP $tag vs $red — model not found: $model" >&2
    return
  fi
  local jobname="eval_${tag}_${red}"
  # CPU-only — eval jobs do not need GPU and should not compete with
  # training jobs for GPU allocation. JAX_PLATFORMS=cpu pins the backend.
  local cmd=(
    sbatch
    --cpus-per-task=8
    --mem=32G
    --time=01:30:00
    --partition=community
    --job-name="${jobname}"
    --output="${SLURM_LOG_DIR}/${jobname}_%j.log"
    --wrap "set -eu
cd ${WORKDIR}
JAX_PLATFORMS=cpu JAXBORG_EXP_DIR=${EXP_DIR} uv run python scripts/eval/cec_phase6_eval_jax.py --model ${model} --eval-red ${red} --episodes ${EPISODES} --seed ${EVAL_SEED}"
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
    tag="cec_phase6_${arm}_seed${seed}"
    for red in "${REDS[@]}"; do
      if [ -n "$RED_FILTER" ] && [ "$red" != "$RED_FILTER" ]; then continue; fi
      submit_one "$tag" "$red"
    done
  done
done

if [ "$DRY" -eq 0 ]; then
  echo
  echo "Submitted. After completion, aggregate with:"
  echo "  uv run python scripts/dev/cec_phase6_aggregate.py --eval-dir ${EXP_DIR}/eval"
fi
