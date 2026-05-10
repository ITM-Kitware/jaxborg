#!/usr/bin/env bash
# Phase 6 Test 2 eval orchestrator — runs JAX-native held-out red sweep.
#
# Per checkpoint × held-out red, runs scripts/eval/cec_phase6_eval_jax.py
# at --episodes 90 (plan default for stat power; --episodes 30 for a smoke
# pass). Each (ckpt, red) job is independent and embarrassingly parallel —
# we submit them as individual sbatch jobs so they fan out across the
# cluster.
#
# Total: 6 ckpts × 5 reds = 30 jobs, ~3 min each on JAX GPU = ~1.5 hr serial,
# ~10 min on 6 GPUs in parallel.
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

ARMS=("C00" "C11")
SEEDS=(42 142 242)
REDS=("fsm" "cia_c" "cia_i" "cia_a" "random")
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
  local cmd=(
    sbatch
    --gres=gpu:1
    --mem=32G
    --time=00:30:00
    --partition=community
    --job-name="${jobname}"
    --output="${SLURM_LOG_DIR}/${jobname}_%j.log"
    --wrap "set -euo pipefail; cd ${WORKDIR}; unset JAX_PLATFORMS; JAXBORG_EXP_DIR=${EXP_DIR} uv run --extra cuda python scripts/eval/cec_phase6_eval_jax.py --model ${model} --eval-red ${red} --episodes ${EPISODES} --seed ${EVAL_SEED}"
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
