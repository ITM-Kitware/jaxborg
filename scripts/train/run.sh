#!/usr/bin/env bash
# Generic recipe launcher.
#
# Usage:
#   ./scripts/train/run.sh <backend:jax|cleanrl> <recipe-name> [seed] [extra-args...]
#
# Reads `algorithm` from the recipe and dispatches to
# scripts/train/algorithms/${algorithm}_${backend}.py.
#
# JAX runs under `srun --gres=gpu:1` (Slurm GPU allocation). CleanRL runs
# as plain bash on CPU (no Slurm — see feedback_no_slurm_for_cpu.md).

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <backend:jax|cleanrl> <recipe-name> [seed] [extra-args...]"
    exit 1
fi

BACKEND="$1"
RECIPE="$2"
SEED="${3:-42}"
shift 3 || shift 2 || true

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

EXP_DIR="${JAXBORG_EXP_DIR:-$(pwd)/jaxborg-exp}"
mkdir -p "$EXP_DIR/logs"
LOG="$EXP_DIR/logs/${RECIPE}_${BACKEND}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

# Resolve algorithm from recipe (the recipe is the source of truth for which
# trainer to invoke).
ALGORITHM=$(uv run python -c "
from jaxborg.recipe import load
print(load('$RECIPE')['algorithm'])
")

SCRIPT="scripts/train/algorithms/${ALGORITHM}_${BACKEND}.py"
if [[ ! -f "$SCRIPT" ]]; then
    echo "No algorithm script for backend=$BACKEND algorithm=$ALGORITHM (expected $SCRIPT)"
    exit 1
fi

echo "=== ${RECIPE} on ${BACKEND} (algo=${ALGORITHM}) seed=${SEED} ==="
echo "Log: $LOG"

case "$BACKEND" in
    jax)
        JAXBORG_EXP_DIR="$EXP_DIR" srun --gres=gpu:1 --mem=64G \
            uv run python "$SCRIPT" \
                --recipe "$RECIPE" --seed "$SEED" "$@" 2>&1 | tee "$LOG"
        ;;
    cleanrl|cyborg)
        JAXBORG_EXP_DIR="$EXP_DIR" \
            uv run python "$SCRIPT" \
                --recipe "$RECIPE" --seed "$SEED" "$@" 2>&1 | tee "$LOG"
        ;;
    *)
        echo "Unknown backend: $BACKEND (expected jax | cleanrl)"
        exit 1
        ;;
esac
