#!/usr/bin/env bash
# Run multiple seeds of a recipe in parallel.
#
# Usage:
#   ./scripts/train/run_seeds.sh <backend:jax|cleanrl> <recipe-name> [num_seeds] [start_seed]
#
# JAX seeds run sequentially (one GPU each via srun queueing).
# CleanRL seeds run in parallel as background bash jobs (no Slurm) — tuned
# so total env workers fit in the 64-core box.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <backend:jax|cleanrl> <recipe-name> [num_seeds=3] [start_seed=42]"
    exit 1
fi

BACKEND="$1"
RECIPE="$2"
NUM_SEEDS="${3:-3}"
START_SEED="${4:-42}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
# Pin all slots to one source/environment before launching any of them.
JAXBORG_EXPECTED_SHA="${JAXBORG_EXPECTED_SHA:-$(git rev-parse HEAD)}"
export JAXBORG_EXPECTED_SHA
JAXBORG_LAUNCH_RECORD="$("$ROOT/.venv/bin/python" -m jaxborg.launch pin)"
export JAXBORG_EXPECTED_SHA JAXBORG_LAUNCH_RECORD

case "$BACKEND" in
    jax)
        for ((i = 0; i < NUM_SEEDS; i++)); do
            SEED=$((START_SEED + i))
            ./scripts/train/run.sh jax "$RECIPE" "$SEED"
        done
        ;;
    cleanrl|cyborg)
        PIDS=()
        for ((i = 0; i < NUM_SEEDS; i++)); do
            SEED=$((START_SEED + i))
            ./scripts/train/run.sh cleanrl "$RECIPE" "$SEED" &
            PIDS+=($!)
        done
        STATUS=0
        for PID in "${PIDS[@]}"; do
            if ! wait "$PID"; then STATUS=1; fi
        done
        exit "$STATUS"
        ;;
    *)
        echo "Unknown backend: $BACKEND"
        exit 1
        ;;
esac
