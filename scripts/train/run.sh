#!/usr/bin/env bash
# ./scripts/train/run.sh <jax|cyborg|cleanrl> <recipe> [seed] [trainer args...]
# Use a dedicated clean checkout and prebuilt isolated .venv. Never sync here.
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <jax|cyborg|cleanrl> <recipe> [seed] [trainer args...]" >&2
    exit 1
fi
BACKEND="$1"
RECIPE="$2"
SEED="${3:-42}"
if [[ $# -ge 3 ]]; then shift 3; else shift 2; fi
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
# Resolve and retain the revision/environment before entering the allocation.
if [[ -z "${JAXBORG_EXPECTED_SHA:-}" ]]; then
    JAXBORG_EXPECTED_SHA="$(git rev-parse HEAD)"
fi
export JAXBORG_EXPECTED_SHA
if [[ -z "${JAXBORG_LAUNCH_RECORD:-}" ]]; then
    JAXBORG_LAUNCH_RECORD="$("$PYTHON" -m jaxborg.launch pin)"
else
    "$PYTHON" -m jaxborg.launch verify "$JAXBORG_LAUNCH_RECORD"
fi
export JAXBORG_LAUNCH_RECORD
ALGORITHM="$("$PYTHON" -c 'import sys; from jaxborg.recipe import load; print(load(sys.argv[1])["algorithm"])' "$RECIPE")"
SCRIPT_BACKEND="${BACKEND/cleanrl/cyborg}"
SCRIPT="scripts/train/algorithms/${ALGORITHM}_${SCRIPT_BACKEND}.py"
[[ -f "$SCRIPT" ]] || { echo "Missing trainer: $SCRIPT" >&2; exit 1; }
case "$BACKEND" in
    jax)
        if [[ "${JAXBORG_ALLOW_CPU:-}" == "1" ]]; then
            export JAX_PLATFORMS=cpu
            "$PYTHON" -m jaxborg.launch verify "$JAXBORG_LAUNCH_RECORD"
            exec "$PYTHON" "$SCRIPT" --recipe "$RECIPE" --seed "$SEED" "$@"
        fi
        # Verification and device discovery occur inside the community allocation.
        exec srun --partition=community --gres=gpu:1 --mem=64G \
            "$ROOT/scripts/train/allocated.sh" "$SCRIPT" --recipe "$RECIPE" --seed "$SEED" "$@"
        ;;
    cleanrl|cyborg)
        export JAX_PLATFORMS=cpu
        "$PYTHON" -m jaxborg.launch verify "$JAXBORG_LAUNCH_RECORD"
        exec "$PYTHON" "$SCRIPT" --recipe "$RECIPE" --seed "$SEED" "$@"
        ;;
    *) echo "Unknown backend: $BACKEND" >&2; exit 1 ;;
esac
