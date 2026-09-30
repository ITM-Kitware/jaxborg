#!/usr/bin/env bash
# ./scripts/sbatch/run_ippo.sh <recipe> [trainer args...]
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=community
set -euo pipefail
[[ $# -ge 1 ]] || { echo "Usage: $0 <recipe> [trainer args...]" >&2; exit 1; }
RECIPE="$1"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    ROOT="$(git rev-parse --show-toplevel)"
    cd "$ROOT"
    if [[ -z "${JAXBORG_EXPECTED_SHA:-}" ]]; then
        JAXBORG_EXPECTED_SHA="$(git rev-parse HEAD)"
    fi
    export JAXBORG_EXPECTED_SHA
    JAXBORG_LAUNCH_RECORD="$("$ROOT/.venv/bin/python" -m jaxborg.launch pin)"
    export JAXBORG_LAUNCH_RECORD
    export JAXBORG_LAUNCH_CHECKOUT="$ROOT"
    # Slurm startup diagnostics are launch logs. Trainer console/metrics belong
    # to the run and are durable even when no final success callback happens.
    mkdir -p "$JAXBORG_EXP_DIR/launches/slurm"
    exec sbatch --partition=community --chdir="$ROOT" --job-name="jaxborg-ippo" \
        --output="$JAXBORG_EXP_DIR/launches/slurm/%j.log" "$ROOT/scripts/sbatch/run_ippo.sh" "$@"
fi
: "${JAXBORG_LAUNCH_CHECKOUT:?Submit through this wrapper so source is pinned before scheduling}"
: "${JAXBORG_EXPECTED_SHA:?Missing submission revision}"
cd "$JAXBORG_LAUNCH_CHECKOUT"
shift
exec "$JAXBORG_LAUNCH_CHECKOUT/scripts/train/allocated.sh" \
    scripts/train/algorithms/ippo_jax.py --recipe "$RECIPE" "$@"
