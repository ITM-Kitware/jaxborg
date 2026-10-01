#!/usr/bin/env bash
# One community GPU, bounded YAML protocol, immutable checkout and explicit Python.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
: "${JAXBORG_DIAGNOSTIC_PYTHON:?Set the compatible GPU Python executable}"
: "${JAXBORG_EXP_DIR:?Set the existing shared MLflow root}"
export PYTHONPATH="$ROOT/src:$ROOT"
if [[ "${1:-}" == --allocated ]]; then
    shift
    : "${SLURM_JOB_ID:?Use Slurm}"
    [[ "$SLURM_JOB_PARTITION" == community ]] || exit 1
    [[ "$(git rev-parse HEAD)" == "$JAXBORG_EXPECTED_SHA" ]] || exit 1
    [[ -z "$(git status --porcelain)" ]] || exit 1
    source scripts/jax_env.sh
    export JAX_PLATFORMS=cuda
    if [[ "${1:-}" == --signal-forks ]]; then
        shift
        exec "$JAXBORG_DIAGNOSTIC_PYTHON" -m scripts.experiments.blue_learning_signal_forks \
            --config "$1" --input-run "$2" --output "$3"
    fi
    exec "$JAXBORG_DIAGNOSTIC_PYTHON" -m scripts.experiments.blue_learning_mechanism \
        --config "$1" --data-root "$2" --output "$3" --reference-repository "$4" "${@:5}"
fi
if [[ "${1:-}" == --signal-forks ]]; then
    [[ $# == 4 ]] || { echo "Usage: $0 --signal-forks config.yaml input-run output" >&2; exit 1; }
    CONFIG="$2"
else
    [[ $# == 4 || $# == 6 ]] || { echo "Usage: $0 config.yaml data-root output reference-repository [--resume-from directory]" >&2; exit 1; }
    CONFIG="$1"
fi
[[ -z "$(git status --porcelain)" ]] || exit 1
export JAXBORG_EXPECTED_SHA="$(git rev-parse HEAD)"
export JAXBORG_MLFLOW_EXPERIMENT=stage-c-blue-learning-mechanism
read -r PARTITION GPUS MEMORY CPUS LIMIT < <(
    "$JAXBORG_DIAGNOSTIC_PYTHON" -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1]))["resources"]; print(c["partition"],c["gpus"],c["memory_gb"],c["cpus"],c["time_limit"])' "$CONFIG"
)
[[ "$PARTITION" == community && "$GPUS" == 1 ]] || exit 1
mkdir -p "$JAXBORG_EXP_DIR/launches/slurm"
exec sbatch --parsable --partition="$PARTITION" --gres="gpu:$GPUS" --mem="${MEMORY}G" \
    --cpus-per-task="$CPUS" --time="$LIMIT" --job-name=blue-learning-mechanism \
    --chdir="$ROOT" --output="$JAXBORG_EXP_DIR/launches/slurm/%j.out" \
    --error="$JAXBORG_EXP_DIR/launches/slurm/%j.err" "$ROOT/scripts/experiments/submit_blue_learning_mechanism.sh" --allocated "$@"
