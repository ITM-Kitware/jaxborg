#!/usr/bin/env bash
# Submit or execute a prespecified checkpoint diagnostic; science runs in its original checkout.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if [[ "${1:-}" == "--allocated" ]]; then
    shift
    : "${SLURM_JOB_ID:?Use a Slurm allocation}"
    [[ "${SLURM_JOB_PARTITION:-community}" == "community" ]] || exit 1
    source scripts/jax_env.sh
    JAX_PLATFORMS=cpu "$ROOT/.venv/bin/python" -m jaxborg.launch verify "$JAXBORG_LAUNCH_RECORD"
    exec env JAX_PLATFORMS=cpu "$ROOT/.venv/bin/python" scripts/experiments/checkpoint_response.py run --manifest "$1" --point "$2"
fi
[[ $# == 2 ]] || { echo "Usage: $0 /absolute/diagnostic-manifest.json point" >&2; exit 1; }
: "${JAXBORG_EXP_DIR:?Set the shared experiment root}"
export JAXBORG_EXPECTED_SHA="$(git rev-parse HEAD)"
export JAXBORG_LAUNCH_RECORD="$(JAX_PLATFORMS=cpu "$ROOT/.venv/bin/python" -m jaxborg.launch pin)"
read -r PARTITION GPUS MEMORY CPUS LIMIT < <(
    "$ROOT/.venv/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1]))["config"]["resources"]; print(r["partition"],r["gpus_per_job"],r["memory_gb"],r["cpus_per_task"],r["time_limit"])' "$1"
)
[[ "$PARTITION" == community && "$GPUS" == 1 ]] || exit 1
export JAXBORG_MLFLOW_EXPERIMENT=stage-c-blue-checkpoint-diagnostic
export JAXBORG_CAMPAIGN="checkpoint-diagnostic-$2"
mkdir -p "$JAXBORG_EXP_DIR/launches/slurm"
exec sbatch --parsable --partition="$PARTITION" --gres="gpu:$GPUS" --mem="${MEMORY}G" --cpus-per-task="$CPUS" --time="$LIMIT" --job-name="$JAXBORG_CAMPAIGN" --chdir="$ROOT" --output="$JAXBORG_EXP_DIR/launches/slurm/%j.out" --error="$JAXBORG_EXP_DIR/launches/slurm/%j.err" "$ROOT/scripts/experiments/submit_checkpoint_response.sh" --allocated "$1" "$2"
