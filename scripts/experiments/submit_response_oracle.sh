#!/usr/bin/env bash
# Submit a YAML-configured response campaign from a clean launch checkout.
set -euo pipefail
[[ $# == 2 && ( "$1" == "smoke" || "$1" == "run" || "$1" == "pilot" ) ]] || {
    echo "Usage: $0 <smoke|run|pilot> /absolute/manifest.json" >&2; exit 1;
}
: "${JAXBORG_EXP_DIR:?Set the explicit absolute external experiment root}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
export JAXBORG_EXPECTED_SHA="$(git rev-parse HEAD)"
export JAXBORG_LAUNCH_RECORD="$(JAX_PLATFORMS=cpu "$ROOT/.venv/bin/python" -m jaxborg.launch pin)"
export JAXBORG_CAMPAIGN="$("$ROOT/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaign"])' "$2")"
read -r PARTITION GPUS MEMORY CPUS RUN_LIMIT SMOKE_LIMIT JAXBORG_MLFLOW_EXPERIMENT < <(
    "$ROOT/.venv/bin/python" -c \
      'import json,sys; m=json.load(open(sys.argv[1])); r=m["resources"]; print(r["partition"],r["gpus_per_job"],r["memory_gb"],r["cpus_per_task"],r["time_limit"],r["smoke_time_limit"],m["tracking"]["experiment"])' "$2"
)
export JAXBORG_MLFLOW_EXPERIMENT
[[ "$PARTITION" == "community" && "$GPUS" == "1" ]] || { echo "Use one GPU on community" >&2; exit 1; }
mkdir -p "$JAXBORG_EXP_DIR/launches/slurm"
if [[ "$1" == "run" ]]; then
    # Validate actual smoke evidence before submitting any research training.
    JAX_PLATFORMS=cpu "$ROOT/.venv/bin/python" -c \
        'import importlib.util,sys; s=importlib.util.spec_from_file_location("pilot", "scripts/experiments/response_oracle.py"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); p=m.load_protocol(sys.argv[1]); m.require_smoke(p,sys.argv[1])' "$2"
    LIMIT="$RUN_LIMIT"
elif [[ "$1" == "pilot" ]]; then
    # The allocated controller must pass this source's smoke before research training.
    LIMIT="$RUN_LIMIT"
else
    LIMIT="$SMOKE_LIMIT"
fi
DEPENDENCY=()
if [[ -n "${JAXBORG_SLURM_DEPENDENCY:-}" ]]; then
    DEPENDENCY=(--dependency="$JAXBORG_SLURM_DEPENDENCY")
fi
exec sbatch --parsable --partition="$PARTITION" --gres="gpu:$GPUS" --mem="${MEMORY}G" --cpus-per-task="$CPUS" \
    "${DEPENDENCY[@]}" \
    --time="$LIMIT" --job-name="$JAXBORG_CAMPAIGN" --chdir="$ROOT" \
    --output="$JAXBORG_EXP_DIR/launches/slurm/%j.out" --error="$JAXBORG_EXP_DIR/launches/slurm/%j.err" \
    "$ROOT/scripts/experiments/allocated_response_oracle.sh" "$1" --manifest "$2"
