#!/usr/bin/env bash
# Submit a verified Stage B smoke or bounded serial pilot from a clean launch checkout.
set -euo pipefail
[[ $# == 2 && ( "$1" == "smoke" || "$1" == "run" ) ]] || {
    echo "Usage: $0 <smoke|run> /absolute/manifest.json" >&2; exit 1;
}
: "${JAXBORG_EXP_DIR:?Set the explicit absolute external experiment root}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
export JAXBORG_EXPECTED_SHA="$(git rev-parse HEAD)"
export JAXBORG_LAUNCH_RECORD="$(JAX_PLATFORMS=cpu "$ROOT/.venv/bin/python" -m jaxborg.launch pin)"
export JAXBORG_CAMPAIGN="$("$ROOT/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaign"])' "$2")"
mkdir -p "$JAXBORG_EXP_DIR/launches/slurm"
if [[ "$1" == "run" ]]; then
    # Validate actual smoke evidence before submitting any research training.
    JAX_PLATFORMS=cpu "$ROOT/.venv/bin/python" -c \
        'import importlib.util,sys; s=importlib.util.spec_from_file_location("pilot", "scripts/experiments/oracle_stage_b.py"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); p=m.load_protocol(sys.argv[1]); m.require_smoke(p,sys.argv[1])' "$2"
    LIMIT=12:00:00
else
    LIMIT=01:00:00
fi
exec sbatch --parsable --partition=community --gres=gpu:1 --mem=64G --cpus-per-task=8 \
    --time="$LIMIT" --job-name="stage-b-$1" --chdir="$ROOT" \
    --output="$JAXBORG_EXP_DIR/launches/slurm/%j.out" --error="$JAXBORG_EXP_DIR/launches/slurm/%j.err" \
    "$ROOT/scripts/experiments/allocated_stage_b.sh" "$1" --manifest "$2"
