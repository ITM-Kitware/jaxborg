#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
EXP_DIR="${JAXBORG_EXP_DIR:-${ROOT}/jaxborg-exp}"

# --remote serves the mirror pulled by scripts/sync/pull_runs.sh.
PORT="${MLFLOW_PORT:-5001}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote) EXP_DIR="${ROOT}/remote/jaxborg-exp"; shift ;;
        --port)   PORT="$2"; shift 2 ;;
        *)        break ;;
    esac
done

DB_PATH="$(realpath -m "${EXP_DIR}/mlflow.db")"

if [[ ! -f "${DB_PATH}" ]]; then
    echo "MLflow database not found: ${DB_PATH}" >&2
    echo "Set JAXBORG_EXP_DIR to the experiment directory used for training," >&2
    echo "or pass --remote to serve ./remote/jaxborg-exp (see scripts/sync/pull_runs.sh)." >&2
    exit 1
fi

echo "Serving MLflow results from ${DB_PATH}"
echo "Open http://127.0.0.1:${PORT}"
uv run mlflow ui --backend-store-uri "sqlite:///${DB_PATH}" --port "${PORT}" "$@"
