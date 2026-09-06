#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
EXP_DIR="${JAXBORG_EXP_DIR:-${ROOT}/jaxborg-exp}"
DB_PATH="$(realpath -m "${EXP_DIR}/mlflow.db")"

if [[ ! -f "${DB_PATH}" ]]; then
    echo "MLflow database not found: ${DB_PATH}" >&2
    echo "Set JAXBORG_EXP_DIR to the experiment directory used for training." >&2
    exit 1
fi

echo "Serving MLflow results from ${DB_PATH}"
echo "Open http://127.0.0.1:5000"
uv run mlflow ui --backend-store-uri "sqlite:///${DB_PATH}" "$@"
