#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
UV_BIN=${UV_BIN:-uv}
"$UV_BIN" sync --frozen --extra cpu
"$UV_BIN" pip install --python .venv/bin/python --no-deps -r studies/defender-transfer/cuda-runtime.txt
