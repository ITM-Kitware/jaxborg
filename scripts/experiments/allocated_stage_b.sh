#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${JAXBORG_LAUNCH_RECORD:?Pin the clean source/environment before submission}"
[[ "${SLURM_JOB_PARTITION:-community}" == "community" ]] || { echo "Use community" >&2; exit 1; }
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
unset JAX_PLATFORMS
source scripts/jax_env.sh
"$ROOT/.venv/bin/python" -m jaxborg.launch verify --gpu "$JAXBORG_LAUNCH_RECORD"
exec "$ROOT/.venv/bin/python" scripts/experiments/oracle_stage_b.py "$@"
