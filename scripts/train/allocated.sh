#!/usr/bin/env bash
# Internal allocation entrypoint shared by interactive and batch wrappers.
set -euo pipefail
: "${SLURM_JOB_ID:?Must run inside a Slurm allocation}"
: "${JAXBORG_LAUNCH_RECORD:?Pin a launch before scheduling}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
"$ROOT/.venv/bin/python" -m jaxborg.launch verify --gpu "$JAXBORG_LAUNCH_RECORD"
exec "$ROOT/.venv/bin/python" "$@"
