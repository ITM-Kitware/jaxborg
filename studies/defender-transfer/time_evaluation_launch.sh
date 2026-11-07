#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?Usage: time_evaluation_launch.sh submit ROOT BENCHMARK_MANIFEST}
if [ "$MODE" = allocated ]; then
  SOURCE_ROOT=${5:?Submitted source checkout required}
else
  SOURCE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
fi
cd "$SOURCE_ROOT"
ROOT=$(realpath -m "${2:?New external result root required}")
MANIFEST=$(realpath "${3:?Existing matched benchmark manifest required}")
case "$ROOT/" in "$SOURCE_ROOT/"*|"$(dirname "$MANIFEST")/"*) exit 2;; esac
SOURCE_SHA=$(git rev-parse HEAD)
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo 'Clean committed source required' >&2; exit 2; }
if [ "$MODE" = submit ]; then
  [ ! -e "$ROOT" ] || { echo 'Use a new result root' >&2; exit 2; }
  mkdir -p "$ROOT/logs"
  JOB=$(sbatch --parsable --partition=community --gres=gpu:1 --cpus-per-task=8 --mem=32G \
    --time=00:30:00 --job-name=eval-cpu-gpu-timing --chdir="$SOURCE_ROOT" \
    --output="$ROOT/logs/%j.out" --error="$ROOT/logs/%j.err" \
    "$SOURCE_ROOT/studies/defender-transfer/time_evaluation_launch.sh" allocated \
    "$ROOT" "$MANIFEST" "$SOURCE_SHA" "$SOURCE_ROOT")
  printf '%s\n' "$JOB" | tee "$ROOT/logs/submitted.job"
  exit 0
fi
[ "$MODE" = allocated ]
[ -n "${SLURM_JOB_ID:-}" ]
[ "${SLURM_JOB_PARTITION:-}" = community ]
[ "$SOURCE_SHA" = "${4:?Pinned source required}" ]
[ -n "${CUDA_VISIBLE_DEVICES:-}" ]
exec 9>"$ROOT/.execution.lock"
flock -n 9 || exit 1
source "$HOME/jaxborg-exp/shared/store.env"
export MLFLOW_TRACKING_URI="sqlite:///$JAXBORG_EXP_DIR/mlflow.db"
export MLFLOW_EXPERIMENT_NAME=defender-eval-throughput
export JAXBORG_MLFLOW_EXPERIMENT="$MLFLOW_EXPERIMENT_NAME"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source scripts/jax_env.sh
srun --ntasks=1 --cpu-bind=cores .venv/bin/python studies/defender-transfer/time_evaluation.py execute \
  --root "$ROOT" --benchmark-manifest "$MANIFEST" --expected-revision "$SOURCE_SHA"
