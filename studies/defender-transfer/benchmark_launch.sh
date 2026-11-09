#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?Usage: benchmark_launch.sh submit STUDY_ROOT TRANSFER_MANIFEST NATIVE_ROOT HMARL_MODEL}
if [ "$MODE" = allocated ]; then
  SOURCE_ROOT=${7:?Submitted source checkout required}
else
  SOURCE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
fi
cd "$SOURCE_ROOT"
STUDY_ROOT=$(realpath -m "${2:?A new external study root is required}")
TRANSFER_MANIFEST=$(realpath "${3:?Transfer-study manifest is required}")
NATIVE_ROOT=$(realpath "${4:?Completed native-matched attempt root is required}")
HMARL_MODEL=$(realpath "${5:?Imported H-MARL actors checkpoint is required}")
case "$STUDY_ROOT/" in "$SOURCE_ROOT/"*|"$NATIVE_ROOT/"*) echo 'Output must be outside source and inputs' >&2; exit 2;; esac
SOURCE_SHA=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  echo 'Benchmark requires clean, committed source' >&2
  exit 2
fi
if [ "$MODE" = submit ]; then
  [ ! -e "$STUDY_ROOT" ] || { echo 'Use a new study root; existing attempts are never resumed automatically' >&2; exit 2; }
  mkdir -p "$STUDY_ROOT/logs"
  JOB_ID=$(sbatch --parsable --partition=community --gres=gpu:1 --cpus-per-task=8 --mem=32G \
    --time=02:00:00 --job-name=defender-benchmark --chdir="$SOURCE_ROOT" \
    --output="$STUDY_ROOT/logs/%j.out" --error="$STUDY_ROOT/logs/%j.err" \
    "$SOURCE_ROOT/studies/defender-transfer/benchmark_launch.sh" allocated \
    "$STUDY_ROOT" "$TRANSFER_MANIFEST" "$NATIVE_ROOT" "$HMARL_MODEL" "$SOURCE_SHA" "$SOURCE_ROOT")
  printf '%s\n' "$JOB_ID" | tee "$STUDY_ROOT/logs/submitted.job"
  exit 0
fi
[ "$MODE" = allocated ]
[ -n "${SLURM_JOB_ID:-}" ]
[ "${SLURM_JOB_PARTITION:-}" = community ]
[ "$SOURCE_SHA" = "${6:?Pinned submission source required}" ]
[ -n "${CUDA_VISIBLE_DEVICES:-}" ]
exec 9>"$STUDY_ROOT/.execution.lock"
flock -n 9 || { echo 'Another execution holds this study root' >&2; exit 1; }
source "$HOME/jaxborg-exp/shared/store.env"
export MLFLOW_TRACKING_URI="sqlite:///$JAXBORG_EXP_DIR/mlflow.db"
export MLFLOW_EXPERIMENT_NAME=defender-matched-system-benchmark
export JAXBORG_MLFLOW_EXPERIMENT="$MLFLOW_EXPERIMENT_NAME"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="$STUDY_ROOT/compilation-cache"
source scripts/jax_env.sh
.venv/bin/python studies/defender-transfer/benchmark.py execute \
  --study-root "$STUDY_ROOT" --transfer-manifest "$TRANSFER_MANIFEST" \
  --native-root "$NATIVE_ROOT" --hmarl-model "$HMARL_MODEL" --expected-revision "$SOURCE_SHA"
