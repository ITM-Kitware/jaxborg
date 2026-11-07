#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?Usage: launch.sh submit STUDY_ROOT final|early|smoke [ARCHIVE_ROOT]}
if [ "$MODE" = allocated ]; then
  SOURCE_ROOT=${6:?Submitted source checkout required}
else
  SOURCE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
fi
cd "$SOURCE_ROOT"
STUDY_ROOT=$(realpath -m "${2:?Study output root required}")
STAGE=${3:?Stage required: final, early, smoke}
ARCHIVE_ROOT=${4:-/data/shared/jaxborg/jaxborg-separate}
case "$STAGE" in final|early|smoke) ;; *) exit 2 ;; esac
case "$STUDY_ROOT/" in "$SOURCE_ROOT/"*) echo 'Output must be outside the source checkout' >&2; exit 2;; esac
SOURCE_SHA=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  echo 'Study execution requires clean, committed source' >&2
  exit 2
fi
if [ "$MODE" = submit ]; then
  mkdir -p "$STUDY_ROOT/logs"
  JOB_ID=$(sbatch --parsable --partition=community --gres=gpu:1 --cpus-per-task=8 --mem=32G \
    --time=06:00:00 --job-name="defender-$STAGE" --chdir="$SOURCE_ROOT" \
    --output="$STUDY_ROOT/logs/%j.out" --error="$STUDY_ROOT/logs/%j.err" \
    "$SOURCE_ROOT/studies/defender-transfer/launch.sh" allocated "$STUDY_ROOT" "$STAGE" "$ARCHIVE_ROOT" "$SOURCE_SHA" "$SOURCE_ROOT")
  printf '%s\n' "$JOB_ID" | tee "$STUDY_ROOT/logs/submitted-$STAGE.job"
  exit 0
fi
[ "$MODE" = allocated ]
[ -n "${SLURM_JOB_ID:-}" ]
[ "${SLURM_JOB_PARTITION:-}" = community ]
[ "$SOURCE_SHA" = "${5:?Pinned submission source required}" ]
[ -n "${CUDA_VISIBLE_DEVICES:-}" ]
exec 9>"$STUDY_ROOT/.execution.lock"
flock -n 9 || { echo "Another stage holds this study root" >&2; exit 1; }
source "$HOME/jaxborg-exp/shared/store.env"
export MLFLOW_TRACKING_URI="sqlite:///$JAXBORG_EXP_DIR/mlflow.db"
export MLFLOW_EXPERIMENT_NAME=defender-transfer-diagnostic
export JAXBORG_MLFLOW_EXPERIMENT="$MLFLOW_EXPERIMENT_NAME"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAXBORG_EVAL_BATCH_SIZE=16
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="$STUDY_ROOT/compilation-cache"
source scripts/jax_env.sh
.venv/bin/python studies/defender-transfer/execute.py \
  --study-root "$STUDY_ROOT" --archive-root "$ARCHIVE_ROOT" --stage "$STAGE" --expected-revision "$SOURCE_SHA"
