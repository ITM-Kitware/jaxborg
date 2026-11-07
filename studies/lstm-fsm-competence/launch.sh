#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?Usage: launch.sh submit ROOT smoke|full SEED [MAX_HOURS]}
if [ "$MODE" = allocated ]; then
  SOURCE_ROOT=${7:?Pinned source checkout required}
else
  SOURCE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
fi
cd "$SOURCE_ROOT"
ROOT=$(realpath "${2:?Prepared external study root required}")
PHASE=${3:?smoke or full required}
SEED=${4:?Training seed required}
MAX_HOURS=${5:-36}
SOURCE_SHA=$(git rev-parse HEAD)
case "$ROOT/" in "$SOURCE_ROOT/"*) echo 'Study root must be external' >&2; exit 2;; esac
case "$PHASE:$SEED" in smoke:42|full:42|full:100) ;; *) echo 'Invalid phase or seed' >&2; exit 2;; esac
[[ "$MAX_HOURS" =~ ^[0-9]+$ ]] && [ "$MAX_HOURS" -gt 0 ] && [ "$MAX_HOURS" -le 48 ]
[ "$PHASE" != smoke ] || MAX_HOURS=1
.venv/bin/python studies/lstm-fsm-competence/prepare.py verify --study-root "$ROOT" --expected-revision "$SOURCE_SHA"
DEADLINE=$(date -d '2026-11-09 13:00:00 UTC' +%s)
REMAINING=$((DEADLINE - $(date +%s) - 600))
LIMIT=$((MAX_HOURS * 3600))
[ "$REMAINING" -ge 600 ] || { echo 'Insufficient time before the fixed deadline' >&2; exit 2; }
[ "$REMAINING" -ge "$LIMIT" ] || LIMIT=$REMAINING
if [ "$MODE" = submit ]; then
  if [ "$PHASE" = full ]; then
    .venv/bin/python studies/lstm-fsm-competence/validate.py \
      --study-root "$ROOT" --expected-revision "$SOURCE_SHA" --require-cpu-eval
  fi
  TRIAL="fsm-seed$SEED"
  [ "$PHASE" != smoke ] || TRIAL="smoke-seed42"
  [ ! -e "$ROOT/runs/$TRIAL" ] || { echo 'Trial already attempted; use explicit recorded retry procedure' >&2; exit 2; }
  mkdir -p "$ROOT/logs"
  exec 8>"$ROOT/.submission-$TRIAL.lock"
  flock -n 8 || { echo 'Concurrent submission for this trial' >&2; exit 2; }
  [ ! -e "$ROOT/logs/$TRIAL.job" ] || { echo 'Trial already submitted' >&2; exit 2; }
  JOB_ID=$(sbatch --parsable --partition=community --gres=gpu:1 --cpus-per-task=8 --mem=32G \
    --time="$((LIMIT / 60))" --signal=INT@180 --job-name="lstm-$TRIAL" --chdir="$SOURCE_ROOT" \
    --output="$ROOT/logs/%j.out" --error="$ROOT/logs/%j.err" \
    "$SOURCE_ROOT/studies/lstm-fsm-competence/launch.sh" allocated \
    "$ROOT" "$PHASE" "$SEED" "$MAX_HOURS" "$SOURCE_SHA" "$SOURCE_ROOT")
  printf '%s\n' "$JOB_ID" | tee "$ROOT/logs/$TRIAL.job"
  exit 0
fi
[ "$MODE" = allocated ]
[ -n "${SLURM_JOB_ID:-}" ]
[ "${SLURM_JOB_PARTITION:-}" = community ]
[ "$SOURCE_SHA" = "${6:?Pinned submission source required}" ]
[ -n "${CUDA_VISIBLE_DEVICES:-}" ]
source "$HOME/jaxborg-exp/shared/store.env"
export MLFLOW_TRACKING_URI="sqlite:///$JAXBORG_EXP_DIR/mlflow.db"
export MLFLOW_EXPERIMENT_NAME=lstm-fsm-competence
export JAXBORG_MLFLOW_EXPERIMENT="$MLFLOW_EXPERIMENT_NAME"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="$ROOT/compilation-cache/$PHASE-$SEED"
source scripts/jax_env.sh
timeout --signal=INT --kill-after=90 "$((LIMIT - 180))" \
  .venv/bin/python studies/lstm-fsm-competence/run.py --study-root "$ROOT" \
    --phase "$PHASE" --seed "$SEED" --expected-revision "$SOURCE_SHA"
