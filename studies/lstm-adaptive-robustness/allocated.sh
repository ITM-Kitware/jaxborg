#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?Study root required}
REVISION=${2:?Source revision required}
KIND=${3:?train or eval required}
TRIAL=${4:?Trial required}
SOURCE_ROOT=${5:?Source checkout required}
WALL_MINUTES=${6:?Allocated minutes required}
cd "$SOURCE_ROOT"
[ "$(git rev-parse HEAD)" = "$REVISION" ]
[ -z "$(git status --porcelain --untracked-files=all)" ]
[ -n "${SLURM_JOB_ID:-}" ]
[ "${SLURM_JOB_PARTITION:-}" = community ]
DEADLINE=$(($(date +%s) + WALL_MINUTES * 60))
if [ "$KIND" = train ]; then
  GPU_DEADLINE=$(date -d '2026-11-09 13:00:00 UTC' +%s)
  [ "$DEADLINE" -le "$GPU_DEADLINE" ] || DEADLINE=$GPU_DEADLINE
fi
REMAINING=$((DEADLINE - $(date +%s) - 180))
[ "$REMAINING" -gt 60 ] || exit 2
source "$HOME/jaxborg-exp/shared/store.env"
export MLFLOW_TRACKING_URI="sqlite:///$JAXBORG_EXP_DIR/mlflow.db"
export MLFLOW_EXPERIMENT_NAME=lstm-adaptive-robustness
export JAXBORG_MLFLOW_EXPERIMENT="$MLFLOW_EXPERIMENT_NAME"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$ROOT/compilation-cache/$KIND"
source scripts/jax_env.sh
case "$KIND" in
  train)
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ]
    export JAX_PLATFORMS=cuda
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
    exec srun --input=none --cpu-bind=cores timeout --signal=INT --kill-after=90 "$REMAINING" \
      .venv/bin/python studies/lstm-adaptive-robustness/run.py \
      --study-root "$ROOT" --expected-revision "$REVISION" --trial "$TRIAL"
    ;;
  eval)
    export JAX_PLATFORMS=cpu JAXBORG_EVAL_BATCH_SIZE=1
    BLUES=(blue42-primary blue42-final blue100-primary blue100-final)
    [ "$TRIAL" != smoke-blue42-primary-red17 ] || BLUES=(blue42-primary)
    STAGES=(primary)
    UPDATES=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["trial_updates"][sys.argv[2]])' "$ROOT/inputs/manifest.json" "$TRIAL")
    if [ "$TRIAL" != smoke-blue42-primary-red17 ] && [ "$UPDATES" -gt 208 ]; then STAGES+=(final); fi
    for STAGE in "${STAGES[@]}"; do
      for BLUE in "${BLUES[@]}"; do
        REMAINING=$((DEADLINE - $(date +%s) - 180))
        [ "$REMAINING" -gt 60 ] || exit 2
        srun --input=none --cpu-bind=cores timeout --signal=INT --kill-after=90 "$REMAINING" \
          .venv/bin/python studies/lstm-adaptive-robustness/evaluate.py \
          --study-root "$ROOT" --expected-revision "$REVISION" --trial "$TRIAL" --blue "$BLUE" --stage "$STAGE"
      done
    done
    ;;
  *) exit 2 ;;
esac
