#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?Usage: eval_launch.sh submit ROOT TRAINING_TRIAL smoke|interim|primary|final TRANSFER_MANIFEST [DEPENDENCY_JOB]}
if [ "$MODE" = allocated ]; then
  SOURCE_ROOT=${7:?Submitted source checkout required}
else
  SOURCE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
fi
cd "$SOURCE_ROOT"
STUDY_ROOT=$(realpath -e "${2:?Study root required}")
TRIAL=${3:?Training trial required}
STAGE=${4:?Evaluation stage required}
TRANSFER=$(realpath -e "${5:?Transfer input manifest required}")
case "$STAGE" in smoke|interim|primary|final) ;; *) exit 2 ;; esac
case "$TRIAL" in smoke-seed42|fsm-seed42|fsm-seed100) ;; *) exit 2 ;; esac
REVISION=$(git rev-parse HEAD)
test -z "$(git status --porcelain --untracked-files=all)"
case "$STUDY_ROOT/" in "$SOURCE_ROOT/"*) echo 'Outputs must remain outside source' >&2; exit 2;; esac
if [ "$MODE" = submit ]; then
  mkdir -p "$STUDY_ROOT/logs"
  DEADLINE=$(TZ=America/New_York date -d '2026-11-09 08:00:00' +%s)
  REMAINING=$(( (DEADLINE - $(date +%s) - 300) / 60 ))
  [ "$REMAINING" -gt 0 ] || { echo 'Study resource window has ended' >&2; exit 1; }
  LIMIT=360
  [ "$STAGE" != smoke ] || LIMIT=60
  [ "$REMAINING" -ge "$LIMIT" ] || LIMIT=$REMAINING
  EXTRA=()
  [ -z "${6:-}" ] || EXTRA=("--dependency=afterok:$6")
  JOB=$(sbatch --parsable --partition=community --cpus-per-task=8 --mem=32G \
    --time="$LIMIT" --job-name="lstm-eval-$TRIAL" --chdir="$SOURCE_ROOT" \
    --output="$STUDY_ROOT/logs/%j.out" --error="$STUDY_ROOT/logs/%j.err" \
    "${EXTRA[@]}" "$SOURCE_ROOT/studies/lstm-fsm-competence/eval_launch.sh" \
    allocated "$STUDY_ROOT" "$TRIAL" "$STAGE" "$TRANSFER" "$REVISION" "$SOURCE_ROOT")
  printf '%s\n' "$JOB" | tee "$STUDY_ROOT/logs/submitted-eval-$TRIAL-$STAGE.job"
  exit 0
fi
[ "$MODE" = allocated ]
[ "$REVISION" = "${6:?Pinned revision required}" ]
[ -n "${SLURM_JOB_ID:-}" ]
[ "${SLURM_JOB_PARTITION:-}" = community ]
exec 9>"$STUDY_ROOT/.eval-$TRIAL.lock"
flock -n 9 || { echo 'Another milestone evaluation of this trial is already running' >&2; exit 1; }
source "$HOME/jaxborg-exp/shared/store.env"
export MLFLOW_TRACKING_URI="sqlite:///$JAXBORG_EXP_DIR/mlflow.db"
export MLFLOW_EXPERIMENT_NAME=lstm-fsm-competence
export JAXBORG_MLFLOW_EXPERIMENT="$MLFLOW_EXPERIMENT_NAME"
export JAX_PLATFORMS=cpu JAXBORG_EVAL_BATCH_SIZE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$STUDY_ROOT/cpu-compilation-cache}"
DEADLINE=$(TZ=America/New_York date -d '2026-11-09 08:00:00' +%s)
REMAINING=$(( DEADLINE - $(date +%s) - 300 ))
[ "$REMAINING" -gt 0 ] || { echo 'Study resource window has ended' >&2; exit 1; }
exec timeout --signal=TERM "$REMAINING" .venv/bin/python studies/lstm-fsm-competence/evaluate.py \
  --study-root "$STUDY_ROOT" --training-trial "$TRIAL" --stage "$STAGE" \
  --transfer-manifest "$TRANSFER" --expected-revision "$REVISION"
