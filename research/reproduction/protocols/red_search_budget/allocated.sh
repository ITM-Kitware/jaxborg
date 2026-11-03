#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${JAXBORG_LAUNCH_RECORD:?Pinned launch record required}"
STUDY_ROOT=${JAXBORG_STUDY_ROOT:?Set the study directory}
CHECKOUT=${JAXBORG_EVALUATION_CHECKOUT:?Set the pinned evaluation checkout}
cd "$CHECKOUT"
source "${JAXBORG_EXP_DIR:?Set the experiment store}/store.env"
export PYTHONPATH="$CHECKOUT/src:$CHECKOUT"
export JAXBORG_MLFLOW_EXPERIMENT=red-search-budget-20261101
export JAXBORG_CAMPAIGN=red-search-budget-20261101
export JAXBORG_EXPECTED_SHA=6f17b5546bc7fd2773d30d33444f44f1bcbb1b7b
export JAXBORG_SKIP_POST_TRAINING_EVAL=1
export JAX_PLATFORMS=cuda
export JAXBORG_EVAL_BATCH_SIZE=64
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$STUDY_ROOT/provenance/matplotlib"
source scripts/jax_env.sh
"$CHECKOUT/.venv/bin/python" -m jaxborg.launch verify --gpu "$JAXBORG_LAUNCH_RECORD"
remaining=$(( $(date -d '2026-11-02T07:55:00-05:00' +%s) - $(date +%s) ))
if (( remaining <= 0 )); then exit 1; fi
exec timeout --signal=TERM --kill-after=30s "${remaining}s" "$CHECKOUT/.venv/bin/python" "$STUDY_ROOT/study.py" run
