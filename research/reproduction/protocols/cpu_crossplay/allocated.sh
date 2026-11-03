#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?CPU Slurm allocation required}"
: "${JAXBORG_LAUNCH_RECORD:?Pinned launch record required}"
STUDY_ROOT=${JAXBORG_STUDY_ROOT:?Set the study directory}
CHECKOUT=${JAXBORG_EVALUATION_CHECKOUT:?Set the pinned evaluation checkout}
cd "$CHECKOUT"
source "${JAXBORG_EXP_DIR:?Set the experiment store}/store.env"
export PYTHONPATH="$CHECKOUT/src:$CHECKOUT"
export JAXBORG_MLFLOW_EXPERIMENT=crossplay-100m-125m-150m-cpu-20261102
export JAXBORG_CAMPAIGN=crossplay-100m-125m-150m-cpu-20261102
export JAXBORG_EXPECTED_SHA=6f17b5546bc7fd2773d30d33444f44f1bcbb1b7b
export MLFLOW_TRACKING_URI="sqlite:///$JAXBORG_EXP_DIR/mlflow.db"
export JAX_PLATFORMS=cpu
export JAXBORG_ALLOW_CPU=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAXBORG_EVAL_BATCH_SIZE=64
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$STUDY_ROOT/provenance/matplotlib"
source scripts/jax_env.sh
"$CHECKOUT/.venv/bin/python" -m jaxborg.launch verify "$JAXBORG_LAUNCH_RECORD"
exec "$CHECKOUT/.venv/bin/python" "$STUDY_ROOT/run.py"
