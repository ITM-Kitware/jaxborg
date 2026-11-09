#!/bin/bash -l
set -euo pipefail
umask 0002
study_root="$1"
study_checkout="$2"
study_python="$3"
study_revision="$4"
study_phase="$5"
study_blue="${6:-}"
export JAXBORG_EXP_DIR="$study_root"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
export MLFLOW_EXPERIMENT_NAME=jaxborg/observation-value-20261109
export UV_CACHE_DIR=/data/users/paul.elliott/.cache/uv/
export JAX_COMPILATION_CACHE_DIR=/data/users/paul.elliott/.cache/jaxborg/xla/
export TMPDIR="$study_root/tmp/$SLURM_JOB_ID"
export JAX_PLATFORMS=cpu
export XLA_FLAGS=""
export JAXBORG_EVAL_BATCH_SIZE=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$study_checkout/src"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$JAX_COMPILATION_CACHE_DIR"
cd "$study_checkout"
exec "$study_python" studies/observation-value/worker.py "$study_phase" \
  --root "$study_root" --revision "$study_revision" --blue "$study_blue"
