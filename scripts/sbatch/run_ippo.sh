#!/usr/bin/env bash
# Generic JAX IPPO runner — picks experiment via Hydra overlay.
# Self-submits to slurm if not already running under it.
#
# Usage:
#   ./scripts/sbatch/run_ippo.sh <experiment> [hydra_overrides...]
#   ./scripts/sbatch/run_ippo.sh pure_g99
#   ./scripts/sbatch/run_ippo.sh pure_g99 SEED=7
#
# Override defaults via env vars (slurm reads these automatically):
#   SBATCH_TIMELIMIT=24:00:00 ./scripts/sbatch/run_ippo.sh pure_g99
#   SBATCH_GRES=gpu:2         ./scripts/sbatch/run_ippo.sh pure_g99
#
# Logs: $JAXBORG_EXP_DIR/slurm/<experiment>_<jobid>.log

#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=community

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <experiment-name> [hydra_overrides...]" >&2
  exit 1
fi

EXPERIMENT="$1"
WORKDIR=$(git rev-parse --show-toplevel)
EXP_DIR="${JAXBORG_EXP_DIR:-$WORKDIR/../jaxborg-exp}"

# Self-submit: if not yet under slurm, resubmit via sbatch with computed --output
if [ -z "${SLURM_JOB_ID:-}" ]; then
  mkdir -p "$EXP_DIR/slurm"
  exec sbatch \
    --job-name="$EXPERIMENT" \
    --output="$EXP_DIR/slurm/${EXPERIMENT}_%j.log" \
    "$0" "$@"
fi

# Running under slurm — do the work
shift
cd "$WORKDIR"

echo "=== JAX IPPO: experiment=$EXPERIMENT (job $SLURM_JOB_ID) ==="
echo "Workdir: $WORKDIR"
echo "Log:     $EXP_DIR/slurm/${EXPERIMENT}_${SLURM_JOB_ID}.log"
echo "Start:   $(date)"

JAXBORG_EXP_DIR="$EXP_DIR" \
  uv run python scripts/train/ippo_jax.py "+experiment=$EXPERIMENT" "$@"

echo "=== Finished: $(date) ==="
