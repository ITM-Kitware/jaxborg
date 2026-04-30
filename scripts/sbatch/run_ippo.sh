#!/usr/bin/env bash
# Generic JAX IPPO runner — dispatches by recipe name.
# Self-submits to slurm if not already running under it.
#
# Usage:
#   ./scripts/sbatch/run_ippo.sh <recipe-name> [extra args to ippo_jax.py...]
#   ./scripts/sbatch/run_ippo.sh singh
#   ./scripts/sbatch/run_ippo.sh singh --seed 7
#
# Override defaults via env vars (slurm reads these automatically):
#   SBATCH_TIMELIMIT=24:00:00 ./scripts/sbatch/run_ippo.sh singh
#   SBATCH_GRES=gpu:2         ./scripts/sbatch/run_ippo.sh singh
#
# Logs: $JAXBORG_EXP_DIR/slurm/<recipe>_<jobid>.log

#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=community

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <recipe-name> [extra args...]" >&2
  exit 1
fi

RECIPE="$1"
WORKDIR=$(git rev-parse --show-toplevel)
# Worktree layout has $WORKDIR two levels above jaxborg-exp/ (e.g.
# .../jaxborg/kernel-split → .../jaxborg-exp); standalone repo has it one
# level up. Prefer the existing one; fall back to the legacy default.
if [ -n "${JAXBORG_EXP_DIR:-}" ]; then
  EXP_DIR="$JAXBORG_EXP_DIR"
elif [ -d "$WORKDIR/../../jaxborg-exp" ]; then
  EXP_DIR="$WORKDIR/../../jaxborg-exp"
else
  EXP_DIR="$WORKDIR/../jaxborg-exp"
fi

# Self-submit: if not yet under slurm, resubmit via sbatch with computed --output
if [ -z "${SLURM_JOB_ID:-}" ]; then
  mkdir -p "$EXP_DIR/slurm"
  exec sbatch \
    --job-name="$RECIPE" \
    --output="$EXP_DIR/slurm/${RECIPE}_%j.log" \
    "$0" "$@"
fi

# Running under slurm — do the work
shift
cd "$WORKDIR"

echo "=== JAX IPPO: recipe=$RECIPE (job $SLURM_JOB_ID) ==="
echo "Workdir: $WORKDIR"
echo "Log:     $EXP_DIR/slurm/${RECIPE}_${SLURM_JOB_ID}.log"
echo "Start:   $(date)"

JAXBORG_EXP_DIR="$EXP_DIR" \
  uv run python scripts/train/algorithms/ippo_jax.py --recipe "$RECIPE" "$@"

echo "=== Finished: $(date) ==="
