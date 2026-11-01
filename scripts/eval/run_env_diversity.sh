
set -euo pipefail

REPO_DIR="${JAXBORG_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}}"
cd "${REPO_DIR}"
source scripts/jax_env.sh

if (( $# < 2 )); then
    echo "Usage: $0 BASELINE_RECIPE DIVERSE_RECIPE [evaluation options]" >&2
    exit 2
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
exec uv run --with "jax-cuda12-plugin[with-cuda]==0.10.2" \
    python scripts/eval/eval_env_diversity.py "$@"
