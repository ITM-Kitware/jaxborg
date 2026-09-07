#!/usr/bin/env bash
# Materialize the shared finite topology pools used by cotraining and
# cotraining_env_diversity. Existing valid snapshots are verified and reused.
#
# Usage:
#   ./scripts/dev/generate_cotraining_topologies.sh
#   ./scripts/dev/generate_cotraining_topologies.sh --dry-run

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

case "${1:-}" in
  "")
    MATERIALIZE_ARGS=()
    ;;
  --dry-run)
    MATERIALIZE_ARGS=(--dry-run)
    ;;
  -h|--help)
    echo "Usage: $0 [--dry-run]"
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
    ;;
esac

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

echo "Preparing cotraining's 5-train/100-eval topology pools..."
JAX_PLATFORMS=cpu uv run materialize-topologies \
  --recipe cotraining \
  --scope all \
  "${MATERIALIZE_ARGS[@]}"

echo "Preparing cotraining_env_diversity's 100-entry train pool..."
# Seeds 0-4 and the held-out evaluation bank are shared with cotraining, so
# this second command only needs to add/check training seeds 5-99.
JAX_PLATFORMS=cpu uv run materialize-topologies \
  --recipe cotraining_env_diversity \
  --scope train \
  "${MATERIALIZE_ARGS[@]}"

if [ "${1:-}" = "--dry-run" ]; then
  echo "Dry run complete; no topology snapshots were written."
else
  echo "Topology generation complete under .bank_cache/topologies/cotraining/."
fi
