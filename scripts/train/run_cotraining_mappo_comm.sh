#!/usr/bin/env bash
# Train matched single-topology and diverse-topology Blue communication arms.
set -euo pipefail
cd "$(dirname "$0")/../.."
for recipe in cotraining_mappo_comm cotraining_mappo_env_diversity_comm; do
    bash scripts/train/run_seeds.sh jax "$recipe" "${1:-3}" "${2:-42}"
done
