#!/usr/bin/env bash
# Phase 4c: train a single topology-only-diversity arm to isolate the
# topology contribution to ZSC against held-out reds.
#
# Same matched conditions as Phase 2 / Phase 3 nomsg arms except:
#   * TOPOLOGY_FIXED_KEY=null      → vary network layout + host count
#   * VARY_MISSION_PROFILE=false   → fix mission to default (1, 1, 1)
#   * BLUE_COMMS=false             → no inter-agent messaging
#
# 1 seed, 20M timesteps, ~2h08m on 1× A6000.

set -euo pipefail

OUT_ROOT="${1:-${JAXBORG_EXP_DIR:-./jaxborg-exp}/cec_phase4c_topology_only}"
OUT_ROOT=$(cd "$(dirname "$OUT_ROOT")" && pwd)/$(basename "$OUT_ROOT")
TIMESTEPS="${TIMESTEPS:-20000000}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEED="${SEED:-1}"
ARM="gen-topology-only-nomsg"

save_dir="$OUT_ROOT/$ARM/seed$SEED"
if [[ -e "$save_dir/checkpoint_final.pkl" ]]; then
    echo "[SKIP] $ARM/seed$SEED already has checkpoint_final.pkl"
    exit 0
fi
mkdir -p "$save_dir"
logfile="$save_dir/train.log"

echo "[QUEUE] $ARM/seed$SEED → $save_dir"
sbatch \
    --partition=community \
    --gres=gpu:1 \
    --mem=128G \
    --job-name="cec4c_${ARM}_s${SEED}" \
    --output="$logfile" \
    --wrap="cd $PWD && uv run python scripts/train/ippo_jax.py \
        SEED=$SEED \
        TOTAL_TIMESTEPS=$TIMESTEPS \
        NUM_ENVS=$NUM_ENVS \
        MLFLOW_ENABLED=false \
        SAVE_DIR=$save_dir \
        CHECKPOINT_EVERY_UPDATES=999999 \
        +TAG=cec_phase4c_${ARM}_s${SEED} \
        TOPOLOGY_FIXED_KEY=null \
        VARY_ROUTER_LINKS=false \
        VARY_PHASE_REWARDS=false \
        VARY_MISSION_PROFILE=false \
        BLUE_COMMS=false"

echo "Submitted.  Watch with: squeue -u \$USER"
echo "Checkpoint will land at: $save_dir/checkpoint_final.pkl"
