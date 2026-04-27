#!/usr/bin/env bash
# Daemon that watches a CEC Phase 1 grid root and submits an eval slurm job
# for each checkpoint as soon as it appears (and no eval JSON exists yet).
# Exits when all 9 evals are present.
#
# Usage: bash scripts/eval/cec_phase1_eval_daemon.sh <grid_root>

set -euo pipefail
ROOT="${1:?need grid root}"
EPISODES="${EPISODES:-30}"

declare -A SUBMITTED

submit_eval() {
    local arm="$1" seed="$2"
    local ckpt="$ROOT/$arm/seed$seed/checkpoint_final.pkl"
    local out="$ROOT/$arm/seed$seed/eval_phase1.json"
    local key="$arm/seed$seed"
    if [[ -n "${SUBMITTED[$key]:-}" ]]; then return 0; fi
    if [[ -f "$out" ]]; then SUBMITTED[$key]=1; return 0; fi
    if [[ ! -f "$ckpt" ]]; then return 0; fi
    SUBMITTED[$key]=1
    local logfile="$ROOT/$arm/seed$seed/eval.log"
    echo "[$(date +%H:%M:%S)] submit eval $key"
    srun --gres=gpu:1 --mem=64G -- \
        uv run python scripts/eval/cec_phase1_eval.py \
            --checkpoint "$ckpt" --arm "$arm" --episodes "$EPISODES" --output "$out" \
            > "$logfile" 2>&1 &
}

while true; do
    for arm in gen-fixed gen-base gen-router; do
        for seed in 0 1 2; do
            submit_eval "$arm" "$seed"
        done
    done
    DONE=$(ls "$ROOT"/*/seed*/eval_phase1.json 2>/dev/null | wc -l)
    if [[ "$DONE" -ge 9 ]]; then
        echo "[$(date +%H:%M:%S)] all 9 evals present"
        wait
        break
    fi
    sleep 60
done
