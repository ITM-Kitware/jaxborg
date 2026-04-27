#!/usr/bin/env bash
# Quick partial-results inspector for an in-progress CEC Phase 1 grid.
#
# Usage:
#   bash scripts/eval/cec_phase1_partial.sh [<grid_root>]
#
# Lists which checkpoints exist, their final training metrics, and which evals
# have completed. Useful to peek at the run while training is still going.

set -euo pipefail
ROOT="${1:-/home/local/KHQ/paul.elliott/src/cyber/jaxborg-exp/cec_phase1}"

if [[ ! -d "$ROOT" ]]; then
    echo "no grid root: $ROOT" >&2
    exit 1
fi

echo "Grid root: $ROOT"
echo
printf "%-12s %5s %12s %15s %15s %s\n" arm seed updates raw_reward exp_var has_eval
printf "%-12s %5s %12s %15s %15s %s\n" "----" "----" "-------" "----------" "-------" "--------"

for arm_dir in "$ROOT"/gen-fixed "$ROOT"/gen-base "$ROOT"/gen-router; do
    [[ -d "$arm_dir" ]] || continue
    arm=$(basename "$arm_dir")
    for seed_dir in "$arm_dir"/seed*; do
        [[ -d "$seed_dir" ]] || continue
        seed=$(basename "$seed_dir" | sed 's/seed//')
        mlog="$seed_dir/metrics.jsonl"
        ckpt="$seed_dir/checkpoint_final.pkl"
        eval_path="$seed_dir/eval_phase1.json"
        if [[ -f "$mlog" ]] && [[ -s "$mlog" ]]; then
            last=$(tail -1 "$mlog")
            upd=$(echo "$last" | jq -r '.update' 2>/dev/null || echo "?")
            rew=$(echo "$last" | jq -r '.raw_episode_reward_mean | round' 2>/dev/null || echo "?")
            exp=$(echo "$last" | jq -r '.explained_var | (.*1000 | round)/1000' 2>/dev/null || echo "?")
        else
            upd="-"; rew="-"; exp="-"
        fi
        ckpt_status="-"
        [[ -f "$ckpt" ]] && ckpt_status="ckpt"
        eval_status=""
        [[ -f "$eval_path" ]] && eval_status="EVAL"
        printf "%-12s %5s %12s %15s %15s %s %s\n" "$arm" "$seed" "$upd" "$rew" "$exp" "$ckpt_status" "$eval_status"
    done
done

echo
echo "Slurm queue:"
squeue -u "$USER" -o "  %.8i %.9P %.8j %.8u %.2t %.10M %b" 2>/dev/null | head
