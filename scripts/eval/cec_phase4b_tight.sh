#!/usr/bin/env bash
# Phase 4b: tighter stats + CIA breakdown.
#
# Runs all 12 ckpts (4 arms x 3 seeds) against `discovery` red only — the
# only red that gave signal in Phase 4a (random was a floor; fsm was the
# training distribution).  Bumps episodes 30 -> 90 to shrink stderr.
# Writes per-episode trajectories so we can score CIA components post-hoc.
#
# Output:
#   <root>/cec_phase4b/<arm>/seed<N>/discovery.json     summary
#   <root>/cec_phase4b/<arm>/seed<N>/discovery_traj/    *.jsonl per ep
#
# Runs locally in parallel (no GPU needed; CPU bound).

set -euo pipefail

ROOT="${1:-jaxborg-exp}"
EPISODES="${EPISODES:-90}"
PARALLEL="${PARALLEL:-12}"

OUT_ROOT="$(cd "$(dirname "$ROOT")" && pwd)/$(basename "$ROOT")/cec_phase4b"
mkdir -p "$OUT_ROOT"

# Arm -> mission_multipliers tail to append to obs (3 zeros for hidden, 1,1,1
# for visible default profile, ignored for 210-dim Phase 2 ckpts).
declare -A MULT
MULT[gen-fixed-nomsg]="0,0,0"
MULT[gen-mission-nomsg]="0,0,0"
MULT[gen-mission-10x-hidden]="0,0,0"
MULT[gen-mission-10x-visible]="1,1,1"

# Map arm -> ckpt path glob root (Phase 2 vs Phase 3).
declare -A CKPT_ROOT
CKPT_ROOT[gen-fixed-nomsg]="cec_phase2_20M"
CKPT_ROOT[gen-mission-nomsg]="cec_phase2_20M"
CKPT_ROOT[gen-mission-10x-hidden]="cec_phase3_20M"
CKPT_ROOT[gen-mission-10x-visible]="cec_phase3_20M"

ARMS=(gen-fixed-nomsg gen-mission-nomsg gen-mission-10x-hidden gen-mission-10x-visible)
SEEDS=(1 2 3)

# Build a job list: one line per (ckpt, arm, seed).
job_list=$(mktemp)
trap 'rm -f $job_list' EXIT
for arm in "${ARMS[@]}"; do
    mult="${MULT[$arm]}"
    glob="${CKPT_ROOT[$arm]}"
    for seed in "${SEEDS[@]}"; do
        ckpt=$(find "$ROOT" -path "*${glob}/$arm/seed$seed/checkpoint_final.pkl" 2>/dev/null | head -1)
        if [[ -z "$ckpt" ]]; then
            echo "[MISS] $arm/seed$seed — no checkpoint" >&2
            continue
        fi
        out_dir="$OUT_ROOT/$arm/seed$seed"
        mkdir -p "$out_dir"
        out_json="$out_dir/discovery.json"
        if [[ -f "$out_json" ]]; then
            echo "[SKIP] $arm/seed$seed — already done" >&2
            continue
        fi
        echo "$ckpt|$arm|$seed|$mult" >> "$job_list"
    done
done

n=$(wc -l < "$job_list")
echo "Launching $n jobs locally with parallelism $PARALLEL ..."

cat "$job_list" | xargs -P "$PARALLEL" -I {} bash -c '
    IFS="|" read -r ckpt arm seed mult <<< "{}"
    out_dir="'"$OUT_ROOT"'/$arm/seed$seed"
    out_json="$out_dir/discovery.json"
    traj_dir="$out_dir/discovery_traj"
    log="$out_dir/discovery.log"
    mkdir -p "$out_dir" "$traj_dir"
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run python scripts/eval/cyborg.py \
        --checkpoint "$ckpt" \
        --episodes '"$EPISODES"' \
        --seed 5000 \
        --red-agent discovery \
        --mission-multipliers "$mult" \
        --output "$out_json" \
        --trajectory-dir "$traj_dir" >"$log" 2>&1
'

echo "All Phase 4b jobs complete."
