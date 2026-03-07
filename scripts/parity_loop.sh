#!/usr/bin/env bash
# Differential parity loop (fuzzer only): run batches until mismatch or clean.
#
# Usage:
#   bash scripts/parity_loop.sh
#
# Environment variables:
#   MAX_ITERATIONS          - max fuzz batches (default: 20)
#   FUZZ_SEEDS              - number of seeds per batch (default: 20)
#   FUZZ_STEPS              - steps per seed (default: 100)
#   FUZZ_MISMATCH_MODE      - "error" (default) or "all"
#   FUZZ_BLUE_AGENT         - "random" (default), "sleep", or "monitor"
#   FUZZ_BLUE_ACTION_SOURCE - "cyborg_policy" (default) or "sleep"

set -euo pipefail
cd "$(dirname "$0")/.."

MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
FUZZ_SEEDS="${FUZZ_SEEDS:-20}"
FUZZ_STEPS="${FUZZ_STEPS:-100}"
FUZZ_MISMATCH_MODE="${FUZZ_MISMATCH_MODE:-error}"
FUZZ_BLUE_AGENT="${FUZZ_BLUE_AGENT:-random}"
FUZZ_BLUE_ACTION_SOURCE="${FUZZ_BLUE_ACTION_SOURCE:-cyborg_policy}"
FUZZ_STRICT_RANDOM_SYNC=1

# Persist JAX compilations between loop iterations so repeated uv/python
# processes avoid recompilation costs.
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-1}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/.cache/jax}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo ""
    echo "============================================"
    echo "  Parity Loop — Iteration $i / $MAX_ITERATIONS"
    echo "============================================"

    echo "Running fuzzer (seeds=0-$((FUZZ_SEEDS-1)), steps=$FUZZ_STEPS, mismatch_mode=$FUZZ_MISMATCH_MODE, blue_agent=$FUZZ_BLUE_AGENT, blue_action_source=$FUZZ_BLUE_ACTION_SOURCE)..."
    output=$(uv run python -u -c "
from tests.differential.fuzzer import run_differential_fuzz
r = run_differential_fuzz(
    seeds=range($FUZZ_SEEDS),
    max_steps_per_seed=$FUZZ_STEPS,
    verbose=True,
    mismatch_mode='$FUZZ_MISMATCH_MODE',
    blue_agent='$FUZZ_BLUE_AGENT',
    blue_action_source='$FUZZ_BLUE_ACTION_SOURCE',
    strict_random_sync=bool(int($FUZZ_STRICT_RANDOM_SYNC)),
)
if r:
    print(f'MISMATCH|{r.seed}|{r.step}|{r.field_name}|{r.host_or_agent}|{r.cyborg_value}|{r.jax_value}')
    print(r.all_diffs_str)
else:
    print('CLEAN')
")

    if echo "$output" | grep -q "^CLEAN$"; then
        echo ""
        echo "No monitored mismatches found across $FUZZ_SEEDS seeds x $FUZZ_STEPS steps (mode=$FUZZ_MISMATCH_MODE)."
        echo "Current monitored parity target reached after $i iterations."
        exit 0
    fi

    mismatch=$(echo "$output" | grep "^MISMATCH|" | head -1)
    diffs=$(echo "$output" | grep -v "^MISMATCH|" | grep -v "^---" | grep -v "^  Seed" | grep -v "^Total:" | grep -v "^Running" || true)
    IFS='|' read -r _ seed step field host cyborg_val jax_val <<< "$mismatch"

    echo ""
    echo "FOUND GAP: seed=$seed step=$step $field [$host] cyborg=$cyborg_val jax=$jax_val"
    if [[ -n "$diffs" ]]; then
        echo "$diffs"
    fi
    echo ""
    echo "Stopping on first mismatch. Add an explicit differential regression, fix, then rerun."
    exit 1
done

echo ""
echo "Completed $MAX_ITERATIONS iterations without CLEAN or MISMATCH sentinel output."
exit 1
