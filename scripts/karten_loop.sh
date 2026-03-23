#!/usr/bin/env bash
# karten_loop.sh — Hierarchical verification loop (Karten et al. approach)
#
# Runs tests by level (L1→L2→L3s→L3i→L4), spawns claude agents to fix
# failures, passes structured handoff between agent invocations.
#
# Usage:
#   bash scripts/karten_loop.sh              # run from L1 upward
#   LEVEL=l3i bash scripts/karten_loop.sh    # start at specific level
#   MAX_ITER=10 bash scripts/karten_loop.sh  # limit iterations

set -euo pipefail

# --- Configuration ---
WORKTREE="$(cd "$(dirname "$0")/.." && pwd)"
HANDOFF_DIR="${WORKTREE}/.agent_handoff"
STATUS_FILE="${HANDOFF_DIR}/verification_status.json"
PROMPT_TEMPLATE="${WORKTREE}/scripts/prompts/karten_parity.md"
MAX_ITER="${MAX_ITER:-50}"
FUZZ_SEEDS="${FUZZ_SEEDS:-20}"
FUZZ_STEPS="${FUZZ_STEPS:-500}"

# CPU-only — keep GPUs free for training
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

# JAX compilation cache (avoid recompiling across iterations)
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR="${WORKTREE}/.jax_cache"

# Ensure handoff directory exists
mkdir -p "${HANDOFF_DIR}/history"

# --- Level Definitions ---
LEVELS=("l1" "l2" "l3")
declare -A LEVEL_NAMES=(
    [l1]="L1 Property Tests"
    [l2]="L2 Interaction Tests"
    [l3]="L3 Rollout (no Category A syncs)"
)
declare -A LEVEL_DESCRIPTIONS=(
    [l1]="Individual component tests in isolation. Each subsystem (topology, red actions,
blue actions, rewards, observations) is tested independently with known
input/output pairs. Failures here indicate a single-module bug.

Test command: uv run pytest tests/subsystems/ -v -x -n auto"
    [l2]="Cross-module interaction tests. These verify that components work correctly
together — action sequencing, state propagation across modules, harness
infrastructure validation. Short traces (2-74 steps).

Test command: uv run pytest tests/differential/ -v -x -n auto"
    [l3]="Full episode rollout with Category A (deterministic) syncs REMOVED from
the harness. Impact, green events, session identity, and abstract rank
are computed by JAX independently — not copied from CybORG. RNG syncs
(Category B) remain active. 20 seeds x 500 steps.

Test command: uv run pytest tests/l3/ -v -x -n auto"
)
declare -A LEVEL_COMMANDS=(
    [l1]="uv run pytest tests/subsystems/ -v -x -n auto"
    [l2]="uv run pytest tests/differential/ -v -x -n auto"
    [l3]="uv run pytest tests/l3/ -v -x -n auto"
)

# --- Helper Functions ---

init_status() {
    if [[ ! -f "$STATUS_FILE" ]]; then
        cat > "$STATUS_FILE" << 'ENDJSON'
{
    "l1":  {"status": "unknown", "iterations": 0},
    "l2":  {"status": "unknown", "iterations": 0},
    "l3s": {"status": "unknown", "iterations": 0},
    "l3i": {"status": "unknown", "iterations": 0},
    "l4":  {"status": "not_started", "iterations": 0}
}
ENDJSON
    fi
}

get_status() {
    local level="$1"
    python3 -c "
import json
with open('${STATUS_FILE}') as f:
    data = json.load(f)
print(data.get('${level}', {}).get('status', 'unknown'))
"
}

get_iterations() {
    local level="$1"
    python3 -c "
import json
with open('${STATUS_FILE}') as f:
    data = json.load(f)
print(data.get('${level}', {}).get('iterations', 0))
"
}

update_status() {
    local level="$1"
    local status="$2"
    shift 2
    # Remaining args are key=value pairs for extra fields
    python3 -c "
import json, sys
with open('${STATUS_FILE}') as f:
    data = json.load(f)
entry = data.setdefault('${level}', {})
entry['status'] = '${status}'
entry['iterations'] = entry.get('iterations', 0) + 1
entry['last_checked'] = __import__('datetime').datetime.now().isoformat()
# Parse extra key=value args
for arg in sys.argv[1:]:
    k, v = arg.split('=', 1)
    try:
        v = json.loads(v)
    except (json.JSONDecodeError, ValueError):
        pass
    entry[k] = v
with open('${STATUS_FILE}', 'w') as f:
    json.dump(data, f, indent=2)
" "$@"
}

get_current_level() {
    # If LEVEL is set, use it; otherwise find lowest non-passing level
    if [[ -n "${LEVEL:-}" ]]; then
        echo "$LEVEL"
        return
    fi
    for lvl in "${LEVELS[@]}"; do
        local status
        status=$(get_status "$lvl")
        if [[ "$status" != "passing" ]]; then
            echo "$lvl"
            return
        fi
    done
    echo "done"
}

archive_handoff() {
    local level="$1"
    local iter="$2"
    local ts
    ts=$(date +%Y-%m-%dT%H%M)
    local prefix="${HANDOFF_DIR}/history/$(printf '%03d' "$iter")_${level}_${ts}"

    # Archive handoff
    if [[ -f "${HANDOFF_DIR}/handoff.md" ]]; then
        cp "${HANDOFF_DIR}/handoff.md" "${prefix}_handoff.md"
    fi
    # Archive test output
    if [[ -f "${HANDOFF_DIR}/test_output.txt" ]]; then
        cp "${HANDOFF_DIR}/test_output.txt" "${prefix}_tests.txt"
    fi
    # Archive agent conversation log
    if [[ -f "${HANDOFF_DIR}/agent_output_${iter}.txt" ]]; then
        cp "${HANDOFF_DIR}/agent_output_${iter}.txt" "${prefix}_agent.txt"
    fi
}

build_prompt() {
    local level="$1"
    local iter="$2"
    local level_name="${LEVEL_NAMES[$level]}"
    local level_desc="${LEVEL_DESCRIPTIONS[$level]}"

    # Read previous handoff if it exists
    local handoff_content="No previous handoff — this is the first iteration."
    if [[ -f "${HANDOFF_DIR}/handoff.md" ]]; then
        handoff_content=$(cat "${HANDOFF_DIR}/handoff.md")
    fi

    # Read test output (truncated to last 150 lines)
    local test_output="No test output yet."
    if [[ -f "${HANDOFF_DIR}/test_output.txt" ]]; then
        test_output=$(tail -150 "${HANDOFF_DIR}/test_output.txt")
    fi

    # Read template and interpolate
    local prompt
    prompt=$(cat "$PROMPT_TEMPLATE")

    # Shell variable substitution
    prompt="${prompt//\$\{WORKTREE\}/$WORKTREE}"
    prompt="${prompt//\$\{LEVEL\}/$level}"
    prompt="${prompt//\$\{LEVEL_NAME\}/$level_name}"
    prompt="${prompt//\$\{ITERATION\}/$iter}"
    prompt="${prompt//\$\{LEVEL_DESCRIPTION\}/$level_desc}"
    prompt="${prompt//\$\{TEST_OUTPUT\}/$test_output}"
    prompt="${prompt//\$\{HANDOFF_CONTENT\}/$handoff_content}"

    echo "$prompt"
}

run_tests() {
    local level="$1"
    local cmd="${LEVEL_COMMANDS[$level]}"
    echo "=== Running ${LEVEL_NAMES[$level]} ==="
    echo "Command: $cmd"
    echo ""

    # Run tests, capture output
    set +e
    (cd "$WORKTREE" && eval "$cmd") > "${HANDOFF_DIR}/test_output.txt" 2>&1
    local exit_code=$?
    set -e

    # Show summary
    if [[ $exit_code -eq 0 ]]; then
        echo "PASSED"
        tail -5 "${HANDOFF_DIR}/test_output.txt"
    else
        echo "FAILED (exit code $exit_code)"
        echo "--- Last 30 lines ---"
        tail -30 "${HANDOFF_DIR}/test_output.txt"
    fi
    echo ""
    return $exit_code
}

spawn_agent() {
    local level="$1"
    local iter="$2"

    echo "=== Spawning agent for ${LEVEL_NAMES[$level]}, iteration ${iter} ==="

    local prompt
    prompt=$(build_prompt "$level" "$iter")

    # Invoke claude in non-interactive mode
    (cd "$WORKTREE" && claude -p "$prompt" \
        --allowedTools "Read,Edit,Write,Bash(uv run*),Bash(git add*),Bash(git commit*),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(ls*),Bash(python3*),Bash(cat .agent_handoff*),Grep,Glob" \
    ) 2>&1 | tee "${HANDOFF_DIR}/agent_output_${iter}.txt"

    echo "=== Agent finished ==="

    # Check if agent hit rate limit
    if grep -qi "hit your limit\|rate.limit\|resets.*America" "${HANDOFF_DIR}/agent_output_${iter}.txt" 2>/dev/null; then
        echo ""
        echo "!!! Agent hit API rate limit — pausing loop"
        echo "$(date -Iseconds) | iter=${iter} | ${level} | RATE LIMITED — pausing" \
            >> "${HANDOFF_DIR}/loop_log.txt"
        echo "!!! Waiting 60 minutes before retrying..."
        sleep 3600
    fi
}

# --- Main Loop ---

main() {
    cd "$WORKTREE"
    init_status

    echo "=========================================="
    echo "  Karten Loop — Hierarchical Verification"
    echo "  Worktree: ${WORKTREE}"
    echo "  Max iterations: ${MAX_ITER}"
    echo "=========================================="
    echo ""

    local total_iter=0

    while [[ $total_iter -lt $MAX_ITER ]]; do
        local current_level
        current_level=$(get_current_level)

        if [[ "$current_level" == "done" ]]; then
            echo "ALL LEVELS PASSING"
            echo ""
            python3 -c "
import json
with open('${STATUS_FILE}') as f:
    data = json.load(f)
for lvl, info in data.items():
    print(f'  {lvl}: {info[\"status\"]} ({info.get(\"iterations\", 0)} iterations)')
"
            exit 0
        fi

        local level_iter
        level_iter=$(get_iterations "$current_level")
        total_iter=$((total_iter + 1))

        echo "--- Iteration ${total_iter}/${MAX_ITER} | Level: ${current_level} | Level iteration: ${level_iter} ---"

        # Run tests for current level
        if run_tests "$current_level"; then
            # Tests passed — mark level and advance
            update_status "$current_level" "passing"
            echo ">>> ${LEVEL_NAMES[$current_level]} PASSED — advancing"
            echo "$(date -Iseconds) | iter=${total_iter} | ${current_level} | PASSED" \
                >> "${HANDOFF_DIR}/loop_log.txt"
            # Clear LEVEL override so we auto-advance
            unset LEVEL 2>/dev/null || true
        else
            # Tests failed — spawn agent to fix
            archive_handoff "$current_level" "$total_iter"
            update_status "$current_level" "failing"
            echo "$(date -Iseconds) | iter=${total_iter} | ${current_level} | FAILED → spawning agent" \
                >> "${HANDOFF_DIR}/loop_log.txt"

            spawn_agent "$current_level" "$total_iter"

            # Check if agent marked itself as stuck
            if [[ -f "${HANDOFF_DIR}/handoff.md" ]] && grep -q "status: stuck" "${HANDOFF_DIR}/handoff.md"; then
                echo ""
                echo "!!! Agent reported STUCK at ${LEVEL_NAMES[$current_level]} ==="
                echo "!!! Review ${HANDOFF_DIR}/handoff.md for details"
                echo "!!! Add targeted tests or refine the prompt, then re-run"
                exit 2
            fi
        fi

        echo ""
    done

    echo "MAX ITERATIONS (${MAX_ITER}) reached without convergence"
    echo "Current status:"
    cat "$STATUS_FILE"
    exit 1
}

# --- Entry Point ---
trap 'echo "Interrupted"; exit 130' INT TERM
main "$@"
