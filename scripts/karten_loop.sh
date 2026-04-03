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

# Tag for slurm jobs submitted by this loop (used for cleanup).
# Exported so spawned agents inherit it and can tag their own srun calls.
KARTEN_JOB_TAG="karten-loop-$$"
export KARTEN_JOB_TAG

# L4 training config
TRAIN_TIMESTEPS="${TRAIN_TIMESTEPS:-5000000}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-1024}"
TOPOLOGY_MODE="${TOPOLOGY_MODE:-cyborg_bank}"
TOPOLOGY_BANK_SIZE="${TOPOLOGY_BANK_SIZE:-32}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EXP_DIR="${JAXBORG_EXP_DIR:-${WORKTREE}/jaxborg-exp}"

# Trained-policy checkpoint for L3/L4 (auto-discover or set via env)
BLUE_CHECKPOINT="${BLUE_CHECKPOINT:-$(find "${EXP_DIR}" -name 'checkpoint_final.pkl' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)}"
export BLUE_CHECKPOINT

# CPU-only for L1-L3 — L4 uses srun for GPU
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

# JAX compilation cache (avoid recompiling across iterations)
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR="${WORKTREE}/.jax_cache"

# Ensure handoff directory exists
mkdir -p "${HANDOFF_DIR}/history"

# --- Level Definitions ---
LEVELS=("l1" "l2" "l3" "l4")
declare -A LEVEL_NAMES=(
    [l1]="L1 Property Tests"
    [l2]="L2 Interaction Tests"
    [l3]="L3 Rollout (trained policy + random blue)"
    [l4]="L4 Cross-Backend Transfer (TOST)"
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
    [l3]="Full episode rollout comparison. Includes:
- Random blue (50 seeds x 500 steps): tests/l3/test_full_episode_fuzzing.py
- Trained IPPO policy (100 seeds x 500 steps): tests/l3/test_trained_blue_policy.py
The trained policy exercises realistic action sequences (heavy Restore, targeted
Monitor/Remove) that random blue never does, catching action-combination bugs.
Requires BLUE_CHECKPOINT env var.

Test command: BLUE_CHECKPOINT=${BLUE_CHECKPOINT} uv run pytest tests/l3/ -v -x -n auto"
    [l4]="Cross-backend policy transfer (TOST). Trains a fresh IPPO policy in JAXborg
via slurm (GPU), then evaluates it independently in both JAXborg and CybORG.
A failing L4 feeds back into targeted L1/L2/L3 tests.

Steps: (1) train via srun --gres=gpu:1, (2) eval_transfer.py --independent-rollouts"
)
declare -A LEVEL_COMMANDS=(
    [l1]="uv run pytest tests/subsystems/ -v -x -n auto"
    [l2]="uv run pytest tests/differential/ -v -x -n auto"
    [l3]='BLUE_CHECKPOINT=${BLUE_CHECKPOINT} uv run pytest tests/l3/ -v -x -n auto'
    [l4]="RUN_L4_TRAINING"
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

build_verification_status() {
    # Build dynamic status summary from verification_status.json and git log
    python3 -c "
import json, subprocess, os
status_file = '${STATUS_FILE}'
try:
    with open(status_file) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}

lines = []
for lvl in ['l1', 'l2', 'l3', 'l4']:
    info = data.get(lvl, {})
    status = info.get('status', 'unknown')
    iters = info.get('iterations', 0)
    lines.append(f'**{lvl.upper()}**: {status} ({iters} iterations)')

# Recent commits for context
try:
    log = subprocess.check_output(
        ['git', 'log', '--oneline', '-5'],
        cwd='${WORKTREE}', text=True, stderr=subprocess.DEVNULL
    ).strip()
    lines.append('')
    lines.append('Recent commits:')
    for line in log.split(chr(10)):
        lines.append(f'- \`{line}\`')
except Exception:
    pass

print(chr(10).join(lines))
"
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

    # Build dynamic verification status
    local verification_status
    verification_status=$(build_verification_status)

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
    prompt="${prompt//\$\{VERIFICATION_STATUS\}/$verification_status}"

    echo "$prompt"
}

run_l4_train_and_eval() {
    # Step 1: Train fresh IPPO policy via srun (GPU)
    local round_dir="${EXP_DIR}/karten_l4/round_$(date +%Y%m%d_%H%M)"
    mkdir -p "$round_dir"

    echo "  L4 Step 1: Training IPPO (${TRAIN_TIMESTEPS} steps, ${TRAIN_NUM_ENVS} envs)..."
    local train_seed=$((RANDOM % 1000))

    # Unset CPU-only vars for GPU training
    (
        unset CUDA_VISIBLE_DEVICES JAX_PLATFORMS
        export JAX_ENABLE_COMPILATION_CACHE=1
        export JAX_COMPILATION_CACHE_DIR="${HOME}/.cache/jaxborg/xla"
        export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
        srun --gres=gpu:1 --mem=64G --partition=community --comment="${KARTEN_JOB_TAG}" \
            uv run python scripts/train_ippo_cc4.py \
                TOTAL_TIMESTEPS="$TRAIN_TIMESTEPS" \
                NUM_ENVS="$TRAIN_NUM_ENVS" \
                TOPOLOGY_MODE="$TOPOLOGY_MODE" \
                +TOPOLOGY_BANK_SIZE="$TOPOLOGY_BANK_SIZE" \
                SEED="$train_seed" \
                hydra.run.dir="$round_dir/hydra" \
                hydra.job.chdir=True
    ) >> "${HANDOFF_DIR}/test_output.txt" 2>&1

    # Find latest checkpoint
    local checkpoint
    checkpoint=$(ls -t "${EXP_DIR}"/ippo_cc4*/checkpoint_final.pkl 2>/dev/null | head -1)
    if [[ -z "$checkpoint" ]]; then
        echo "ERROR: No checkpoint found after training" >> "${HANDOFF_DIR}/test_output.txt"
        return 1
    fi
    cp "$checkpoint" "$round_dir/checkpoint_final.pkl"
    BLUE_CHECKPOINT="$checkpoint"
    export BLUE_CHECKPOINT
    echo "  Checkpoint: $checkpoint"

    # Step 2: Eval transfer (needs GPU for JAX rollouts)
    echo "  L4 Step 2: Evaluating transfer (${EVAL_EPISODES} episodes, independent)..."
    (
        unset CUDA_VISIBLE_DEVICES JAX_PLATFORMS
        export JAX_ENABLE_COMPILATION_CACHE=1
        export JAX_COMPILATION_CACHE_DIR="${HOME}/.cache/jaxborg/xla"
        export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
        srun --gres=gpu:1 --mem=64G --partition=community --comment="${KARTEN_JOB_TAG}" \
            uv run python scripts/eval_transfer.py \
                --checkpoint "$checkpoint" \
                --episodes "$EVAL_EPISODES" \
                --independent-rollouts \
                --baselines \
                --seed "$train_seed"
    ) >> "${HANDOFF_DIR}/test_output.txt" 2>&1
}

run_tests() {
    local level="$1"
    local cmd="${LEVEL_COMMANDS[$level]}"
    echo "=== Running ${LEVEL_NAMES[$level]} ==="

    # L4 has special handling: train + eval via GPU
    if [[ "$level" == "l4" ]]; then
        echo "Training + evaluating via slurm GPU..."
        echo ""
        > "${HANDOFF_DIR}/test_output.txt"
        set +e
        run_l4_train_and_eval
        local exit_code=$?
        set -e

        # Check TOST verdict.  L4 passes if:
        #   (a) policy TOST says EQUIVALENT, OR
        #   (b) policy TOST says NOT EQUIVALENT but sleep TOST confirms
        #       simulation equivalence (gap is from policy transfer, not sim bug)
        if grep -q "EQUIVALENT" "${HANDOFF_DIR}/test_output.txt" && \
           ! grep -q "NOT EQUIVALENT" "${HANDOFF_DIR}/test_output.txt"; then
            exit_code=0
        elif grep -q "simulation equivalence" "${HANDOFF_DIR}/test_output.txt"; then
            echo "  (simulation equivalent — policy transfer gap only)"
            exit_code=0
        else
            exit_code=1
        fi
    else
        echo "Command: $cmd"
        echo ""
        set +e
        (cd "$WORKTREE" && eval "$cmd") > "${HANDOFF_DIR}/test_output.txt" 2>&1
        local exit_code=$?
        set -e
    fi

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

    # Write prompt to temp file (too large for shell arg on some systems)
    local prompt_file
    prompt_file=$(mktemp "${HANDOFF_DIR}/prompt_XXXXXX.md")
    echo "$prompt" > "$prompt_file"

    # Invoke claude in non-interactive mode with 2-hour timeout.
    # Broad tool permissions — the agent needs grep, find, python, etc.
    timeout 7200 claude -p "$(cat "$prompt_file")" \
        --allowedTools "Read,Edit,Write,Bash,Grep,Glob" \
        2>&1 | tee "${HANDOFF_DIR}/agent_output_${iter}.txt"
    local agent_exit=${PIPESTATUS[0]}
    rm -f "$prompt_file"

    echo "=== Agent finished (exit=$agent_exit) ==="

    # Cancel GPU jobs the agent left running.  Only target jobs tagged
    # with our KARTEN_JOB_TAG comment so we never touch the user's own
    # srun sessions in other terminals.
    local stale_jobs
    stale_jobs=$(squeue -u "$(whoami)" -h -o "%i %k" 2>/dev/null \
        | grep "${KARTEN_JOB_TAG}" | awk '{print $1}')
    if [[ -n "$stale_jobs" ]]; then
        echo "  Cleaning up karten-tagged GPU jobs: $stale_jobs"
        echo "$stale_jobs" | xargs -r scancel 2>/dev/null || true
    fi

    if [[ $agent_exit -eq 124 ]]; then
        echo "!!! Agent timed out after 2 hours"
        echo "$(date -Iseconds) | iter=${iter} | ${level} | TIMEOUT" \
            >> "${HANDOFF_DIR}/loop_log.txt"
    fi

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

            # L4 failure feedback: agent fixed code, reset L1-L3 to re-verify
            # before retraining (paper's "feed back into earlier stages")
            if [[ "$current_level" == "l4" ]]; then
                echo ">>> L4 fix applied — resetting L1-L3 for re-verification"
                for reset_lvl in l1 l2 l3; do
                    update_status "$reset_lvl" "unknown"
                done
                # L4 always trains fresh, so no checkpoint invalidation needed.
                # L3 reuses old checkpoints (tests parity, not transfer).
                unset LEVEL 2>/dev/null || true
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
