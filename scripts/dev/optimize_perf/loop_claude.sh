#!/usr/bin/env bash
# Autoresearch-style optimization loop for JAXborg performance.
#
# The loop asks Claude Code for one focused optimization, then the script runs
# correctness and performance gates. Passing improvements are committed; failed
# candidates are reverted from the editable scope.
#
# Usage:
#   bash scripts/dev/optimize_perf/loop_claude.sh
#   bash scripts/dev/optimize_perf/loop_claude.sh --prompt-only
#   bash scripts/dev/optimize_perf/loop_claude.sh --benchmark-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$WORKTREE"

BENCHMARK_RECIPE="${BENCHMARK_RECIPE:-default}"
RUN_DIR="${RUN_DIR:-${WORKTREE}/.agent_handoff/optimize_perf_${BENCHMARK_RECIPE}}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-${SCRIPT_DIR}/prompt.md}"
RESULTS_TSV="${RESULTS_TSV:-${RUN_DIR}/results.tsv}"
MEMORY_FILE="${MEMORY_FILE:-${RUN_DIR}/experiment_memory.md}"

MAX_ITER="${MAX_ITER:-20}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-3600}"
CORRECTNESS_TIMEOUT_SEC="${CORRECTNESS_TIMEOUT_SEC:-1800}"
BENCHMARK_TIMEOUT_SEC="${BENCHMARK_TIMEOUT_SEC:-1800}"

BENCHMARK_SEED="${BENCHMARK_SEED:-0}"
BENCHMARK_UPDATES="${BENCHMARK_UPDATES:-5}"
BENCHMARK_NUM_ENVS="${BENCHMARK_NUM_ENVS:-}"
BENCHMARK_NUM_STEPS="${BENCHMARK_NUM_STEPS:-}"
SLURM_MEM="${SLURM_MEM:-64G}"

MIN_SPS_REL_IMPROVE="${MIN_SPS_REL_IMPROVE:-0.01}"
SPS_NOISE_TOLERANCE="${SPS_NOISE_TOLERANCE:-0.005}"
MIN_COMPILE_REL_IMPROVE="${MIN_COMPILE_REL_IMPROVE:-0.05}"

CORRECTNESS_CMD="${CORRECTNESS_CMD:-uv run pytest tests/subsystems tests/differential tests/test_cc4_env_differential.py tests/test_action_mask_differential.py -q -x -n auto}"
CLAUDE_ALLOWED_TOOLS="${CLAUDE_ALLOWED_TOOLS:-Read,Edit,Write,Bash(git status*),Bash(git diff*),Bash(git log*),Bash(rg*),Bash(sed*),Bash(ls*),Bash(find*),Bash(uv run pytest*),Bash(uv run python*),Grep,Glob}"
CLAUDE_MODEL="${CLAUDE_MODEL:-}"

EDITABLE_PATHS=(
    "src/jaxborg"
    "scripts/train/algorithms/ippo_jax.py"
)

GUARDED_PATHS=(
    "scripts/dev/optimize_perf/benchmark.py"
    "scripts/dev/optimize_perf/loop_claude.sh"
    "scripts/dev/optimize_perf/prompt.md"
    "scripts/eval"
    "tests"
    "recipes"
    "pyproject.toml"
    "uv.lock"
)

trap 'echo "Interrupted"; exit 130' INT TERM

init_run_dir() {
    mkdir -p "$RUN_DIR/logs"
    if [[ ! -f "$RESULTS_TSV" ]]; then
        printf "timestamp\titeration\tcommit\tstatus\tsteady_sps_median\tcompile_time_estimate_s\tfirst_update_time_s\tdescription\n" \
            > "$RESULTS_TSV"
    fi
    if [[ ! -f "$MEMORY_FILE" ]]; then
        {
            echo "# JAXborg Perf Experiment Memory"
            echo ""
            echo "Benchmark recipe: ${BENCHMARK_RECIPE}"
            echo ""
            echo "Claude must not repeat discarded ideas unless it can explain what is materially different."
            echo ""
        } > "$MEMORY_FILE"
    fi
}

editable_scope_status() {
    git status --porcelain --untracked-files=all -- "${EDITABLE_PATHS[@]}"
}

ensure_editable_scope_clean() {
    local status
    status="$(editable_scope_status)"
    if [[ -n "$status" ]]; then
        echo "Editable scope is not clean. Commit, stash, or remove these changes before starting:"
        echo "$status"
        return 1
    fi
}

guard_hash() {
    {
        git status --porcelain --untracked-files=all -- "${GUARDED_PATHS[@]}"
        git diff --binary -- "${GUARDED_PATHS[@]}"
        git diff --cached --binary -- "${GUARDED_PATHS[@]}"
    } | sha256sum | awk '{print $1}'
}

benchmark_args() {
    local json_out="$1"
    local label="$2"
    local args=(
        "uv" "run" "python" "scripts/dev/optimize_perf/benchmark.py"
        "--recipe" "$BENCHMARK_RECIPE"
        "--seed" "$BENCHMARK_SEED"
        "--num-updates" "$BENCHMARK_UPDATES"
        "--json-out" "$json_out"
        "--label" "$label"
    )
    if [[ -n "$BENCHMARK_NUM_ENVS" ]]; then
        args+=("--num-envs" "$BENCHMARK_NUM_ENVS")
    fi
    if [[ -n "$BENCHMARK_NUM_STEPS" ]]; then
        args+=("--num-steps" "$BENCHMARK_NUM_STEPS")
    fi
    printf "%q " "${args[@]}"
}

benchmark_display_cmd() {
    local json_out="$1"
    local label="$2"
    local inner
    inner="$(benchmark_args "$json_out" "$label")"
    if [[ -n "${SLURM_JOB_ID:-}" || "${OPT_LOOP_NO_SLURM:-}" == "1" ]]; then
        echo "$inner"
    else
        echo "srun --gres=gpu:1 --mem=${SLURM_MEM} -- $inner"
    fi
}

run_benchmark() {
    local iteration="$1"
    local label="$2"
    local json_out="${RUN_DIR}/logs/bench_${iteration}.json"
    local log_out="${RUN_DIR}/logs/bench_${iteration}.log"
    local inner=(
        uv run python scripts/dev/optimize_perf/benchmark.py
        --recipe "$BENCHMARK_RECIPE"
        --seed "$BENCHMARK_SEED"
        --num-updates "$BENCHMARK_UPDATES"
        --json-out "$json_out"
        --label "$label"
    )
    if [[ -n "$BENCHMARK_NUM_ENVS" ]]; then
        inner+=(--num-envs "$BENCHMARK_NUM_ENVS")
    fi
    if [[ -n "$BENCHMARK_NUM_STEPS" ]]; then
        inner+=(--num-steps "$BENCHMARK_NUM_STEPS")
    fi
    local cmd=()
    if [[ -n "${SLURM_JOB_ID:-}" || "${OPT_LOOP_NO_SLURM:-}" == "1" ]]; then
        cmd=("${inner[@]}")
    else
        cmd=(srun --gres=gpu:1 --mem="$SLURM_MEM" -- "${inner[@]}")
    fi

    echo "Benchmark command: $(printf "%q " "${cmd[@]}")" >&2
    set +e
    (
        export JAX_ENABLE_COMPILATION_CACHE=1
        export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${RUN_DIR}/xla_cache}"
        export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
        timeout "$BENCHMARK_TIMEOUT_SEC" "${cmd[@]}"
    ) > "$log_out" 2>&1
    local exit_code=$?
    set -e

    if [[ $exit_code -ne 0 ]]; then
        echo "Benchmark failed with exit code $exit_code. Last 40 log lines:" >&2
        tail -40 "$log_out" >&2 || true
        return "$exit_code"
    fi
    if [[ ! -f "$json_out" ]]; then
        echo "Benchmark did not write JSON output: $json_out" >&2
        tail -40 "$log_out" >&2 || true
        return 1
    fi
    echo "$json_out"
}

json_field() {
    local json_file="$1"
    local field="$2"
    python3 - "$json_file" "$field" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
value = data[sys.argv[2]]
if isinstance(value, float):
    print(f"{value:.6f}")
else:
    print(value)
PY
}

run_correctness() {
    local iteration="$1"
    local log_out="${RUN_DIR}/logs/correctness_${iteration}.log"
    echo "Correctness command: $CORRECTNESS_CMD"
    set +e
    (timeout "$CORRECTNESS_TIMEOUT_SEC" bash -lc "$CORRECTNESS_CMD") > "$log_out" 2>&1
    local exit_code=$?
    set -e
    if [[ $exit_code -ne 0 ]]; then
        echo "Correctness failed with exit code $exit_code. Last 40 log lines:"
        tail -40 "$log_out" || true
        return "$exit_code"
    fi
}

build_prompt() {
    local iteration="$1"
    local best_sps="$2"
    local best_compile="$3"
    local prompt_file="$4"
    local recent_results
    recent_results="$(tail -12 "$RESULTS_TSV" 2>/dev/null || true)"
    local experiment_memory
    experiment_memory="$(tail -160 "$MEMORY_FILE" 2>/dev/null || true)"
    local editable_scope
    editable_scope="$(printf "%s\n" "${EDITABLE_PATHS[@]}")"
    local benchmark_cmd
    benchmark_cmd="$(benchmark_display_cmd "${RUN_DIR}/logs/candidate.json" "candidate")"

    WORKTREE_VALUE="$WORKTREE" \
    ITERATION_VALUE="$iteration" \
    BEST_STEADY_SPS_VALUE="$best_sps" \
    BEST_COMPILE_TIME_VALUE="$best_compile" \
    RECENT_RESULTS_VALUE="$recent_results" \
    EXPERIMENT_MEMORY_VALUE="$experiment_memory" \
    CORRECTNESS_CMD_VALUE="$CORRECTNESS_CMD" \
    BENCHMARK_CMD_VALUE="$benchmark_cmd" \
    EDITABLE_SCOPE_VALUE="$editable_scope" \
    python3 - "$PROMPT_TEMPLATE" "$prompt_file" <<'PY'
import os
import sys
from pathlib import Path

template = Path(sys.argv[1]).read_text()
replacements = {
    "WORKTREE": os.environ["WORKTREE_VALUE"],
    "ITERATION": os.environ["ITERATION_VALUE"],
    "BEST_STEADY_SPS": os.environ["BEST_STEADY_SPS_VALUE"],
    "BEST_COMPILE_TIME": os.environ["BEST_COMPILE_TIME_VALUE"],
    "RECENT_RESULTS": os.environ["RECENT_RESULTS_VALUE"],
    "EXPERIMENT_MEMORY": os.environ["EXPERIMENT_MEMORY_VALUE"],
    "CORRECTNESS_CMD": os.environ["CORRECTNESS_CMD_VALUE"],
    "BENCHMARK_CMD": os.environ["BENCHMARK_CMD_VALUE"],
    "EDITABLE_SCOPE": os.environ["EDITABLE_SCOPE_VALUE"],
}
for key, value in replacements.items():
    template = template.replace("${" + key + "}", value)
Path(sys.argv[2]).write_text(template)
PY
}

run_claude_iteration() {
    local iteration="$1"
    local best_sps="$2"
    local best_compile="$3"
    local prompt_file="${RUN_DIR}/logs/prompt_${iteration}.md"
    local output_file="${RUN_DIR}/logs/claude_${iteration}.log"
    build_prompt "$iteration" "$best_sps" "$best_compile" "$prompt_file"

    local claude_args=(-p "$(cat "$prompt_file")" --allowedTools "$CLAUDE_ALLOWED_TOOLS")
    if [[ -n "$CLAUDE_MODEL" ]]; then
        claude_args+=(--model "$CLAUDE_MODEL")
    fi

    echo "Launching Claude for iteration $iteration..."
    set +e
    timeout "$AGENT_TIMEOUT_SEC" claude "${claude_args[@]}" > "$output_file" 2>&1
    local exit_code=$?
    set -e
    if [[ $exit_code -ne 0 ]]; then
        echo "Claude exited with code $exit_code. Last 40 log lines:"
        tail -40 "$output_file" || true
        return "$exit_code"
    fi
}

experiment_description() {
    experiment_field "EXPERIMENT_DESCRIPTION" "$1"
}

experiment_field() {
    local field="$1"
    local output_file="$2"
    local value
    value="$(grep "^${field}:" "$output_file" | tail -1 | sed "s/^${field}:[[:space:]]*//" || true)"
    if [[ -z "$value" ]]; then
        value="not provided"
    fi
    echo "$value" | tr '\t' ' ' | head -c 300
}

discard_candidate() {
    local start_ref="$1"
    git restore --source="$start_ref" --staged --worktree -- "${EDITABLE_PATHS[@]}" || true
    git clean -fd -- "${EDITABLE_PATHS[@]}" >/dev/null || true
}

commit_candidate() {
    local desc="$1"
    git add -- "${EDITABLE_PATHS[@]}"
    if git diff --cached --quiet; then
        echo "No staged editable changes to commit."
        return 1
    fi
    git commit -m "perf: ${desc}"
}

append_result() {
    local iteration="$1"
    local commit="$2"
    local status="$3"
    local sps="$4"
    local compile="$5"
    local first_update="$6"
    local desc="$7"
    local expected="${8:-not provided}"
    local risk="${9:-not provided}"
    local timestamp
    timestamp="$(date -Iseconds)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$timestamp" "$iteration" "$commit" "$status" "$sps" "$compile" "$first_update" "$desc" \
        >> "$RESULTS_TSV"
    {
        echo "## ${timestamp} iteration=${iteration} status=${status}"
        echo ""
        echo "- commit: ${commit}"
        echo "- steady_sps_median: ${sps}"
        echo "- compile_time_estimate_s: ${compile}"
        echo "- first_update_time_s: ${first_update}"
        echo "- description: ${desc}"
        echo "- expected_impact: ${expected}"
        echo "- risk: ${risk}"
        echo ""
    } >> "$MEMORY_FILE"
}

decision_for_candidate() {
    local candidate_sps="$1"
    local best_sps="$2"
    local candidate_compile="$3"
    local best_compile="$4"
    python3 - "$candidate_sps" "$best_sps" "$candidate_compile" "$best_compile" \
        "$MIN_SPS_REL_IMPROVE" "$SPS_NOISE_TOLERANCE" "$MIN_COMPILE_REL_IMPROVE" <<'PY'
import sys

candidate_sps = float(sys.argv[1])
best_sps = float(sys.argv[2])
candidate_compile = float(sys.argv[3])
best_compile = float(sys.argv[4])
min_sps_rel = float(sys.argv[5])
sps_noise = float(sys.argv[6])
min_compile_rel = float(sys.argv[7])

if candidate_sps >= best_sps * (1.0 + min_sps_rel):
    print("keep:steps_per_second")
elif (
    candidate_sps >= best_sps * (1.0 - sps_noise)
    and candidate_compile <= best_compile * (1.0 - min_compile_rel)
):
    print("keep:compile_time_tiebreaker")
else:
    print("discard:no_improvement")
PY
}

prompt_only() {
    init_run_dir
    local prompt_file="${RUN_DIR}/prompt_preview.md"
    build_prompt "preview" "unknown" "unknown" "$prompt_file"
    echo "Wrote prompt preview: $prompt_file"
}

benchmark_only() {
    init_run_dir
    local json_file
    json_file="$(run_benchmark "manual" "manual")"
    echo "steady_sps_median=$(json_field "$json_file" steady_sps_median)"
    echo "compile_time_estimate_s=$(json_field "$json_file" compile_time_estimate_s)"
}

main() {
    init_run_dir
    ensure_editable_scope_clean

    echo "============================================================"
    echo "JAXborg Claude optimization loop"
    echo "Worktree: $WORKTREE"
    echo "Run dir:  $RUN_DIR"
    echo "Recipe:   $BENCHMARK_RECIPE"
    echo "Max iter: $MAX_ITER"
    echo "============================================================"

    echo "Running baseline correctness..."
    run_correctness "baseline"

    echo "Running baseline benchmark..."
    local baseline_json
    baseline_json="$(run_benchmark "baseline" "baseline")"
    local best_sps
    local best_compile
    local baseline_first
    best_sps="$(json_field "$baseline_json" steady_sps_median)"
    best_compile="$(json_field "$baseline_json" compile_time_estimate_s)"
    baseline_first="$(json_field "$baseline_json" first_update_time_s)"
    append_result "baseline" "$(git rev-parse --short HEAD)" "baseline" "$best_sps" "$best_compile" "$baseline_first" "baseline"

    local iteration
    for iteration in $(seq 1 "$MAX_ITER"); do
        echo ""
        echo "============================================================"
        echo "Iteration $iteration / $MAX_ITER"
        echo "Current best: ${best_sps} steady_sps_median, ${best_compile}s compile estimate"
        echo "============================================================"

        ensure_editable_scope_clean
        local start_ref
        start_ref="$(git rev-parse HEAD)"
        local guard_before
        guard_before="$(guard_hash)"

        if ! run_claude_iteration "$iteration" "$best_sps" "$best_compile"; then
            append_result "$iteration" "$(git rev-parse --short HEAD)" "agent_failed" "NA" "NA" "NA" "claude failed"
            discard_candidate "$start_ref"
            continue
        fi

        if [[ "$(git rev-parse HEAD)" != "$start_ref" ]]; then
            echo "Claude created a commit despite instructions; resetting branch to keep evaluation script-owned."
            git reset --soft "$start_ref"
        fi

        local guard_after
        guard_after="$(guard_hash)"
        if [[ "$guard_after" != "$guard_before" ]]; then
            echo "Guarded files changed. Stopping so the harness is not silently altered."
            echo "Review guarded changes, then restore or commit intentionally."
            exit 3
        fi

        local claude_log="${RUN_DIR}/logs/claude_${iteration}.log"
        local desc
        desc="$(experiment_description "$claude_log")"
        local expected
        expected="$(experiment_field "EXPECTED_IMPACT" "$claude_log")"
        local risk
        risk="$(experiment_field "RISK" "$claude_log")"

        if [[ -z "$(editable_scope_status)" ]]; then
            echo "Claude made no editable changes."
            append_result "$iteration" "$(git rev-parse --short HEAD)" "discard:no_change" "NA" "NA" "NA" "$desc" "$expected" "$risk"
            continue
        fi

        echo "Candidate editable changes:"
        git status --short -- "${EDITABLE_PATHS[@]}"

        if ! run_correctness "$iteration"; then
            append_result "$iteration" "$(git rev-parse --short HEAD)" "discard:correctness_failed" "NA" "NA" "NA" "$desc" "$expected" "$risk"
            discard_candidate "$start_ref"
            continue
        fi

        local candidate_json
        if ! candidate_json="$(run_benchmark "$iteration" "candidate_${iteration}")"; then
            append_result "$iteration" "$(git rev-parse --short HEAD)" "discard:benchmark_failed" "NA" "NA" "NA" "$desc" "$expected" "$risk"
            discard_candidate "$start_ref"
            continue
        fi

        local candidate_sps
        local candidate_compile
        local candidate_first
        candidate_sps="$(json_field "$candidate_json" steady_sps_median)"
        candidate_compile="$(json_field "$candidate_json" compile_time_estimate_s)"
        candidate_first="$(json_field "$candidate_json" first_update_time_s)"

        local decision
        decision="$(decision_for_candidate "$candidate_sps" "$best_sps" "$candidate_compile" "$best_compile")"
        echo "Decision: $decision"

        if [[ "$decision" == keep:* ]]; then
            commit_candidate "$desc"
            local kept_commit
            kept_commit="$(git rev-parse --short HEAD)"
            append_result "$iteration" "$kept_commit" "$decision" "$candidate_sps" "$candidate_compile" "$candidate_first" "$desc" "$expected" "$risk"
            best_sps="$candidate_sps"
            best_compile="$candidate_compile"
        else
            append_result "$iteration" "$(git rev-parse --short HEAD)" "$decision" "$candidate_sps" "$candidate_compile" "$candidate_first" "$desc" "$expected" "$risk"
            discard_candidate "$start_ref"
        fi
    done
}

case "${1:-}" in
    --prompt-only)
        prompt_only
        ;;
    --benchmark-only)
        benchmark_only
        ;;
    ""|run)
        main
        ;;
    *)
        echo "Unknown argument: $1"
        echo "Usage: bash scripts/dev/optimize_perf/loop_claude.sh [--prompt-only|--benchmark-only]"
        exit 2
        ;;
esac
