#!/usr/bin/env bash
# Train→Eval→Debug loop for policy transfer gap fixing.
#
# Each iteration:
#   1. Train IPPO from scratch (200K steps)
#   2. Eval trained policy on JAXborg + CybORG (10 episodes)
#   3. Run baselines for context
#   4. Feed diagnosis to claude -p to investigate and fix
#   5. Run tests to validate fix
#   6. Commit and repeat
#
# Usage:
#   bash scripts/transfer_loop.sh
#
# Environment variables:
#   MAX_ROUNDS          - number of train→fix iterations (default: 10)
#   TRAIN_TIMESTEPS     - training timesteps per round (default: 200000)
#   EVAL_EPISODES       - eval episodes per round (default: 10)
#   MAX_FIX_ATTEMPTS    - claude retries per round (default: 3)

set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure Ctrl+C kills child processes (uv/python/JAX)
trap 'echo ""; echo "Interrupted. Killing children..."; kill 0; exit 130' INT TERM

MAX_ROUNDS="${MAX_ROUNDS:-10}"
TRAIN_TIMESTEPS="${TRAIN_TIMESTEPS:-100000}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
MAX_FIX_ATTEMPTS="${MAX_FIX_ATTEMPTS:-3}"

EXP_DIR="$(pwd)/jaxborg-exp"

export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-1}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/.cache/jax}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

for round in $(seq 1 "$MAX_ROUNDS"); do
    echo ""
    echo "================================================================"
    echo "  Transfer Loop — Round $round / $MAX_ROUNDS"
    echo "================================================================"

    ROUND_DIR="$EXP_DIR/transfer_loop/round_${round}"
    mkdir -p "$ROUND_DIR"

    # --- 1. Train from scratch ---
    echo ""
    echo "--- Step 1: Training IPPO ($TRAIN_TIMESTEPS steps) ---"
    uv run python scripts/train_ippo_cc4.py \
        TOTAL_TIMESTEPS="$TRAIN_TIMESTEPS" \
        NUM_ENVS="$TRAIN_NUM_ENVS" \
        SEED="$round" \
        hydra.run.dir="$ROUND_DIR/hydra" \
        hydra.job.chdir=True

    # Training writes to EXP_DIR/ippo_cc4/; archive to per-round dir
    TRAIN_SRC="$EXP_DIR/ippo_cc4"
    CHECKPOINT="$TRAIN_SRC/checkpoint_final.pkl"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint at $CHECKPOINT after training"
        exit 1
    fi
    cp "$CHECKPOINT" "$ROUND_DIR/checkpoint_final.pkl"
    cp "$TRAIN_SRC/metrics.jsonl" "$ROUND_DIR/metrics.jsonl" 2>/dev/null || true
    cp "$TRAIN_SRC/config.json" "$ROUND_DIR/config.json" 2>/dev/null || true

    # --- 2. Eval on JAXborg + CybORG ---
    echo ""
    echo "--- Step 2: Evaluating transfer ($EVAL_EPISODES episodes) ---"
    EVAL_OUTPUT=$(uv run python scripts/eval_transfer.py \
        --checkpoint "$CHECKPOINT" \
        --episodes "$EVAL_EPISODES" \
        --baselines \
        --seed "$round" 2>&1) || true

    echo "$EVAL_OUTPUT"

    echo "$EVAL_OUTPUT" > "$ROUND_DIR/eval_output.txt"

    # --- 3. Feed to claude for diagnosis and fix ---
    echo ""
    echo "--- Step 3: Claude diagnosis and fix ---"

    ATTEMPT=0
    FIXED=false

    while [ $ATTEMPT -lt $MAX_FIX_ATTEMPTS ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo "--- Fix attempt $ATTEMPT/$MAX_FIX_ATTEMPTS ---"

        claude -p "You are debugging policy transfer gaps in JAXborg (JAX port of CybORG CC4).
Working directory: $(pwd)

## Context

A policy was just trained via IPPO in JAXborg for $TRAIN_TIMESTEPS steps (round $round),
then evaluated on both JAXborg and CybORG. The eval output is below.

## Eval Output

$EVAL_OUTPUT

## Your Task

Analyze the transfer gap between JAXborg and CybORG reward/behavior.
The gap means either:
1. **Observation gap** — JAXborg obs encoding differs from CybORG BlueFlatWrapper obs
2. **Reward gap** — JAXborg reward computation differs from CybORG
3. **Dynamics gap** — JAXborg state transitions differ (action effects, green agents, red FSM)
4. **Action masking gap** — JAXborg action masks differ from CybORG valid actions
5. **Action translation gap** — jax_blue_to_cyborg in translate.py maps incorrectly

## Investigation Steps

1. Read the eval output carefully. Focus on:
   - Reward gap (JAXborg mean vs CybORG mean)
   - Action distribution differences (which action types are over/under-used?)
   - Trajectory shape (does CybORG get worse at a specific phase?)
2. If the reward gap is large, diagnose WHY the same policy produces different rewards.
   - Run scripts/diagnose_reward_parity.py to compare step-by-step rewards with sleep policy.
   - Check if obs encodings match: compare JAXborg obs vs CybORG BlueFlatWrapper obs for the same state.
   - Check action masking: are the same actions valid in both?
3. Read relevant source:
   - CybORG source: .venv/lib/python3.11/site-packages/CybORG/
   - JAXborg source: src/jaxborg/
   - Translation: src/jaxborg/translate.py
   - Observations: src/jaxborg/observations.py
   - Rewards: src/jaxborg/rewards.py
4. Write an explicit differential test that reproduces the specific gap.
5. Fix the JAXborg code (not harness workarounds).
6. Run: uv run pytest tests/ -v -x
7. Commit with a clear message.

## Rules

- One gap at a time. Fix the most impactful issue first.
- Every fix needs a differential test.
- Do not change CybORG or harness to hide gaps.
- Prefer functional programming.
- Run ruff: uv run ruff check --fix . && uv run ruff format .
" \
            --allowedTools "Read,Edit,Write,Bash(uv run*),Bash(git add*),Bash(git commit*),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(ls*),Bash(python3*),Grep,Glob"

        # Validate: run tests
        echo "--- Validating fix (tests) ---"
        if uv run pytest tests/ -v -x --timeout=120 2>&1 | tail -20; then
            FIXED=true
            break
        else
            echo "Tests failed on attempt $ATTEMPT"
        fi
    done

    if [ "$FIXED" = false ]; then
        echo ""
        echo "FAILED to fix transfer gap after $MAX_FIX_ATTEMPTS attempts in round $round."
        echo "Stopping. Review $ROUND_DIR/eval_output.txt and fix manually."
        exit 1
    fi

    echo ""
    echo "Round $round complete. Fix committed. Proceeding to next training run."
done

echo ""
echo "================================================================"
echo "  Transfer Loop Complete ($MAX_ROUNDS rounds)"
echo "================================================================"
