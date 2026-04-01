#!/usr/bin/env bash
# Train→Eval→Verify loop for L4 cross-backend policy transfer (Karten et al.).
#
# Each iteration:
#   1. Train IPPO in JAXborg
#   2. Eval trained policy independently in JAXborg + CybORG (runs TOST)
#   3. Feed full eval output to claude -p to review/diagnose
#   4. If claude finds issues → fix, run tests, commit, retrain
#   5. If claude confirms TOST passes and everything looks good → done
#
# Usage:
#   srun --gres=gpu:1 --mem=64G bash scripts/transfer_loop.sh
#
# Environment variables:
#   MAX_ROUNDS          - train→verify iterations (default: 10)
#   TRAIN_TIMESTEPS     - training timesteps per round (default: 5000000)
#   TRAIN_NUM_ENVS      - parallel envs for training (default: 1024)
#   EVAL_EPISODES       - eval episodes per round (default: 10)
#   MAX_FIX_ATTEMPTS    - claude retries per round (default: 3)
#   TOPOLOGY_MODE       - topology mode for training (default: bank)

set -euo pipefail
# Resolve repo root. Under sbatch, BASH_SOURCE points to spool dir, so
# prefer JAXBORG_ROOT env var or SLURM_SUBMIT_DIR, falling back to script path.
if [ -n "${JAXBORG_ROOT:-}" ]; then
    cd "$JAXBORG_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR/.."
fi

# Ensure Ctrl+C kills child processes (uv/python/JAX)
trap 'echo ""; echo "Interrupted. Killing children..."; kill 0; exit 130' INT TERM

MAX_ROUNDS="${MAX_ROUNDS:-10}"
TRAIN_TIMESTEPS="${TRAIN_TIMESTEPS:-5000000}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-1024}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
MAX_FIX_ATTEMPTS="${MAX_FIX_ATTEMPTS:-3}"
TOPOLOGY_MODE="${TOPOLOGY_MODE:-cyborg_bank}"

EXP_DIR="${JAXBORG_EXP_DIR:-$(pwd)/jaxborg-exp}"

export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-1}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jaxborg/xla}"
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
    echo "--- Step 1: Training IPPO ($TRAIN_TIMESTEPS steps, $TRAIN_NUM_ENVS envs, topology=$TOPOLOGY_MODE) ---"
    uv run python scripts/train_ippo_cc4.py \
        TOTAL_TIMESTEPS="$TRAIN_TIMESTEPS" \
        NUM_ENVS="$TRAIN_NUM_ENVS" \
        TOPOLOGY_MODE="$TOPOLOGY_MODE" \
        +TOPOLOGY_BANK_SIZE=32 \
        SEED="$round" \
        hydra.run.dir="$ROUND_DIR/hydra" \
        hydra.job.chdir=True

    # Training writes to EXP_DIR/ippo_cc4_{timestamp}/; find latest checkpoint
    CHECKPOINT=$(ls -t "$EXP_DIR"/ippo_cc4*/checkpoint_final.pkl 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found in $EXP_DIR/ippo_cc4*/"
        exit 1
    fi
    TRAIN_SRC="$(dirname "$CHECKPOINT")"
    cp "$CHECKPOINT" "$ROUND_DIR/checkpoint_final.pkl"
    cp "$TRAIN_SRC/metrics.jsonl" "$ROUND_DIR/metrics.jsonl" 2>/dev/null || true
    cp "$TRAIN_SRC/config.json" "$ROUND_DIR/config.json" 2>/dev/null || true

    # --- 2. Eval on JAXborg + CybORG (includes TOST) ---
    echo ""
    echo "--- Step 2: Evaluating transfer ($EVAL_EPISODES episodes, independent) ---"
    EVAL_OUTPUT=$(uv run python scripts/eval_transfer.py \
        --checkpoint "$CHECKPOINT" \
        --episodes "$EVAL_EPISODES" \
        --baselines \
        --seed "$round" 2>&1) || true

    echo "$EVAL_OUTPUT"
    echo "$EVAL_OUTPUT" > "$ROUND_DIR/eval_output.txt"

    # --- 3. Send full eval to claude for review ---
    echo ""
    echo "--- Step 3: Claude review (round $round) ---"

    ATTEMPT=0
    RESOLVED=false

    while [ $ATTEMPT -lt $MAX_FIX_ATTEMPTS ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo "--- Review attempt $ATTEMPT/$MAX_FIX_ATTEMPTS ---"

        CLAUDE_OUTPUT=$(claude -p "You are verifying and debugging L4 cross-backend policy transfer for JAXborg (JAX port of CybORG CC4).
Working directory: $(pwd)

## Context

Round $round of the Karten verification loop. A policy was trained via IPPO in
JAXborg for $TRAIN_TIMESTEPS steps, then evaluated independently in both JAXborg
and CybORG. The full eval output (including TOST equivalence test) is below.

## Full Eval Output

$EVAL_OUTPUT

## Your Task

Review the ENTIRE eval output. Check:

1. **TOST result** — did the equivalence test pass (p<0.05 within margin)?
2. **Reward gap** — JAXborg mean vs CybORG mean. Is it directional or noise?
3. **Action distributions** — are the same action types used at similar rates?
4. **Baselines** — do sleep/random baselines look reasonable in both?
5. **Training quality** — did the policy actually learn (reward >> sleep baseline)?
6. **Any anomalies** — unexpected patterns, NaN, crashes, suspicious numbers?

## Decision

If TOST passes AND everything looks healthy:
- Write 'VERDICT: PASS' on its own line
- Summarize why you're confident in the equivalence

If TOST fails OR you find issues:
- Write 'VERDICT: FAIL' on its own line
- Diagnose the specific gap (obs? reward? dynamics? masking?)
- Read relevant source code to understand the root cause
- Write a differential test that reproduces the gap
- Fix the JAXborg code
- Run: uv run pytest tests/ -v -x
- Run: uv run ruff check --fix . && uv run ruff format .
- Commit with a clear message

## Key Files

- CybORG source: .venv/lib/python3.11/site-packages/CybORG/
- JAXborg source: src/jaxborg/
- Translation: src/jaxborg/translate.py
- Observations: src/jaxborg/observations.py
- Rewards: src/jaxborg/rewards.py
- Diagnostics: scripts/diagnose_*.py
- CYBORG_DIFFERENCES.md — known intentional divergences

## Rules

- Review everything, not just one number
- One gap at a time if fixing — most impactful first
- Every fix needs a differential test
- Do not change CybORG source or hide gaps with harness syncs
- Run ruff before committing
" \
            --allowedTools "Read,Edit,Write,Bash(uv run*),Bash(git add*),Bash(git commit*),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(ls*),Bash(python3*),Bash(CUDA_VISIBLE_DEVICES*),Grep,Glob" 2>&1) || true

        echo "$CLAUDE_OUTPUT" > "$ROUND_DIR/claude_review_${ATTEMPT}.txt"

        # Check verdict
        if echo "$CLAUDE_OUTPUT" | grep -q "VERDICT: PASS"; then
            echo ""
            echo "================================================================"
            echo "  TOST PASSED — L4 equivalence confirmed (round $round)"
            echo "================================================================"
            echo "$CLAUDE_OUTPUT" | grep -A 20 "VERDICT: PASS" | head -25
            RESOLVED=true
            break
        fi

        if echo "$CLAUDE_OUTPUT" | grep -q "VERDICT: FAIL"; then
            echo "Claude found issues, checking if fix was committed..."
            # Validate: run tests after fix
            echo "--- Validating fix (tests) ---"
            if uv run pytest tests/ -v -x --timeout=120 2>&1 | tail -20; then
                echo "Tests pass after fix. Will retrain next round."
                RESOLVED=true
                break
            else
                echo "Tests failed on attempt $ATTEMPT"
            fi
        else
            echo "No clear verdict from claude. Retrying..."
        fi
    done

    if [ "$RESOLVED" = true ]; then
        # Check if it was a PASS (done!) or FAIL (need retrain)
        if echo "$CLAUDE_OUTPUT" | grep -q "VERDICT: PASS"; then
            echo ""
            echo "L4 cross-backend transfer verified. Done!"
            exit 0
        fi
        echo ""
        echo "Round $round: fix applied. Retraining in next round..."
    else
        echo ""
        echo "FAILED to resolve after $MAX_FIX_ATTEMPTS attempts in round $round."
        echo "Review $ROUND_DIR/ and fix manually."
        exit 1
    fi
done

echo ""
echo "================================================================"
echo "  Transfer Loop Complete ($MAX_ROUNDS rounds without TOST pass)"
echo "  Review results in $EXP_DIR/transfer_loop/"
echo "================================================================"
exit 1
