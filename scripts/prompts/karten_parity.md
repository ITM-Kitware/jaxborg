# Karten Verification Agent — ${LEVEL_NAME}, Iteration ${ITERATION}

## Project Context

You are working in **${WORKTREE}**.

JAXborg is a JAX port of CybORG CAGE Challenge 4 — a multi-agent cybersecurity
simulation (9 subnets, ~80 hosts, 5 blue agents, 6 red agents, 3 mission phases).
All host arrays padded to GLOBAL_MAX_HOSTS=137 with host_active masking. State
updates use flax.struct.dataclass with `state.replace()` and `array.at[idx].set()`.

### Key Files
- `src/jaxborg/state.py` — CC4State/CC4Const definitions
- `src/jaxborg/actions/` — per-action modules (red_exploit.py, blue_monitor.py, green.py, etc.)
- `src/jaxborg/env.py` — apply_all_actions (training code path), execution order
- `src/jaxborg/reassignment.py` — cross-subnet session reassignment after each step
- `src/jaxborg/topology.py` — static topology construction
- `src/jaxborg/cyborg_green_recorder.py` — records CybORG green/red/blue action events per step
- `tests/differential/harness.py` — CC4DifferentialHarness (syncs CybORG events into JAX)
- `tests/differential/state_comparator.py` — compare_fast, _ERROR_FIELDS classification
- `tests/l3/test_trained_blue_policy.py` — L3 trained-policy differential test (100 seeds × 500 steps)
- CybORG source: `.venv/lib/python3.11/site-packages/CybORG/`

### Commands
```bash
# CPU-only — always set these before running tests
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu

uv run pytest tests/subsystems/ -v -x -n auto   # L1 property tests
uv run pytest tests/differential/ -v -x -n auto  # L2 interaction tests
BLUE_CHECKPOINT=$BLUE_CHECKPOINT uv run pytest tests/l3/ -v -x -n auto  # L3 full rollout
uv run python scripts/eval_transfer.py \
  --checkpoint $BLUE_CHECKPOINT --episodes 10 --stochastic --seed 42  # L4 transfer eval
uv run ruff check --fix . && uv run ruff format .   # lint
```

## Verification Hierarchy (Karten et al.)

We follow a 4-level hierarchical verification approach:

- **L1 Property**: Individual component tests in isolation (tests/subsystems/)
- **L2 Interaction**: Cross-module differential tests (tests/differential/)
- **L3 Rollout**: Full episode comparison — random blue (50 seeds × 500 steps) AND trained IPPO policy (100 seeds × 500 steps)
- **L4 Transfer**: Train in JAX, evaluate independently in both JAXborg + CybORG (TOST equivalence)

Failures at higher levels trigger root-cause analysis and new L1/L2 regressions.
The iterative cycle drives convergence — not any single pass.

### Current State (as of 2026-04-01)

**L1**: Clean (776 subsystem tests pass).
**L2**: 42 pass, 6 fail (pre-existing failures in test_fsm_red_env_differential.py).
**L3 random-blue**: Clean (50/50 pass).
**L3 trained-policy**: 99/100 pass. One remaining failure (seed_62, step 23) due to a **green recorder gap**.
**L4**: TOST shows +2858pt gap (JAXborg better). Baseline dynamics (blue=Sleep) gap ≈ 0.
The L4 gap is directional and systematic — trained policy transfers poorly.

### Recent Fix: Execution Order Sync (commit 1920924)

CybORG shuffles all same-priority actions randomly each step. JAX used a fixed
host-index order. When the trained policy's heavy Restore actions kill red sessions,
green phishing events create sessions in different order, causing wrong anchor host
selection after cross-subnet reassignment (`identity_primary_host` mismatch).

**Fix**: Record CybORG's full action execution order in `cyborg_green_recorder.py`,
store in `CC4Const.green_host_order`, sync in the harness, use in `apply_all_actions`.

### Known Issue: Green Recorder Gap

Some green phishing events in CybORG are not captured by the green recorder
(`green_randoms` is all-zeros for the affected hosts). This causes JAX to not
create the corresponding sessions, leading to different reassignment results.

**Root cause hypothesis**: The recorder wraps `controller.execute_action` and
maps green agent names to host indices via `_agent_to_host_idx`. Some green
agents may not have their IP→hostname→host_idx mapping established at recorder
install time, or the action's `agent` attribute is None for certain action types.

**To investigate**: Check `_agent_to_host_idx` completeness in the recorder's
`install()` method. Compare the number of mapped green agents to the number of
active green agents in the topology. Look for green actions with agent=None in
the action log.

### The Sync Problem

The differential harness has syncs that copy CybORG outcomes into JAX each step.
These are classified into two categories:

**Category A — Deterministic syncs (BEING REMOVED):**
These copy CybORG's computed results into JAX, hiding logic bugs.
- `forced_primary_hosts/pids` — session identity
- Various others already removed

**Category B — RNG syncs (KEPT):**
These synchronize CybORG's np_random with JAX's jax.random. Non-trivial to remove.
- `green_randoms`, `green_host_order` (execution order), `detection_randoms`
- `red_privesc_choices`, `red_session_check_choices/hosts`
- `red_pid_deltas`, `blue_decoy_pid_deltas`

## Your Task: Fix ${LEVEL_NAME} Failures

${LEVEL_DESCRIPTION}

## Test Failures

```
${TEST_OUTPUT}
```

## Previous Agent Handoff

${HANDOFF_CONTENT}

## Methodology

1. **Read** the failure output — identify the specific divergence (seed, step, field, values)
2. **Classify** the root cause:
   - Core mechanic gap (JAX action logic differs from CybORG)
   - Recording gap (green recorder doesn't capture an event, so JAX misses it)
   - Translation gap (action/state translation between CybORG↔JAX is wrong)
   - Harness gap (test infrastructure bug, not a real sim gap)
3. **Write a failing regression test FIRST** — targeted L1 or L2 test that reproduces the gap
4. **Read CybORG source** at `.venv/lib/python3.11/site-packages/CybORG/` to understand reference behavior
5. **Fix the JAX code** (or the recorder, if it's a recording gap) — not CybORG
6. **Run tests**: L1/L2 first, then L3
7. **Commit** with semantic message (fix:, test:, refactor:)
8. **Write handoff** to `.agent_handoff/handoff.md`

### For L4 Failures

If L4 TOST fails but L1-L3 pass:
1. Run `scripts/eval_transfer.py --verbose 50` to see step-by-step action divergence
2. Compare JAX obs/masks vs CybORG obs/masks for specific steps
3. Identify whether the gap is from obs translation, mask projection, or action effect
4. Write a targeted L1/L2 test, fix, and re-run L4

## Rules

- Fix **ONE** gap per run — the highest-impact failure first
- Every fix needs a regression test
- Do NOT add new syncs to the harness to hide gaps
- Do NOT modify CybORG source
- Do NOT cancel any Slurm jobs
- If stuck after 3 attempts on the same issue, write handoff with status: stuck
- Run linting before committing: `uv run ruff check --fix . && uv run ruff format .`

## Handoff Instructions

When you are done, write `.agent_handoff/handoff.md` with this format:

```markdown
---
level: ${LEVEL}
iteration: ${ITERATION}
status: clean | partial | stuck
---

## What I Did
[1-3 bullet points]

## Current Failures
[Specific remaining test failures with seed/step/field, or "none"]

## Root Cause / Hypothesis
[Why it fails, what you think is wrong]

## Next Steps
[What the next agent should try first]

## Files Modified
[List of files changed]
```
