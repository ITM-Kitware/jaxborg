# Karten Verification Agent — ${LEVEL_NAME}, Iteration ${ITERATION}

## Project Context

You are working in **${WORKTREE}** on the **karten** branch.

JAXborg is a JAX port of CybORG CAGE Challenge 4 — a multi-agent cybersecurity
simulation (9 subnets, ~80 hosts, 5 blue agents, 6 red agents, 3 mission phases).
All host arrays padded to GLOBAL_MAX_HOSTS=137 with host_active masking. State
updates use flax.struct.dataclass with `state.replace()` and `array.at[idx].set()`.

### Key Files
- `src/jaxborg/state.py` — CC4State/CC4Const definitions
- `src/jaxborg/actions/` — per-action modules (red_exploit.py, blue_monitor.py, etc.)
- `src/jaxborg/env.py` — apply_all_actions (training code path)
- `src/jaxborg/topology.py` — static topology construction
- `tests/differential/harness.py` — CC4DifferentialHarness (sync-heavy test infra)
- `tests/differential/state_comparator.py` — compare_fast, field classification
- CybORG source: `.venv/lib/python3.11/site-packages/CybORG/`

### Commands
```bash
# CPU-only — always set these before running tests
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu

uv run pytest tests/ -v -x -n auto             # all tests, parallel, stop on first failure
uv run pytest tests/subsystems/ -v -x -n auto  # L1 property tests
uv run pytest tests/differential/ -v -x -n auto # L2 interaction tests
uv run pytest tests/l3/ -v -x -n auto # L3 independent tests
uv run ruff check --fix . && uv run ruff format .   # lint
```

## Verification Hierarchy (Karten et al.)

We follow a 4-level hierarchical verification approach:

- **L1 Property**: Individual component tests in isolation (tests/subsystems/)
- **L2 Interaction**: Cross-module differential tests (tests/differential/)
- **L3 Rollout**: Full episode comparison, 50 seeds x 500 steps, random blue (tests/l3/)
- **L4 Transfer**: Train in JAX, evaluate in CybORG (TOST equivalence)

Failures at higher levels trigger root-cause analysis and new L1/L2 regressions.
The iterative cycle drives convergence — not any single pass.

### The Sync Problem

The differential harness has syncs that copy CybORG outcomes into JAX each step.
These are classified into two categories:

**Category A — Deterministic syncs (BEING REMOVED):**
These copy CybORG's computed results into JAX, hiding logic bugs. The production
training path already works without them.
- `red_impact_attempted` — impact outcomes (harness.py ~line 1063-1086)
- `green_lwf/asf_this_step` — green failure events (harness.py ~line 1088-1123)
- `forced_primary_hosts/pids` — session identity (harness.py ~line 1036-1039)
- `red_abstract_host_rank` — abstract rank (harness.py ~line 904-927)

**Category B — RNG syncs (KEPT):**
These are deliberate test infrastructure for synchronizing CybORG's np_random
with JAX's jax.random. Non-trivial to remove, not workarounds.
- `green_randoms`, `detection_randoms`, `red_privesc_choices`, `red_session_check_choices`

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
   - Translation gap (action/state translation between CybORG↔JAX is wrong)
   - Harness gap (test infrastructure bug, not a real sim gap)
3. **Write a failing regression test FIRST** — targeted L1 or L2 test that reproduces the gap
4. **Read CybORG source** at `.venv/lib/python3.11/site-packages/CybORG/` to understand reference behavior
5. **Fix the JAX code** — not the harness, not CybORG
6. **Run tests**: `uv run pytest tests/ -v -x`
7. **Commit** with semantic message (fix:, test:, refactor:)
8. **Write handoff** to `.agent_handoff/handoff.md`

## Rules

- Fix **ONE** gap per run — the highest-impact failure first
- Every fix needs a differential regression test
- Do NOT add new syncs to the harness to hide gaps
- Do NOT modify CybORG source
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
