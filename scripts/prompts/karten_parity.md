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
- `src/jaxborg/env.py` — apply_all_actions_typed (training path), apply_all_actions_in_order (harness path), execution order shuffling
- `src/jaxborg/reassignment.py` — cross-subnet session reassignment after each step
- `src/jaxborg/topology.py` — static topology construction
- `src/jaxborg/cyborg_green_recorder.py` — records CybORG green/red/blue action events per step
- `tests/differential/harness.py` — CC4DifferentialHarness (syncs CybORG events into JAX)
- `tests/differential/state_comparator.py` — compare_fast, _ERROR_FIELDS classification
- `tests/l3/` — L3 rollout tests (random blue + trained policy)
- CybORG source: `.venv/lib/python3.11/site-packages/CybORG/`

### Commands
```bash
# CPU-only — always set these before running tests
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu

uv run pytest tests/subsystems/ -v -x -n auto   # L1 property tests
uv run pytest tests/differential/ -v -x -n auto  # L2 interaction tests
BLUE_CHECKPOINT=$BLUE_CHECKPOINT uv run pytest tests/l3/ -v -x -n auto  # L3 full rollout
uv run python scripts/eval/transfer.py \
  --checkpoint $BLUE_CHECKPOINT --episodes 30 --independent-rollouts --seed 42  # L4 transfer eval
uv run ruff check --fix . && uv run ruff format .   # lint
```

## Verification Hierarchy (Karten et al.)

We follow a 4-level hierarchical verification approach:

- **L1 Property**: Individual component tests in isolation (tests/subsystems/)
- **L2 Interaction**: Cross-module differential tests (tests/differential/)
- **L3 Rollout**: Full episode comparison — random blue (50 seeds × 500 steps) AND trained IPPO policy (100 seeds × 500 steps)
- **L4 Transfer**: Train in JAX, evaluate independently in both JAXborg + CybORG (TOST equivalence, Δ=200)

Failures at higher levels trigger root-cause analysis and new L1/L2 regressions.
The iterative cycle drives convergence — not any single pass.

### Current Verification Status

${VERIFICATION_STATUS}

### Architecture Notes

**Execution order**: `apply_all_actions_typed` (training path) shuffles blue/green/red
agent execution order within each phase per step, matching CybORG's random shuffle
of same-priority actions. The shuffle key is derived from `key_green` with distinct
fold-in constants per phase.

**FSM host knowledge**: `fsm_host_entered` is updated from discover, scan, exploit,
privesc, reassignment, and a post-step bulk `|= red_sessions`. This matches CybORG's
`_process_new_observations` which adds ALL hosts from the observation to `host_states`.
The harness asserts (not syncs) parity via `_assert_fsm_host_entered`.

**Session selection**: CybORG's FSM picks a random session from `server_session`
(P(success) = 1/N where N = len(server_session)). CybORG's `server_session`
dict never removes destroyed sessions — after Blue Restore, phantom IDs persist,
inflating N and making exploits fail more often. JAXborg replicates this via
`red_server_session_count` (monotonic high-water mark updated end-of-step).
In the harness, `red_exploit_session_choices` syncs the choice index from CybORG.
See `CYBORG_DIFFERENCES.md` ("Exploit source-session selection") for details.

### The Sync Problem

The differential harness syncs CybORG outcomes into JAX each step. Classification:

**RNG syncs (KEPT — bridge different RNG implementations):**
- `green_randoms`, `green_host_order` (execution order), `detection_randoms`
- `red_privesc_choices`, `red_session_check_choices/hosts`
- `red_pid_deltas`, `blue_decoy_pid_deltas`
- `red_exploit_session_ok` (harness limitation — always True, see above)

**Removed (JAX computes its own):**
- `forced_primary_hosts/pids`, `red_impact_attempted`, `green_lwf/asf_this_step`

**Assertions (verify parity, don't sync):**
- `_assert_fsm_host_entered` — warns on mismatch, does not copy

### Common Pitfalls

- **Shape bugs in shuffles/permutations**: When shuffling arrays of length
  `GLOBAL_MAX_HOSTS` that have inactive padding, only permute active entries.
  A raw `jax.random.permutation(key, GLOBAL_MAX_HOSTS)` will mix active and
  inactive indices — the `fori_loop` only processes `num_green_agents` entries,
  so active hosts shuffled beyond that range silently disappear. Always verify
  with a test that counts how many unique active agents actually execute.
- **Duration parity**: CybORG sets `self.duration` in `__init__` before `execute()`.
  Duration is committed at scheduling time regardless of success/failure.
- **Post-step fsm_host_entered**: Must be updated AFTER reassignment and session
  checks, before the next step's FSM action selection.

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
   - Shape/index bug (silent data loss from array shape mismatches — see Common Pitfalls)
3. **Write a failing regression test FIRST** — targeted L1 or L2 test that reproduces the gap
4. **Read CybORG source** at `.venv/lib/python3.11/site-packages/CybORG/` to understand reference behavior
5. **Fix the JAX code** (or the recorder, if it's a recording gap) — not CybORG
6. **Run tests**: L1/L2 first, then L3
7. **Commit** with semantic message (fix:, test:, refactor:)
8. **Write handoff** to `.agent_handoff/handoff.md`

### For L4 Failures

L4 runs each backend FULLY INDEPENDENTLY — same policy weights, matched topology
seeds, but independent RNG for everything (red, green, detection, etc.). It
compares population mean rewards via TOST. A failing L4 means the simulation
produces different reward *distributions* for the same policy.

**The key diagnostic is directionality of the mean reward difference (JAXborg - CybORG).**
Independent RNG adds noise but should NOT add bias. If the mean difference is
consistently positive or negative across runs, that is a real simulation bug —
regardless of what any sleep or random baseline shows. Sleep baselines only test
the null-policy path; the trained policy exercises different code paths (Restore,
Monitor, Remove) that may diverge in ways sleep never triggers.

1. Check the SIGN of the mean reward gap. A consistent direction (e.g. JAXborg
   always worse) means the sim diverges under active blue play — find the mechanism
2. Instrument per-step reward breakdowns (RIA, LWF, ASF) to find which component diverges
3. Compare per-step state snapshots at the first divergence point
4. Read CybORG source for that subsystem, write a targeted L1/L2 test, fix it
5. Do NOT dismiss the gap as "expected RNG divergence" or build workarounds
   (sleep baselines, margin increases, wider TOST margins)
6. Do NOT conclude "simulation is correct" based on sleep baseline equivalence —
   sleep does not exercise Restore/Remove/Monitor code paths

## Rules

- Fix **ONE** gap per run — the highest-impact failure first
- Every fix needs a regression test
- Do NOT add new syncs to the harness to hide gaps
- Do NOT modify CybORG source
- Do NOT diverge JAXborg's action masks from CybORG's. The environment's
  `compute_blue_action_mask` MUST match CybORG's BlueFlatWrapper masks exactly.
  If you want to improve training (e.g. mask out no-op actions), do it in the
  training script or a policy wrapper — NOT in the environment code. Parity is
  the top priority.
- Do NOT cancel any Slurm jobs
- When submitting srun jobs, always add `--comment="$KARTEN_JOB_TAG"` so the loop can clean up after you
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
