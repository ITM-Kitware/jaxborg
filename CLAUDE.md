# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

JAX port of CybORG CAGE Challenge 4 (CC4) — a multi-agent cybersecurity simulation (9 subnets, ~80 hosts, 5 blue agents, 6 red agents, 3 mission phases). Re-implements CC4 as JIT-compilable JAX arrays for GPU-accelerated parallel RL training via JaxMARL's `MultiAgentEnv` interface.

## Worktree Layout

This repo uses a bare-repo worktree setup:

```
jaxborg/
  .bare/     ← bare git repo
  .git       ← points to .bare
  main/      ← worktree on main branch
  parity/    ← worktree on parity branch
```

**When setting up a new worktree:**

```bash
cd /home/paulhax/src/cyber/jaxborg
git worktree add <name> <branch>
cd <name>
uv sync
```

**Always be aware of which worktree you are working in.** Keep the specific worktree path in mind across conversations and when making plans — file paths, test commands, and commits all target a specific worktree. Reference the full worktree path (e.g. `/home/paulhax/src/cyber/jaxborg/parity/`) rather than just `jaxborg/` to avoid ambiguity.

## Commands

```bash
uv sync                                              # install deps
uv run pytest tests/ -v -m "not slow"  # fast tests only (~10 min)
uv run pytest tests/ -v  # all tests including slow integration
uv run pytest tests/subsystems/test_red_discover.py -v     # single test file
uv run pytest tests/subsystems/test_red_discover.py::TestClassName::test_name -v  # single test
```

Training output goes to `../jaxborg-exp/`.

**Do NOT use pytest-xdist (`-n auto`, `-n 4`, etc.).** Each worker loads JAX + CybORG into a separate process, which exhausts WSL memory and crashes the system. Always run tests serially.

Tests marked `@pytest.mark.slow` are integration/smoke tests (full episode runs, JIT training, bugcheck catalog). Skip them during core logic iteration with `-m "not slow"`.

## Architecture

### Core Data Structures (`src/jaxborg/state.py`)

Two `flax.struct.dataclass` PyTrees:

- **CC4Const** — static topology: host properties, subnet adjacency, data links, agent assignments, phase rewards. Built once per episode via `build_topology()` (pure JAX) or `build_const_from_cyborg()` (extracts from CybORG instance).
- **CC4State** — dynamic per-step state: compromise levels, red sessions/privilege/discovery, activity tracking, decoys, blocked zones, messages, FSM states. Created via `create_initial_state()`.

All host-indexed arrays are padded to `GLOBAL_MAX_HOSTS=137` with `host_active` masking. State updates use `state.replace(field=new_value)` and `array.at[idx].set(value)`.

### Action System (`src/jaxborg/actions/`)

Actions are integer-encoded. `encoding.py` defines the action space layout (ranges of ints mapping to action type + target). Red actions dispatch through `apply_red_action()` which decodes then branches via `jax.lax.cond` to per-action handlers (discover, scan, 8 exploit types, privesc, impact). Blue actions dispatch through `apply_blue_action()`.

Each action module (e.g., `red_exploit.py`, `blue_monitor.py`) exports an `apply_*` function: `(CC4State, CC4Const, agent_id, target) -> CC4State`.

### Topology (`src/jaxborg/topology.py`)

Builds the static `CC4Const` from seed. Hardcoded subnet adjacency (NACLs), router backbone, host generation with per-subnet counts. Two entry points: `build_topology(seeds, num_steps)` for pure JAX, `build_const_from_cyborg(env)` for extracting from a live CybORG instance.

## Development Workflow

Implementation is driven by a 22-subsystem catalog (`tests/catalog.py`) with dependency ordering. `scripts/ralph_wiggum.sh` automates the loop: get next subsystem, implement JAX code + differential tests, validate, commit.

Check progress: `tests/catalog_status.json` tracks which subsystems are passing.

### Differential Testing

Every test must compare JAX output against CybORG. Pattern:
1. Create CybORG env via `cyborg_env` fixture (conftest.py): SleepAgent blue, EnterpriseGreenAgent green, FiniteStateRedAgent red, seed=42, 500 steps
2. Build JAX const: `build_const_from_cyborg(cyborg_env)`
3. Create JAX state: `create_initial_state()`
4. Execute identical action sequences in both environments
5. Compare relevant state fields (host_compromised, red_sessions, red_privilege, etc.)

Test infrastructure lives in `tests/differential/` (harness, action translator, state comparator).

Differential harness replay is strict-only: do not add CybORG->JAX state synchronization/patching to hide parity gaps.

**Tests must exercise the training code path.** Differential tests should call the same functions used during training (`FsmRedCC4Env.step_env`, `apply_red_action`, `apply_green_agents`, `fsm_red_post_step_update`, etc.) rather than reimplementing logic in test code. Shared functions like `fsm_red_post_step_update` are extracted so both the training env and the differential harness use the same code. If a test needs custom orchestration (e.g. action duration tracking for CybORG parity), the JAX-side logic should still call into production functions.

### TDD + Differential Test Quality (Required)

For every fuzzer-found parity bug, follow strict red-green-refactor:
1. **Red**: add a failing regression test first.
2. **Green**: implement the smallest JAX fix that makes that exact test pass.
3. **Refactor**: clean up only after parity is restored and tests stay green.

Primary regression tests for parity bugs must be **explicit differential tests**:
- Run both CybORG and JAX in the same test (typically via `CC4DifferentialHarness`).
- Pin the exact reproduction context (seed, step, agent/action, hostnames/indices, field).
- Assert concrete before/after state on both sides, then assert no diff for the target field.
- Do not rely only on generic assertions like "no diffs at step N" without explicit context checks.

Unit tests are allowed as secondary guardrails, but they do **not** replace the required differential regression for parity bugs.

Hard rule for parity regressions:
- Seed/step replay is triage only; do not commit replay-only regressions.
- Do not commit tests that call `run_differential_fuzz(...)` as the main assertion.
- Committed regression must be mechanism-explicit:
  - set concrete preconditions that trigger the bug
  - run matching CybORG and JAX actions
  - assert the exact divergent field/value parity
- If CybORG has a concrete mechanic/model that JAXborg lacks (e.g., PID/session identity, action lifecycle state), implement that model in `src/jaxborg/` rather than adding heuristic workaround logic.

#### Good 4-step TDD loop (example)

Example gap: `seed=0 step=130 host_compromised [host_80] cyborg=1 jax=0` caused by Blue `Remove`.

1. **Reproduce and isolate context**
   - Re-run the failing seed/step and inspect the exact CybORG action and state preconditions.
   - In this example: `Remove` was called with `sus_pids` present, but PID was stale (no live process), so CybORG did not clear the red session.
2. **Write one explicit failing differential test**
   - Add a subsystem differential test that sets up only the required state and compares CybORG and JAX for that mechanic.
   - In this example: create a stale suspicious PID case for `Remove` and assert both sides keep the user session.
3. **Apply minimal production fix**
   - Change only `src/jaxborg/` logic needed for parity; do not rewrite broad behavior.
   - In this example: gate JAX `Remove` clearing on stronger, CybORG-consistent preconditions.
4. **Verify and iterate**
   - Run the new regression first, then nearby subsystem tests, then rerun fuzzing to get the next first mismatch.
   - Keep one-gap-at-a-time discipline: reproduce -> test -> fix -> rerun.

### Precomputed Randoms for Deterministic Testing

CybORG and JAX use independent RNG streams, making direct comparison of random-dependent behavior (green agents, detection rolls) impossible without synchronization. The solution is precomputed random arrays stored in `CC4State`:

- **Green agents**: `green_randoms` `(MAX_STEPS, GLOBAL_MAX_HOSTS, 7)` float array + `use_green_randoms` bool. The 7 fields encode action choice, service selection, reliability roll, FP roll, phishing roll, dest host, and access FP roll as `[0,1)` uniforms. `sample_green_random()` in `actions/rng.py` uses `jax.lax.cond` to read from the array or fall back to JAX RNG. `tests/differential/green_recorder.py` wraps CybORG's `np_random` to capture values, then the harness injects them into JAX state via `sync_green_rng=True`.
- **Detection**: `detection_randoms` flat sequence + `detection_random_index` counter. `sample_detection_random()` in `actions/rng.py`.

To run green parity tests: `uv run pytest tests/test_green_parity.py -v` (requires CybORG).
To run green unit tests (no CybORG): `uv run pytest tests/test_green_unit.py -v`.

## Fixing Differential Gaps

The differential fuzzer (`tests/differential/fuzzer.py`) systematically finds CybORG/JAX state divergences by running full episodes across many seeds. Run it:

```bash
uv run python -m tests.differential.fuzzer          # 20 seeds × 100 steps
```

It returns a `MismatchReport` on the first error: seed, step, field name, CybORG vs JAX values.

### Investigation workflow

1. **Reproduce (TDD red)**: write a failing **explicit differential regression** that runs CybORG and JAX to the failing seed/step and asserts the exact mismatch context.
   - Prefer the relevant subsystem file and explicit state setup; avoid seed/step replay-only tests.
   - Keep tests differential (pure CybORG + JAX in the same test), not JAX-only.
   - One gap at a time: stop on first mismatch, add test, fix, rerun.
2. Place the test in the appropriate file:
   - Red action bugs → `tests/subsystems/test_red_*.py` (exploit, discover, privesc, etc.)
   - Blue action bugs → `tests/subsystems/test_blue_*.py`
   - Green/phishing bugs → `tests/test_green_parity.py`
   - Session/topology bugs → `tests/subsystems/test_dynamic_topology.py`
   - FSM state machine bugs → `tests/subsystems/test_fsm_red_agent.py`
   - Reward/phase bugs → `tests/subsystems/test_rewards.py` or `tests/subsystems/test_phase_transitions.py`
   - Cross-cutting or unclear → place in the closest subsystem differential test file and make the context explicit in the test name/body
3. **Read CybORG source** at `.venv/lib/python3.11/site-packages/CybORG/` to understand the mechanic causing the divergence
4. **Fix the JAX code** in `src/jaxborg/` to match CybORG's behavior
5. **Verify targeted tests first** (new regression + closest subsystem tests), then run:
   - `uv run pytest tests/ -v -m "not slow" -x`
   - rerun differential fuzzing to find the next gap
6. **Lint**: `uv run ruff check --fix . && uv run ruff format .`
7. **Commit** with a message describing the gap and fix

### Common root causes

- JAX doesn't model a CybORG mechanic (action duration, session reassignment, observation-based tracking)
- JAX success conditions differ from CybORG (missing service check, wrong subnet logic, no decoy handling)
- Translator creates wrong CybORG action type or missing parameters (e.g. ExploitRemoteService needs a FixedExploitSelector)
- Timing: CybORG multi-tick actions (duration > 1) need deferred JAX execution via `_red_pending_jax` in the harness

### Key CybORG entry points for investigation

- `SimulationController.step()` — action submission, `actions_in_progress` duration tracking
- `ActionSpace.update()` — observation-based validity tracking (subnets, IPs)
- `replace_action_if_invalid()` — action type/parameter validation
- `ExploitRemoteService.execute()` — delegates to concrete exploit via selector
- `FiniteStateRedAgent` — CybORG's FSM, uses `ExploitRemoteService` (duration=4)
- `different_subnet_agent_reassignment()` — green phishing session transfers

## Linting

Run `uv run ruff check --fix . && uv run ruff format .` before committing.

## JAX Constraints

- `jax.lax.cond()` for branching (no Python if/else in JIT code)
- No Python loops over dynamic values in JIT code
- `flax.struct.dataclass` for PyTree-compatible state
- Use `numpy` for host indexing in tests; `jax.numpy` for JIT-compiled logic

## Reference

CC2 JAX port at `/home/paulhax/src/cyber/jaxmarl/integration/jaxmarl/environments/cage/` for patterns.

CybORG source installed at `.venv/lib/python3.11/site-packages/CybORG/`.
