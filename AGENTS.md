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
uv run pytest                                        # fast suite (~7 min, excludes `-m slow`)
uv run pytest -m slow                                # slow L3 fuzz + full-episode parity (~40 min)
uv run pytest -m ""                                  # everything
uv run pytest tests/subsystems/test_red_discover.py -v     # single test file
uv run pytest tests/subsystems/test_red_discover.py::TestClassName::test_name -v  # single test
```

Training output goes to `$JAXBORG_EXP_DIR` (defaults to `./jaxborg-exp` relative to cwd). Set this in your shell profile.

pytest-xdist can be used for parallel test execution (e.g., `-n 4`). On WSL, avoid xdist — each worker loads JAX + CybORG into a separate process which may exhaust memory.


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

## Testing

```bash
uv run pytest                    # fast suite (~7 min, excludes slow)
uv run pytest -m slow            # slow-only (L3 rollouts, full-episode fuzz)
uv run pytest -m "" -x           # everything, stop on first failure
```

Test infrastructure lives in `tests/differential/` (harness, action translator, state comparator) for CybORG↔JAX comparison. CybORG source is at `.venv/lib/python3.11/site-packages/CybORG/`.

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
