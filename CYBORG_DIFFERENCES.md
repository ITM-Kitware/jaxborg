# CybORG CC4 JAX Port: Open Differences

This file tracks active parity gaps only. Resolved items have been removed to keep the
list focused on remaining work.

## Open Session-Identity Gaps

1. Remaining scan-memory parity gaps still appear in long random-policy fuzz runs
   (current example: `seed=5 step=198`, multi-field drift on `red_agent_3`:
   `red_sessions`, `red_privilege`, `red_scanned_hosts`, `host_compromised`).
   This indicates remaining per-session reassignment/restore semantics are still
   collapsing incorrectly into host-level state.
   - `src/jaxborg/reassignment.py`
   - `src/jaxborg/actions/blue_restore.py`

## Additional Open Differences

### Observation Layout: Fixed vs Variable Body Size

CybORG `BlueFlatWrapper` uses a variable-length observation body based on how many
subnets the blue agent controls, then appends messages and pads to 210.

JAX keeps a fixed 3-subnet body for all blue agents, then appends messages. This keeps
training input shape uniform but means raw vector indices differ from CybORG for
agents that monitor fewer than 3 subnets.

- JAX: `src/jaxborg/observations.py`
- CybORG: `BlueFlatWrapper.observation_change()`

### CC4Env Agent Interface: Blue-Only vs Exposed Red Actions

CybORG CC4 is blue-controlled with red behavior produced by internal FSM agents.

JAX `CC4Env` currently accepts both blue and red actions in the action dict (while
`FsmRedCC4Env` exists for blue-only control). This is still a behavior/interface
difference for users of `CC4Env`.

- JAX: `src/jaxborg/env.py`
- CybORG: enterprise simulation controller + FSM red agents
