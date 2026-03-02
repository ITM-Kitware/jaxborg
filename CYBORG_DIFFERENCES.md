# CybORG CC4 JAX Port: Open Differences

This file tracks active parity gaps only. Resolved items have been removed to keep the
list focused on remaining work.

## Open Session-Identity Gaps

1. Remaining scan-memory parity gaps still appear in long random-policy fuzz runs
   (examples: `seed=0 step=211`, `seed=2 step=215`; `red_scanned_hosts` mismatch for `red_agent_3`).
   This indicates there are still edge cases in session-id lifecycle binding for
   deferred/red FSM scan flows.
   - `src/jaxborg/actions/duration.py`
   - `tests/differential/harness.py`

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
