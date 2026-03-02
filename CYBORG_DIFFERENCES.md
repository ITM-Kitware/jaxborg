# CybORG CC4 JAX Port: Open Differences

This file tracks active parity gaps only. Resolved items have been removed to keep the
list focused on remaining work.

## Open Heuristic Implementations (To Remove)

1. Scan-memory owner/source selection is based on argmax/mode over host-index arrays
   instead of concrete per-session identity/lifecycle matching.
   - `src/jaxborg/actions/red_common.py`

2. Cross-subnet reassignment merges transferred session attributes via aggregate
   `max/sum` reductions rather than strict per-session reassignment semantics.
   - `src/jaxborg/reassignment.py`

3. Green phishing source-agent selection is simplified (first-valid) and does not yet
   fully mirror CybORG `PhishingEmail` source selection behavior.
   - `src/jaxborg/actions/green.py`

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
