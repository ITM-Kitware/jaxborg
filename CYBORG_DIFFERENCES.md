# CybORG CC4 JAX Port: Open Differences

This file tracks active parity gaps only. Resolved items have been removed to keep the
list focused on remaining work.

## Open Session-Identity Gaps

1. CybORG PID memory is unbounded, while JAX PID identity storage is bounded by
   fixed capacities (`MAX_TRACKED_SESSION_PIDS` and `MAX_TRACKED_SUSPICIOUS_PIDS`,
   currently 64 each). This affects:
   - Red session PID identity memory (`red_session_pids`)
   - Blue suspicious PID memory (`blue_suspicious_pids`)
   Differential harness/test setup now fail fast on overflow instead of silently truncating,
   but core JAX state is still bounded.
   - `src/jaxborg/constants.py`
   - `tests/differential/harness.py`

## Additional Open Differences

### CC4Env Agent Interface: Blue-Only vs Exposed Red Actions

CybORG CC4 is blue-controlled with red behavior produced by internal FSM agents.

JAX `CC4Env` currently accepts both blue and red actions in the action dict (while
`FsmRedCC4Env` exists for blue-only control). This is still a behavior/interface
difference for users of `CC4Env`.

- JAX: `src/jaxborg/env.py`
- CybORG: enterprise simulation controller + FSM red agents
