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

## Stubbed Exploits: EternalBlue & BlueKeep

`apply_exploit_eternalblue` and `apply_exploit_bluekeep` in `src/jaxborg/actions/red_exploit.py`
are no-ops (return state unchanged). CybORG's `FiniteStateRedAgent` never selects these
exploit types, so this has no effect on FSM-driven training via `FsmRedCC4Env`.

## Intentionally Matched CybORG Quirks

These are CybORG behaviors that look like bugs but are intentionally replicated in
JAXborg for parity. Do not "fix" them in JAXborg without also verifying CybORG changed.

### Impact rewards penalize all attempts, not just successes

`BlueRewardMachine` line 118: `elif 'red' in agent_name and success and isinstance(action, Impact)`.
`success` is a `TernaryEnum` where `bool(TernaryEnum.FALSE)` is `True`, so ALL Impact
actions that reach execution get penalized regardless of outcome. JAXborg matches via
`red_impact_attempted` flag set unconditionally in `apply_impact`.

- CybORG: `Shared/BlueRewardMachine.py:118`
- JAXborg: `src/jaxborg/actions/red_impact.py`, `src/jaxborg/env.py`

### GreenAccessService never checks destination service availability

`if not self.available_dest_service:` (line 176) checks the method object (always truthy),
not its return value — should be `self.available_dest_service(state)`. The service
availability check is dead code, so GreenAccessService only fails on blocked traffic.
JAXborg matches by only triggering ASF on blocked traffic.

- CybORG: `Simulator/Actions/GreenActions/GreenAccessService.py:176`
- JAXborg: `src/jaxborg/actions/green.py` (ASF block at line ~205)

### GreenAccessService always assumes reachable hosts exist

`if len(reachable_hosts) < 0:` (line 112) is never true (len is never negative), so
`None` is never returned. If reachable_hosts were empty, `np_random.choice([])`
would crash. In practice there are always reachable servers.

- CybORG: `Simulator/Actions/GreenActions/GreenAccessService.py:112`

## Additional Open Differences

### CC4Env Agent Interface: Blue-Only vs Exposed Red Actions

CybORG CC4 is blue-controlled with red behavior produced by internal FSM agents.

JAX `CC4Env` currently accepts both blue and red actions in the action dict (while
`FsmRedCC4Env` exists for blue-only control). This is still a behavior/interface
difference for users of `CC4Env`.

- JAX: `src/jaxborg/env.py`
- CybORG: enterprise simulation controller + FSM red agents
