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

## CybORG Bugs Affecting Reward Semantics

### GreenAccessService.available_dest_service never called (line 176)

`if not self.available_dest_service:` checks the method object (always truthy), not its
return value. Should be `self.available_dest_service(state)`. The service availability
check is dead code — GreenAccessService only fails on blocked traffic, never on
unavailable destination services. JAXborg matches this behavior by only triggering ASF
on blocked traffic.

- CybORG: `Simulator/Actions/GreenActions/GreenAccessService.py:176`

### GreenAccessService.random_reachable_ip unreachable None branch (line 112)

`if len(reachable_hosts) < 0:` is never true (len is never negative), so `None` is
never returned. If reachable_hosts is empty, `np_random.choice([])` crashes instead.
In practice there are always reachable servers, so this doesn't fire.

- CybORG: `Simulator/Actions/GreenActions/GreenAccessService.py:112`

## Additional Open Differences

### CC4Env Agent Interface: Blue-Only vs Exposed Red Actions

CybORG CC4 is blue-controlled with red behavior produced by internal FSM agents.

JAX `CC4Env` currently accepts both blue and red actions in the action dict (while
`FsmRedCC4Env` exists for blue-only control). This is still a behavior/interface
difference for users of `CC4Env`.

- JAX: `src/jaxborg/env.py`
- CybORG: enterprise simulation controller + FSM red agents
