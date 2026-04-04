# CybORG CC4 JAX Port: Open Differences

This file tracks active parity gaps only. Resolved items have been removed to keep the
list focused on remaining work.

## Open Session-Identity Gaps

1. CybORG PID memory is unbounded, while JAX PID identity storage is bounded by
   fixed capacities (`MAX_TRACKED_SESSION_PIDS` and `MAX_TRACKED_SUSPICIOUS_PIDS`,
   currently 16 each). This affects:
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

### Impact rewards penalize failed attempts only while the red agent still has a session

`BlueRewardMachine` line 118: `elif 'red' in agent_name and success and isinstance(action, Impact)`.
`success` is a `TernaryEnum` where `bool(TernaryEnum.FALSE)` is `True`, so ALL Impact
actions with at least one active red session still get penalized regardless of outcome.
If the red agent has no active sessions, `BlueRewardMachine` skips the reward entirely.
JAXborg matches via `red_impact_attempted` only when the red agent still has a session.

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

## Intentional Divergences (JAXborg is more correct)

These are cases where JAXborg deliberately does NOT replicate a CybORG behavior
because the CybORG behavior is buggy/inefficient and matching it would hurt training.

### Exploit source-session selection

CybORG's `FiniteStateRedAgent` picks a **random session ID** from `server_session`
(the action-space session dict) when creating `ExploitRemoteService`. This is an
implementation quirk: the `session` parameter falls through to the generic
`np_random.choice(options)` path (no special handling like `hostname` or
`ip_address` get). Scan results (`ports` dict) are stored **per-session** — only
the session that performed the scan has the target's port data. When the FSM picks
a session that didn't scan the target, the exploit fails with
`self.ip_address not in session.ports`. The effect only manifests when green
phishing is active (adding extra abstract sessions to `server_session`).

**Key mechanism:** `server_session` only contains `RedAbstractSession`-type sessions
(primary + green phishing). Concrete exploit sessions (SSH, RED_REVERSE_SHELL) have
session types not in `SESSION_TYPES`, so `ActionSpace.update()` never adds them.
Additionally, `_filter_obs` strips sessions on hosts outside the agent's
`allowed_subnets`. This keeps `server_session` small (~3-5 entries at step 500)
despite 10-15+ total active sessions. CybORG exploit failure rate is ~73%.

JAXborg uses **deterministic anchor-based source selection**
(`select_scan_execution_source_host`) and a per-(agent, target, source) scan
ownership matrix (`red_scanned_source_hosts`). This always finds the correct
source host regardless of how many other sessions exist.

**JAXborg replication:** JAXborg replicates CybORG's session selection at FSM
action-creation time (matching CybORG's `get_action()` timing). It counts N =
abstract sessions in allowed subnets at the step the exploit is queued, then
rolls 1/N at execution time — exactly one of the N sessions holds scan data,
so `P(success) = 1/N`. In the differential harness, the outcome is synced from
CybORG via the `red_exploit_session_choices` array, which provides the choice
index so JAXborg exercises its own N computation.

- CybORG: `Actions/AbstractActions/ExploitRemoteService.py:175` (`session.ports` check)
- CybORG: `Agents/SimpleAgents/FiniteStateRedAgent.py:330` (session selection)
- CybORG: `Shared/ActionSpace.py:208-211` (`SESSION_TYPES` filter)
- CybORG: `Simulator/SimulationController.py:1054-1066` (`_filter_obs` by allowed subnets)
- CybORG: `Actions/ConcreteActions/RedSessionCheck.py:58-65` (end-turn reports all sessions)
- JAXborg: `src/jaxborg/actions/red_common.py` (`compute_visible_sessions`, `exploit_common_preconditions`)
- JAXborg: `src/jaxborg/actions/duration.py` (`process_red_with_duration`)
- JAXborg: `src/jaxborg/state.py` (`red_abstract_session_count`, `red_pending_visible_sessions`)

### action_cost not modeled

CybORG's `SimulationController` adds an `action_cost` reward component that sums
`action.cost` for each blue agent (e.g., Restore costs −1). JAXborg does not model
this component — it uses only `BlueRewardMachine` for rewards. The eval pipeline
(`scripts/eval_transfer.py`) extracts only `BlueRewardMachine` from CybORG rewards
to match.

- CybORG: `Simulator/SimulationController.py:310–311`
- JAXborg: `src/jaxborg/rewards.py` (no action_cost term)

## Additional Open Differences

### CC4Env Agent Interface: Blue-Only vs Exposed Red Actions

CybORG CC4 is blue-controlled with red behavior produced by internal FSM agents.

JAX `CC4Env` currently accepts both blue and red actions in the action dict (while
`FsmRedCC4Env` exists for blue-only control). This is still a behavior/interface
difference for users of `CC4Env`.

- JAX: `src/jaxborg/env.py`
- CybORG: enterprise simulation controller + FSM red agents
