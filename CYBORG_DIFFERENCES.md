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

CybORG's `FiniteStateRedAgent` picks a **random session ID** from the action space
when creating `ExploitRemoteService`. Scan results (`ports` dict) are stored
**per-session** — only the session that performed the scan has the target's port
data. When green phishing adds extra sessions to a source host, the FSM may
randomly pick a phishing session (empty `ports`) instead of the scanning session,
causing the exploit to fail (`self.ip_address not in session.ports`).

JAXborg uses **deterministic anchor-based source selection**
(`select_scan_execution_source_host`) and a per-(agent, target, source) scan
ownership matrix (`red_scanned_source_hosts`). This always finds the correct
source host regardless of how many other sessions exist.

**Impact:** With green agents active, CybORG's red agents waste exploit turns
picking wrong sessions, slowing red progression. JAXborg's red spreads ~15%
faster to unique hosts (68 vs 60 unique (agent,host) pairs) and reaches ROOT on
~35% more hosts (67 vs 50 ROOT pairs) over 500 steps with blue=Sleep. This
produces a systematic ~1900-point reward gap (JAX worse = harder environment for
blue) that shrinks to ~300 points with a trained blue policy.

With green disabled (red-only), the gap is ≈ 0 — confirming the divergence is
purely from the green-phishing ↔ exploit-session interaction.

**Parity flag:** `FsmRedCC4Env(cyborg_random_exploit_source=True)` adds a
probabilistic exploit penalty based on session distribution. The exact
CybORG mechanism is still under investigation — the simple model
over-corrects, suggesting CybORG's FSM action_space may only expose
session 0 (abstract) rather than all sessions.

- CybORG: `Actions/AbstractActions/ExploitRemoteService.py:175` (`session.ports` check)
- CybORG: `Agents/SimpleAgents/FiniteStateRedAgent.py:330` (session selection)
- JAXborg: `src/jaxborg/actions/red_common.py:150` (`exploit_common_preconditions` + `cyborg_random_exploit_source` flag)

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
