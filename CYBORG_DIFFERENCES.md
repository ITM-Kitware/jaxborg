# CybORG CC4 JAX Port: Parity Status

This file tracks parity between JAXborg and the CybORG reference implementation:
active gaps, intentionally matched quirks, intentional divergences, and the
historical workaround inventory.

## Open Parity Gaps

### PID memory is bounded in JAX
CybORG PID memory is unbounded, while JAX PID identity storage is bounded by
fixed capacities (`MAX_TRACKED_SESSION_PIDS` and `MAX_TRACKED_SUSPICIOUS_PIDS`,
currently 34 each). This affects:
- Red session PID identity memory (`red_session_pids`)
- Blue suspicious PID memory (`blue_suspicious_pids`)

The differential harness/test setup fail fast on overflow instead of silently
truncating, but core JAX state is still bounded.
- `src/jaxborg/constants.py`
- `tests/differential/harness.py`

### Stubbed exploits: EternalBlue & BlueKeep
`apply_exploit_eternalblue` and `apply_exploit_bluekeep` in
`src/jaxborg/actions/red_exploit.py` are no-ops (return state unchanged).
CybORG's `FiniteStateRedAgent` never selects these exploit types, so this has
no effect on FSM-driven training via `FsmRedCC4Env`.

### `CC4Env` accepts red actions
CybORG CC4 is blue-controlled with red behavior produced by internal FSM agents.
JAX `CC4Env` currently accepts both blue and red actions in the action dict
(while `FsmRedCC4Env` exists for blue-only control). This is still a
behavior/interface difference for users of `CC4Env`.
- JAX: `src/jaxborg/env.py`
- CybORG: enterprise simulation controller + FSM red agents

### Open workarounds
None currently tracked.

## Intentionally Matched CybORG Quirks

These are CybORG behaviors that look like bugs but are intentionally replicated
in JAXborg for parity. Do not "fix" them in JAXborg without also verifying
CybORG changed.

### Impact rewards penalize failed attempts only while the red agent still has a session
`BlueRewardMachine` line 118: `elif 'red' in agent_name and success and isinstance(action, Impact)`.
`success` is a `TernaryEnum` where `bool(TernaryEnum.FALSE)` is `True`, so ALL
Impact actions with at least one active red session still get penalized
regardless of outcome. If the red agent has no active sessions,
`BlueRewardMachine` skips the reward entirely. JAXborg matches via
`red_impact_attempted` only when the red agent still has a session.

- CybORG: `Shared/BlueRewardMachine.py:118`
- JAXborg: `src/jaxborg/actions/red_impact.py`, `src/jaxborg/env.py`

### GreenAccessService never checks destination service availability
`if not self.available_dest_service:` (line 176) checks the method object
(always truthy), not its return value — should be
`self.available_dest_service(state)`. The service availability check is dead
code, so GreenAccessService only fails on blocked traffic. JAXborg matches by
only triggering ASF on blocked traffic.

- CybORG: `Simulator/Actions/GreenActions/GreenAccessService.py:176`
- JAXborg: `src/jaxborg/actions/green.py` (ASF block at line ~205)

### GreenAccessService always assumes reachable hosts exist
`if len(reachable_hosts) < 0:` (line 112) is never true (len is never
negative), so `None` is never returned. If reachable_hosts were empty,
`np_random.choice([])` would crash. In practice there are always reachable
servers.

- CybORG: `Simulator/Actions/GreenActions/GreenAccessService.py:112`

## Intentional Divergences (JAXborg is more correct)

These are cases where JAXborg deliberately does NOT replicate a CybORG behavior
because the CybORG behavior is buggy/inefficient and matching it would hurt
training.

### Exploit source-session selection
CybORG's `FiniteStateRedAgent` picks a **random session ID** from
`server_session` (the action-space session dict) when creating
`ExploitRemoteService`. This is an implementation quirk: the `session`
parameter falls through to the generic `np_random.choice(options)` path (no
special handling like `hostname` or `ip_address` get). Scan results (`ports`
dict) are stored **per-session** — only the session that performed the scan
has the target's port data. When the FSM picks a session that didn't scan the
target, the exploit fails with `self.ip_address not in session.ports`. The
effect only manifests when green phishing is active (adding extra abstract
sessions to `server_session`).

**Key mechanism:** `server_session` only contains `RedAbstractSession`-type
sessions (primary + green phishing). Concrete exploit sessions (SSH,
RED_REVERSE_SHELL) have session types not in `SESSION_TYPES`, so
`ActionSpace.update()` never adds them. Additionally, `_filter_obs` strips
sessions on hosts outside the agent's `allowed_subnets`. This keeps
`server_session` small (~3-5 entries at step 500) despite 10-15+ total active
sessions. CybORG exploit failure rate is ~73%.

`server_session` also never **removes** entries when sessions are destroyed by
Blue Restore/Remove (`ActionSpace.update()` only adds; `reset()` only runs at
episode end; `RestoreFromBackup` returns an empty `Observation()` so the
update loop has no signal to remove destroyed sessions). Phantom entries stay
with `value=True`, inflating N in the `1/N` session-selection roll.

JAXborg uses **deterministic anchor-based source selection**
(`select_scan_execution_source_host`) and a per-(agent, target, source) scan
ownership matrix (`red_scanned_source_hosts`). This always finds the correct
source host regardless of how many other sessions exist.

**JAXborg replication of the 1/N failure:** JAXborg replicates CybORG's
session selection at FSM action-creation time (matching CybORG's
`get_action()` timing). It counts N = abstract sessions in allowed subnets at
the step the exploit is queued, then rolls 1/N at execution time — exactly
one of the N sessions holds scan data, so `P(success) = 1/N`. The N count is
phantom-inclusive to match CybORG's stale `server_session`. The retired
runtime replay tape that forced CybORG's choice index has been replaced by
live differential traces and explicit translated red actions in targeted
debug tests.

- CybORG: `Actions/AbstractActions/ExploitRemoteService.py:175` (`session.ports` check)
- CybORG: `Agents/SimpleAgents/FiniteStateRedAgent.py:330` (session selection)
- CybORG: `Shared/ActionSpace.py:208-212` (`SESSION_TYPES` filter + monotonic add)
- CybORG: `Shared/ActionSpace.py:139` (`reset()` clears `server_session` at episode end only)
- CybORG: `Simulator/SimulationController.py:1054-1066` (`_filter_obs` by allowed subnets)
- CybORG: `Actions/ConcreteActions/RedSessionCheck.py:58-65` (end-turn reports all sessions)
- CybORG: `Simulator/Actions/ConcreteActions/RestoreFromBackup.py` (empty `Observation()`)
- JAXborg: `src/jaxborg/actions/red_common.py` (`compute_visible_sessions`, `exploit_common_preconditions`)
- JAXborg: `src/jaxborg/actions/duration.py` (`process_red_with_duration`)
- JAXborg: `src/jaxborg/state.py` (`red_abstract_session_count`, `red_pending_visible_sessions`)

### `action_cost` mirrors CybORG's caller-submission accounting
CybORG's `SimulationController._step:310` sums `actions.get(agent, Action()).cost`
across the caller-submitted action dict every step. With Restore's
`duration == 5`, a policy that re-submits Restore on the 4 follow-up busy
ticks is charged −1 per submission even though the busy ticks execute
`Sleep()`. CC4's headline scorer
(`BlueFixedActionWrapper.step:171–175`, `Evaluation/evaluation.py:110`)
inherits this via `sum(reward.values())`.

JAXborg now mirrors that contract in `compute_reward_breakdown`: −1 for
every step a Restore action is submitted by the caller, regardless of busy
state. Earlier versions gated on `is_initiating = (blue_pending_ticks == 0)`
(charging only at logical-Restore initiation), which under-charged by ~+4
per Restore episode and produced a +109..+175 directional headline-reward
gap vs CybORG.

- CybORG: `Simulator/SimulationController.py:310–311`
- JAXborg: `src/jaxborg/rewards.py:62–69`
- Regression test: `tests/differential/test_reward_cc4_contract.py`
- Differential harness now compares `sum(controller.reward["Blue"].values())`
  against the JAX full-contract reward (`tests/differential/harness.py:1167–1190`).

## Closed Workarounds

Historical JAXborg shims that were not direct CybORG mechanics and have since
been removed.

1. **Anchor/abstract minimum-retention guard in Remove**
   - Removed from: `src/jaxborg/actions/blue_remove.py`
   - Prior behavior: prevented clearing last session based on anchor/abstract heuristics.
   - Reason removed: not supported by CybORG `Remove`/`StopProcess` source behavior.

2. **Reusing stale blue suspicious PID for phishing-created session PID**
   - Removed from: `src/jaxborg/actions/green.py`
   - Prior behavior: phishing could allocate a stale suspicious PID.
   - Reason removed: not supported by CybORG `PhishingEmail` and host PID allocation path.

3. **`blue_suspicious_pid_budget` drives Remove loop span**
   - Removed from: `src/jaxborg/actions/blue_remove.py`
   - Prior behavior: Remove iterated up to `max(row_max_slot, pid_budget)` instead of concrete suspicious entries.
   - Reason removed: CybORG iterates `sus_pids[hostname]` list entries only.

4. **Global budget field for suspicious PIDs**
   - Removed from: `src/jaxborg/state.py` and dependent action paths.
   - Prior behavior: state carried a separate suspicious-PID budget path.
   - Reason removed: CybORG stores only `sus_pids` list data; budget was non-source-backed.

5. **Exploit success broad writes directly into blue suspicious memory**
   - Removed from: `src/jaxborg/actions/red_common.py`
   - Prior behavior: exploit appended PID sets directly to blue suspicious rows.
   - Reason removed: CybORG suspicious memory is monitor/event-driven, not exploit-write driven.

## Rules For New Fixes

- Any new non-source-backed shim must be documented under "Open workarounds"
  in the same PR, with a replacement plan and removal condition.
- When a shim is removed, move its entry to "Closed Workarounds".
