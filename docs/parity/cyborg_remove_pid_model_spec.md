# CybORG Remove/PID Parity Spec

## Scope
This spec defines the CybORG source-of-truth model for:
- red session identity and PID lifecycle
- blue suspicious PID memory (`sus_pids`)
- Blue `Remove` behavior (`Remove -> StopProcess`)

It is intentionally narrow and is the authoritative contract for parity work in JAXborg.

## CybORG Source-Of-Truth

### Session and PID model
- Sessions are concrete objects keyed as `state.sessions[agent][session_id]`.
- Each session has its own `pid`, `hostname`, `username`, `session_type`, `parent`, `active`.
- Host-local session registration is maintained in `Host.sessions[agent]` and process list in `Host.processes`.

Sources:
- `.venv/lib/python3.11/site-packages/CybORG/Simulator/State.py:85`
- `.venv/lib/python3.11/site-packages/CybORG/Shared/Session.py:9`
- `.venv/lib/python3.11/site-packages/CybORG/Simulator/Host.py:189`

### Blue suspicious PID memory
- `VelociraptorServer` owns `sus_pids: Dict[str, List[int]]`.
- `add_sus_pids` appends directly (no dedupe/normalization).
- `Monitor` (blue default end-turn action in Enterprise) populates suspicious PID memory from host events:
  - network connection events with `event.pid`
  - process creation events with `event['pid']`

Sources:
- `.venv/lib/python3.11/site-packages/CybORG/Shared/Session.py:154`
- `.venv/lib/python3.11/site-packages/CybORG/Simulator/Actions/AbstractActions/Monitor.py:35`
- `.venv/lib/python3.11/site-packages/CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py:695`
- `.venv/lib/python3.11/site-packages/CybORG/Simulator/SimulationController.py:280`

### Remove mechanics
- `Remove` chooses a blue session on target host and iterates `sus_pids[target_host]`.
- For each suspicious PID, it executes `StopProcess(target_session=<chosen session>, pid=<sus_pid>)`.
- `StopProcess` only kills when PID resolves to a live process and process user is not privileged (`root`/`SYSTEM`) unless `stop_all`.
- Kill path is PID-based: `state.get_session_from_pid(hostname, pid)` determines exactly which session is removed.

Sources:
- `.venv/lib/python3.11/site-packages/CybORG/Simulator/Actions/AbstractActions/Remove.py:42`
- `.venv/lib/python3.11/site-packages/CybORG/Simulator/Actions/ConcreteActions/StopProcess.py:23`
- `.venv/lib/python3.11/site-packages/CybORG/Simulator/State.py:420`

### Phishing mechanics
- `PhishingEmail` creates a new `RedAbstractSession` (new PID via host/session add path).
- `PhishingEmail` itself does not add suspicious PID entries.
- Any suspicious PID visibility for phishing-created sessions must come from monitor/event flow.

Source:
- `.venv/lib/python3.11/site-packages/CybORG/Simulator/Actions/ConcreteActions/PhishingEmail.py:94`

## Required JAX Parity Semantics
1. Treat session identity and PID as first-class state, not inferred counters.
2. Update blue suspicious PID memory from event-equivalent mechanics, not host compromise heuristics.
3. Implement `Remove` as PID-driven kill attempts over blue suspicious PID list for that host.
4. Never preserve/remove sessions based on JAX-only guardrails (anchor floors, abstract retention heuristics) unless directly traceable to CybORG.
5. Keep stale suspicious PIDs as stale entries; they can fail to match and should not implicitly mutate live sessions.

## Anti-Patterns (Do Not Add)
- Session retention floors tied to anchor/session type when PID match says kill.
- Reusing stale suspicious PID values as new session PIDs.
- Conflating suspicious PID cardinality with live session count.

## Acceptance Criteria For This Spec
- For any parity bug in `host_compromised`/`red_sessions`/`red_privilege` around `Remove`, the fix must be explainable as one of the source-backed rules above.
- New tests must cite relevant source rule(s) and validate behavior with explicit CybORG differential assertions.
