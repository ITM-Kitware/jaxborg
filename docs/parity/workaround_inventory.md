# Parity Workaround Inventory

This file tracks temporary JAXborg shims that are not direct CybORG mechanics and must be removed.

## Open Items

None currently tracked.

## Closed Items

1. Anchor/abstract minimum-retention guard in Remove
- Removed from: `src/jaxborg/actions/blue_remove.py`
- Prior behavior: prevented clearing last session based on anchor/abstract heuristics.
- Reason removed: not supported by CybORG `Remove`/`StopProcess` source behavior.

2. Reusing stale blue suspicious PID for phishing-created session PID
- Removed from: `src/jaxborg/actions/green.py`
- Prior behavior: phishing could allocate a stale suspicious PID.
- Reason removed: not supported by CybORG `PhishingEmail` and host PID allocation path.

3. `blue_suspicious_pid_budget` drives Remove loop span
- Removed from: `src/jaxborg/actions/blue_remove.py`
- Prior behavior: Remove iterated up to `max(row_max_slot, pid_budget)` instead of concrete suspicious entries.
- Reason removed: CybORG iterates `sus_pids[hostname]` list entries only.

4. Global budget field for suspicious PIDs
- Removed from: `src/jaxborg/state.py` and dependent action paths.
- Prior behavior: state carried a separate suspicious-PID budget path.
- Reason removed: CybORG stores only `sus_pids` list data; budget was non-source-backed.

5. Exploit success broad writes directly into blue suspicious memory
- Removed from: `src/jaxborg/actions/red_common.py`
- Prior behavior: exploit appended PID sets directly to blue suspicious rows.
- Reason removed: CybORG suspicious memory is monitor/event-driven, not exploit-write driven.

## Rules For New Fixes
- Any new non-source-backed shim must be documented here in the same PR.
- Every shim entry must include a replacement plan and removal condition.
