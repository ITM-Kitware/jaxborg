# CybORG Bug: server_session Never Removes Destroyed Sessions

## Summary

CybORG's `ActionSpace.server_session` dict accumulates session IDs
monotonically and never removes entries when sessions are destroyed by
Blue Restore (or Blue Remove). After Restore clears all red sessions on
a host, the phantom session IDs remain in `server_session` with
`known=True`. The red FSM then picks from these phantoms when selecting
a session for `ExploitRemoteService`, causing the exploit to fail because
the chosen session no longer exists.

## Impact on Exploit Success Rate

The FSM selects a session uniformly at random from
`server_session` (filtered to `value=True` entries):

```python
# FiniteStateRedAgent.py, line ~320
options = [i for i, v in action_space['session'].items() if v]
action_params['session'] = self.np_random.choice(options)
```

Only the session that performed the scan has port data for the target.
If N = len(options), the exploit succeeds with probability 1/N.

After Restore destroys sessions, CybORG's N stays inflated (phantom
entries persist), while the true live session count drops. This makes
exploits fail more often in CybORG than they should, giving blue
artificially better rewards.

## Root Cause

`ActionSpace.update()` (ActionSpace.py:208-212) only **adds** entries:

```python
for session in info.get("Sessions", []):
    if "session_id" in session and session['agent'] in self.agent:
        if "Type" in session and (session["Type"] in SESSION_TYPES):
            self.server_session[session["session_id"]] = known
```

`server_session` is only cleared in `ActionSpace.reset()` (line 139),
which is called at episode end, never mid-episode.

`RestoreFromBackup.execute_targeteted_local_action()` returns an empty
`Observation()`, so the observation update loop has no signal to remove
the destroyed sessions from `server_session`.

## Reproduction

```python
# After 100 steps of sleep blue (red spreads freely), do Restore:
# Pre-restore:
#   red_agent_5: server_session active=6, live_sessions=9
# Post-restore:
#   red_agent_5: server_session active=6, live_sessions=0
#   *** STALE: 6 phantom sessions in server_session! ***
```

See `scripts/diagnose_server_session.py` for the full diagnostic.

## JAXborg Parity Fix

JAXborg must replicate this behavior: track a "server_session count"
per red agent that grows when new RedAbstractSession types are observed
but is NOT decremented by Restore or Remove. The exploit 1/N roll should
use this phantom-inclusive count instead of the true live abstract
session count.

## Affected Files

- `CybORG/Shared/ActionSpace.py` — `server_session` dict, `update()`, `reset()`
- `CybORG/Agents/SimpleAgents/FiniteStateRedAgent.py` — session selection
- `CybORG/Simulator/Actions/ConcreteActions/RestoreFromBackup.py` — empty observation
