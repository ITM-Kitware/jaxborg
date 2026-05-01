import jax.numpy as jnp

from jaxborg.constants import (
    ACTIVITY_NONE,
    ACTIVITY_SCAN,
    MAX_TRACKED_SUSPICIOUS_PIDS,
    NUM_BLUE_AGENTS,
)
from jaxborg.state import SimulatorConst, SimulatorState


def apply_blue_monitor(state: SimulatorState, const: SimulatorConst, agent_id: int | None = None) -> SimulatorState:
    if agent_id is None:
        for b in range(NUM_BLUE_AGENTS):
            state = apply_blue_monitor(state, const, b)
        return state

    covers = const.blue_agent_hosts[agent_id]
    has_any_activity = state.red_activity_this_step != ACTIVITY_NONE
    has_scan_activity = state.red_activity_this_step == ACTIVITY_SCAN
    newly_detected = has_any_activity & covers
    # Check for any process creation event including no-PID sentinels (-2).
    # CybORG's events.process_creation includes green FP events (no PID).
    has_process_creation_events = jnp.any(state.host_process_creation_pids != -1, axis=1)
    # CybORG: scans create network_connection events, exploits create process_creation events
    host_activity_detected = state.host_activity_detected | (has_scan_activity & covers)
    # CybORG stores process_creation events on the host object regardless of blue
    # coverage. Monitor only ages/clears events on covered hosts. On uncovered hosts
    # the events (and the derived detection flag) persist indefinitely.
    host_exploit_detected = state.host_exploit_detected | has_process_creation_events
    # CybORG Monitor ages events: old = current, then clear current for covered hosts.
    # Observations read old | current, giving events 2-cycle persistence.
    old_host_activity_detected = jnp.where(covers, host_activity_detected, state.old_host_activity_detected)
    aged_host_activity_detected = jnp.where(covers, False, host_activity_detected)
    old_host_exploit_detected = jnp.where(covers, host_exploit_detected, state.old_host_exploit_detected)
    aged_host_exploit_detected = jnp.where(covers, False, host_exploit_detected)
    host_suspicious_process = state.host_suspicious_process | newly_detected

    # Vectorized equivalent of repeatedly appending event_row[0..K-1] into the
    # next empty slot of suspicious_row (where empty := < 0 and append is a
    # no-op when pid < 0 or row is full). Concatenate then stable-sort by
    # validity so all valid PIDs (>= 0) move to the front in their original
    # left-to-right order; truncate to the suspicious row width.
    suspicious = state.blue_suspicious_pids[agent_id]
    events = state.host_process_creation_pids
    combined = jnp.concatenate([suspicious, events], axis=1)
    order = jnp.argsort((combined < 0).astype(jnp.int32), axis=1, stable=True)
    sorted_combined = jnp.take_along_axis(combined, order, axis=1)
    updated_rows = sorted_combined[:, :MAX_TRACKED_SUSPICIOUS_PIDS]
    updated_agent_rows = jnp.where(covers[:, None], updated_rows, suspicious)
    blue_suspicious_pids = state.blue_suspicious_pids.at[agent_id].set(updated_agent_rows)

    cleared_events = jnp.where(covers[:, None], -1, state.host_process_creation_pids)
    return state.replace(
        host_activity_detected=aged_host_activity_detected,
        old_host_activity_detected=old_host_activity_detected,
        host_exploit_detected=aged_host_exploit_detected,
        old_host_exploit_detected=old_host_exploit_detected,
        host_suspicious_process=host_suspicious_process,
        blue_suspicious_pids=blue_suspicious_pids,
        host_process_creation_pids=cleared_events,
    )
