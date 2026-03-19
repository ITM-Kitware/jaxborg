import jax
import jax.numpy as jnp

from jaxborg.actions.pids import append_pid_to_row_allow_duplicates
from jaxborg.constants import (
    ACTIVITY_NONE,
    ACTIVITY_SCAN,
    MAX_TRACKED_SUSPICIOUS_PIDS,
    NUM_BLUE_AGENTS,
)
from jaxborg.state import CC4Const, CC4State


def apply_blue_monitor(state: CC4State, const: CC4Const, agent_id: int | None = None) -> CC4State:
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

    def _ingest_host(h, blue_suspicious_pids):
        event_row = state.host_process_creation_pids[h]
        suspicious_row = blue_suspicious_pids[agent_id, h]

        def _append_slot(slot, pid_row):
            return append_pid_to_row_allow_duplicates(pid_row, event_row[slot])

        updated_row = jax.lax.fori_loop(0, MAX_TRACKED_SUSPICIOUS_PIDS, _append_slot, suspicious_row)
        updated_row = jnp.where(covers[h], updated_row, suspicious_row)
        blue_suspicious_pids = blue_suspicious_pids.at[agent_id, h].set(updated_row)
        return blue_suspicious_pids

    blue_suspicious_pids = jax.lax.fori_loop(
        0,
        state.host_process_creation_pids.shape[0],
        _ingest_host,
        state.blue_suspicious_pids,
    )

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
