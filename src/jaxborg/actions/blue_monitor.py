import jax
import jax.numpy as jnp

from jaxborg.actions.pids import append_pid_to_row_allow_duplicates
from jaxborg.constants import ACTIVITY_NONE, MAX_TRACKED_SUSPICIOUS_PIDS, NUM_BLUE_AGENTS
from jaxborg.state import CC4Const, CC4State


def apply_blue_monitor(state: CC4State, const: CC4Const, agent_id: int | None = None) -> CC4State:
    if agent_id is None:
        for b in range(NUM_BLUE_AGENTS):
            state = apply_blue_monitor(state, const, b)
        return state

    covers = const.blue_agent_hosts[agent_id]
    has_activity = state.red_activity_this_step != ACTIVITY_NONE
    newly_detected = has_activity & covers
    host_activity_detected = state.host_activity_detected | newly_detected
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
        host_activity_detected=host_activity_detected,
        host_suspicious_process=host_suspicious_process,
        blue_suspicious_pids=blue_suspicious_pids,
        host_process_creation_pids=cleared_events,
    )
