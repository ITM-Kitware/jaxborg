import jax
import jax.numpy as jnp

from jaxborg.actions.pids import first_valid_pid, pid_row_contains, remove_pid_from_row
from jaxborg.actions.red_common import recompute_scan_anchor_hosts
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import COMPROMISE_NONE, COMPROMISE_USER, MAX_TRACKED_SUSPICIOUS_PIDS, NUM_RED_AGENTS
from jaxborg.state import CC4Const, CC4State


def apply_blue_remove(state: CC4State, const: CC4Const, agent_id: int, target_host: int) -> CC4State:
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    suspicious_pid_row = state.blue_suspicious_pids[agent_id, target_host]

    session_count_before = effective_session_counts(state)
    new_session_count = session_count_before
    new_suspicious_count = state.red_suspicious_process_count
    new_privilege = state.red_privilege
    new_session_pids = state.red_session_pids
    any_removed = jnp.array(False)

    for slot in range(MAX_TRACKED_SUSPICIOUS_PIDS):
        sus_pid = suspicious_pid_row[slot]
        has_pid = sus_pid >= 0

        match_by_red = jnp.zeros(NUM_RED_AGENTS, dtype=jnp.bool_)
        for r in range(NUM_RED_AGENTS):
            is_user = new_privilege[r, target_host] == COMPROMISE_USER
            has_sessions = new_session_count[r, target_host] > 0
            has_live_pid = pid_row_contains(new_session_pids[r, target_host], sus_pid)
            match_by_red = match_by_red.at[r].set(covers_host & has_pid & is_user & has_sessions & has_live_pid)

        any_match = jnp.any(match_by_red)
        matched_red = jnp.argmax(match_by_red)
        any_removed = any_removed | any_match

        matched_count = new_session_count[matched_red, target_host]
        count_after = jnp.maximum(matched_count - 1, 0)
        matched_suspicious = new_suspicious_count[matched_red, target_host]
        suspicious_after = jnp.maximum(matched_suspicious - 1, 0)

        matched_pid_row = new_session_pids[matched_red, target_host]
        updated_pid_row = remove_pid_from_row(matched_pid_row, sus_pid)
        priv_after = jnp.where(count_after > 0, COMPROMISE_USER, COMPROMISE_NONE)

        new_session_count = jnp.where(
            any_match,
            new_session_count.at[matched_red, target_host].set(count_after),
            new_session_count,
        )
        new_suspicious_count = jnp.where(
            any_match,
            new_suspicious_count.at[matched_red, target_host].set(suspicious_after),
            new_suspicious_count,
        )
        new_privilege = jnp.where(
            any_match,
            new_privilege.at[matched_red, target_host].set(priv_after),
            new_privilege,
        )
        new_session_pids = jnp.where(
            any_match,
            new_session_pids.at[matched_red, target_host].set(updated_pid_row),
            new_session_pids,
        )

    new_sessions = new_session_count > 0
    new_multiple = new_session_count > 1
    new_many = new_session_count > 2
    new_session_pid = jax.vmap(jax.vmap(first_valid_pid))(new_session_pids)
    new_session_pid = jnp.where(new_sessions, new_session_pid, -1)

    remaining_max_priv = jnp.max(new_privilege[:, target_host])
    new_host_compromised = jnp.where(
        covers_host & any_removed,
        state.host_compromised.at[target_host].set(remaining_max_priv),
        state.host_compromised,
    )
    had_any_sessions = jnp.any(session_count_before > 0, axis=1)
    has_any_sessions_now = jnp.any(new_session_count > 0, axis=1)
    cleared_all_sessions = had_any_sessions & ~has_any_sessions_now
    sessions_lost_on_target = (session_count_before[:, target_host] > 0) & (new_session_count[:, target_host] == 0)
    via_target = state.red_scanned_via == jnp.int32(target_host)
    via_clear = sessions_lost_on_target[:, None] & via_target
    full_clear = cleared_all_sessions[:, None]
    new_scanned_hosts = state.red_scanned_hosts & ~(full_clear | via_clear)
    new_scanned_via = jnp.where(full_clear | via_clear, -1, state.red_scanned_via)
    any_suspicious_after = jnp.any(new_suspicious_count[:, target_host] > 0)
    new_suspicious_process = jnp.where(
        covers_host,
        state.host_suspicious_process.at[target_host].set(any_suspicious_after),
        state.host_suspicious_process,
    )
    sessions_cleared_on_host = (session_count_before[:, target_host] > 0) & (new_session_count[:, target_host] == 0)
    abstract_update = state.red_session_is_abstract.at[:, target_host].set(
        state.red_session_is_abstract[:, target_host] & ~sessions_cleared_on_host
    )
    red_session_is_abstract = jnp.where(
        covers_host,
        abstract_update,
        state.red_session_is_abstract,
    )
    red_scan_anchor_host = recompute_scan_anchor_hosts(
        state.red_scan_anchor_host,
        new_sessions,
        red_session_is_abstract,
        const.host_active,
    )
    return state.replace(
        red_sessions=new_sessions,
        red_session_count=new_session_count,
        red_session_multiple=new_multiple,
        red_session_many=new_many,
        red_session_pid=new_session_pid,
        red_session_pids=new_session_pids,
        red_suspicious_process_count=new_suspicious_count,
        red_privilege=new_privilege,
        red_scan_anchor_host=red_scan_anchor_host,
        red_scanned_hosts=new_scanned_hosts,
        red_scanned_via=new_scanned_via,
        host_compromised=new_host_compromised,
        host_suspicious_process=new_suspicious_process,
        blue_suspicious_pid_budget=state.blue_suspicious_pid_budget,
        red_session_is_abstract=red_session_is_abstract,
    )
