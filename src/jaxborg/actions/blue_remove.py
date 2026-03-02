import jax
import jax.numpy as jnp

from jaxborg.actions.pids import pid_row_contains, remove_pid_from_row
from jaxborg.actions.red_common import recompute_scan_anchor_hosts, sync_scan_memory_fields
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import ABSTRACT_RANK_NONE, COMPROMISE_NONE, COMPROMISE_USER, MAX_TRACKED_SUSPICIOUS_PIDS
from jaxborg.state import CC4Const, CC4State


def apply_blue_remove(state: CC4State, const: CC4Const, agent_id: int, target_host: int) -> CC4State:
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    suspicious_pid_row = state.blue_suspicious_pids[agent_id, target_host]

    session_count_before = effective_session_counts(state)
    row_indices = jnp.arange(MAX_TRACKED_SUSPICIOUS_PIDS, dtype=jnp.int32)
    row_max_slot = jnp.max(jnp.where(suspicious_pid_row >= 0, row_indices, -1)) + 1
    pid_budget = jnp.clip(state.blue_suspicious_pid_budget[agent_id, target_host], 0, MAX_TRACKED_SUSPICIOUS_PIDS)
    slot_limit = jnp.clip(jnp.maximum(row_max_slot, pid_budget), 0, MAX_TRACKED_SUSPICIOUS_PIDS)

    def _remove_slot(slot, carry):
        new_session_count, new_suspicious_count, new_privilege, new_session_pids, any_removed = carry
        sus_pid = suspicious_pid_row[slot]
        has_pid = sus_pid >= 0

        is_user = new_privilege[:, target_host] == COMPROMISE_USER
        has_sessions = new_session_count[:, target_host] > 0
        has_live_pid = jax.vmap(pid_row_contains, in_axes=(0, None))(new_session_pids[:, target_host, :], sus_pid)
        match_by_red = covers_host & has_pid & is_user & has_sessions & has_live_pid

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
        return new_session_count, new_suspicious_count, new_privilege, new_session_pids, any_removed

    init = (
        session_count_before,
        state.red_suspicious_process_count,
        state.red_privilege,
        state.red_session_pids,
        jnp.array(False),
    )
    new_session_count, new_suspicious_count, new_privilege, new_session_pids, any_removed = jax.lax.fori_loop(
        0, slot_limit, _remove_slot, init
    )

    new_sessions = new_session_count > 0
    remaining_max_priv = jnp.max(new_privilege[:, target_host])
    new_host_compromised = jnp.where(
        covers_host & any_removed,
        state.host_compromised.at[target_host].set(remaining_max_priv),
        state.host_compromised,
    )
    had_any_sessions = jnp.any(session_count_before > 0, axis=1)
    has_any_sessions_now = jnp.any(new_session_count > 0, axis=1)
    cleared_all_sessions = had_any_sessions & ~has_any_sessions_now
    full_clear = cleared_all_sessions[:, None]
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
    rank_update = state.red_abstract_host_rank.at[:, target_host].set(
        jnp.where(sessions_cleared_on_host, jnp.int32(ABSTRACT_RANK_NONE), state.red_abstract_host_rank[:, target_host])
    )
    red_abstract_host_rank = jnp.where(
        covers_host,
        rank_update,
        state.red_abstract_host_rank,
    )
    red_scan_anchor_host = recompute_scan_anchor_hosts(
        state.red_scan_anchor_host,
        new_sessions,
        red_session_is_abstract,
        const.host_active,
    )
    scan_synced = sync_scan_memory_fields(
        state.replace(
            red_sessions=new_sessions,
            red_session_is_abstract=red_session_is_abstract,
            red_abstract_host_rank=red_abstract_host_rank,
        ),
        const,
    )
    new_scanned_hosts = jnp.where(full_clear, False, scan_synced.red_scanned_hosts)
    new_scanned_source_hosts = jnp.where(full_clear[:, :, None], False, scan_synced.red_scanned_source_hosts)
    return state.replace(
        red_sessions=new_sessions,
        red_session_count=new_session_count,
        red_session_pids=new_session_pids,
        red_suspicious_process_count=new_suspicious_count,
        red_privilege=new_privilege,
        red_scan_anchor_host=red_scan_anchor_host,
        red_scanned_hosts=new_scanned_hosts,
        red_scanned_source_hosts=new_scanned_source_hosts,
        host_compromised=new_host_compromised,
        host_suspicious_process=new_suspicious_process,
        blue_suspicious_pid_budget=state.blue_suspicious_pid_budget,
        red_session_is_abstract=red_session_is_abstract,
        red_abstract_host_rank=red_abstract_host_rank,
    )
