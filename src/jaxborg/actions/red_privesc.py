import chex
import jax
import jax.numpy as jnp

from jaxborg.actions.pids import append_pid_to_row, count_valid_pids, first_valid_pid, nth_valid_pid
from jaxborg.actions.red_common import bound_source_is_abstract, sync_scan_memory_fields
from jaxborg.actions.rng import sample_red_privesc_choice
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import ABSTRACT_RANK_NONE, ACTIVITY_EXPLOIT, COMPROMISE_PRIVILEGED
from jaxborg.state import CC4Const, CC4State


def apply_privesc(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
    key: jax.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    session_counts = effective_session_counts(state)
    target_count = session_counts[agent_id, target_host]
    has_session = target_count > 0
    not_already_privileged = state.red_privilege[agent_id, target_host] < COMPROMISE_PRIVILEGED
    is_abstract = bound_source_is_abstract(state, const, agent_id)

    target_pid_row = state.red_session_pids[agent_id, target_host]
    tracked_pid_count = count_valid_pids(target_pid_row)
    total_safe = jnp.maximum(target_count, jnp.int32(1))
    # CybORG: np_random.choice(sessions) picks a random session on the host.
    # Sync via precomputed choices when available to match CybORG's selection.
    chosen_slot = sample_red_privesc_choice(const, state.time, agent_id, key, total_safe)

    # CybORG samples one concrete target session object on the host. The older
    # host-level sandbox flag is only safe to apply when a single session exists.
    single_session_sandboxed = state.red_session_sandboxed[agent_id, target_host] & (target_count == 1)
    success = is_active & has_session & not_already_privileged & is_abstract & ~single_session_sandboxed

    # Sandboxed sessions are removed on escalation attempt (CybORG PrivilegeEscalate behavior)
    red_sessions = jnp.where(
        is_active & has_session & single_session_sandboxed,
        state.red_sessions.at[agent_id, target_host].set(False),
        state.red_sessions,
    )
    red_session_count = jnp.where(
        is_active & has_session & single_session_sandboxed,
        session_counts.at[agent_id, target_host].set(0),
        session_counts,
    )
    red_suspicious_process_count = jnp.where(
        is_active & has_session & single_session_sandboxed,
        state.red_suspicious_process_count.at[agent_id, target_host].set(0),
        state.red_suspicious_process_count,
    )
    red_session_is_abstract = jnp.where(
        is_active & has_session & single_session_sandboxed,
        state.red_session_is_abstract.at[agent_id, target_host].set(False),
        state.red_session_is_abstract,
    )
    red_abstract_host_rank = jnp.where(
        is_active & has_session & single_session_sandboxed,
        state.red_abstract_host_rank.at[agent_id, target_host].set(jnp.int32(ABSTRACT_RANK_NONE)),
        state.red_abstract_host_rank,
    )
    red_session_pids = jnp.where(
        is_active & has_session & single_session_sandboxed,
        state.red_session_pids.at[agent_id, target_host].set(-1),
        state.red_session_pids,
    )
    red_session_abstract_pids = jnp.where(
        is_active & has_session & single_session_sandboxed,
        state.red_session_abstract_pids.at[agent_id, target_host].set(-1),
        state.red_session_abstract_pids,
    )
    red_session_privileged_pids = jnp.where(
        is_active & has_session & single_session_sandboxed,
        state.red_session_privileged_pids.at[agent_id, target_host].set(-1),
        state.red_session_privileged_pids,
    )

    fallback_pid = first_valid_pid(target_pid_row)
    tracked_pid = nth_valid_pid(target_pid_row, chosen_slot)
    escalate_pid = jnp.where(tracked_pid_count == target_count, tracked_pid, fallback_pid)
    target_priv_pid_row = red_session_privileged_pids[agent_id, target_host]
    escalated_priv_pid_row = append_pid_to_row(target_priv_pid_row, escalate_pid)
    red_session_privileged_pids = jnp.where(
        success,
        red_session_privileged_pids.at[agent_id, target_host].set(escalated_priv_pid_row),
        red_session_privileged_pids,
    )

    new_priv = jnp.where(success, COMPROMISE_PRIVILEGED, state.red_privilege[agent_id, target_host])
    red_privilege = jnp.where(
        success,
        state.red_privilege.at[agent_id, target_host].set(new_priv),
        state.red_privilege,
    )

    host_compromised = jnp.where(
        success,
        state.host_compromised.at[target_host].set(
            jnp.maximum(state.host_compromised[target_host], COMPROMISE_PRIVILEGED)
        ),
        state.host_compromised,
    )
    discovered_row = state.red_discovered_hosts[agent_id]
    discovered_with_info = discovered_row | const.host_info_links[target_host]
    red_discovered_hosts = jnp.where(
        success,
        state.red_discovered_hosts.at[agent_id].set(discovered_with_info),
        state.red_discovered_hosts,
    )

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )
    had_any_sessions = jnp.any(session_counts > 0, axis=1)
    has_any_sessions_now = jnp.any(red_session_count > 0, axis=1)
    cleared_all_sessions = had_any_sessions & ~has_any_sessions_now
    full_clear = cleared_all_sessions[:, None]
    scan_synced = sync_scan_memory_fields(
        state.replace(
            red_sessions=red_sessions,
            red_session_is_abstract=red_session_is_abstract,
            red_abstract_host_rank=red_abstract_host_rank,
        ),
        const,
    )
    red_scanned_hosts = jnp.where(full_clear, False, scan_synced.red_scanned_hosts)
    red_scanned_source_hosts = jnp.where(full_clear[:, :, None], False, scan_synced.red_scanned_source_hosts)
    any_suspicious = jnp.any(red_suspicious_process_count[:, target_host] > 0)
    host_suspicious_process = jnp.where(
        is_active & has_session & single_session_sandboxed,
        state.host_suspicious_process.at[target_host].set(any_suspicious),
        state.host_suspicious_process,
    )
    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_session_pids=red_session_pids,
        red_session_abstract_pids=red_session_abstract_pids,
        red_session_privileged_pids=red_session_privileged_pids,
        red_suspicious_process_count=red_suspicious_process_count,
        red_session_is_abstract=red_session_is_abstract,
        red_abstract_host_rank=red_abstract_host_rank,
        red_privilege=red_privilege,
        red_discovered_hosts=red_discovered_hosts,
        red_scanned_hosts=red_scanned_hosts,
        red_scanned_source_hosts=red_scanned_source_hosts,
        host_compromised=host_compromised,
        host_suspicious_process=host_suspicious_process,
        red_activity_this_step=activity,
    )
