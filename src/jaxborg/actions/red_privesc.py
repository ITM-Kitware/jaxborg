import chex
import jax.numpy as jnp

from jaxborg.actions.red_common import bound_source_is_abstract, sync_scan_memory_fields
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import ACTIVITY_EXPLOIT, COMPROMISE_PRIVILEGED
from jaxborg.state import CC4Const, CC4State


def apply_privesc(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    session_counts = effective_session_counts(state)
    has_session = session_counts[agent_id, target_host] > 0
    not_already_privileged = state.red_privilege[agent_id, target_host] < COMPROMISE_PRIVILEGED
    is_sandboxed = state.red_session_sandboxed[agent_id, target_host]
    is_abstract = bound_source_is_abstract(state, const, agent_id)
    success = is_active & has_session & not_already_privileged & ~is_sandboxed & is_abstract

    # Sandboxed sessions are removed on escalation attempt (CybORG PrivilegeEscalate behavior)
    red_sessions = jnp.where(
        is_active & has_session & is_sandboxed,
        state.red_sessions.at[agent_id, target_host].set(False),
        state.red_sessions,
    )
    red_session_count = jnp.where(
        is_active & has_session & is_sandboxed,
        session_counts.at[agent_id, target_host].set(0),
        session_counts,
    )
    red_session_multiple = jnp.where(
        is_active & has_session & is_sandboxed,
        state.red_session_multiple.at[agent_id, target_host].set(False),
        state.red_session_multiple,
    )
    red_session_many = jnp.where(
        is_active & has_session & is_sandboxed,
        state.red_session_many.at[agent_id, target_host].set(False),
        state.red_session_many,
    )
    red_suspicious_process_count = jnp.where(
        is_active & has_session & is_sandboxed,
        state.red_suspicious_process_count.at[agent_id, target_host].set(0),
        state.red_suspicious_process_count,
    )
    red_session_is_abstract = jnp.where(
        is_active & has_session & is_sandboxed,
        state.red_session_is_abstract.at[agent_id, target_host].set(False),
        state.red_session_is_abstract,
    )
    red_abstract_host_rank = jnp.where(
        is_active & has_session & is_sandboxed,
        state.red_abstract_host_rank.at[agent_id, target_host].set(jnp.int32(1_000_000)),
        state.red_abstract_host_rank,
    )
    red_session_pids = jnp.where(
        is_active & has_session & is_sandboxed,
        state.red_session_pids.at[agent_id, target_host].set(-1),
        state.red_session_pids,
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
        is_active & has_session & is_sandboxed,
        state.host_suspicious_process.at[target_host].set(any_suspicious),
        state.host_suspicious_process,
    )
    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_session_multiple=red_session_multiple,
        red_session_many=red_session_many,
        red_session_pids=red_session_pids,
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
        blue_suspicious_pid_budget=state.blue_suspicious_pid_budget,
    )
