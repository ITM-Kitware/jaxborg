import jax.numpy as jnp

from jaxborg.actions.red_common import recompute_scan_anchor_hosts, sync_scan_memory_fields
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import COMPROMISE_NONE
from jaxborg.state import CC4Const, CC4State


def apply_blue_restore(state: CC4State, const: CC4Const, agent_id: int, target_host: int) -> CC4State:
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    session_counts = effective_session_counts(state)

    host_compromised = jnp.where(
        covers_host,
        state.host_compromised.at[target_host].set(COMPROMISE_NONE),
        state.host_compromised,
    )

    red_sessions = jnp.where(
        covers_host,
        state.red_sessions.at[:, target_host].set(False),
        state.red_sessions,
    )
    red_session_count = jnp.where(
        covers_host,
        session_counts.at[:, target_host].set(0),
        session_counts,
    )

    red_privilege = jnp.where(
        covers_host,
        state.red_privilege.at[:, target_host].set(COMPROMISE_NONE),
        state.red_privilege,
    )
    red_session_multiple = jnp.where(
        covers_host,
        state.red_session_multiple.at[:, target_host].set(False),
        state.red_session_multiple,
    )
    red_session_many = jnp.where(
        covers_host,
        state.red_session_many.at[:, target_host].set(False),
        state.red_session_many,
    )
    red_suspicious_process_count = jnp.where(
        covers_host,
        state.red_suspicious_process_count.at[:, target_host].set(0),
        state.red_suspicious_process_count,
    )
    red_session_is_abstract = jnp.where(
        covers_host,
        state.red_session_is_abstract.at[:, target_host].set(False),
        state.red_session_is_abstract,
    )
    red_abstract_host_rank = jnp.where(
        covers_host,
        state.red_abstract_host_rank.at[:, target_host].set(jnp.int32(1_000_000)),
        state.red_abstract_host_rank,
    )
    red_session_pid = jnp.where(
        covers_host,
        state.red_session_pid.at[:, target_host].set(-1),
        state.red_session_pid,
    )
    red_session_pids = jnp.where(
        covers_host,
        state.red_session_pids.at[:, target_host].set(-1),
        state.red_session_pids,
    )
    had_any_sessions = jnp.any(session_counts > 0, axis=1)
    has_any_sessions_now = jnp.any(red_session_count > 0, axis=1)
    cleared_all_sessions = had_any_sessions & ~has_any_sessions_now
    full_clear = cleared_all_sessions[:, None]
    red_scan_anchor_host = recompute_scan_anchor_hosts(
        state.red_scan_anchor_host,
        red_sessions,
        red_session_is_abstract,
        const.host_active,
    )
    scan_synced = sync_scan_memory_fields(
        state.replace(
            red_sessions=red_sessions,
            red_session_is_abstract=red_session_is_abstract,
            red_abstract_host_rank=red_abstract_host_rank,
        ),
        const,
    )
    red_scanned_hosts = jnp.where(full_clear, False, scan_synced.red_scanned_hosts)
    red_scanned_via = jnp.where(full_clear, -1, scan_synced.red_scanned_via)
    red_scanned_source_hosts = jnp.where(full_clear[:, :, None], False, scan_synced.red_scanned_source_hosts)

    host_services = jnp.where(
        covers_host,
        state.host_services.at[target_host].set(const.initial_services[target_host]),
        state.host_services,
    )

    host_has_malware = jnp.where(
        covers_host,
        state.host_has_malware.at[target_host].set(False),
        state.host_has_malware,
    )

    host_decoys = jnp.where(
        covers_host,
        state.host_decoys.at[target_host].set(False),
        state.host_decoys,
    )

    host_activity_detected = jnp.where(
        covers_host,
        state.host_activity_detected.at[target_host].set(False),
        state.host_activity_detected,
    )
    host_suspicious_process = jnp.where(
        covers_host,
        state.host_suspicious_process.at[target_host].set(False),
        state.host_suspicious_process,
    )

    ot_service_stopped = jnp.where(
        covers_host,
        state.ot_service_stopped.at[target_host].set(False),
        state.ot_service_stopped,
    )

    host_service_reliability = jnp.where(
        covers_host,
        state.host_service_reliability.at[target_host].set(100),
        state.host_service_reliability,
    )
    return state.replace(
        host_compromised=host_compromised,
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_session_multiple=red_session_multiple,
        red_session_many=red_session_many,
        red_session_pid=red_session_pid,
        red_session_pids=red_session_pids,
        red_suspicious_process_count=red_suspicious_process_count,
        red_privilege=red_privilege,
        red_scanned_hosts=red_scanned_hosts,
        red_scanned_via=red_scanned_via,
        red_scanned_source_hosts=red_scanned_source_hosts,
        red_scan_anchor_host=red_scan_anchor_host,
        host_services=host_services,
        host_has_malware=host_has_malware,
        host_decoys=host_decoys,
        host_activity_detected=host_activity_detected,
        host_suspicious_process=host_suspicious_process,
        blue_suspicious_pid_budget=state.blue_suspicious_pid_budget,
        ot_service_stopped=ot_service_stopped,
        host_service_reliability=host_service_reliability,
        red_session_is_abstract=red_session_is_abstract,
        red_abstract_host_rank=red_abstract_host_rank,
    )
