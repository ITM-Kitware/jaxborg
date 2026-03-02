import chex
import jax
import jax.numpy as jnp

from jaxborg.actions.pids import append_pid_to_row, first_valid_pid
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import ACTIVITY_EXPLOIT, COMPROMISE_USER, NUM_BLUE_AGENTS, NUM_SUBNETS
from jaxborg.state import CC4Const, CC4State


def has_any_session(session_hosts: chex.Array, const: CC4Const) -> chex.Array:
    return jnp.any(session_hosts & const.host_active)


def has_abstract_session(state: CC4State, agent_id: int) -> chex.Array:
    return jnp.any(state.red_session_is_abstract[agent_id] & state.red_sessions[agent_id])


def select_bound_source_host(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
) -> chex.Array:
    """Return the host of the bound source session for red actions.

    CybORG red abstract actions execute against a concrete bound session id
    (session 0 for FSM-driven CC4 actions). In JAX we track that binding via
    `red_scan_anchor_host`.
    """
    anchor = state.red_scan_anchor_host[agent_id]
    anchor_idx = jnp.clip(anchor, 0, state.red_sessions.shape[1] - 1)
    anchor_valid = (anchor >= 0) & state.red_sessions[agent_id, anchor_idx] & const.host_active[anchor_idx]

    # When no explicit anchor has been set, session 0 maps to the red start host.
    start_host = const.red_start_hosts[agent_id]
    start_idx = jnp.clip(start_host, 0, state.red_sessions.shape[1] - 1)
    start_valid = (start_host >= 0) & state.red_sessions[agent_id, start_idx] & const.host_active[start_idx]

    return jnp.where(anchor_valid, anchor, jnp.where((anchor < 0) & start_valid, start_host, jnp.int32(-1)))


def bound_source_is_abstract(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
) -> chex.Array:
    source_host = select_bound_source_host(state, const, agent_id)
    source_idx = jnp.clip(source_host, 0, state.red_sessions.shape[1] - 1)
    return (source_host >= 0) & state.red_session_is_abstract[agent_id, source_idx]


def can_reach_subnet(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_subnet: chex.Array,
) -> chex.Array:
    session_hosts = state.red_sessions[agent_id]
    has_session = has_any_session(session_hosts, const)
    active_sessions = session_hosts & const.host_active
    subnet_one_hot = jax.nn.one_hot(const.host_subnet, NUM_SUBNETS, dtype=jnp.bool_)
    session_subnets = jnp.any(active_sessions[:, None] & subnet_one_hot, axis=0)
    not_blocked = ~state.blocked_zones[target_subnet]
    can_route = jnp.any(session_subnets & not_blocked)
    return has_session & can_route


def exploit_common_preconditions(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> chex.Array:
    is_active = const.host_active[target_host]
    source_host = select_scan_execution_source_host(state, const, agent_id, target_host)
    owns_target_scan = (
        (source_host >= 0)
        & state.red_scanned_hosts[agent_id, target_host]
        & (state.red_scanned_via[agent_id, target_host] == source_host)
    )
    target_subnet = const.host_subnet[target_host]
    can_reach = can_reach_subnet(state, const, agent_id, target_subnet)
    is_abstract = source_host >= 0
    return is_active & owns_target_scan & can_reach & is_abstract


def select_scan_source_host(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
) -> chex.Array:
    """Choose source host for abstract red actions.

    Priority:
    1) Bound source session host (anchor) when it exists and is abstract.
    2) If anchor is absent/unset, first available abstract session host.
    """
    source_host = select_bound_source_host(state, const, agent_id)
    source_idx = jnp.clip(source_host, 0, state.red_sessions.shape[1] - 1)
    source_is_abstract = (source_host >= 0) & state.red_session_is_abstract[agent_id, source_idx]

    abstract_hosts = state.red_session_is_abstract[agent_id] & state.red_sessions[agent_id] & const.host_active
    has_fallback = jnp.any(abstract_hosts)
    fallback = jnp.where(has_fallback, jnp.argmax(abstract_hosts), -1)

    return jnp.where(source_host >= 0, jnp.where(source_is_abstract, source_host, fallback), fallback)


def select_scan_execution_source_host(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> chex.Array:
    """Choose source host for target-specific scan/exploit actions.

    Prefer the existing owner of the target's scan memory when that owner is a
    live abstract session; otherwise fall back to generic scan source selection.
    """
    target_idx = jnp.clip(target_host, 0, state.red_scanned_hosts.shape[1] - 1)
    via = state.red_scanned_via[agent_id, target_idx]
    via_idx = jnp.clip(via, 0, state.red_sessions.shape[1] - 1)
    via_valid = (
        (via >= 0)
        & state.red_scanned_hosts[agent_id, target_idx]
        & state.red_sessions[agent_id, via_idx]
        & state.red_session_is_abstract[agent_id, via_idx]
        & const.host_active[via_idx]
    )
    fallback = select_scan_source_host(state, const, agent_id)
    return jnp.where(via_valid, via, fallback)


def apply_exploit_success(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
    success: chex.Array,
) -> CC4State:
    session_counts = effective_session_counts(state)
    had_count = session_counts[agent_id, target_host]
    new_count = jnp.where(success, had_count + 1, had_count)
    red_session_count = jnp.where(
        success,
        session_counts.at[agent_id, target_host].set(new_count),
        session_counts,
    )
    red_sessions = jnp.where(
        success,
        state.red_sessions.at[agent_id, target_host].set(new_count > 0),
        state.red_sessions,
    )
    red_session_multiple = jnp.where(
        success,
        state.red_session_multiple.at[agent_id, target_host].set(new_count > 1),
        state.red_session_multiple,
    )
    red_session_many = jnp.where(
        success,
        state.red_session_many.at[agent_id, target_host].set(new_count > 2),
        state.red_session_many,
    )
    prior_suspicious = state.red_suspicious_process_count[agent_id, target_host]
    new_suspicious = jnp.where(
        success,
        prior_suspicious + 1,
        prior_suspicious,
    )
    red_suspicious_process_count = jnp.where(
        success,
        state.red_suspicious_process_count.at[agent_id, target_host].set(new_suspicious),
        state.red_suspicious_process_count,
    )

    new_priv = jnp.where(
        success,
        jnp.maximum(state.red_privilege[agent_id, target_host], COMPROMISE_USER),
        state.red_privilege[agent_id, target_host],
    )
    red_privilege = jnp.where(
        success,
        state.red_privilege.at[agent_id, target_host].set(new_priv),
        state.red_privilege,
    )

    host_compromised = jnp.where(
        success,
        state.host_compromised.at[target_host].set(jnp.maximum(state.host_compromised[target_host], COMPROMISE_USER)),
        state.host_compromised,
    )

    host_has_malware = jnp.where(
        success,
        state.host_has_malware.at[target_host].set(True),
        state.host_has_malware,
    )
    host_suspicious_process = jnp.where(
        success,
        state.host_suspicious_process.at[target_host].set(True),
        state.host_suspicious_process,
    )

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )
    blue_budget_inc = const.blue_agent_hosts[:, target_host].astype(jnp.int32)
    blue_suspicious_pid_budget = jnp.where(
        success,
        state.blue_suspicious_pid_budget.at[:, target_host].add(blue_budget_inc),
        state.blue_suspicious_pid_budget,
    )
    new_pid = state.red_next_pid
    red_next_pid = jnp.where(success, state.red_next_pid + 1, state.red_next_pid)
    session_pid_row = state.red_session_pids[agent_id, target_host]
    pid_row_updated = append_pid_to_row(session_pid_row, new_pid)
    red_session_pids = jnp.where(
        success,
        state.red_session_pids.at[agent_id, target_host].set(pid_row_updated),
        state.red_session_pids,
    )
    red_session_pid = jnp.where(
        success,
        state.red_session_pid.at[agent_id, target_host].set(first_valid_pid(pid_row_updated)),
        state.red_session_pid,
    )
    blue_suspicious_pids = state.blue_suspicious_pids
    for b in range(NUM_BLUE_AGENTS):
        covers = const.blue_agent_hosts[b, target_host]
        pid_row = blue_suspicious_pids[b, target_host]
        updated_row = append_pid_to_row(pid_row, new_pid)
        blue_suspicious_pids = blue_suspicious_pids.at[b, target_host].set(
            jnp.where(success & covers, updated_row, pid_row)
        )

    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_session_multiple=red_session_multiple,
        red_session_many=red_session_many,
        red_suspicious_process_count=red_suspicious_process_count,
        red_privilege=red_privilege,
        red_session_pid=red_session_pid,
        red_session_pids=red_session_pids,
        red_next_pid=red_next_pid,
        host_compromised=host_compromised,
        host_has_malware=host_has_malware,
        host_suspicious_process=host_suspicious_process,
        red_activity_this_step=activity,
        blue_suspicious_pid_budget=blue_suspicious_pid_budget,
        blue_suspicious_pids=blue_suspicious_pids,
    )
