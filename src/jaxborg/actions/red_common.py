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

    return jnp.where(anchor_valid, anchor, jnp.int32(-1))


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


def scan_sources_with_fallback(state: CC4State) -> chex.Array:
    """Return scan-memory ownership matrix.

    CybORG tracks per-session source ownership for each scanned target through
    red-session `ports` memory. JAX mirrors that state in
    `red_scanned_source_hosts`; ownership should not be reconstructed from
    derived fields (`red_scanned_hosts` / `red_scanned_via`).
    """
    return state.red_scanned_source_hosts


def recompute_scan_views_from_sources(scan_sources: chex.Array) -> tuple[chex.Array, chex.Array]:
    """Derive scanned-host and primary-owner views from source ownership."""
    red_scanned_hosts = jnp.any(scan_sources, axis=2)
    primary_owner = jnp.argmax(scan_sources, axis=2).astype(jnp.int32)
    red_scanned_via = jnp.where(red_scanned_hosts, primary_owner, -1)
    return red_scanned_hosts, red_scanned_via


def sync_scan_memory_fields(
    state: CC4State,
    const: CC4Const,
    scan_sources: chex.Array | None = None,
) -> CC4State:
    """Project scan-memory ownership onto active source sessions.

    CybORG derives scanned-host memory from session `ports` maps. That logic is
    not gated on JAX-specific abstract-session bookkeeping, so scan-memory
    validity should depend on live source sessions and active hosts only.
    """
    sources = scan_sources if scan_sources is not None else scan_sources_with_fallback(state)
    valid_sources = (
        sources
        & state.red_sessions[:, None, :]
        & const.host_active[None, None, :]
    )
    red_scanned_hosts, red_scanned_via = recompute_scan_views_from_sources(valid_sources)
    return state.replace(
        red_scanned_source_hosts=valid_sources,
        red_scanned_hosts=red_scanned_hosts,
        red_scanned_via=red_scanned_via,
    )


def exploit_common_preconditions(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> chex.Array:
    is_active = const.host_active[target_host]
    source_host = select_scan_execution_source_host(state, const, agent_id, target_host)
    target_idx = jnp.clip(target_host, 0, state.red_scanned_hosts.shape[1] - 1)
    source_idx = jnp.clip(source_host, 0, state.red_sessions.shape[1] - 1)
    scan_sources = scan_sources_with_fallback(state)
    owns_target_scan = (source_host >= 0) & scan_sources[agent_id, target_idx, source_idx]
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
    abstract_ranks = state.red_abstract_host_rank[agent_id]
    rank_scores = jnp.where(abstract_hosts, abstract_ranks, jnp.int32(1_000_000))
    has_rank_fallback = jnp.any(abstract_hosts & (abstract_ranks < jnp.int32(1_000_000)))
    fallback_by_rank = jnp.argmin(rank_scores).astype(jnp.int32)
    has_any_fallback = jnp.any(abstract_hosts)
    fallback_any = jnp.where(has_any_fallback, jnp.argmax(abstract_hosts), -1)
    fallback = jnp.where(has_rank_fallback, fallback_by_rank, fallback_any)

    return jnp.where(source_host >= 0, jnp.where(source_is_abstract, source_host, fallback), fallback)


def recompute_scan_anchor_hosts(
    prior_anchor_hosts: chex.Array,
    red_sessions: chex.Array,
    red_session_is_abstract: chex.Array,
    host_active: chex.Array,
) -> chex.Array:
    """Invalidate anchors that no longer reference a live session host.

    CybORG's `RedSessionCheck` promotes a new session 0 using RNG. Anchor
    promotion is therefore handled in red turn processing, not here.
    """
    del red_session_is_abstract
    anchor_idx = jnp.clip(prior_anchor_hosts, 0, red_sessions.shape[1] - 1)
    anchor_valid = (
        (prior_anchor_hosts >= 0)
        & red_sessions[jnp.arange(prior_anchor_hosts.shape[0]), anchor_idx]
        & host_active[anchor_idx]
    )
    has_any_sessions = jnp.any(red_sessions & host_active[None, :], axis=1)
    return jnp.where(has_any_sessions & anchor_valid, prior_anchor_hosts, -1)


def select_new_primary_session_host(
    session_counts: chex.Array,
    host_active: chex.Array,
    key: jax.Array,
) -> chex.Array:
    """Mirror CybORG RedSessionCheck primary-session promotion.

    CybORG randomly chooses a new session id from all active sessions when
    session 0 is missing. We model that by sampling among host session slots
    weighted by per-host session multiplicity.
    """
    active_counts = jnp.where(host_active, session_counts, jnp.int32(0))
    total = jnp.sum(active_counts)
    total_safe = jnp.maximum(total, jnp.int32(1))
    draw = jax.random.randint(key, (), minval=0, maxval=total_safe, dtype=jnp.int32)
    cumulative = jnp.cumsum(active_counts)
    chosen_host = jnp.argmax(draw < cumulative)
    return jnp.where(total > 0, chosen_host.astype(jnp.int32), jnp.int32(-1))


def apply_red_session_check(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    key: jax.Array,
) -> CC4State:
    """Ensure each active red agent has a valid primary-session anchor host."""
    session_counts = effective_session_counts(state)[agent_id]
    has_any_sessions = jnp.any((session_counts > 0) & const.host_active)
    anchor = state.red_scan_anchor_host[agent_id]
    anchor_idx = jnp.clip(anchor, 0, state.red_sessions.shape[1] - 1)
    anchor_valid = (anchor >= 0) & state.red_sessions[agent_id, anchor_idx] & const.host_active[anchor_idx]
    needs_primary = has_any_sessions & ~anchor_valid
    forced = state.red_session_check_forced_host[agent_id]
    forced_idx = jnp.clip(forced, 0, state.red_sessions.shape[1] - 1)
    forced_valid = (forced >= 0) & (session_counts[forced_idx] > 0) & const.host_active[forced_idx]
    sampled = select_new_primary_session_host(session_counts, const.host_active, key)
    promoted = jnp.where(forced_valid, forced, sampled)
    next_anchor = jnp.where(has_any_sessions, jnp.where(needs_primary, promoted, anchor), jnp.int32(-1))
    return state.replace(red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(next_anchor))


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
    target_sources = scan_sources_with_fallback(state)[agent_id, target_idx]
    active_abstract_sources = (
        target_sources & state.red_sessions[agent_id] & state.red_session_is_abstract[agent_id] & const.host_active
    )
    has_owner = jnp.any(active_abstract_sources)
    owner_host = jnp.where(has_owner, jnp.argmax(active_abstract_sources), -1)

    # During duration processing, scans can be pre-bound to a specific source host.
    # Respect that explicit binding (including "bound to none") instead of recomputing
    # against potentially changed same-step state (e.g. after green updates).
    pending_source = state.red_pending_source_host[agent_id]
    pending_idx = jnp.clip(pending_source, 0, state.red_sessions.shape[1] - 1)
    pending_valid = (
        (pending_source >= 0)
        & state.red_sessions[agent_id, pending_idx]
        & state.red_session_is_abstract[agent_id, pending_idx]
        & const.host_active[pending_idx]
    )
    has_pending_binding = pending_source != jnp.int32(-1)

    fallback = select_scan_source_host(state, const, agent_id)
    computed = jnp.where(has_owner, owner_host, fallback)
    return jnp.where(
        has_pending_binding,
        jnp.where(pending_valid, pending_source, jnp.int32(-1)),
        computed,
    )


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
