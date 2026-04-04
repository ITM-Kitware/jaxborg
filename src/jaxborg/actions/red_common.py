import chex
import jax
import jax.numpy as jnp

from jaxborg.actions.pending_source import (
    PENDING_SOURCE_KIND_BOUND_NONE,
    PENDING_SOURCE_KIND_HOST,
    PENDING_SOURCE_KIND_NONE,
    PENDING_SOURCE_KIND_SESSION_BINDING,
)
from jaxborg.actions.pids import (
    allocate_host_pid_from_delta,
    append_pid_to_row,
    append_process_event,
    move_pid_to_row_end,
    nth_valid_pid,
    pid_row_contains,
)
from jaxborg.actions.rng import (
    sample_detection_random,
    sample_exploit_session_choice,
    sample_red_pid_delta,
    sample_red_session_check_choice,
)
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import (
    ABSTRACT_RANK_NONE,
    ACTIVITY_EXPLOIT,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_SUBNETS,
)
from jaxborg.state import CC4Const, CC4State

EXPLOIT_ROUTE_DETECTION_RATE = 0.95
EXPLOIT_PROCESS_EVENT_DETECTION_RATE = 0.95
UNKNOWN_PRIMARY_PID = jnp.int32(-2)


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
    # CybORG's PrivilegeEscalate checks isinstance(session_0, RedAbstractSession).
    # When RedSessionCheck promotes a non-abstract session to session 0, the
    # host may still have abstract sessions but the primary is not abstract.
    # Use the per-agent primary-is-abstract flag for this check.
    return (source_host >= 0) & state.red_primary_is_abstract[agent_id]


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


def can_reach_subnet_from_source_host(
    state: CC4State,
    const: CC4Const,
    source_host: chex.Array,
    target_subnet: chex.Array,
) -> chex.Array:
    source_idx = jnp.clip(source_host, 0, state.red_sessions.shape[1] - 1)
    source_subnet = const.host_subnet[source_idx]
    source_active = (source_host >= 0) & const.host_active[source_idx]
    return source_active & ~state.blocked_zones[target_subnet, source_subnet]


def scan_sources(state: CC4State) -> chex.Array:
    """Return scan-memory ownership matrix.

    CybORG tracks per-session source ownership for each scanned target through
    red-session `ports` memory. JAX mirrors that state in
    `red_scanned_source_hosts`; ownership should not be reconstructed from
    derived fields.
    """
    return state.red_scanned_source_hosts


def recompute_scanned_hosts_from_sources(source_matrix: chex.Array) -> chex.Array:
    """Derive scanned-host view from source ownership."""
    return jnp.any(source_matrix, axis=2)


def sync_scan_memory_fields(
    state: CC4State,
    const: CC4Const,
    source_matrix: chex.Array | None = None,
) -> CC4State:
    """Project scan-memory ownership onto active abstract source sessions.

    CybORG derives scanned-host memory from RedAbstractSession `ports` maps.
    Only abstract sessions can own scan memory; non-abstract sessions (plain
    Session objects) do not have ports dicts and cannot own scan results.
    """
    sources = source_matrix if source_matrix is not None else scan_sources(state)
    valid_sources = (
        sources
        & state.red_sessions[:, None, :]
        & state.red_session_is_abstract[:, None, :]
        & const.host_active[None, None, :]
    )
    red_scanned_hosts = recompute_scanned_hosts_from_sources(valid_sources)
    return state.replace(
        red_scanned_source_hosts=valid_sources,
        red_scanned_hosts=red_scanned_hosts,
    )


def compute_visible_sessions(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
) -> chex.Array:
    """Count abstract sessions in allowed subnets (CybORG's server_session size)."""
    abstract_counts = state.red_abstract_session_count[agent_id]
    allowed_hosts = const.red_agent_subnets[agent_id, const.host_subnet]
    return jnp.maximum(jnp.sum(abstract_counts * allowed_hosts), jnp.int32(1))


def exploit_common_preconditions(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
    key: chex.Array = None,
) -> chex.Array:
    """Check base preconditions + CybORG session-selection logic for exploits.

    CybORG's FSM picks a random session ID from server_session (which only
    contains RedAbstractSession-type sessions).  Only the session that scanned
    the target has ports — so P(success) = 1/N where N = abstract sessions in
    allowed subnets.

    The 1/N roll uses the creation-time N saved in
    ``red_pending_visible_sessions`` (snapshotted when the action was first
    queued in ``process_red_with_duration``).  In the differential harness the
    choice index is synced from CybORG via ``red_exploit_session_choices``;
    in training it comes from JAX RNG.  Both paths exercise the same N.
    """
    is_active = const.host_active[target_host]
    source_host = select_scan_execution_source_host(state, const, agent_id, target_host)
    target_idx = jnp.clip(target_host, 0, state.red_scanned_hosts.shape[1] - 1)
    source_idx = jnp.clip(source_host, 0, state.red_sessions.shape[1] - 1)
    source_matrix = scan_sources(state)
    owns_target_scan = (source_host >= 0) & source_matrix[agent_id, target_idx, source_idx]
    target_subnet = const.host_subnet[target_host]
    can_reach = can_reach_subnet_from_source_host(state, const, source_host, target_subnet)
    is_abstract = source_host >= 0
    base = is_active & owns_target_scan & can_reach & is_abstract

    if key is None:
        return base

    # CybORG's FSM picks uniformly from server_session (abstract sessions in
    # allowed subnets).  Only 1 of N sessions has scan data for the target.
    # N is snapshotted at action creation time (matching CybORG's get_action()
    # timing).  The choice index is either synced from CybORG or from JAX RNG.
    visible_sessions = state.red_pending_visible_sessions[agent_id]
    time_idx = jnp.minimum(jnp.int32(state.time), jnp.int32(const.red_exploit_session_choices.shape[0] - 1))
    cyborg_roll = sample_exploit_session_choice(const, time_idx, agent_id, key, visible_sessions)
    roll_ok = cyborg_roll == 0

    return base & roll_ok


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
    rank_scores = jnp.where(abstract_hosts, abstract_ranks, jnp.int32(ABSTRACT_RANK_NONE))
    has_rank_fallback = jnp.any(abstract_hosts & (abstract_ranks < jnp.int32(ABSTRACT_RANK_NONE)))
    fallback_by_rank = jnp.argmin(rank_scores).astype(jnp.int32)
    has_any_fallback = jnp.any(abstract_hosts)
    fallback_any = jnp.where(has_any_fallback, jnp.argmax(abstract_hosts), -1)
    fallback = jnp.where(has_rank_fallback, fallback_by_rank, fallback_any)

    return jnp.where(source_host >= 0, jnp.where(source_is_abstract, source_host, jnp.int32(-1)), fallback)


def recompute_scan_anchor_hosts(
    prior_anchor_hosts: chex.Array,
    red_sessions: chex.Array,
    red_session_is_abstract: chex.Array,
    host_active: chex.Array,
    red_abstract_host_rank: chex.Array | None = None,
) -> chex.Array:
    """Recompute scan anchor hosts after session reassignment.

    Keeps valid anchors, invalidates stale ones, and promotes a new anchor
    for agents that gained sessions (e.g. via green phishing reassignment)
    but have no valid anchor yet.

    When ``red_abstract_host_rank`` is provided, the fallback prefers the
    host with the lowest rank (ident=0 in CybORG), matching CybORG's
    ``add_session`` order during ``different_subnet_agent_reassignment``.
    """
    del red_session_is_abstract
    anchor_idx = jnp.clip(prior_anchor_hosts, 0, red_sessions.shape[1] - 1)
    anchor_valid = (
        (prior_anchor_hosts >= 0)
        & red_sessions[jnp.arange(prior_anchor_hosts.shape[0]), anchor_idx]
        & host_active[anchor_idx]
    )
    active_sessions = red_sessions & host_active[None, :]
    has_any_sessions = jnp.any(active_sessions, axis=1)
    # Prefer the host with the lowest abstract rank (ident=0 = CybORG primary).
    # Fall back to lowest host index when no rank data is available.
    fallback_by_idx = jnp.argmax(active_sessions.astype(jnp.int32), axis=1).astype(jnp.int32)
    if red_abstract_host_rank is not None:
        rank_scores = jnp.where(active_sessions, red_abstract_host_rank, jnp.int32(ABSTRACT_RANK_NONE))
        has_ranked = jnp.any(active_sessions & (red_abstract_host_rank < jnp.int32(ABSTRACT_RANK_NONE)), axis=1)
        fallback_by_rank = jnp.argmin(rank_scores, axis=1).astype(jnp.int32)
        fallback_hosts = jnp.where(has_ranked, fallback_by_rank, fallback_by_idx)
    else:
        fallback_hosts = fallback_by_idx
    needs_promotion = has_any_sessions & ~anchor_valid
    return jnp.where(
        anchor_valid,
        prior_anchor_hosts,
        jnp.where(needs_promotion, fallback_hosts, jnp.int32(-1)),
    )


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
    forced_primary_host: chex.Array = jnp.int32(-1),
    forced_primary_pid: chex.Array = UNKNOWN_PRIMARY_PID,
) -> CC4State:
    """Ensure each active red agent has a valid primary-session anchor host."""
    session_counts = effective_session_counts(state)[agent_id]
    has_any_sessions = jnp.any((session_counts > 0) & const.host_active)
    anchor = state.red_scan_anchor_host[agent_id]
    anchor_idx = jnp.clip(anchor, 0, state.red_sessions.shape[1] - 1)
    current_primary_pid = state.red_primary_pid[agent_id]
    anchor_valid = (anchor >= 0) & state.red_sessions[agent_id, anchor_idx] & const.host_active[anchor_idx]
    primary_pid_tracked = pid_row_contains(state.red_session_pids[agent_id, anchor_idx], current_primary_pid)
    # When CybORG PID allocation diverges from JAX, the synced primary PID may
    # not appear in JAX's session_pids even though session 0 is still alive.
    # Fall back to abstract-session existence so PID drift doesn't trigger a
    # spurious primary invalidation (which clears scan memory).
    primary_pid_or_abstract = primary_pid_tracked | (
        state.red_session_is_abstract[agent_id, anchor_idx] & (current_primary_pid >= 0)
    )
    primary_valid = anchor_valid & primary_pid_or_abstract
    needs_primary = has_any_sessions & ~primary_valid
    forced_idx = jnp.clip(forced_primary_host, 0, state.red_sessions.shape[1] - 1)
    forced_valid = (forced_primary_host >= 0) & (session_counts[forced_idx] > 0) & const.host_active[forced_idx]
    forced_pid_valid = forced_valid & pid_row_contains(state.red_session_pids[agent_id, forced_idx], forced_primary_pid)
    # When CybORG session-check host is synced, use it instead of JAX RNG
    # sampling. CybORG's _choose_new_primary_session picks uniformly from all
    # sessions via np_random.choice — this is an RNG sync (Category B).
    synced_host = jax.lax.cond(
        const.use_red_session_check_choices,
        lambda: const.red_session_check_hosts[state.time, agent_id],
        lambda: jnp.int32(-1),
    )
    synced_host_idx = jnp.clip(synced_host, 0, state.red_sessions.shape[1] - 1)
    synced_host_valid = (synced_host >= 0) & (session_counts[synced_host_idx] > 0) & const.host_active[synced_host_idx]
    rng_sampled = select_new_primary_session_host(session_counts, const.host_active, key)
    sampled = jnp.where(synced_host_valid, synced_host, rng_sampled)
    promoted = jnp.where(needs_primary, sampled, anchor)
    next_anchor = jnp.where(
        has_any_sessions,
        jnp.where(forced_valid, forced_primary_host, promoted),
        jnp.int32(-1),
    )
    next_idx = jnp.clip(next_anchor, 0, state.red_sessions.shape[1] - 1)
    # When CybORG PID allocation diverges from JAX, the forced PID may not
    # appear in JAX's session_pids row even though session 0 is still alive.
    # Accept the current primary PID when the forced PID confirms it.
    forced_confirms_current_pid = (
        (forced_primary_pid != UNKNOWN_PRIMARY_PID)
        & (forced_primary_pid == current_primary_pid)
        & (next_anchor >= 0)
        & (next_anchor == anchor)
    )
    current_pid_matches_next_anchor = (
        (next_anchor >= 0)
        & (next_anchor == anchor)
        & (
            pid_row_contains(state.red_session_pids[agent_id, next_idx], current_primary_pid)
            | forced_confirms_current_pid
        )
    )
    # CybORG's _choose_new_primary_session picks a random session via
    # np_random.choice(all_sessions).  Match this by picking a random
    # session on the chosen host (uniform) rather than always first_valid_pid.
    session_check_key = jax.random.fold_in(key, jnp.int32(7723))
    host_session_count = jnp.maximum(session_counts[next_idx], jnp.int32(1))
    within_host_slot = sample_red_session_check_choice(
        const, state.time, agent_id, session_check_key, host_session_count
    )
    selected_primary_pid = jax.lax.cond(
        next_anchor >= 0,
        lambda _: nth_valid_pid(state.red_session_pids[agent_id, next_idx], within_host_slot),
        lambda _: jnp.int32(-1),
        operand=None,
    )
    next_primary_pid = jnp.where(
        has_any_sessions,
        jnp.where(
            current_pid_matches_next_anchor,
            current_primary_pid,
            jnp.where(forced_pid_valid, forced_primary_pid, selected_primary_pid),
        ),
        jnp.int32(-1),
    )
    should_reorder_primary_row = needs_primary & (next_anchor >= 0) & (next_primary_pid >= 0)
    reordered_pid_row = move_pid_to_row_end(state.red_session_pids[agent_id, next_idx], next_primary_pid)
    reordered_abstract_pid_row = move_pid_to_row_end(
        state.red_session_abstract_pids[agent_id, next_idx], next_primary_pid
    )
    reordered_privileged_pid_row = move_pid_to_row_end(
        state.red_session_privileged_pids[agent_id, next_idx], next_primary_pid
    )
    red_session_pids = jnp.where(
        should_reorder_primary_row,
        state.red_session_pids.at[agent_id, next_idx].set(reordered_pid_row),
        state.red_session_pids,
    )
    red_session_abstract_pids = jnp.where(
        should_reorder_primary_row,
        state.red_session_abstract_pids.at[agent_id, next_idx].set(reordered_abstract_pid_row),
        state.red_session_abstract_pids,
    )
    red_session_privileged_pids = jnp.where(
        should_reorder_primary_row,
        state.red_session_privileged_pids.at[agent_id, next_idx].set(reordered_privileged_pid_row),
        state.red_session_privileged_pids,
    )
    next_primary_is_abstract = jax.lax.cond(
        next_anchor >= 0,
        lambda _: pid_row_contains(red_session_abstract_pids[agent_id, next_idx], next_primary_pid),
        # CybORG defaults to True when session 0 is absent.  A stale False
        # left by blue Restore/Remove must not persist once all sessions are gone.
        lambda _: jnp.bool_(True),
        operand=None,
    )
    promoted_abstract_primary = (
        has_any_sessions & (next_anchor >= 0) & (next_anchor != anchor) & next_primary_is_abstract
    )
    red_abstract_host_rank = jax.lax.cond(
        promoted_abstract_primary,
        lambda ranks: ranks.at[agent_id, next_idx].set(jnp.int32(0)),
        lambda ranks: ranks,
        state.red_abstract_host_rank,
    )
    anchor_changed = has_any_sessions & (next_anchor >= 0) & (next_anchor != anchor)

    # CybORG scan memory (ports dict) is per-session-object.  Ports survive
    # as long as the session that performed the scan is alive, regardless of
    # primary identity.  Only clear when the scan-owning session is dead.
    anchor_changed_host = anchor_changed & (anchor >= 0)
    primary_invalidated_same_host = (
        (needs_primary | ~has_any_sessions) & ~anchor_changed_host & (anchor >= 0) & (current_primary_pid >= 0)
    )
    old_anchor_idx = jnp.clip(anchor, 0, state.red_scanned_source_hosts.shape[2] - 1)
    scan_owner_pid = state.red_scan_source_pid[agent_id, old_anchor_idx]
    scan_owner_alive = pid_row_contains(state.red_session_pids[agent_id, old_anchor_idx], scan_owner_pid)
    # Preserve scan memory when the owning session is alive.  When no owner
    # is recorded (pid == -1), fall back to clearing (conservative).
    preserve_scan = scan_owner_alive & (scan_owner_pid >= 0)
    should_clear_scan = (anchor_changed_host | primary_invalidated_same_host) & ~preserve_scan
    cleared_source_hosts = state.red_scanned_source_hosts.at[agent_id, :, old_anchor_idx].set(False)
    cleared_scanned_hosts = jnp.any(cleared_source_hosts[agent_id], axis=1)
    red_scanned_source_hosts = jnp.where(
        should_clear_scan,
        cleared_source_hosts,
        state.red_scanned_source_hosts,
    )
    red_scanned_hosts = jnp.where(
        should_clear_scan,
        state.red_scanned_hosts.at[agent_id].set(cleared_scanned_hosts),
        state.red_scanned_hosts,
    )
    # Clear scan-owning PID on old anchor when scan memory is cleared.
    red_scan_source_pid = jnp.where(
        should_clear_scan,
        state.red_scan_source_pid.at[agent_id, old_anchor_idx].set(-1),
        state.red_scan_source_pid,
    )

    return state.replace(
        red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(next_anchor),
        red_abstract_host_rank=red_abstract_host_rank,
        red_primary_is_abstract=state.red_primary_is_abstract.at[agent_id].set(next_primary_is_abstract),
        red_primary_pid=state.red_primary_pid.at[agent_id].set(next_primary_pid),
        red_session_pids=red_session_pids,
        red_session_abstract_pids=red_session_abstract_pids,
        red_session_privileged_pids=red_session_privileged_pids,
        red_scanned_source_hosts=red_scanned_source_hosts,
        red_scanned_hosts=red_scanned_hosts,
        red_scan_source_pid=red_scan_source_pid,
    )


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
    target_sources = scan_sources(state)[agent_id, target_idx]
    active_abstract_sources = (
        target_sources & state.red_sessions[agent_id] & state.red_session_is_abstract[agent_id] & const.host_active
    )
    has_owner = jnp.any(active_abstract_sources)
    owner_host = jnp.where(has_owner, jnp.argmax(active_abstract_sources), -1)

    # During duration processing, scans can be pre-bound to a specific source host.
    # Respect that explicit binding (including "bound to none") instead of recomputing
    # against potentially changed same-step state (e.g. after green updates).
    pending_source_kind = state.red_pending_source_kind[agent_id]
    pending_source_host = state.red_pending_source_host[agent_id]
    pending_idx = jnp.clip(pending_source_host, 0, state.red_sessions.shape[1] - 1)
    pending_host_valid = (
        (pending_source_host >= 0)
        & state.red_sessions[agent_id, pending_idx]
        & state.red_session_is_abstract[agent_id, pending_idx]
        & const.host_active[pending_idx]
    )
    bound_source_host = select_bound_source_host(state, const, agent_id)
    bound_idx = jnp.clip(bound_source_host, 0, state.red_sessions.shape[1] - 1)
    pending_bound_valid = (
        (bound_source_host >= 0)
        & state.red_session_is_abstract[agent_id, bound_idx]
        & state.red_sessions[agent_id, bound_idx]
        & const.host_active[bound_idx]
    )
    pending_source = jnp.select(
        [
            pending_source_kind == PENDING_SOURCE_KIND_HOST,
            pending_source_kind == PENDING_SOURCE_KIND_SESSION_BINDING,
            pending_source_kind == PENDING_SOURCE_KIND_BOUND_NONE,
        ],
        [
            jnp.where(pending_host_valid, pending_source_host, jnp.int32(-1)),
            jnp.where(pending_bound_valid, bound_source_host, jnp.int32(-1)),
            jnp.int32(-1),
        ],
        default=jnp.int32(-1),
    )
    has_pending_binding = pending_source_kind != PENDING_SOURCE_KIND_NONE

    fallback = select_scan_source_host(state, const, agent_id)
    computed = jnp.where(has_owner, owner_host, fallback)
    return jnp.where(
        has_pending_binding,
        pending_source,
        computed,
    )


def shortest_path_nodes(
    data_links: chex.Array,
    host_active: chex.Array,
    source_host: chex.Array,
    target_host: chex.Array,
) -> tuple[chex.Array, chex.Array]:
    """Return ordered shortest-path hosts from source to target, padded with -1."""
    n_hosts = data_links.shape[0]
    empty_path = jnp.full((GLOBAL_MAX_HOSTS,), -1, dtype=jnp.int32)
    source_idx = jnp.clip(source_host, 0, n_hosts - 1)
    target_idx = jnp.clip(target_host, 0, n_hosts - 1)
    valid = (source_host >= 0) & (target_host >= 0) & host_active[source_idx] & host_active[target_idx]

    def _compute_path(_):
        visited = jnp.zeros((n_hosts,), dtype=jnp.bool_).at[source_idx].set(True)
        frontier = jnp.zeros((n_hosts,), dtype=jnp.bool_).at[source_idx].set(True)
        parent = jnp.full((n_hosts,), -1, dtype=jnp.int32)
        found = visited[target_idx]

        def _cond(carry):
            _, frontier_mask, _, _, found_target = carry
            return (~found_target) & jnp.any(frontier_mask)

        def _body(carry):
            step, frontier_mask, visited_mask, parent_idx, _ = carry
            edge_mask = frontier_mask[:, None] & data_links
            new_frontier = jnp.any(edge_mask, axis=0) & ~visited_mask & host_active
            new_parents = jnp.argmax(edge_mask, axis=0).astype(jnp.int32)
            parent_idx = jnp.where(new_frontier, new_parents, parent_idx)
            visited_mask = visited_mask | new_frontier
            return step + 1, new_frontier, visited_mask, parent_idx, visited_mask[target_idx]

        _, _, visited, parent, found = jax.lax.while_loop(
            _cond,
            _body,
            (jnp.int32(0), frontier, visited, parent, found),
        )

        def _reconstruct(_):
            reverse_path = jnp.full((GLOBAL_MAX_HOSTS,), -1, dtype=jnp.int32)

            def _rev_cond(carry):
                idx, current, _ = carry
                return (current >= 0) & (idx < GLOBAL_MAX_HOSTS)

            def _rev_body(carry):
                idx, current, rev = carry
                rev = rev.at[idx].set(current)
                next_current = jnp.where(current == source_idx, jnp.int32(-1), parent[current])
                return idx + 1, next_current, rev

            rev_len, _, reverse_path = jax.lax.while_loop(
                _rev_cond,
                _rev_body,
                (jnp.int32(0), target_idx, reverse_path),
            )
            ordered = jnp.full((GLOBAL_MAX_HOSTS,), -1, dtype=jnp.int32)

            def _write_path(i, arr):
                return arr.at[i].set(reverse_path[rev_len - i - 1])

            ordered = jax.lax.fori_loop(0, rev_len, _write_path, ordered)
            return ordered, rev_len

        return jax.lax.cond(found, _reconstruct, lambda _: (empty_path, jnp.int32(0)), operand=None)

    return jax.lax.cond(valid, _compute_path, lambda _: (empty_path, jnp.int32(0)), operand=None)


def apply_exploit_route_detection(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
    enabled: chex.Array,
    key: jax.Array,
) -> CC4State:
    """Mirror SSHBruteForce route-level network connection events."""
    source_host = select_scan_execution_source_host(state, const, agent_id, target_host)
    path_nodes, path_len = shortest_path_nodes(const.data_links, const.host_active, source_host, target_host)

    def _apply(state_in):
        def _body(i, current_state):
            host_idx = path_nodes[i]
            draw_key = jax.random.fold_in(key, i)
            rand_val, current_state = sample_detection_random(current_state, const, draw_key)
            detected = rand_val > jnp.float32(1.0 - EXPLOIT_ROUTE_DETECTION_RATE)
            host_activity_detected = jnp.where(
                detected,
                current_state.host_activity_detected.at[host_idx].set(True),
                current_state.host_activity_detected,
            )
            return current_state.replace(host_activity_detected=host_activity_detected)

        return jax.lax.fori_loop(0, path_len, _body, state_in)

    return jax.lax.cond(enabled & (path_len > 0), _apply, lambda s: s, state)


def sample_sim_exploit_success_roll(
    state: CC4State,
    const: CC4Const,
    enabled: chex.Array,
    key: jax.Array,
) -> tuple[chex.Array, CC4State]:
    """Mirror ExploitAction.sim_exploit success-rate RNG consumption.

    CC4 exploit success rates are effectively 1.0, but CybORG still consumes
    one random draw when an exploit reaches the success-rate gate.
    """

    def _sample(state_in):
        rand_val, next_state = sample_detection_random(state_in, const, key)
        return rand_val > jnp.float32(0.0), next_state

    return jax.lax.cond(
        enabled,
        _sample,
        lambda state_in: (jnp.bool_(False), state_in),
        state,
    )


def sample_exploit_process_event_roll(
    state: CC4State,
    const: CC4Const,
    enabled: chex.Array,
    key: jax.Array,
) -> tuple[chex.Array, CC4State]:
    """Mirror ExploitAction process_creation event detection roll."""

    def _sample(state_in):
        rand_val, next_state = sample_detection_random(state_in, const, key)
        detected = rand_val > jnp.float32(1.0 - EXPLOIT_PROCESS_EVENT_DETECTION_RATE)
        return detected, next_state

    return jax.lax.cond(
        enabled,
        _sample,
        lambda state_in: (jnp.bool_(False), state_in),
        state,
    )


def apply_exploit_success(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
    success: chex.Array,
    key: jax.Array,
    process_event_detected: chex.Array | None = None,
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
    pid_delta = sample_red_pid_delta(const, state.time, agent_id, key)
    new_pid = allocate_host_pid_from_delta(state, const, target_host, pid_delta)
    red_next_pid = jnp.where(success, jnp.maximum(state.red_next_pid, new_pid + 1), state.red_next_pid)
    host_max_pid = jnp.where(
        success,
        state.host_max_pid.at[target_host].set(jnp.maximum(state.host_max_pid[target_host], new_pid)),
        state.host_max_pid,
    )
    session_pid_row = state.red_session_pids[agent_id, target_host]
    pid_row_updated = append_pid_to_row(session_pid_row, new_pid)
    red_session_pids = jnp.where(
        success,
        state.red_session_pids.at[agent_id, target_host].set(pid_row_updated),
        state.red_session_pids,
    )
    emit_process_event = success if process_event_detected is None else (success & process_event_detected)
    event_row = state.host_process_creation_pids[target_host]
    updated_event_row = append_process_event(event_row, new_pid)
    host_process_creation_pids = jnp.where(
        emit_process_event,
        state.host_process_creation_pids.at[target_host].set(updated_event_row),
        state.host_process_creation_pids,
    )

    red_exploit_success = jnp.where(
        success,
        state.red_exploit_success.at[agent_id].set(True),
        state.red_exploit_success,
    )
    # CybORG's _process_new_observations adds hosts from ANY observation to
    # host_states.  A successful exploit reveals the target host in the
    # observation, so the FSM should know about it.
    fsm_host_entered = jnp.where(
        success,
        state.fsm_host_entered.at[agent_id, target_host].set(True),
        state.fsm_host_entered,
    )
    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_suspicious_process_count=red_suspicious_process_count,
        red_privilege=red_privilege,
        red_session_pids=red_session_pids,
        red_next_pid=red_next_pid,
        host_max_pid=host_max_pid,
        host_compromised=host_compromised,
        host_has_malware=host_has_malware,
        host_suspicious_process=host_suspicious_process,
        red_activity_this_step=activity,
        host_process_creation_pids=host_process_creation_pids,
        red_exploit_success=red_exploit_success,
        fsm_host_entered=fsm_host_entered,
    )
