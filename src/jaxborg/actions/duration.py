import jax
import jax.numpy as jnp

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.encoding import (
    ACTION_TYPE_AGGRESSIVE_SCAN,
    ACTION_TYPE_DISCOVER_DECEPTION,
    ACTION_TYPE_EXPLOIT_BLUEKEEP,
    ACTION_TYPE_EXPLOIT_SSH,
    ACTION_TYPE_SCAN,
    ACTION_TYPE_STEALTH_SCAN,
    BLUE_ACTION_DURATIONS,
    RED_ACTION_DURATIONS,
    decode_blue_action,
    decode_red_action,
)
from jaxborg.actions.pending_source import (
    PENDING_SOURCE_KIND_HOST,
    PENDING_SOURCE_KIND_NONE,
    PENDING_SOURCE_KIND_SESSION_BINDING,
)
from jaxborg.actions.pids import pid_row_contains
from jaxborg.actions.red_common import (
    apply_red_session_check,
    scan_sources,
    select_bound_source_host,
    select_scan_execution_source_host,
)
from jaxborg.state import SimulatorConst, SimulatorState

UNKNOWN_PRIMARY_HOST = jnp.int32(-2)
UNKNOWN_PRIMARY_PID = jnp.int32(-2)


def process_red_with_duration(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    action_idx: int,
    key: jax.Array,
    forced_primary_host: jnp.int32 = UNKNOWN_PRIMARY_HOST,
    forced_primary_pid: jnp.int32 = UNKNOWN_PRIMARY_PID,
    *,
    run_session_check: bool = True,
    creation_visible_sessions_override: jnp.int32 | None = None,
) -> SimulatorState:
    is_busy = state.red_pending_ticks[agent_id] > 0

    effective_action = jnp.where(is_busy, state.red_pending_action[agent_id], action_idx)
    effective_key = jnp.where(
        is_busy,
        state.red_pending_key[agent_id],
        jnp.asarray(key, dtype=jnp.uint32),
    )

    action_type, _, target_host = decode_red_action(effective_action, agent_id, const)
    duration = RED_ACTION_DURATIONS[action_type]
    current_ticks = jnp.where(is_busy, state.red_pending_ticks[agent_id], duration)
    is_scan_action = (
        (action_type == ACTION_TYPE_SCAN)
        | (action_type == ACTION_TYPE_AGGRESSIVE_SCAN)
        | (action_type == ACTION_TYPE_STEALTH_SCAN)
    )
    is_exploit_action = (action_type >= ACTION_TYPE_EXPLOIT_SSH) & (action_type <= ACTION_TYPE_EXPLOIT_BLUEKEEP)
    is_target_source_action = is_scan_action | is_exploit_action
    pending_source_kind = state.red_pending_source_kind[agent_id]
    pending_source_host = state.red_pending_source_host[agent_id]
    source_is_bound = pending_source_kind != PENDING_SOURCE_KIND_NONE
    primary_snapshot_known = forced_primary_host != UNKNOWN_PRIMARY_HOST
    primary_missing = primary_snapshot_known & (forced_primary_host < 0)
    forced_source_idx = jnp.clip(forced_primary_host, 0, state.red_sessions.shape[1] - 1)
    forced_primary_pid_known = forced_primary_pid != UNKNOWN_PRIMARY_PID
    forced_primary_pid_live = pid_row_contains(state.red_session_pids[agent_id, forced_source_idx], forced_primary_pid)
    # When CybORG PID allocation diverges from JAX (different deltas), the
    # forced PID won't appear in JAX's session_pids even though session 0 is
    # still alive.  Fall back to abstract-session existence so PID drift
    # doesn't falsely invalidate the source.  Blue Remove already clears
    # red_primary_is_abstract when it kills session 0, so the downstream
    # source_valid gate (which checks that flag) still correctly blocks
    # scans after session 0 destruction.
    forced_primary_pid_ok = forced_primary_pid_live | (
        state.red_session_is_abstract[agent_id, forced_source_idx] & state.red_sessions[agent_id, forced_source_idx]
    )
    forced_source_valid = (
        (forced_primary_host >= 0)
        & state.red_sessions[agent_id, forced_source_idx]
        & const.host_active[forced_source_idx]
        # Distinguish "session 0 survived" from "another session still exists on
        # the same host after blue Remove killed session 0".
        & jnp.where(forced_primary_pid_known, forced_primary_pid_ok, jnp.array(True))
    )
    live_bound_source_host = select_bound_source_host(state, const, agent_id)
    bound_source_host = jnp.where(
        primary_snapshot_known,
        jnp.where(forced_source_valid, forced_primary_host, jnp.int32(-1)),
        live_bound_source_host,
    )
    queued_source_host = jnp.where(
        is_scan_action,
        jnp.where(
            source_is_bound,
            pending_source_host,
            jnp.where(
                primary_missing,
                jnp.int32(-1),
                select_scan_execution_source_host(state, const, agent_id, target_host),
            ),
        ),
        jnp.where(
            is_exploit_action,
            jnp.where(primary_missing, jnp.int32(-1), bound_source_host),
            jnp.int32(-1),
        ),
    )
    queued_source_host = jnp.where(
        is_scan_action & ~source_is_bound & forced_source_valid,
        bound_source_host,
        queued_source_host,
    )
    target_idx = jnp.clip(target_host, 0, state.red_scanned_hosts.shape[1] - 1)
    queued_source_idx = jnp.clip(queued_source_host, 0, state.red_sessions.shape[1] - 1)
    source_matrix = scan_sources(state)
    source_from_scan_memory = (
        is_scan_action & (queued_source_host >= 0) & source_matrix[agent_id, target_idx, queued_source_idx]
    )
    source_from_bound_session = is_scan_action & (queued_source_host >= 0) & (queued_source_host == bound_source_host)
    queued_source_kind = jnp.where(
        source_is_bound,
        pending_source_kind,
        jnp.where(
            is_scan_action & (queued_source_host >= 0) & ~source_from_scan_memory & source_from_bound_session,
            PENDING_SOURCE_KIND_SESSION_BINDING,
            jnp.where(
                is_scan_action & (queued_source_host >= 0),
                PENDING_SOURCE_KIND_HOST,
                jnp.where(
                    is_exploit_action & (queued_source_host >= 0),
                    PENDING_SOURCE_KIND_SESSION_BINDING,
                    PENDING_SOURCE_KIND_NONE,
                ),
            ),
        ),
    )
    queued_source_binding_host = jnp.where(
        source_is_bound & (pending_source_kind == PENDING_SOURCE_KIND_HOST),
        pending_source_host,
        jnp.where(
            (queued_source_kind == PENDING_SOURCE_KIND_HOST)
            | (queued_source_kind == PENDING_SOURCE_KIND_SESSION_BINDING),
            queued_source_host,
            jnp.int32(-1),
        ),
    )
    effective_source_kind = jnp.where(is_busy, pending_source_kind, queued_source_kind)
    effective_source_binding_host = jnp.where(is_busy, pending_source_host, queued_source_binding_host)
    anchor_source_host = bound_source_host
    # CybORG binds scan actions to session 0 (a session ID), not a fixed host.
    # When RedSessionCheck promotes a new session to slot 0 between queuing and
    # execution, CybORG follows the updated session.  When forced_primary_host
    # is available (Category A sync), use it directly.  Otherwise, follow the
    # live anchor host — this tracks session 0's current location after any
    # RedSessionCheck promotions.  If session 0 was destroyed (blue Restore),
    # the anchor will be invalid (-1) and the scan correctly fails.
    execution_source_host = jnp.where(
        is_busy & (pending_source_kind == PENDING_SOURCE_KIND_SESSION_BINDING),
        jnp.where(
            primary_snapshot_known,
            jnp.where(forced_source_valid, forced_primary_host, jnp.int32(-1)),
            anchor_source_host,
        ),
        effective_source_binding_host,
    )
    effective_source_host = jnp.where(
        is_scan_action & (execution_source_host < 0) & (anchor_source_host >= 0),
        anchor_source_host,
        execution_source_host,
    )
    source_idx = jnp.clip(effective_source_host, 0, state.red_sessions.shape[1] - 1)
    # CybORG's DiscoverNetworkServices.execute() checks
    # isinstance(session_0, RedAbstractSession) (line 67).  Session 0 can be
    # a regular Session when RedSessionCheck promotes a non-abstract session.
    # The per-agent red_primary_is_abstract flag tracks this exactly, whereas
    # the per-host red_session_is_abstract flag would be True if *any* session
    # on the host is abstract (masking a non-abstract session 0).
    source_is_abstract = state.red_primary_is_abstract[agent_id]
    source_valid = (
        (effective_source_host >= 0)
        & state.red_sessions[agent_id, source_idx]
        & source_is_abstract
        & const.host_active[source_idx]
    )

    new_ticks = current_ticks - 1
    should_execute = new_ticks <= 0
    is_discover_deception = action_type == ACTION_TYPE_DISCOVER_DECEPTION
    # CybORG's DiscoverDeception.execute() checks that session 0 exists (line 70).
    # After blue restore clears the session 0 host, the action must fail.
    # DiscoverDeception always executes as a pending action (duration=2), so
    # forced_source_valid (which checks the live JAX state at forced_primary_host)
    # tells us whether blue restore destroyed session 0 during this step.
    # When forced_primary_host is unavailable (-1, production mode), fall back to
    # bound_source_host which uses the current anchor.
    deception_source_host = jnp.where(
        is_discover_deception & (forced_primary_host >= 0),
        forced_primary_host,
        jnp.where(is_discover_deception, bound_source_host, jnp.int32(-1)),
    )
    deception_source_idx = jnp.clip(deception_source_host, 0, state.red_sessions.shape[1] - 1)
    deception_source_valid = (
        (deception_source_host >= 0)
        & state.red_sessions[agent_id, deception_source_idx]
        & const.host_active[deception_source_idx]
    )
    requires_bound_source = is_target_source_action | is_discover_deception
    source_ok = jnp.where(is_discover_deception, deception_source_valid, source_valid)
    can_execute = should_execute & ((~requires_bound_source) | source_ok)
    execution_pending_kind = jnp.where(
        is_target_source_action & (effective_source_host >= 0),
        PENDING_SOURCE_KIND_HOST,
        PENDING_SOURCE_KIND_NONE,
    )
    # Snapshot visible_sessions at creation time for the exploit 1/N roll.
    # CybORG's FSM picks from server_session which accumulates unique session
    # IDs monotonically — use the cumulative counter that mirrors this.
    # When an override is provided, use it — this lets the caller snapshot
    # the count before the green phase, matching CybORG's get_action() timing
    # where server_session reflects the previous step's observation.
    creation_visible_sessions = (
        creation_visible_sessions_override
        if creation_visible_sessions_override is not None
        else state.red_server_session_count[agent_id]
    )
    pending_visible_sessions = jnp.where(
        is_busy, state.red_pending_visible_sessions[agent_id], creation_visible_sessions
    )
    state_with_source = state.replace(
        red_pending_source_kind=state.red_pending_source_kind.at[agent_id].set(execution_pending_kind),
        red_pending_source_host=state.red_pending_source_host.at[agent_id].set(effective_source_host),
        red_pending_visible_sessions=state.red_pending_visible_sessions.at[agent_id].set(pending_visible_sessions),
    )

    new_state = jax.lax.cond(
        can_execute,
        lambda s: apply_red_action(s, const, agent_id, effective_action, effective_key),
        lambda s: s,
        state_with_source,
    )

    final_ticks = jnp.where(should_execute, jnp.int32(0), new_ticks)
    final_source_kind = jnp.where(should_execute, PENDING_SOURCE_KIND_NONE, effective_source_kind)
    preserve_source = ~should_execute & (
        (final_source_kind == PENDING_SOURCE_KIND_HOST) | (final_source_kind == PENDING_SOURCE_KIND_SESSION_BINDING)
    )
    final_source_host = jnp.where(
        preserve_source,
        effective_source_binding_host,
        jnp.int32(-1),
    )
    final_visible_sessions = jnp.where(should_execute, jnp.int32(1), pending_visible_sessions)
    new_state = new_state.replace(
        red_pending_ticks=new_state.red_pending_ticks.at[agent_id].set(final_ticks),
        red_pending_action=new_state.red_pending_action.at[agent_id].set(effective_action),
        red_pending_key=new_state.red_pending_key.at[agent_id].set(effective_key),
        red_pending_source_kind=new_state.red_pending_source_kind.at[agent_id].set(final_source_kind),
        red_pending_source_host=new_state.red_pending_source_host.at[agent_id].set(final_source_host),
        red_pending_visible_sessions=new_state.red_pending_visible_sessions.at[agent_id].set(final_visible_sessions),
    )

    if run_session_check:
        session_check_key = jax.random.fold_in(jnp.asarray(key, dtype=jnp.uint32), jnp.int32(931))
        new_state = apply_red_session_check(
            new_state,
            const,
            agent_id,
            session_check_key,
            forced_primary_host=forced_primary_host,
            forced_primary_pid=forced_primary_pid,
        )

    return new_state


def process_blue_with_duration(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    action_idx: int,
    key=None,
) -> SimulatorState:
    is_busy = state.blue_pending_ticks[agent_id] > 0

    effective_action = jnp.where(is_busy, state.blue_pending_action[agent_id], action_idx)

    action_type, _, _, _, _ = decode_blue_action(effective_action, agent_id, const)
    duration = BLUE_ACTION_DURATIONS[action_type]
    current_ticks = jnp.where(is_busy, state.blue_pending_ticks[agent_id], duration)

    new_ticks = current_ticks - 1
    should_execute = new_ticks <= 0

    new_state = jax.lax.cond(
        should_execute,
        lambda s: apply_blue_action(s, const, agent_id, effective_action, key),
        lambda s: s,
        state,
    )

    final_ticks = jnp.where(should_execute, jnp.int32(0), new_ticks)
    new_state = new_state.replace(
        blue_pending_ticks=new_state.blue_pending_ticks.at[agent_id].set(final_ticks),
        blue_pending_action=new_state.blue_pending_action.at[agent_id].set(effective_action),
    )

    return new_state
