import jax
import jax.numpy as jnp

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.encoding import (
    ACTION_TYPE_AGGRESSIVE_SCAN,
    ACTION_TYPE_SCAN,
    ACTION_TYPE_STEALTH_SCAN,
    BLUE_ACTION_DURATIONS,
    RED_ACTION_DURATIONS,
    decode_blue_action,
    decode_red_action,
)
from jaxborg.actions.red_common import (
    apply_red_session_check,
    scan_sources_with_fallback,
    select_bound_source_host,
    select_scan_execution_source_host,
)
from jaxborg.state import CC4Const, CC4State

PENDING_SOURCE_UNSET = jnp.int32(-1)
PENDING_SOURCE_NONE = jnp.int32(-2)


def process_red_with_duration(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    action_idx: int,
    key: jax.Array,
) -> CC4State:
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
    pending_source = state.red_pending_source_host[agent_id]
    source_is_bound = pending_source != PENDING_SOURCE_UNSET
    queued_source_host = jnp.where(
        is_scan_action,
        jnp.where(
            source_is_bound,
            pending_source,
            select_scan_execution_source_host(state, const, agent_id, target_host),
        ),
        PENDING_SOURCE_UNSET,
    )
    target_idx = jnp.clip(target_host, 0, state.red_scanned_hosts.shape[1] - 1)
    queued_source_idx = jnp.clip(queued_source_host, 0, state.red_sessions.shape[1] - 1)
    scan_sources = scan_sources_with_fallback(state)
    source_from_scan_memory_for_new_action = (
        is_scan_action & (queued_source_host >= 0) & scan_sources[agent_id, target_idx, queued_source_idx]
    )
    source_from_scan_memory = jnp.where(
        is_busy | source_is_bound,
        state.red_pending_source_from_scan_memory[agent_id],
        source_from_scan_memory_for_new_action,
    )
    effective_source_host = jnp.where(is_busy, pending_source, queued_source_host)
    anchor_source_host = select_bound_source_host(state, const, agent_id)
    effective_source_host = jnp.where(
        is_busy & is_scan_action & (effective_source_host == PENDING_SOURCE_UNSET) & (anchor_source_host >= 0),
        anchor_source_host,
        effective_source_host,
    )
    source_idx_pre = jnp.clip(effective_source_host, 0, state.red_sessions.shape[1] - 1)
    source_valid_pre = (
        (effective_source_host >= 0)
        & state.red_sessions[agent_id, source_idx_pre]
        & state.red_session_is_abstract[agent_id, source_idx_pre]
        & const.host_active[source_idx_pre]
    )
    effective_source_host = jnp.where(
        is_busy
        & is_scan_action
        & (effective_source_host >= 0)
        & ~source_valid_pre
        & (anchor_source_host >= 0)
        & ~source_from_scan_memory,
        anchor_source_host,
        effective_source_host,
    )
    source_idx = jnp.clip(effective_source_host, 0, state.red_sessions.shape[1] - 1)
    source_valid = (
        (effective_source_host >= 0)
        & state.red_sessions[agent_id, source_idx]
        & state.red_session_is_abstract[agent_id, source_idx]
        & const.host_active[source_idx]
    )

    new_ticks = current_ticks - 1
    should_execute = new_ticks <= 0
    requires_bound_source = is_scan_action
    can_execute = should_execute & ((~requires_bound_source) | source_valid)
    state_with_source = state.replace(
        red_pending_source_host=state.red_pending_source_host.at[agent_id].set(effective_source_host)
    )

    new_state = jax.lax.cond(
        can_execute,
        lambda s: apply_red_action(s, const, agent_id, effective_action, effective_key),
        lambda s: s,
        state_with_source,
    )

    final_ticks = jnp.where(should_execute, jnp.int32(0), new_ticks)
    final_source_host = jnp.where(should_execute, PENDING_SOURCE_UNSET, effective_source_host)
    final_source_from_scan_memory = jnp.where(should_execute, jnp.bool_(False), source_from_scan_memory)
    new_state = new_state.replace(
        red_pending_ticks=new_state.red_pending_ticks.at[agent_id].set(final_ticks),
        red_pending_action=new_state.red_pending_action.at[agent_id].set(effective_action),
        red_pending_key=new_state.red_pending_key.at[agent_id].set(effective_key),
        red_pending_source_host=new_state.red_pending_source_host.at[agent_id].set(final_source_host),
        red_pending_source_from_scan_memory=new_state.red_pending_source_from_scan_memory.at[agent_id].set(
            final_source_from_scan_memory
        ),
    )

    session_check_key = jax.random.fold_in(jnp.asarray(key, dtype=jnp.uint32), jnp.int32(931))
    new_state = apply_red_session_check(new_state, const, agent_id, session_check_key)

    return new_state


def process_blue_with_duration(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    action_idx: int,
) -> CC4State:
    is_busy = state.blue_pending_ticks[agent_id] > 0

    effective_action = jnp.where(is_busy, state.blue_pending_action[agent_id], action_idx)

    action_type, _, _, _, _ = decode_blue_action(effective_action, agent_id, const)
    duration = BLUE_ACTION_DURATIONS[action_type]
    current_ticks = jnp.where(is_busy, state.blue_pending_ticks[agent_id], duration)

    new_ticks = current_ticks - 1
    should_execute = new_ticks <= 0

    new_state = jax.lax.cond(
        should_execute,
        lambda s: apply_blue_action(s, const, agent_id, effective_action),
        lambda s: s,
        state,
    )

    final_ticks = jnp.where(should_execute, jnp.int32(0), new_ticks)
    new_state = new_state.replace(
        blue_pending_ticks=new_state.blue_pending_ticks.at[agent_id].set(final_ticks),
        blue_pending_action=new_state.blue_pending_action.at[agent_id].set(effective_action),
    )

    return new_state
