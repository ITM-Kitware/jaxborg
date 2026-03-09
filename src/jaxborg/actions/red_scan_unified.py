import chex
import jax
import jax.numpy as jnp

from jaxborg.actions.red_common import (
    can_reach_subnet,
    scan_sources,
    select_scan_execution_source_host,
    sync_scan_memory_fields,
)
from jaxborg.actions.rng import sample_detection_random
from jaxborg.constants import ACTIVITY_SCAN
from jaxborg.state import CC4Const, CC4State


def apply_scan_unified(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
    key: jax.Array,
    has_detection_roll: chex.Array,
    detection_rate: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    is_discovered = state.red_discovered_hosts[agent_id, target_host]
    target_subnet = const.host_subnet[target_host]
    can_reach = can_reach_subnet(state, const, agent_id, target_subnet)

    source_host = select_scan_execution_source_host(state, const, agent_id, target_host)
    has_abstract_source = source_host >= 0
    success = is_active & is_discovered & can_reach & has_abstract_source

    source_matrix = scan_sources(state)
    source_idx = jnp.clip(source_host, 0, state.red_sessions.shape[1] - 1)
    source_matrix = jnp.where(
        success,
        source_matrix.at[agent_id, target_host, source_idx].set(True),
        source_matrix,
    )

    def with_roll(s: CC4State):
        rand_val, next_state = sample_detection_random(s, key)
        return success & (rand_val < detection_rate), next_state

    def without_roll(s: CC4State):
        return success, s

    detected, state = jax.lax.cond(has_detection_roll, with_roll, without_roll, state)

    activity = jnp.where(
        detected,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_SCAN),
        state.red_activity_this_step,
    )
    red_scan_anchor_host = jnp.where(
        success & (state.red_scan_anchor_host[agent_id] < 0),
        state.red_scan_anchor_host.at[agent_id].set(source_host),
        state.red_scan_anchor_host,
    )

    next_state = state.replace(
        red_scan_anchor_host=red_scan_anchor_host,
        red_activity_this_step=activity,
    )
    return sync_scan_memory_fields(next_state, const, source_matrix=source_matrix)
