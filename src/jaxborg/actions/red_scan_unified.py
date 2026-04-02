import chex
import jax
import jax.numpy as jnp

from jaxborg.actions.red_common import (
    can_reach_subnet_from_source_host,
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
    source_host = select_scan_execution_source_host(state, const, agent_id, target_host)
    can_reach = can_reach_subnet_from_source_host(state, const, source_host, target_subnet)
    has_abstract_source = source_host >= 0
    success = is_active & is_discovered & can_reach & has_abstract_source

    source_matrix = scan_sources(state)
    source_idx = jnp.clip(source_host, 0, state.red_sessions.shape[1] - 1)
    source_matrix = jnp.where(
        success,
        source_matrix.at[agent_id, target_host, source_idx].set(True),
        source_matrix,
    )

    should_roll = has_detection_roll & success

    # CybORG Portscan: decoy processes always trigger detection regardless of random
    has_decoy = jnp.any(state.host_decoys[target_host])

    def with_roll(s: CC4State):
        rand_val, next_state = sample_detection_random(s, const, key)
        return (rand_val < detection_rate) | has_decoy, next_state

    def without_roll(s: CC4State):
        return success, s

    detected, state = jax.lax.cond(should_roll, with_roll, without_roll, state)

    activity = jnp.where(
        detected,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_SCAN),
        state.red_activity_this_step,
    )
    # CybORG Portscan creates network_connections events independently of exploit
    # events. Set host_activity_detected directly so a later exploit overwriting
    # red_activity_this_step doesn't erase the scan detection.
    scan_detected = jnp.where(
        detected,
        state.host_activity_detected.at[target_host].set(True),
        state.host_activity_detected,
    )
    red_scan_anchor_host = jnp.where(
        success & (state.red_scan_anchor_host[agent_id] < 0),
        state.red_scan_anchor_host.at[agent_id].set(source_host),
        state.red_scan_anchor_host,
    )
    # Record scan-owning PID on the source host (see apply_scan).
    executing_pid = state.red_primary_pid[agent_id]
    red_scan_source_pid = jnp.where(
        success,
        state.red_scan_source_pid.at[agent_id, source_idx].set(executing_pid),
        state.red_scan_source_pid,
    )

    red_scan_success = jnp.where(
        success,
        state.red_scan_success.at[agent_id].set(True),
        state.red_scan_success,
    )
    # CybORG's _process_new_observations adds hosts from ANY observation to
    # host_states.  A successful scan reveals the target in the observation.
    fsm_host_entered = jnp.where(
        success,
        state.fsm_host_entered.at[agent_id, target_host].set(True),
        state.fsm_host_entered,
    )

    next_state = state.replace(
        red_scan_anchor_host=red_scan_anchor_host,
        red_scan_source_pid=red_scan_source_pid,
        red_scan_success=red_scan_success,
        red_activity_this_step=activity,
        host_activity_detected=scan_detected,
        fsm_host_entered=fsm_host_entered,
    )
    return sync_scan_memory_fields(next_state, const, source_matrix=source_matrix)
