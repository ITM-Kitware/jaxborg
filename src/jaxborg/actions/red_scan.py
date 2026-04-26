import chex
import jax.numpy as jnp

from jaxborg.actions.red_common import (
    can_reach_subnet_from_source_host,
    scan_sources,
    select_scan_execution_source_host,
    sync_scan_memory_fields,
)
from jaxborg.constants import ACTIVITY_SCAN
from jaxborg.state import SimulatorConst, SimulatorState


def apply_scan(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    target_host: chex.Array,
) -> SimulatorState:
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

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_SCAN),
        state.red_activity_this_step,
    )
    red_scan_anchor_host = jnp.where(
        success & (state.red_scan_anchor_host[agent_id] < 0),
        state.red_scan_anchor_host.at[agent_id].set(source_host),
        state.red_scan_anchor_host,
    )
    # Record the PID of the session performing the scan on the source host.
    # CybORG stores scan results (ports) in the executing session object;
    # when that session is killed the scan knowledge is lost.  The scan is
    # performed by the primary session (session 0), whose PID is tracked in
    # red_primary_pid.  After promotions, session 0's PID is NOT necessarily
    # the smallest — it's the promoted session's original PID.
    executing_pid = state.red_primary_pid[agent_id]
    red_scan_source_pid = jnp.where(
        success,
        state.red_scan_source_pid.at[agent_id, source_idx].set(executing_pid),
        state.red_scan_source_pid,
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
        red_activity_this_step=activity,
        fsm_host_entered=fsm_host_entered,
    )
    next_state = sync_scan_memory_fields(next_state, const, source_matrix=source_matrix)
    return next_state
