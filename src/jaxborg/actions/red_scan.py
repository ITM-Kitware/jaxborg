import chex
import jax.numpy as jnp

from jaxborg.actions.red_common import (
    can_reach_subnet,
    scan_via_owner_alive,
    select_scan_execution_source_host,
)
from jaxborg.constants import ACTIVITY_SCAN
from jaxborg.state import CC4Const, CC4State


def apply_scan(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    is_discovered = state.red_discovered_hosts[agent_id, target_host]
    target_subnet = const.host_subnet[target_host]
    can_reach = can_reach_subnet(state, const, agent_id, target_subnet)

    source_host = select_scan_execution_source_host(state, const, agent_id, target_host)
    has_abstract_source = source_host >= 0
    success = is_active & is_discovered & can_reach & has_abstract_source

    red_scanned_hosts = state.red_scanned_hosts.at[agent_id, target_host].set(
        state.red_scanned_hosts[agent_id, target_host] | success
    )

    current_owner_alive = scan_via_owner_alive(state, const, agent_id, target_host)
    should_update_owner = success & (~state.red_scanned_hosts[agent_id, target_host] | ~current_owner_alive)
    red_scanned_via = jnp.where(
        should_update_owner,
        state.red_scanned_via.at[agent_id, target_host].set(source_host),
        state.red_scanned_via,
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

    return state.replace(
        red_scanned_hosts=red_scanned_hosts,
        red_scanned_via=red_scanned_via,
        red_scan_anchor_host=red_scan_anchor_host,
        red_activity_this_step=activity,
    )
