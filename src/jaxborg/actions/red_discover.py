import chex

from jaxborg.actions.red_common import select_bound_source_host
from jaxborg.state import CC4Const, CC4State


def apply_discover(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_subnet: chex.Array,
) -> CC4State:
    source_host = select_bound_source_host(state, const, agent_id)
    has_bound_source = source_host >= 0
    # CybORG DiscoverRemoteSystems delegates to Pingsweep, which does not
    # consult state.blocks / firewall rules when resolving routes.
    can_reach = has_bound_source

    in_subnet = (const.host_subnet == target_subnet) & const.host_active
    pingable = in_subnet & const.host_respond_to_ping
    newly_discovered = pingable & can_reach

    new_discovered = state.red_discovered_hosts[agent_id] | newly_discovered
    red_discovered_hosts = state.red_discovered_hosts.at[agent_id].set(new_discovered)

    # Mark action-discovered hosts as entered in the FSM.  This mirrors
    # CybORG's _process_new_observations adding observed hosts to
    # host_states after the discover action's observation is returned.
    fsm_host_entered = state.fsm_host_entered.at[agent_id].set(state.fsm_host_entered[agent_id] | newly_discovered)

    return state.replace(
        red_discovered_hosts=red_discovered_hosts,
        fsm_host_entered=fsm_host_entered,
    )
