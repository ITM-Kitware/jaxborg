"""Optional decoy connection events, containing only observable remote IPs."""

import jax.numpy as jnp


def record_decoy_connection(state, const, agent_id, target_host, detected):
    if state.blue_decoy_sources is None:
        return state
    from jaxborg.actions.red_common import select_scan_execution_source_host

    source = select_scan_execution_source_host(state, const, agent_id, target_host)
    old = state.blue_decoy_sources[target_host, agent_id]
    events = state.blue_decoy_sources.at[target_host, agent_id].set(jnp.where(detected & (source >= 0), source, old))
    return state.replace(blue_decoy_sources=events)
