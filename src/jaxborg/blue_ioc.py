"""Observed IOC history and bounded sharing, shared with the released H-MARL adapter.

Only observed Analyse evidence, Monitor alerts and decoy connection origins enter
this memory. One source-host reference per sender is queued for the next tick.
"""

import jax
import jax.numpy as jnp


def initial_ioc_memory(n_hosts, n_blue=5):
    return dict(
        process=jnp.zeros(n_hosts, dtype=jnp.bool_),
        network=jnp.zeros(n_hosts, dtype=jnp.bool_),
        files=jnp.zeros(n_hosts, dtype=jnp.int32),
        decoys=jnp.zeros(n_hosts, dtype=jnp.bool_),
        pending=jnp.zeros((n_blue, n_hosts), dtype=jnp.bool_),
        delivered=jnp.zeros(n_hosts, dtype=jnp.bool_),
    )


def initialize_blue_ioc(state):
    """Allocate only for enhanced v2 environments or pretrained H-MARL."""
    n_red, n_hosts = state.red_sessions.shape
    events = jnp.full((n_hosts, n_red), -1, dtype=jnp.int32)
    return state.replace(
        blue_decoy_sources=events,
        blue_old_decoy_sources=events,
        blue_ioc_memory=initial_ioc_memory(n_hosts),
        blue_ioc_codes=jnp.zeros((5, 48), dtype=jnp.int32),
    )


def _host_slots(const, agent):
    n_subnets = 3 if agent == 4 else 1
    subnets = const.blue_obs_subnets[agent, :n_subnets]
    hosts = const.obs_host_map[subnets, :16]
    safe = jnp.clip(hosts, 0, const.host_active.size - 1)
    valid = (hosts < const.host_active.size) & const.host_active[safe]
    return subnets, safe, valid


def update_ioc_memory(state, const, memory):
    """Accumulate visible events and IOC messages; never inspect Red sessions."""
    recovered = state.blue_recovered_this_step
    process = (memory["process"] & ~recovered) | state.host_exploit_detected | state.old_host_exploit_detected
    network = (memory["network"] & ~recovered) | state.host_activity_detected | state.old_host_activity_detected
    files = jnp.where(recovered, 0, memory["files"])
    # Empty Analyse results do not erase upstream's file history.
    files = jnp.where(state.blue_file_evidence > 0, 3 - state.blue_file_evidence, files)
    decoys = (memory["decoys"] | memory["delivered"]) & ~recovered
    pending = memory["pending"]
    ioc_rows = []
    n_hosts = const.host_active.size
    events = jnp.concatenate([state.blue_old_decoy_sources, state.blue_decoy_sources], axis=1)
    blue_owned = jnp.any(const.blue_agent_hosts, axis=0) & ~const.host_is_router

    for agent in range(5):
        subnets, host_slots, valid_slots = _host_slots(const, agent)
        rows = []
        for subnet_slot in range(len(subnets)):
            hosts, valid = host_slots[subnet_slot], valid_slots[subnet_slot]

            def visit(slot, carry):
                flags, outgoing, first = carry
                host = hosts[slot]
                visit_host = valid[slot] & (first < 0)
                origins = events[host]
                safe_origins = jnp.clip(origins, 0, n_hosts - 1)
                observed = visit_host & (origins >= 0) & blue_owned[safe_origins]
                local = const.blue_agent_hosts[agent, safe_origins]
                # Scatter-max prevents duplicate/invalid source slots erasing evidence.
                flags = flags.at[safe_origins].max(observed & local)
                outgoing = outgoing.at[safe_origins].max(observed & ~local)
                first = jnp.where(visit_host & flags[host], slot, first)
                return flags, outgoing, first

            decoys, outgoing, first = jax.lax.fori_loop(0, 16, visit, (decoys, pending[agent], jnp.int32(-1)))
            pending = pending.at[agent].set(outgoing)
            iocs = jnp.where(jnp.arange(16) == first, 3, 0)
            iocs = jnp.where(files[hosts] > 0, files[hosts], iocs)
            rows.append(jnp.where(valid, iocs, 0))
        ioc_rows.append(jnp.stack(rows))

    # Broadcast at most one (subnet, host slot) per sender for the NEXT tick.
    # Stable subnet/slot priority replaces upstream's unspecified set iteration.
    order = const.obs_host_map[jnp.array([5, 4, 8, 6, 2, 3, 7, 0, 1]), :16].reshape(-1)
    safe_order = jnp.clip(order, 0, n_hosts - 1)
    candidates = pending[:, safe_order] & (order < n_hosts)
    chosen = safe_order[jnp.argmax(candidates, axis=1)]
    has_message = jnp.any(candidates, axis=1)
    delivered = jnp.zeros(n_hosts, dtype=jnp.bool_).at[chosen].max(has_message)
    pending = pending.at[jnp.arange(5), chosen].set(False)
    return dict(
        process=process, network=network, files=files, decoys=decoys, pending=pending, delivered=delivered
    ), ioc_rows
