import jax
import jax.numpy as jnp

from jaxborg.actions.blue_decoys import host_decoy_compatibility_mask
from jaxborg.actions.encoding import (
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_ANALYSE_START,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_DECOY_START,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
)
from jaxborg.constants import ACTION_HOST_SLOTS, GLOBAL_MAX_HOSTS, NUM_DECOY_TYPES, NUM_SUBNETS
from jaxborg.state import CC4Const, CC4State


def compute_blue_action_mask(const: CC4Const, agent_id: int, state: CC4State | None = None) -> jnp.ndarray:
    """Return (BLUE_ALLOW_TRAFFIC_END,) bool mask of valid actions for a blue agent.

    Uses canonical (subnet, slot) encoding via obs_host_map for topology invariance.
    When `state` is provided, decoy validity reflects live services/decoys.
    Otherwise it falls back to reset-time services from `const`.
    """
    # (NUM_SUBNETS, OBS_HOSTS_PER_SUBNET) — True where slot has a valid host
    obs_valid = const.obs_host_map < GLOBAL_MAX_HOSTS
    # (NUM_SUBNETS,) — True where agent controls this subnet
    agent_subnets = const.blue_agent_subnets[agent_id]
    # (NUM_SUBNETS, OBS_HOSTS_PER_SUBNET) — valid slots for this agent
    slot_valid = obs_valid & agent_subnets[:, None]
    # Flatten to (ACTION_HOST_SLOTS,) matching canonical encoding
    slot_valid_flat = slot_valid.reshape(-1)

    mask = jnp.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=jnp.bool_)

    # Sleep (0) and Monitor (1) always valid
    mask = mask.at[0].set(True)
    mask = mask.at[1].set(True)

    # Analyse, Remove, Restore: same canonical slot mask
    mask = mask.at[BLUE_ANALYSE_START : BLUE_ANALYSE_START + ACTION_HOST_SLOTS].set(slot_valid_flat)
    mask = mask.at[BLUE_REMOVE_START : BLUE_REMOVE_START + ACTION_HOST_SLOTS].set(slot_valid_flat)
    mask = mask.at[BLUE_RESTORE_START : BLUE_RESTORE_START + ACTION_HOST_SLOTS].set(slot_valid_flat)

    safe_host_idx = jnp.where(obs_valid, const.obs_host_map, 0)
    if state is None:
        host_services = const.initial_services[safe_host_idx]
        host_decoys = jnp.zeros(host_services.shape[:-1] + (NUM_DECOY_TYPES,), dtype=jnp.bool_)
    else:
        host_services = state.host_services[safe_host_idx]
        host_decoys = state.host_decoys[safe_host_idx]

    flat_services = host_services.reshape(ACTION_HOST_SLOTS, -1)
    flat_decoys = host_decoys.reshape(ACTION_HOST_SLOTS, -1)
    flat_decoy_compat = jax.vmap(host_decoy_compatibility_mask)(flat_services, flat_decoys)

    # Decoy: host slot validity further gated by decoy/service compatibility
    for d in range(NUM_DECOY_TYPES):
        offset = BLUE_DECOY_START + d * ACTION_HOST_SLOTS
        mask = mask.at[offset : offset + ACTION_HOST_SLOTS].set(slot_valid_flat & flat_decoy_compat[:, d])

    # Block/Allow Traffic: agent controls dst subnet, src != dst
    src_idx = jnp.arange(NUM_SUBNETS)
    dst_idx = jnp.arange(NUM_SUBNETS)
    traffic_valid = agent_subnets[None, :] & (src_idx[:, None] != dst_idx[None, :])
    traffic_flat = traffic_valid.reshape(-1)

    mask = mask.at[BLUE_BLOCK_TRAFFIC_START : BLUE_BLOCK_TRAFFIC_START + NUM_SUBNETS * NUM_SUBNETS].set(traffic_flat)
    mask = mask.at[BLUE_ALLOW_TRAFFIC_START : BLUE_ALLOW_TRAFFIC_START + NUM_SUBNETS * NUM_SUBNETS].set(traffic_flat)

    return mask
