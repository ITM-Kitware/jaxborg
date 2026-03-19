import chex
import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import (
    BLUE_OBS_SIZE,
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_MESSAGES,
    NUM_SUBNETS,
    OBS_VECTOR_HOSTS_PER_SUBNET,
)
from jaxborg.state import CC4Const, CC4State

CYBORG_POS_TO_JAX_ID = jnp.array([5, 4, 8, 6, 2, 3, 7, 0, 1], dtype=jnp.int32)

_inv = np.empty(NUM_SUBNETS, dtype=np.int32)
_inv[np.array([5, 4, 8, 6, 2, 3, 7, 0, 1])] = np.arange(NUM_SUBNETS)
JAX_ID_TO_CYBORG_POS = jnp.array(_inv, dtype=jnp.int32)

SUBNET_BLOCK_SIZE = NUM_SUBNETS * 3 + OBS_VECTOR_HOSTS_PER_SUBNET * 2


def _subnet_block(
    state: CC4State,
    const: CC4Const,
    subnet_id: int,
) -> chex.Array:
    cyborg_pos = JAX_ID_TO_CYBORG_POS[subnet_id]
    subnet_one_hot = jnp.zeros(NUM_SUBNETS, dtype=jnp.float32).at[cyborg_pos].set(1.0)

    blocked_jax = state.blocked_zones[subnet_id]
    blocked_cyborg = blocked_jax[CYBORG_POS_TO_JAX_ID]

    comms_jax = const.comms_policy[state.mission_phase, subnet_id]
    comms_cyborg = comms_jax[CYBORG_POS_TO_JAX_ID]

    # Observation vector uses only server+user slots (matching CybORG BlueFlatWrapper
    # which excludes routers). obs_host_map has an extra router slot used for action
    # encoding but not for observations.
    host_indices = const.obs_host_map[subnet_id, :OBS_VECTOR_HOSTS_PER_SUBNET]
    is_active = host_indices < GLOBAL_MAX_HOSTS

    safe_indices = jnp.where(is_active, host_indices, 0)
    # CybORG observations read old + current events (2-cycle persistence via Monitor aging)
    raw_exploit = state.host_exploit_detected[safe_indices] | state.old_host_exploit_detected[safe_indices]
    malicious_processes = jnp.where(is_active, raw_exploit, False).astype(jnp.float32)

    raw_detected = state.host_activity_detected[safe_indices] | state.old_host_activity_detected[safe_indices]
    network_connections = jnp.where(is_active, raw_detected, False).astype(jnp.float32)

    return jnp.concatenate(
        [
            subnet_one_hot,
            blocked_cyborg.astype(jnp.float32),
            comms_cyborg.astype(jnp.float32),
            malicious_processes,
            network_connections,
        ]
    )


def get_blue_obs(state: CC4State, const: CC4Const, agent_id: int) -> chex.Array:
    mission_phase = state.mission_phase.astype(jnp.float32)

    blocks = []
    for slot in range(3):
        sid = const.blue_obs_subnets[agent_id, slot]
        block = jnp.where(
            sid >= 0,
            _subnet_block(state, const, sid),
            jnp.zeros(SUBNET_BLOCK_SIZE, dtype=jnp.float32),
        )
        blocks.append(block)
    all_blocks = jnp.concatenate(blocks)
    num_subnets = jnp.sum(const.blue_obs_subnets[agent_id] >= 0).astype(jnp.int32)
    subnet_len = num_subnets * SUBNET_BLOCK_SIZE
    subnet_mask = jnp.arange(3 * SUBNET_BLOCK_SIZE, dtype=jnp.int32) < subnet_len
    subnet_section = jnp.where(subnet_mask, all_blocks, 0.0)

    other_agents = jnp.array([i for i in range(NUM_BLUE_AGENTS) if i != agent_id])
    msg_parts = []
    for idx in range(NUM_MESSAGES):
        sender = other_agents[idx]
        msg_parts.append(state.messages[sender, agent_id, :])
    message_section = jnp.concatenate(msg_parts)

    obs = jnp.zeros(BLUE_OBS_SIZE, dtype=jnp.float32)
    obs = obs.at[0].set(mission_phase)
    obs = obs.at[1 : 1 + 3 * SUBNET_BLOCK_SIZE].set(subnet_section)
    message_start = jnp.int32(1) + subnet_len
    obs = jax.lax.dynamic_update_slice(obs, message_section, (message_start,))
    return obs


def get_red_obs(state: CC4State, const: CC4Const, agent_id: int) -> chex.Array:
    return jnp.zeros(BLUE_OBS_SIZE, dtype=jnp.float32)
