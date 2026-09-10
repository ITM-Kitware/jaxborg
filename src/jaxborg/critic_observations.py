"""Training-only CC4 world-state inputs for Blue's centralized value function.

Like JaxMARL's MAPPO world-state wrapper, each agent gets the same centralized
features plus its identity. These arrays never enter the actor observation dict.
"""

import jax.numpy as jnp

from jaxborg.constants import (
    BLUE_OBS_SIZE,
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_DECOY_TYPES,
    NUM_RED_AGENTS,
    NUM_SERVICES,
    NUM_SUBNETS,
)
from jaxborg.observations import get_blue_obs
from jaxborg.state import SimulatorConst, SimulatorState

CRITIC_INPUTS = ("joint_observations", "global_state")


def blue_critic_obs_size(critic_input: str) -> int:
    if critic_input not in CRITIC_INPUTS:
        raise ValueError(f"critic_input must be one of {CRITIC_INPUTS}, got {critic_input!r}")
    size = NUM_BLUE_AGENTS * BLUE_OBS_SIZE + NUM_BLUE_AGENTS
    if critic_input == "global_state":
        size += (
            1  # normalized time
            + 4 * GLOBAL_MAX_HOSTS  # active, subnet, compromise, stopped OT service
            + 2 * GLOBAL_MAX_HOSTS * (NUM_SERVICES + NUM_DECOY_TYPES)
            + 2 * NUM_RED_AGENTS  # active and pending ticks
            + NUM_BLUE_AGENTS  # pending ticks
            + NUM_SUBNETS * 3  # current phase's reward weights
        )
    return size


def get_blue_critic_obs(state: SimulatorState, const: SimulatorConst, critic_input: str = "global_state"):
    """Return (5, critic_obs_dim) features from this exact pre-action state.

    global_state is a compact representation, not the complete simulator state:
    it adds current compromise, service/decoy health, timing and reward weights
    to pooled Blue observations. It excludes RNG keys and future information.
    Inactive hosts and reliability of absent services/decoys are zeroed.
    """
    expected_size = blue_critic_obs_size(critic_input)
    observations = jnp.stack([get_blue_obs(state, const, i) for i in range(NUM_BLUE_AGENTS)])
    features = [observations.reshape(-1)]
    if critic_input == "global_state":
        active = const.host_active.astype(jnp.float32)
        services = state.host_services.astype(jnp.float32) * active[:, None]
        decoys = state.host_decoys.astype(jnp.float32) * active[:, None]
        features.extend(
            [
                jnp.asarray(state.time, dtype=jnp.float32).reshape(1) / jnp.maximum(const.max_steps, 1),
                active,
                const.host_subnet.astype(jnp.float32) * active / (NUM_SUBNETS - 1),
                state.host_compromised.astype(jnp.float32) * active / 2.0,
                state.ot_service_stopped.astype(jnp.float32) * active,
                services.reshape(-1),
                (state.host_service_reliability * services / 100.0).reshape(-1),
                decoys.reshape(-1),
                (state.host_decoy_reliability * decoys / 100.0).reshape(-1),
                state.red_agent_active.astype(jnp.float32),
                state.blue_pending_ticks.astype(jnp.float32) / 5.0,
                state.red_pending_ticks.astype(jnp.float32) / 4.0,
                const.phase_rewards[state.mission_phase].astype(jnp.float32).reshape(-1) / 10.0,
            ]
        )
    world_state = jnp.concatenate(features)
    result = jnp.concatenate(
        [jnp.broadcast_to(world_state, (NUM_BLUE_AGENTS, world_state.size)), jnp.eye(NUM_BLUE_AGENTS)], axis=-1
    )
    if result.shape != (NUM_BLUE_AGENTS, expected_size):
        raise ValueError(f"centralized Blue observation has shape {result.shape}, expected width {expected_size}")
    return result
