"""Training-only CC4 world-state inputs for each team's centralized critic.

Like JaxMARL's MAPPO world-state wrapper, each agent gets the same centralized
features plus its identity. These arrays never enter the actor observation dict.
"""

import jax.numpy as jnp

from jaxborg.blue_observation_contract import blue_obs_size
from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_DECOY_TYPES,
    NUM_RED_AGENTS,
    NUM_SERVICES,
    NUM_SUBNETS,
    RED_OBS_SIZE,
)
from jaxborg.learned_red import get_red_policy_obs
from jaxborg.observations import get_blue_obs
from jaxborg.state import SimulatorConst, SimulatorState

CRITIC_INPUTS = ("joint_observations", "global_state")


def _team_observation_spec(team: str, cage4_enhanced_obs: bool = False):
    if team == "blue":
        return NUM_BLUE_AGENTS, blue_obs_size(cage4_enhanced_obs), get_blue_obs
    if team == "red":
        return NUM_RED_AGENTS, RED_OBS_SIZE, get_red_policy_obs
    raise ValueError(f"critic team must be 'blue' or 'red', got {team!r}")


def critic_obs_size(critic_input: str, *, team: str = "blue", cage4_enhanced_obs: bool = False) -> int:
    if critic_input not in CRITIC_INPUTS:
        raise ValueError(f"critic_input must be one of {CRITIC_INPUTS}, got {critic_input!r}")
    num_agents, obs_size, _ = _team_observation_spec(team, cage4_enhanced_obs)
    size = num_agents * obs_size + num_agents
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


def get_critic_obs(
    state: SimulatorState, const: SimulatorConst, critic_input: str = "global_state", *, team: str = "blue"
):
    """Return (team_agents, critic_obs_dim) features from the pre-action state.

    global_state is a compact representation, not the complete simulator state:
    it adds current compromise, service/decoy health, timing and reward weights
    to that team's pooled policy observations. It excludes RNG keys and future information.
    Inactive hosts and reliability of absent services/decoys are zeroed.
    """
    expected_size = critic_obs_size(critic_input, team=team, cage4_enhanced_obs=const.cage4_enhanced_obs)
    num_agents, _, observe = _team_observation_spec(team)
    observations = jnp.stack([observe(state, const, i) for i in range(num_agents)])
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
        [jnp.broadcast_to(world_state, (num_agents, world_state.size)), jnp.eye(num_agents)], axis=-1
    )
    if result.shape != (num_agents, expected_size):
        raise ValueError(f"centralized {team} observation has shape {result.shape}, expected width {expected_size}")
    return result


def blue_critic_obs_size(critic_input: str) -> int:
    """The original Blue layout, retained for existing callers and checkpoints."""
    return critic_obs_size(critic_input, team="blue")


def get_blue_critic_obs(state: SimulatorState, const: SimulatorConst, critic_input: str = "global_state"):
    return get_critic_obs(state, const, critic_input, team="blue")
