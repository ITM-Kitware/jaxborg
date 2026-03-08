import chex
import jax.numpy as jnp
from flax import struct

from jaxborg.constants import MISSION_PHASES
from jaxborg.state import CC4Const, CC4State

LWF = 0
ASF = 1
RIA = 2


@struct.dataclass
class RewardBreakdown:
    total: chex.Array
    ria_reward: chex.Array
    lwf_reward: chex.Array
    asf_reward: chex.Array
    ria_count: chex.Array
    lwf_count: chex.Array
    asf_count: chex.Array


def compute_reward_breakdown(
    state: CC4State,
    const: CC4Const,
    impact_hosts: chex.Array,
    green_lwf_hosts: chex.Array,
    green_asf_hosts: chex.Array,
) -> RewardBreakdown:
    """Compute blue team shared reward for this step.

    Args:
        state: current CC4State (uses mission_phase, host_subnet)
        const: CC4Const (uses phase_rewards, host_active)
        impact_hosts: (GLOBAL_MAX_HOSTS,) bool - hosts where red Impact succeeded
        green_lwf_hosts: (GLOBAL_MAX_HOSTS,) bool - source hosts where GreenLocalWork failed
        green_asf_hosts: (GLOBAL_MAX_HOSTS,) bool - source hosts where GreenAccessService failed

    Returns:
        RewardBreakdown with total reward and per-term counts.
    """
    phase = state.mission_phase
    subnets = const.host_subnet

    ria_weights = const.phase_rewards[phase, subnets, RIA]
    lwf_weights = const.phase_rewards[phase, subnets, LWF]
    asf_weights = const.phase_rewards[phase, subnets, ASF]

    active = const.host_active.astype(jnp.float32)

    ria_reward = jnp.sum(impact_hosts.astype(jnp.float32) * ria_weights * active)
    lwf_reward = jnp.sum(green_lwf_hosts.astype(jnp.float32) * lwf_weights * active)
    asf_reward = jnp.sum(green_asf_hosts.astype(jnp.float32) * asf_weights * active)

    return RewardBreakdown(
        total=ria_reward + lwf_reward + asf_reward,
        ria_reward=ria_reward,
        lwf_reward=lwf_reward,
        asf_reward=asf_reward,
        ria_count=jnp.sum(impact_hosts.astype(jnp.float32) * active),
        lwf_count=jnp.sum(green_lwf_hosts.astype(jnp.float32) * active),
        asf_count=jnp.sum(green_asf_hosts.astype(jnp.float32) * active),
    )


def compute_rewards(
    state: CC4State,
    const: CC4Const,
    impact_hosts: chex.Array,
    green_lwf_hosts: chex.Array,
    green_asf_hosts: chex.Array,
) -> chex.Array:
    """Compute blue team shared reward for this step."""
    return compute_reward_breakdown(
        state,
        const,
        impact_hosts,
        green_lwf_hosts,
        green_asf_hosts,
    ).total


def advance_mission_phase(state: CC4State, const: CC4Const) -> CC4State:
    """Update mission_phase based on current time step."""
    new_phase = jnp.int32(0)
    for p in range(1, MISSION_PHASES):
        new_phase = jnp.where(state.time >= const.phase_boundaries[p], jnp.int32(p), new_phase)
    return state.replace(mission_phase=new_phase)
