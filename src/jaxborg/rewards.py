import chex
import jax.numpy as jnp
from flax import struct

from jaxborg.actions.encoding import BLUE_RESTORE_END, BLUE_RESTORE_START
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
    action_cost: chex.Array
    ria_count: chex.Array
    lwf_count: chex.Array
    asf_count: chex.Array


def compute_reward_breakdown(
    state: CC4State,
    const: CC4Const,
    impact_hosts: chex.Array,
    green_lwf_hosts: chex.Array,
    green_asf_hosts: chex.Array,
    blue_actions: chex.Array = None,
) -> RewardBreakdown:
    """Compute blue team shared reward for this step.

    Args:
        state: current CC4State (uses mission_phase, host_subnet)
        const: CC4Const (uses phase_rewards, host_active)
        impact_hosts: (GLOBAL_MAX_HOSTS,) bool - hosts where red Impact succeeded
        green_lwf_hosts: (GLOBAL_MAX_HOSTS,) bool - source hosts where GreenLocalWork failed
        green_asf_hosts: (GLOBAL_MAX_HOSTS,) bool - source hosts where GreenAccessService failed
        blue_actions: (NUM_BLUE_AGENTS,) int32 - caller-submitted blue actions this step

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

    # Action cost mirrors CybORG's CC4 contract: -1 per caller-submitted
    # Restore each step (SimulationController._step:310 sums
    # `actions.get(agent, Action()).cost`), regardless of whether the agent
    # is already busy executing a prior Restore. CC4's headline scorer
    # inherits this via `BlueFixedActionWrapper.step`'s `sum(reward.values())`.
    if blue_actions is not None:
        is_restore = (blue_actions >= BLUE_RESTORE_START) & (blue_actions < BLUE_RESTORE_END)
        action_cost = -jnp.sum(is_restore.astype(jnp.float32))
    else:
        action_cost = jnp.float32(0.0)

    return RewardBreakdown(
        total=ria_reward + lwf_reward + asf_reward + action_cost,
        ria_reward=ria_reward,
        lwf_reward=lwf_reward,
        asf_reward=asf_reward,
        action_cost=action_cost,
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
    blue_actions: chex.Array = None,
) -> chex.Array:
    """Compute blue team shared reward for this step."""
    return compute_reward_breakdown(
        state,
        const,
        impact_hosts,
        green_lwf_hosts,
        green_asf_hosts,
        blue_actions,
    ).total


def advance_mission_phase(state: CC4State, const: CC4Const) -> CC4State:
    """Update mission_phase based on current time step."""
    new_phase = jnp.int32(0)
    for p in range(1, const.phase_boundaries.shape[0]):
        new_phase = jnp.where(state.time >= const.phase_boundaries[p], jnp.int32(p), new_phase)
    return state.replace(mission_phase=new_phase)
