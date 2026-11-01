"""Example payoff rule for the existing Blue/Red policy environment."""

import jax.numpy as jnp

from jaxborg.evaluation.jax_env_factory import variant_env_settings
from jaxborg.joint_env import JointPolicyCC4Env


def offset_payoff(rewards, info, blue_agents, offset):
    """Preserve shared zero-sum team payoffs and raw return accounting."""
    offset = jnp.float32(offset)
    rewards = {name: reward + (offset if name in blue_agents else -offset) for name, reward in rewards.items()}
    return rewards, {**info, "reward_ria": info["reward_ria"] + offset}


class RewardOffsetEnv(JointPolicyCC4Env):
    def __init__(self, *, offset, **settings):
        super().__init__(**settings)
        self.offset = float(offset)

    def step_env(self, key, state, actions):
        obs, state, rewards, dones, info = super().step_env(key, state, actions)
        rewards, info = offset_payoff(rewards, info, self.blue_agents, self.offset)
        return obs, state, rewards, dones, info


def build_env(variant, *, topology_mode="generative", training_mode=False, topology_path=None, offset=1.0):
    """Use the same partially bound builder for training and evaluation."""
    if variant.resilience_roles:
        raise ValueError("reward-offset example does not implement resilience roles")
    return RewardOffsetEnv(
        offset=offset,
        **variant_env_settings(variant),
        topology_mode=topology_mode,
        training_mode=training_mode,
        topology_path=topology_path,
    )
