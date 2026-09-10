"""Local actor and centralized critic, following JaxMARL's MAPPO baseline.

Reference: JaxMARL 545d8f9708831a8022bd0b5a16618679173cece5,
baselines/MAPPO/mappo_ff_hanabi.py (ActorFF/CriticFF) and
mappo_rnn_smax.py (world state + agent identity and separate gradient clipping).
The installed JaxMARL package does not ship its baselines, so we adapt that
design using this repo's existing Flax trunks and categorical distribution.

Differences: configurable depth, repo value-head initialization, CC4 busy masks,
and a combined checkpoint/Adam tree with independently clipped actor/critic
gradients. No Hydra, WandB, or distrax runtime dependency is introduced.

Omitting critic_obs explicitly selects actor-only inference: it does not feed
dummy global information through the trained critic. The returned zero value
is only a placeholder for evaluation callers that discard values.
"""

import jax.numpy as jnp

from jaxborg.critic_observations import blue_critic_obs_size

from .base import BUFFER_LAYOUT_FLAT, CentralizedCriticPolicy
from .separate_actor_critic import _ActorTrunk, _CriticTrunk


class _JaxMAPPOActorCritic(CentralizedCriticPolicy):
    action_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 2
    activation: str = "tanh"
    critic_input: str = "global_state"

    @property
    def critic_obs_dim(self):
        return blue_critic_obs_size(self.critic_input)

    def setup(self):
        self.actor_head = _ActorTrunk(self.action_dim, self.hidden_dim, self.hidden_layers, self.activation)
        self.critic_head = _CriticTrunk(self.hidden_dim, self.hidden_layers, self.activation)

    def __call__(self, obs, avail_actions=None, *, critic_obs=None):
        pi = self.actor_head(obs, avail_actions)
        if critic_obs is None:
            return pi, jnp.zeros(obs.shape[:-1], dtype=jnp.float32)
        if critic_obs.shape != obs.shape[:-1] + (self.critic_obs_dim,):
            raise ValueError(f"critic_obs must match actor batch axes with width {self.critic_obs_dim}")
        return pi, self.critic_head(critic_obs)


def jax_factory(action_dim, hidden_dim, hidden_layers, activation, *, critic_input="global_state"):
    blue_critic_obs_size(critic_input)  # Fail before compiling for an invalid input mode.
    return _JaxMAPPOActorCritic(action_dim, hidden_dim, hidden_layers, activation, critic_input)


def torch_factory(**_):
    raise NotImplementedError("MAPPO training is currently supported by the joint JAX trainer only")


JAX_FACTORY = jax_factory
TORCH_FACTORY = torch_factory
BUFFER_LAYOUT = BUFFER_LAYOUT_FLAT
