"""Recurrent MAPPO: independent local actor and centralized critic memories.

Uses the existing masked recurrent trunk and MAPPO heads. Each role has its
own encoder, recurrent cell and parameters; centralized information never
enters the actor, including through its hidden state. Omitting critic_obs
runs only the actor and returns a zero placeholder value for evaluation.
"""

import flax.linen as nn
import jax.numpy as jnp

from jaxborg.critic_observations import critic_obs_size

from .base import BUFFER_LAYOUT_SEQUENCE, CentralizedCriticPolicy, RecurrentPolicy
from .recurrent_actor_critic import CELLS, ScannedRNN, _Trunk
from .separate_actor_critic import _ActorTrunk, _CriticTrunk


class _JaxRecurrentMAPPOActorCritic(RecurrentPolicy, CentralizedCriticPolicy):
    action_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 1
    activation: str = "tanh"
    cell: str = "lstm"
    critic_input: str = "global_state"
    team: str = "blue"
    cage4_enhanced_obs: bool = False

    @property
    def critic_obs_dim(self):
        return critic_obs_size(self.critic_input, team=self.team, cage4_enhanced_obs=self.cage4_enhanced_obs)

    @nn.nowrap
    def initialize_carry(self, batch_size: int):
        return {
            role: ScannedRNN.initialize_carry(batch_size, self.hidden_dim, self.cell) for role in ("actor", "critic")
        }

    def setup(self):
        args = (self.hidden_dim, self.hidden_layers, self.activation, self.cell)
        self.actor_trunk = _Trunk(*args)
        self.critic_trunk = _Trunk(*args)
        self.actor_head = _ActorTrunk(self.action_dim, self.hidden_dim, 1, self.activation)
        self.critic_head = _CriticTrunk(self.hidden_dim, 1, self.activation)

    def __call__(self, carry, obs, avail_actions=None, resets=None, *, critic_obs=None):
        if getattr(obs, "ndim", 0) != 3:
            raise ValueError("recurrent MAPPO requires time-major (T, B, obs_dim) input and a carry")
        if resets is None:
            resets = jnp.zeros(obs.shape[:2], dtype=jnp.bool_)
        actor_carry, actor_features = self.actor_trunk(carry["actor"], obs, resets)
        pi = self.actor_head(actor_features, avail_actions)
        if critic_obs is None:
            return {"actor": actor_carry, "critic": carry["critic"]}, pi, jnp.zeros(obs.shape[:2], dtype=jnp.float32)
        if critic_obs.shape != obs.shape[:-1] + (self.critic_obs_dim,):
            raise ValueError(f"critic_obs must match actor batch axes with width {self.critic_obs_dim}")
        critic_carry, critic_features = self.critic_trunk(carry["critic"], critic_obs, resets)
        return {"actor": actor_carry, "critic": critic_carry}, pi, self.critic_head(critic_features)


def jax_factory(
    action_dim,
    hidden_dim,
    hidden_layers,
    activation,
    *,
    cell="lstm",
    critic_input="global_state",
    team="blue",
    cage4_enhanced_obs=False,
):
    critic_obs_size(critic_input, team=team)
    if cell not in CELLS:
        raise ValueError(f"arch.cell must be one of {list(CELLS)}, got {cell!r}")
    if hidden_layers < 1:
        raise ValueError(f"recurrent MAPPO arch.hidden_layers must be >= 1, got {hidden_layers}")
    return _JaxRecurrentMAPPOActorCritic(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        cell=cell,
        critic_input=critic_input,
        team=team,
        cage4_enhanced_obs=cage4_enhanced_obs,
    )


def torch_factory(**_):
    raise NotImplementedError("recurrent MAPPO training is supported by the joint JAX trainer only")


JAX_FACTORY = jax_factory
TORCH_FACTORY = torch_factory
BUFFER_LAYOUT = BUFFER_LAYOUT_SEQUENCE
