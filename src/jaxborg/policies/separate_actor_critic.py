"""Separate actor / critic networks. arch.name = "separate".

Two independent trunks of (hidden_layers × hidden_dim), one feeding the
action head and one feeding the value head. Matches the architecture in
Singh et al. (arXiv:2410.17351): "actor and critic are represented by two
feedforward neural networks with two hidden layers and 256 neurons per
layer".
"""

from __future__ import annotations

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as tnn
from flax.linen.initializers import constant, orthogonal
from torch.distributions import Categorical

from .base import BUFFER_LAYOUT_FLAT


class _ActorTrunk(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 2
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, avail_actions=None):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        h = x
        for _ in range(self.hidden_layers):
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(h)
            h = act_fn(h)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(h)
        if avail_actions is not None:
            logits = logits - (1 - avail_actions) * 1e10
        return distrax.Categorical(logits=logits)


class _CriticTrunk(nn.Module):
    hidden_dim: int = 256
    hidden_layers: int = 2
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        h = x
        for _ in range(self.hidden_layers):
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(h)
            h = act_fn(h)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h)
        return jnp.squeeze(v, axis=-1)


class _JaxSeparateActorCritic(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 2
    activation: str = "tanh"

    def setup(self):
        self.actor_head = _ActorTrunk(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
        )
        self.critic_head = _CriticTrunk(
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
        )

    def __call__(self, x, avail_actions=None):
        return self.actor_head(x, avail_actions), self.critic_head(x)


def jax_factory(action_dim: int, hidden_dim: int, hidden_layers: int, activation: str):
    return _JaxSeparateActorCritic(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation=activation,
    )


def _build_torch_mlp(in_dim: int, hidden_dims: tuple[int, ...]) -> tnn.Sequential:
    layers: list[tnn.Module] = []
    d = in_dim
    for h in hidden_dims:
        lin = tnn.Linear(d, h)
        tnn.init.orthogonal_(lin.weight, gain=float(np.sqrt(2)))
        tnn.init.constant_(lin.bias, 0.0)
        layers.append(lin)
        layers.append(tnn.Tanh())
        d = h
    return tnn.Sequential(*layers)


class _TorchSeparateActorCritic(tnn.Module):
    arch_name = "separate"

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, hidden_layers: int = 2):
        super().__init__()
        hidden_dims = tuple([hidden_dim] * hidden_layers)
        self.actor_features = _build_torch_mlp(obs_dim, hidden_dims)
        self.critic_features = _build_torch_mlp(obs_dim, hidden_dims)
        head_in = hidden_dims[-1]
        self.actor = tnn.Linear(head_in, action_dim)
        self.critic = tnn.Linear(head_in, 1)
        tnn.init.orthogonal_(self.actor.weight, gain=0.01)
        tnn.init.constant_(self.actor.bias, 0.0)
        tnn.init.orthogonal_(self.critic.weight, gain=1.0)
        tnn.init.constant_(self.critic.bias, 0.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_features(obs)).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        logits = self.actor(self.actor_features(obs)) + (action_mask.float() - 1.0) * 1e10
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(self.critic_features(obs)).squeeze(-1)
        return action, log_prob, entropy, value

    def deterministic_action(self, obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        logits = self.actor(self.actor_features(obs)) + (action_mask.float() - 1.0) * 1e10
        return logits.argmax(dim=-1)


def torch_factory(obs_dim: int, action_dim: int, hidden_dim: int, hidden_layers: int, **_) -> tnn.Module:
    return _TorchSeparateActorCritic(obs_dim, action_dim, hidden_dim, hidden_layers)


JAX_FACTORY = jax_factory
TORCH_FACTORY = torch_factory
BUFFER_LAYOUT = BUFFER_LAYOUT_FLAT
