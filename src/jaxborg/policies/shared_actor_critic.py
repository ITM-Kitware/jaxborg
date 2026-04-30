"""Shared-trunk actor-critic. arch.name = "shared".

Single 2-layer trunk feeding two single-layer heads (one for action logits,
one for state value). This is the historical CC4 default and the closest
match to CybORG's PPOAgent before the Singh recipe.
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


class _JaxSharedActorCritic(nn.Module):
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
        pi = distrax.Categorical(logits=logits)

        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h)
        value = jnp.squeeze(value, axis=-1)
        return pi, value


def jax_factory(action_dim: int, hidden_dim: int, hidden_layers: int, activation: str):
    return _JaxSharedActorCritic(
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


class _TorchSharedActorCritic(tnn.Module):
    """Shared-trunk PyTorch actor-critic, action-masked."""

    arch_name = "shared"

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, hidden_layers: int = 2):
        super().__init__()
        hidden_dims = tuple([hidden_dim] * hidden_layers)
        self.features = _build_torch_mlp(obs_dim, hidden_dims)
        head_in = hidden_dims[-1]
        self.actor = tnn.Linear(head_in, action_dim)
        self.critic = tnn.Linear(head_in, 1)
        tnn.init.orthogonal_(self.actor.weight, gain=0.01)
        tnn.init.constant_(self.actor.bias, 0.0)
        tnn.init.orthogonal_(self.critic.weight, gain=1.0)
        tnn.init.constant_(self.critic.bias, 0.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(obs)).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        feat = self.features(obs)
        logits = self.actor(feat) + (action_mask.float() - 1.0) * 1e8
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(feat).squeeze(-1)
        return action, log_prob, entropy, value

    def deterministic_action(self, obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        logits = self.actor(self.features(obs)) + (action_mask.float() - 1.0) * 1e8
        return logits.argmax(dim=-1)


def torch_factory(obs_dim: int, action_dim: int, hidden_dim: int, hidden_layers: int, **_) -> tnn.Module:
    return _TorchSharedActorCritic(obs_dim, action_dim, hidden_dim, hidden_layers)


JAX_FACTORY = jax_factory
TORCH_FACTORY = torch_factory
BUFFER_LAYOUT = BUFFER_LAYOUT_FLAT
