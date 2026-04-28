"""Flax policy networks for JAXborg IPPO training and evaluation.

Contains ActorCritic (separate heads), SharedActorCritic (shared trunk),
and LegacyActor (actor-only, for loading old checkpoints).
"""

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from jaxborg.constants import MESSAGE_LENGTH


class ActorHead(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, avail_actions=None):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        actor_mean = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        if avail_actions is not None:
            unavail_actions = 1 - avail_actions
            action_logits = action_logits - (unavail_actions * 1e10)

        return distrax.Categorical(logits=action_logits)


class MessageHead(nn.Module):
    hidden_dim: int = 256
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        h = activation(h)
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
        h = activation(h)
        msg = nn.Dense(MESSAGE_LENGTH, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(h)
        return jnp.tanh(msg)


class CriticHead(nn.Module):
    hidden_dim: int = 256
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        critic = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    activation: str = "tanh"

    def setup(self):
        self.actor_head = ActorHead(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
        )
        self.critic_head = CriticHead(
            hidden_dim=self.hidden_dim,
            activation=self.activation,
        )
        self.message_head = MessageHead(
            hidden_dim=self.hidden_dim,
            activation=self.activation,
        )

    def actor(self, x, avail_actions=None):
        return self.actor_head(x, avail_actions)

    def critic(self, x):
        return self.critic_head(x)

    def message(self, x):
        return self.message_head(x)

    def __call__(self, x, avail_actions=None):
        pi = self.actor_head(x, avail_actions)
        value = self.critic_head(x)
        message = self.message_head(x)
        return pi, value, message


class SharedActorCritic(nn.Module):
    """Actor-critic with shared trunk, matching CybORG's PPOAgent architecture.

    Shared 2-layer [hidden_dim, hidden_dim] trunk, then single-layer actor and
    critic projections branching from it.
    """

    action_dim: int
    hidden_dim: int = 256
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, avail_actions=None):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        # Shared trunk (matches CybORG's self.features)
        trunk = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        trunk = activation(trunk)
        trunk = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(trunk)
        trunk = activation(trunk)

        # Actor projection (single layer)
        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(trunk)
        if avail_actions is not None:
            unavail_actions = 1 - avail_actions
            action_logits = action_logits - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        # Critic projection (single layer)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(trunk)
        value = jnp.squeeze(value, axis=-1)

        # Message projection (single layer, tanh-bounded)
        message = nn.Dense(MESSAGE_LENGTH, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(trunk)
        message = jnp.tanh(message)

        return pi, value, message

    def actor(self, x, avail_actions=None):
        pi, _, _ = self(x, avail_actions)
        return pi

    def critic(self, x):
        _, value, _ = self(x)
        return value

    def message(self, x):
        _, _, msg = self(x)
        return msg


class LegacyActor(nn.Module):
    """Actor-only network for loading old checkpoints (pre-ActorCritic)."""

    action_dim: int
    hidden_dim: int = 256
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, avail_actions=None):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        actor_mean = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        if avail_actions is not None:
            action_logits = action_logits - ((1 - avail_actions) * 1e10)

        return distrax.Categorical(logits=action_logits)
