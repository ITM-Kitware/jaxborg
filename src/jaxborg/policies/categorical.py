"""Minimal categorical distribution for JAX policies.

Drop-in replacement for the subset of ``distrax.Categorical`` that jaxborg
uses (``.logits``, ``.sample(seed=)``, ``.log_prob(a)``, ``.entropy()``).
Implemented directly on top of ``jax.random.categorical`` and
``jax.nn.log_softmax`` so we don't drag in ``distrax`` (and through it
``tensorflow-probability``, which pins us below JAX 0.7 because of a
removed-symbol import — see pyproject.toml jax section).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Categorical:
    """Categorical distribution over the last axis of ``logits``.

    Pytree-compatible (flax struct) so it can be returned from ``nn.Module``
    forward passes and threaded through ``jit`` / ``vmap`` / ``scan``.
    """

    logits: jax.Array

    def sample(self, seed: jax.Array) -> jax.Array:
        return jax.random.categorical(seed, self.logits, axis=-1)

    def log_prob(self, action: jax.Array) -> jax.Array:
        log_p = jax.nn.log_softmax(self.logits, axis=-1)
        return jnp.take_along_axis(log_p, action[..., None], axis=-1).squeeze(-1)

    def entropy(self) -> jax.Array:
        log_p = jax.nn.log_softmax(self.logits, axis=-1)
        return -jnp.sum(jnp.exp(log_p) * log_p, axis=-1)
