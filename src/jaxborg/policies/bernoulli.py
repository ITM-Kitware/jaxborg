"""Independent binary message bits, reduced over the message axis."""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Bernoulli:
    logits: jax.Array

    def sample(self, seed):
        return jax.random.bernoulli(seed, jax.nn.sigmoid(self.logits)).astype(jnp.float32)

    def mode(self):
        return (self.logits > 0).astype(jnp.float32)

    def log_prob(self, bits):
        return jnp.sum(jnp.where(bits > 0, jax.nn.log_sigmoid(self.logits), jax.nn.log_sigmoid(-self.logits)), axis=-1)

    def entropy(self):
        p = jax.nn.sigmoid(self.logits)
        return -jnp.sum(p * jax.nn.log_sigmoid(self.logits) + (1 - p) * jax.nn.log_sigmoid(-self.logits), axis=-1)
