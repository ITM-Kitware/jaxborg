"""Topology-bank sampling rules for parallel JAX training environments."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def validate_training_topology_coverage(bank_size: int, num_envs: int) -> None:
    """Require enough diversified snapshots for one unique draw per environment.

    A singleton bank is the fixed-topology control and is intentionally shared
    by every parallel environment. Any bank with more than one entry represents
    a diversity treatment and must cover the complete parallel batch.
    """
    if bank_size < 1:
        raise ValueError("training topology bank must contain at least one snapshot")
    if num_envs < 1:
        raise ValueError("NUM_ENVS must be positive")
    if 1 < bank_size < num_envs:
        raise ValueError(
            f"training topology bank has {bank_size} entries, but NUM_ENVS={num_envs}; "
            "without-replacement sampling requires bank_size >= NUM_ENVS "
            "(a singleton bank remains the fixed-topology control)"
        )


def sample_training_topology_indices(key, bank_size: int, num_envs: int) -> jax.Array:
    """Draw one topology index per environment without replacement.

    Sampling is without replacement within one parallel reset batch. A fresh
    permutation is drawn at the next episode reset, so sampling across batches
    remains independent. The singleton control necessarily returns all zeros.
    """
    validate_training_topology_coverage(bank_size, num_envs)
    if bank_size == 1:
        return jnp.zeros((num_envs,), dtype=jnp.int32)
    return jax.random.permutation(key, bank_size)[:num_envs].astype(jnp.int32)


__all__ = ["sample_training_topology_indices", "validate_training_topology_coverage"]
