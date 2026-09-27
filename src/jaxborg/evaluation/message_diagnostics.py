"""Summaries of learned Blue messages using only policy-visible evidence."""

import jax
import jax.numpy as jnp
import numpy as np


def message_statistics(pi, message, obs, const):
    """Entropy sum, per-bit joint counts with observed evidence, sample count."""
    if pi.messages is None:
        return jnp.zeros(34, dtype=jnp.float32)
    return statistics_from_observations(pi.messages.entropy(), message, obs, const)


def statistics_from_observations(entropy, message, obs, const):
    if obs.shape[-1] >= 450:
        evidence = jnp.any(obs[:, 402:450] > 0, axis=-1)
    elif obs.shape[-1] >= 402:
        evidence = jnp.any(obs[:, 258:402] > 0, axis=-1)
    else:
        # Stock observations: process/network alerts within owned subnet blocks.
        blocks = obs[:, 1:178].reshape(-1, 3, 59)
        valid = const.blue_obs_subnets >= 0
        evidence = jnp.any((blocks[:, :, 27:] > 0) & valid[:, :, None], axis=(1, 2))
    bit_values = jax.nn.one_hot(message.astype(jnp.int32), 2)
    labels = jax.nn.one_hot(evidence.astype(jnp.int32), 2)
    counts = jnp.einsum("abi,aj->bij", bit_values, labels)
    return jnp.concatenate([entropy.sum()[None], counts.reshape(-1), jnp.array([obs.shape[0]])])


def summarize_message_statistics(stats):
    stats = np.asarray(stats, dtype=np.float64)
    counts = stats[1:33].reshape(8, 2, 2)
    count = float(stats[-1])
    if count == 0:
        return dict(mean_entropy=0.0, bit_mean=[0.0] * 8, evidence_mutual_information_bits=[0.0] * 8, samples=0)
    probabilities = counts / count
    independent = probabilities.sum(axis=2, keepdims=True) * probabilities.sum(axis=1, keepdims=True)
    ratio = np.divide(probabilities, independent, out=np.ones_like(probabilities), where=probabilities > 0)
    mutual_information = (probabilities * np.log2(ratio)).sum(axis=(1, 2))
    return dict(
        mean_entropy=float(stats[0] / count),
        bit_mean=probabilities[:, 1].sum(-1).tolist(),
        evidence_mutual_information_bits=mutual_information.tolist(),
        samples=int(count),
    )
