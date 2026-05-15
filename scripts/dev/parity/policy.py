"""Policy loading and batched inference helpers."""

# ruff: noqa: E402,I001

from __future__ import annotations

from scripts.dev.parity.bootstrap import configure_runtime

configure_runtime()

import jax
import jax.numpy as jnp

from jaxborg.policies.categorical import Categorical as JaxCategorical


def load_checkpoint(path):
    """Load a recipe-driven JAX checkpoint via jax_runner (sidecar required)."""
    from jaxborg.evaluation.jax_runner import load_jax_checkpoint

    policy, params, _recipe = load_jax_checkpoint(path)
    return policy, params


def policy_dist(policy, params, obs_jax, mask):
    pi, _ = policy.apply(params, obs_jax, mask)
    return pi


def make_batched_inference_fn(policy, params, deterministic):
    """Build a JIT-compiled function that runs policy inference for all agents at once.

    Returns batched_step(obs_stack, mask_stack, keys) -> (actions, logits)
    where obs_stack/mask_stack are (NUM_BLUE_AGENTS, ...) and keys is (NUM_BLUE_AGENTS, 2).
    """

    def _fwd(o, m):
        pi, _ = policy.apply(params, o, m)
        return pi.logits

    if deterministic:

        @jax.jit
        def batched_step(obs_stack, mask_stack, _keys):
            logits = jax.vmap(_fwd)(obs_stack, mask_stack)
            return jnp.argmax(logits, axis=-1), logits
    else:

        @jax.jit
        def batched_step(obs_stack, mask_stack, keys):
            logits = jax.vmap(_fwd)(obs_stack, mask_stack)
            actions = jax.vmap(lambda lg, k: JaxCategorical(logits=lg).sample(seed=k))(logits, keys)
            return actions, logits

    return batched_step
