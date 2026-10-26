"""Construction boundary for experiment-local joint CC4 environments."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

JointEnvFactory = Callable[..., Any]


def _signature(value):
    leaves, tree = jax.tree_util.tree_flatten(value)
    return tree, [(leaf.shape, leaf.dtype) for leaf in leaves]


def construct_joint_env(
    variant, *, env_factory: JointEnvFactory | None = None, default_factory: JointEnvFactory, **settings
):
    """Construct outside rollout and check the canonical joint CC4 ABI.

    Builders receive the variant and explicit construction settings. Reset and
    step must be pure JAX functions with explicit keys and state. This validates
    structure, shapes and dtypes, not scientific correctness or absence of hidden
    Python side effects. Custom rules must preserve canonical action meanings,
    observations, state, masks and the per-team shared reward convention.
    """
    if env_factory is None:
        return default_factory(variant, **settings)
    if variant.red_reward != "zero_sum":
        raise ValueError("custom joint environment evaluation requires red_reward=zero_sum")
    env = env_factory(variant, **settings)
    reference = default_factory(variant, **settings)
    try:
        for team in ("blue", "red"):
            names = tuple(getattr(reference, f"{team}_agents"))
            if tuple(getattr(env, f"{team}_agents")) != names:
                raise ValueError(f"{team} agents must preserve canonical order")
            for name in names:
                if env.action_space(name).n != reference.action_space(name).n:
                    raise ValueError(f"{name} action space must preserve canonical dimensions")
                if env.observation_space(name).shape != reference.observation_space(name).shape:
                    raise ValueError(f"{name} observation space must preserve canonical dimensions")
        key = jax.random.PRNGKey(0)
        actions = {name: jnp.int32(0) for name in (*env.blue_agents, *env.red_agents)}
        expected_obs, expected_state = jax.eval_shape(reference.reset, key)
        obs, state = jax.eval_shape(env.reset, key)

        def same(actual, expected, label):
            if _signature(actual) != _signature(expected):
                raise ValueError(f"{label} must preserve canonical pytree, shapes and dtypes")

        same((obs, state), (expected_obs, expected_state), "reset output")
        expected = jax.eval_shape(reference.step, key, expected_state, actions)
        actual = jax.eval_shape(env.step, key, state, actions)
        same(actual, expected, "step output")
        same(
            jax.eval_shape(env.step_env, key, state, actions),
            jax.eval_shape(reference.step_env, key, state, actions),
            "step_env output",
        )
        same(actual[:2], (obs, state), "step/reset state and observations")
        same(
            jax.eval_shape(env.get_avail_actions, state),
            jax.eval_shape(reference.get_avail_actions, state),
            "action masks",
        )
        for method in ("reset_batch", "step_batch", "reset_at_topology"):
            if hasattr(env, method):
                # Alternate reset/step paths must preserve the same scalar ABI.
                keys = jax.random.split(key, 2)
                batch_state = jax.tree.map(lambda x: jax.ShapeDtypeStruct((2, *x.shape), x.dtype), state)
                batch_actions = jax.tree.map(lambda x: jnp.repeat(x[None], 2, axis=0), actions)
                if method == "reset_batch":
                    output = jax.eval_shape(env.reset_batch, keys, key)
                    target = jax.tree.map(lambda x: jax.ShapeDtypeStruct((2, *x.shape), x.dtype), (obs, state))
                elif method == "step_batch":
                    output = jax.eval_shape(env.step_batch, keys, batch_state, batch_actions, key)
                    target = jax.tree.map(lambda x: jax.ShapeDtypeStruct((2, *x.shape), x.dtype), actual)
                else:
                    if settings.get("topology_path") is None:
                        continue
                    output = jax.eval_shape(env.reset_at_topology, key, jnp.int32(0))
                    target = (obs, state)
                same(output, target, method)
    except Exception as exc:
        raise ValueError(f"unsupported custom joint JAX environment contract: {exc}") from exc
    return env
