"""Check the production batch reset against its original scalar step API."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from jaxborg.joint_env import JointPolicyCC4Env


@struct.dataclass
class _State:
    time: jax.Array
    topology: jax.Array
    value: jax.Array


class _CheapJointEnv(JointPolicyCC4Env):
    def __init__(self):
        # Exercise the production step and step_batch with cheap transitions;
        # random reset topology and multidimensional leaves catch key/mask bugs.
        self.resets = []
        self._env = SimpleNamespace(_reset_state=self._reset_state)

    def _reset_state(self, state, key):
        jax.debug.callback(lambda: self.resets.append(True))
        return state.replace(
            time=jnp.int32(0), topology=jax.random.randint(key, (), 0, 100), value=jax.random.uniform(key, (2, 3))
        )

    def get_obs(self, state):
        return {"blue_0": state.value, "red_0": state.value + state.topology}

    def step_env(self, key, state, actions):
        state = state.replace(
            time=state.time + 1, value=state.value + jax.random.uniform(key, (2, 3)) + actions["blue_0"]
        )
        done = state.time >= 3
        reward = state.value.sum()
        return self.get_obs(state), state, {"blue_0": reward}, {"__all__": done}, {"time": state.time}


@pytest.mark.parametrize("times", [(0, 0, 0), (2, 2, 2), (0, 2, 1)])
@pytest.mark.parametrize("typed_keys", [False, True])
def test_batch_step_matches_scalar_step_including_rng_and_mixed_termination(times, typed_keys):
    env = _CheapJointEnv()
    key = jax.random.key(17) if typed_keys else jax.random.PRNGKey(17)
    keys = jax.random.split(key, 3)
    states = _State(jnp.asarray(times), jnp.asarray([5, 7, 11]), jnp.zeros((3, 2, 3)))
    actions = {"blue_0": jnp.asarray([0, 1, 2])}
    expected = jax.vmap(env.step)(keys, states, actions)
    jax.block_until_ready(expected)
    jax.effects_barrier()
    env.resets.clear()
    actual = env.step_batch(keys, states, actions)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
    assert bool(env.resets) == any(t == 2 for t in times)


def test_batch_reset_stays_conditional_inside_training_scan():
    env = _CheapJointEnv()
    states = _State(jnp.zeros(3, dtype=jnp.int32), jnp.arange(3), jnp.zeros((3, 2, 3)))
    keys = jax.random.split(jax.random.PRNGKey(9), (6, 3))

    @jax.jit
    def rollout(states):
        def tick(states, step_keys):
            _, states, rewards, dones, _ = env.step_batch(step_keys, states, {"blue_0": jnp.zeros(3)})
            return states, (rewards, dones)

        return jax.lax.scan(tick, states, keys)

    final, (_, dones) = jax.block_until_ready(rollout(states))
    jax.effects_barrier()
    np.testing.assert_array_equal(dones["__all__"][:, 0], [False, False, True, False, False, True])
    np.testing.assert_array_equal(final.time, [0, 0, 0])
    # The callback has no batched inputs, so it executes once per reset batch.
    assert len(env.resets) == 2
