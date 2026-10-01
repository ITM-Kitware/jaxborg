"""Scientific controls for loss attribution and paired trajectory forks."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from jaxborg.blue_learning_probe import make_fork
from jaxborg.policies.categorical import Categorical
from scripts.train.algorithms import ippo_jax_joint as trainer


def test_full_ppo_gradient_equals_actor_value_entropy_sum_with_masked_rows():
    config = {"CLIP_EPS": 0.2, "VF_COEF": 0.5, "ENT_COEF": 0.01}
    transitions = SimpleNamespace(
        action=jnp.array([0, 1, 0]),
        log_prob=jnp.log(jnp.array([0.5, 0.5, 0.5])),
        actor_mask=jnp.array([1.0, 1.0, 0.0]),
        critic_mask=jnp.ones(3),
        value=jnp.zeros(3),
    )
    params = {"logits": jnp.array([[0.2, -0.1], [1.0, -1.0], [0.7, 0.3]]), "value": jnp.array([0.3, 0.8, -0.4])}
    advantages, targets = jnp.array([0.4, -0.8, 100.0]), jnp.array([-0.2, 0.3, -0.5])

    def loss(p, component):
        return trainer.ppo_objective(
            Categorical(p["logits"]), p["value"], transitions, advantages, targets, config, loss_component=component
        )[0]

    grads = {c: jax.grad(loss)(params, c) for c in ("full", "actor", "critic", "entropy", "zero")}
    for name in params:
        np.testing.assert_allclose(
            grads["full"][name], sum(grads[c][name] for c in ("actor", "critic", "entropy")), atol=1e-7
        )
        np.testing.assert_array_equal(grads["zero"][name], np.zeros_like(params[name]))
    # A busy row contributes critic loss but no actor/entropy learning.
    np.testing.assert_array_equal(grads["actor"]["logits"][2], [0.0, 0.0])
    assert float(grads["critic"]["value"][2]) != 0


@struct.dataclass
class ToyState:
    time: jax.Array
    total: jax.Array


class ToyPolicy:
    def apply(self, params, obs, masks=None):
        del params, masks
        return Categorical(jnp.zeros((*obs.shape[:-1], 2))), jnp.zeros(obs.shape[:-1])


class ToyEnv:
    blue_agents = ("blue_0", "blue_1")
    red_agents = ("red_0",)

    def get_avail_actions(self, state):
        del state
        return {n: jnp.ones(2, dtype=bool) for n in (*self.blue_agents, *self.red_agents)}

    def step_env(self, key, state, actions):
        # Other intervention actions affect the result, catching accidental
        # whole-team replacement and omitted first-step reward.
        reward = actions["blue_0"] + 10 * actions["blue_1"] + 100 * actions["red_0"]
        state = state.replace(time=state.time + 1, total=state.total + reward)
        obs = {n: jnp.array([state.time], dtype=float) for n in (*self.blue_agents, *self.red_agents)}
        return (
            obs,
            state,
            {},
            {"__all__": state.time >= 3},
            {
                "reward_ria": reward.astype(float),
                "reward_asf": jnp.float32(0),
                "reward_lwf": jnp.float32(0),
                "action_cost": jnp.float32(0),
            },
        )


def test_fork_preserves_other_actions_and_includes_intervention_reward_and_terminal():
    env = ToyEnv()
    fork = make_fork(env, {"blue": ToyPolicy(), "red": ToyPolicy()})
    actions = {"blue_0": jnp.int32(0), "blue_1": jnp.int32(1), "red_0": jnp.int32(1)}
    kwargs = dict(agent=0, discount=0.99, horizon=20)
    inputs = (ToyState(jnp.int32(2), jnp.float32(0)), actions, jax.random.PRNGKey(1))
    natural = fork(*inputs, jnp.int32(0), jax.random.PRNGKey(2), {"blue": None, "red": None}, **kwargs)
    changed = fork(*inputs, jnp.int32(1), jax.random.PRNGKey(2), {"blue": None, "red": None}, **kwargs)
    np.testing.assert_array_equal(natural["undiscounted"], [110.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(changed["undiscounted"], [111.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(changed["discounted"], changed["undiscounted"])
