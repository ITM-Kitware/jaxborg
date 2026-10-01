"""Scientific controls for loss attribution and paired trajectory forks."""

import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from jaxborg.blue_learning_probe import fork_coordinates, make_fork
from jaxborg.policies.categorical import Categorical
from scripts.experiments.blue_learning_mechanism import advance_rollout_rng, restore_tree, save_tree
from scripts.train.algorithms import ippo_jax_joint as trainer


def test_capture_roundtrips_numeric_optimizer_state_without_unpicklable_functions(tmp_path):
    params = {"weights": jnp.array([0.1, 0.2])}
    state = TrainState.create(apply_fn=lambda: None, params=params, tx=optax.adam(0.0003))
    state = state.apply_gradients(grads={"weights": jnp.array([0.3, -0.4])})
    save_tree(tmp_path / "optimizer", state)
    arrays = np.load(tmp_path / "optimizer.npz")
    structure = pickle.loads((tmp_path / "optimizer.tree.pkl").read_bytes())
    restored = jax.tree.unflatten(structure, [arrays[f"leaf{i}"] for i in range(len(arrays))])
    expected = {"params": state.params, "opt_state": state.opt_state, "step": state.step}
    assert jax.tree.structure(restored) == jax.tree.structure(expected)
    for actual, value in zip(jax.tree.leaves(restored), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(actual, value)
    for actual, value in zip(jax.tree.leaves(restore_tree(tmp_path / "optimizer")), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(actual, value)


def test_rng_recovery_matches_canonical_blue_only_rollout_split_schedule():
    original = jax.random.PRNGKey(6100001)
    expected = original
    for _ in range(4):
        for _ in range(7):
            expected, _, _, _ = jax.random.split(expected, 4)
        expected, _ = jax.random.split(expected)
    np.testing.assert_array_equal(advance_rollout_rng(original, 4, 7), expected)


def test_fork_selection_accepts_read_only_arrays_and_preserves_phase_strata():
    group = np.ones((8, 2, 3), dtype=bool)
    group.flags.writeable = False
    phases = np.repeat(np.array([0, 0, 1, 1, 1, 2, 2, 2])[:, None], 2, axis=1)
    selected = fork_coordinates(group, phases, 6)
    assert len(selected) == 6
    assert all(phases[t, e] == 2 for t, e, _ in selected)
    assert group.all()


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
