from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from flax.training.train_state import TrainState

from jaxborg.policies import make_jax_policy
from scripts.train.algorithms import ippo_jax_joint as joint


@struct.dataclass
class _FakeSimState:
    time: jax.Array
    blue_pending_ticks: jax.Array
    red_pending_ticks: jax.Array
    red_agent_active: jax.Array


@struct.dataclass
class _FakeEnvState:
    state: _FakeSimState


class _TinyJointEnv:
    """Small JAX-compatible environment used to exercise trainer plumbing."""

    blue_agents = ("blue_0",)
    red_agents = ("red_0",)

    def __init__(self, *, blue_obs_dim: int, red_obs_dim: int, blue_actions: int, red_actions: int):
        self._obs_dims = {"blue_0": blue_obs_dim, "red_0": red_obs_dim}
        self._action_dims = {"blue_0": blue_actions, "red_0": red_actions}

    def observation_space(self, agent: str):
        return SimpleNamespace(shape=(self._obs_dims[agent],))

    def _obs(self, state: _FakeEnvState):
        value = state.state.time.astype(jnp.float32) / 10.0
        return {agent: jnp.full((obs_dim,), value, dtype=jnp.float32) for agent, obs_dim in self._obs_dims.items()}

    def reset(self, key):
        del key
        state = _FakeEnvState(
            state=_FakeSimState(
                time=jnp.array(0, dtype=jnp.int32),
                blue_pending_ticks=jnp.zeros((1,), dtype=jnp.int32),
                red_pending_ticks=jnp.zeros((1,), dtype=jnp.int32),
                red_agent_active=jnp.ones((1,), dtype=jnp.bool_),
            )
        )
        return self._obs(state), state

    def get_avail_actions(self, state):
        del state
        return {agent: jnp.ones((action_dim,), dtype=jnp.bool_) for agent, action_dim in self._action_dims.items()}

    def step(self, key, state, actions):
        del key
        next_state = state.replace(state=state.state.replace(time=state.state.time + 1))
        # A non-constant payoff makes both the policy and value objectives
        # meaningful while preserving the game's zero-sum reward contract.
        blue_reward = 1.0 + 0.1 * actions["blue_0"].astype(jnp.float32)
        rewards = {"blue_0": blue_reward, "red_0": -blue_reward}
        done = jnp.array(False)
        dones = {"blue_0": done, "red_0": done, "__all__": done}
        zero = jnp.array(0.0, dtype=jnp.float32)
        infos = {
            "reward_ria": blue_reward,
            "reward_lwf": zero,
            "reward_asf": zero,
            "action_cost": zero,
            "impact_count": zero,
            "green_lwf_count": zero,
            "green_asf_count": zero,
        }
        return self._obs(next_state), next_state, rewards, dones, infos


def _config() -> dict:
    return {
        "SEED": 42,
        "NUM_ENVS": 1,
        "NUM_STEPS": 2,
        "TOTAL_TIMESTEPS": 2,
        "NUM_MINIBATCHES": 1,
        "UPDATE_EPOCHS": 1,
        "TRAIN_VARIANT": object(),
        "TOPOLOGY_MODE": "generative",
        "TRAINING_MODE": True,
        "LR": 1e-2,
        "GAMMA": 0.9,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "MAX_GRAD_NORM": 1.0,
        "CLIP_VALUE_LOSS": False,
        "ANNEAL_LR": False,
        "NORM_REWARDS": False,
        "REWARD_SCALE": 1.0,
    }


def _tree_changed(before, after) -> bool:
    before_leaves, before_def = jax.tree.flatten(before)
    after_leaves, after_def = jax.tree.flatten(after)
    assert before_def == after_def
    return any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(before_leaves, after_leaves))


def _assert_tree_exact(before, after) -> None:
    before_leaves, before_def = jax.tree.flatten(before)
    after_leaves, after_def = jax.tree.flatten(after)
    assert before_def == after_def
    for expected, actual in zip(before_leaves, after_leaves):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.fixture
def tiny_joint(monkeypatch):
    blue_obs_dim, red_obs_dim = 4, 6
    blue_action_dim, red_action_dim = 3, 5
    env = _TinyJointEnv(
        blue_obs_dim=blue_obs_dim,
        red_obs_dim=red_obs_dim,
        blue_actions=blue_action_dim,
        red_actions=red_action_dim,
    )
    monkeypatch.setattr(joint, "make_joint_jax_env", lambda *_args, **_kwargs: env)
    networks = {
        "blue": make_jax_policy("separate", action_dim=blue_action_dim, hidden_dim=8, hidden_layers=1),
        "red": make_jax_policy("separate", action_dim=red_action_dim, hidden_dim=8, hidden_layers=1),
    }
    configs = {"blue": _config(), "red": _config()}
    return networks, configs


def _one_joint_update(tiny_joint, trainable_teams):
    networks, configs = tiny_joint
    _, obs, env_state, init_states, collect_and_update = joint.make_joint_train(
        configs,
        networks,
        trainable_teams=trainable_teams,
    )
    states = init_states(jax.random.PRNGKey(3))
    before = {team: states[team].params for team in joint.TEAMS}
    reward_norm = {team: joint.initial_reward_norm_state(1) for team in joint.TEAMS}
    with jax.disable_jit():
        states, _, _, _, _, metrics = collect_and_update(
            states,
            env_state,
            obs,
            jax.random.PRNGKey(7),
            reward_norm,
        )
    return before, states, metrics


def test_both_mode_updates_both_independent_policy_trees(tiny_joint):
    before, after, metrics = _one_joint_update(tiny_joint, ("blue", "red"))

    assert _tree_changed(before["blue"], after["blue"].params)
    assert _tree_changed(before["red"], after["red"].params)
    assert int(after["blue"].step) == 1
    assert int(after["red"].step) == 1
    np.testing.assert_allclose(metrics["red"]["raw_rollout_return"], -metrics["blue"]["raw_rollout_return"])


def test_single_team_mode_keeps_frozen_opponent_byte_identical(tiny_joint):
    before, after, metrics = _one_joint_update(tiny_joint, ("blue",))

    assert _tree_changed(before["blue"], after["blue"].params)
    _assert_tree_exact(before["red"], after["red"].params)
    assert int(after["blue"].step) == 1
    assert int(after["red"].step) == 0
    assert float(metrics["red"]["total_loss"]) == 0.0


def test_jitted_update_accepts_aliased_reward_normalizer_leaves(tiny_joint):
    networks, configs = tiny_joint
    configs["blue"]["NORM_REWARDS"] = True
    configs["red"]["NORM_REWARDS"] = True
    _, obs, env_state, init_states, collect_and_update = joint.make_joint_train(
        configs,
        networks,
        trainable_teams=("blue", "red"),
    )
    states = init_states(jax.random.PRNGKey(13))
    # This deliberately reuses one pytree (and therefore the same scalar
    # buffers) for both teams. Donating the nested argument makes XLA reject
    # precisely this valid construction as a duplicate donated buffer.
    shared_norm = joint.initial_reward_norm_state(1)
    norm_states = {"blue": shared_norm, "red": shared_norm}

    states, _, _, _, norm_states, metrics = collect_and_update(
        states,
        env_state,
        obs,
        jax.random.PRNGKey(17),
        norm_states,
    )
    jax.block_until_ready(metrics)

    assert int(states["blue"].step) == 1
    assert int(states["red"].step) == 1
    assert float(norm_states["blue"].mean) > 0.0
    assert float(norm_states["red"].mean) < 0.0


def test_joint_train_forwards_shared_topology_bank(tiny_joint, monkeypatch):
    networks, configs = tiny_joint
    fake_env = joint.make_joint_jax_env(None)
    bank = (Path("shape_00.snapshot.npz"), Path("shape_01.snapshot.npz"))
    for config in configs.values():
        config["TOPOLOGY_BANK"] = bank

    captured = {}

    def capture_factory(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_env

    monkeypatch.setattr(joint, "make_joint_jax_env", capture_factory)
    joint.make_joint_train(configs, networks, trainable_teams=("blue", "red"))

    assert captured["kwargs"]["topology_path"] == list(bank)


def test_joint_train_rejects_different_team_topology_banks(tiny_joint):
    networks, configs = tiny_joint
    configs["blue"]["TOPOLOGY_BANK"] = (Path("blue.snapshot.npz"),)
    configs["red"]["TOPOLOGY_BANK"] = (Path("red.snapshot.npz"),)

    with pytest.raises(ValueError, match="red must share TOPOLOGY_BANK"):
        joint.make_joint_train(configs, networks, trainable_teams=("blue", "red"))


def test_actor_and_critic_masks_update_only_their_separate_heads():
    network = make_jax_policy("separate", action_dim=3, hidden_dim=8, hidden_layers=1)
    params = network.init(jax.random.PRNGKey(11), jnp.zeros((4,), dtype=jnp.float32))
    config = _config()
    updater = joint._make_team_updater(network, config)

    obs = jnp.arange(2 * 1 * 2 * 4, dtype=jnp.float32).reshape((2, 1, 2, 4)) / 10.0
    avail = jnp.ones((2, 1, 2, 3), dtype=jnp.bool_)
    pi, value = network.apply(params, obs, avail)
    action = jnp.array([[[0, 1]], [[2, 0]]], dtype=jnp.int32)
    common = dict(
        done=jnp.zeros((2, 1, 2), dtype=jnp.float32),
        action=action,
        value=value,
        reward=jnp.array([[[1.0, -0.5]], [[-0.25, 2.0]]], dtype=jnp.float32),
        log_prob=pi.log_prob(action),
        obs=obs,
        avail_actions=avail,
    )

    def run(actor_mask, critic_mask):
        state = TrainState.create(apply_fn=network.apply, params=params, tx=optax.adam(1e-2))
        trajectory = joint.TeamTransition(
            **common,
            actor_mask=jnp.full((2, 1, 2), actor_mask, dtype=jnp.float32),
            critic_mask=jnp.full((2, 1, 2), critic_mask, dtype=jnp.float32),
        )
        with jax.disable_jit():
            updated, _, _ = updater(state, trajectory, jnp.zeros((1, 2)), jax.random.PRNGKey(12))
        return updated.params

    critic_only = run(actor_mask=0.0, critic_mask=1.0)
    actor_only = run(actor_mask=1.0, critic_mask=0.0)

    _assert_tree_exact(params["params"]["actor_head"], critic_only["params"]["actor_head"])
    assert _tree_changed(params["params"]["critic_head"], critic_only["params"]["critic_head"])
    assert _tree_changed(params["params"]["actor_head"], actor_only["params"]["actor_head"])
    _assert_tree_exact(params["params"]["critic_head"], actor_only["params"]["critic_head"])
