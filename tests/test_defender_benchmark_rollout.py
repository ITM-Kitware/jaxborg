"""Game invariants and policy information boundaries for the matched benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.pretrained.hmarl import HMARLPolicy, actor_logits, policy_observations
from jaxborg.pretrained.hmarl_import import actor_arrays

_PATH = Path(__file__).resolve().parents[1] / "studies/defender-transfer/benchmark_rollout.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_rollout", _PATH)
rollout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rollout)


def test_hmarl_initializer_does_not_change_stock_world(jax_const):
    const = jax_const.replace(cage4_enhanced_obs=True, blue_observation_version=2)
    observations, before = rollout._initial_state(rollout._environment(500), const)
    assert observations[rollout._environment(500).agents[0]].shape == (450,)
    after, _ = HMARLPolicy("expert").initialize(before)
    assert bool(rollout._same_tree(before, after))
    tampered = after.replace(state=after.state.replace(host_services=after.state.host_services + 1))
    assert not bool(rollout._same_tree(before, tampered))
    changed_reward = after.replace(const=after.const.replace(phase_rewards=after.const.phase_rewards + 1))
    assert not bool(rollout._same_tree(before, changed_reward))


def test_hmarl_inputs_do_not_reveal_latent_red_state(jax_const):
    const = jax_const.replace(cage4_enhanced_obs=True, blue_observation_version=2)
    _, env = rollout._initial_state(rollout._environment(500), const)
    state = env.state
    altered = state.replace(
        red_sessions=~state.red_sessions,
        red_privilege=jnp.full_like(state.red_privilege, 2),
        host_compromised=jnp.full_like(state.host_compromised, 2),
    )
    for agent in range(5):
        width = 48 if agent == 4 else 16
        codes = state.blue_ioc_codes[agent, :width].reshape(-1, 16)
        before = policy_observations(state, const, agent, state.blue_ioc_memory, codes)
        after = policy_observations(altered, const, agent, state.blue_ioc_memory, codes)
        for actual, expected in zip(after, before, strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("agent", [0, 4])
@pytest.mark.parametrize("branch", ["master", "investigate", "recover"])
def test_imported_actor_preserves_released_linear_tanh_math(agent, branch):
    subnets = 3 if agent == 4 else 1
    input_size = 1 + subnets * {"master": 75, "investigate": 48, "recover": 16}[branch]
    output_size = 2 if branch == "master" else 2 + 80 * subnets
    rng = np.random.default_rng(123)
    weights = {}
    sizes = ((input_size, 256), (256, 256), (256, output_size))
    names = ("_hidden_layers.0", "_hidden_layers.1", "_logits")
    for name, (inputs, outputs) in zip(names, sizes, strict=True):
        prefix = f"internal_model.{name}._model.0"
        weights[prefix + ".weight"] = rng.normal(0, 0.04, (outputs, inputs)).astype(np.float32)
        weights[prefix + ".bias"] = rng.normal(0, 0.04, outputs).astype(np.float32)
    observations = rng.normal(size=(4, input_size)).astype(np.float32)
    expected = observations
    for index, name in enumerate(names):
        prefix = f"internal_model.{name}._model.0"
        expected = expected @ weights[prefix + ".weight"].T + weights[prefix + ".bias"]
        if index < 2:
            expected = np.tanh(expected)
    imported = actor_arrays(weights, agent=agent, branch=branch)
    np.testing.assert_allclose(actor_logits(imported, agent, branch, observations), expected, atol=2e-6, rtol=2e-5)


def _valid_result():
    components = np.zeros((500, 4))
    components[:, 0] = -1
    return {
        "reward": components.sum(axis=1),
        "components": components,
        "actions": np.zeros((500, 5), dtype=np.int32),
        "done": np.arange(500) == 499,
    }, {
        "initializer_preserved_world": True,
        "initial_services_match": True,
        "initial_pids_match": True,
        "roles_absent": True,
        "const_unchanged": True,
        "observation_width": 450,
        "final_time": 500,
    }


def test_reward_or_horizon_changes_rejected():
    rows, checks = _valid_result()
    rollout._validate_rows(rows, checks, 500)
    rows["reward"][0] += 1
    with pytest.raises(RuntimeError, match="reward"):
        rollout._validate_rows(rows, checks, 500)
    rows, checks = _valid_result()
    rows["done"][498] = True
    with pytest.raises(RuntimeError, match="horizon"):
        rollout._validate_rows(rows, checks, 500)


@pytest.mark.parametrize("check", ["initializer_preserved_world", "roles_absent", "const_unchanged"])
def test_game_changes_rejected(check):
    rows, checks = _valid_result()
    checks[check] = False
    with pytest.raises(RuntimeError, match=check):
        rollout._validate_rows(rows, checks, 500)


def test_seed_and_fixed_horizon_validation():
    for invalid in (True, -1, 2**32, 1.5):
        with pytest.raises(ValueError, match="integer"):
            rollout._seed(invalid, "dynamics_seed")
    with pytest.raises(ValueError, match="500"):
        rollout.run_episode(None, snapshot_path="unused", policy_seed=1, dynamics_seed=2, num_steps=499)


def test_policy_seed_cannot_change_environment_key_ledger():
    environment = jax.random.PRNGKey(22000000)
    for tick in (0, 1, 499):
        action_a, dynamics_a = rollout._step_keys(jax.random.PRNGKey(21000000), environment, tick)
        action_b, dynamics_b = rollout._step_keys(jax.random.PRNGKey(21000001), environment, tick)
        assert not np.array_equal(action_a, action_b)
        np.testing.assert_array_equal(dynamics_a, dynamics_b)
