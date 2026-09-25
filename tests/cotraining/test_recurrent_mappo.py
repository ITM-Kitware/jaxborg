"""LSTM MAPPO memory, information boundaries, checkpoints and cotraining recipes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.blue_observation_contract import blue_obs_size
from jaxborg.checkpoint import PolicyBundleEntry, save_jax_bundle
from jaxborg.constants import RED_OBS_SIZE
from jaxborg.critic_observations import critic_obs_size
from jaxborg.evaluation.matchup_runner import TEAM_DIMS, _jax_actions, load_matchup_policy
from jaxborg.policies import (
    buffer_layout,
    has_centralized_critic,
    init_policy_params,
    initial_carry,
    is_recurrent,
    policy_from_arch,
    policy_sequence,
    policy_step,
)
from jaxborg.recipe import load, project_jax, team_recipe


def _arch(team="blue", **options):
    return dict(name="recurrent_mappo", team=team, hidden_dim=8, hidden_layers=1, cell="lstm", **options)


def _assert_close(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("mode", ["global_state", "joint_observations"])
def test_lstm_replay_matches_rollout_and_resets_both_memories(team, mode):
    module = policy_from_arch(_arch(team, critic_input=mode), action_dim=3)
    params = init_policy_params(module, jax.random.PRNGKey(0), 4)
    assert is_recurrent(module) and has_centralized_critic(module)
    assert buffer_layout("recurrent_mappo") == "sequence"
    carry = initial_carry(module, 2)
    assert len(jax.tree.leaves(carry)) == 4  # actor (c, h), critic (c, h)
    # A nonzero start also checks replay of an arbitrary rollout window.
    carry = jax.tree.map(lambda x: x + 0.3, carry)
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, 2, 4))
    world = jax.random.normal(jax.random.PRNGKey(2), (4, 2, module.critic_obs_dim))
    masks = jnp.broadcast_to(jnp.array([True, False, True]), (4, 2, 3))
    resets = jnp.zeros((4, 2), dtype=bool).at[2, 0].set(True)
    seq_pi, seq_value, seq_carry = policy_sequence(
        module, params, obs, masks, carry=carry, reset=resets, critic_obs=world
    )
    logits, values = [], []
    for t in range(4):
        pi, value, carry = policy_step(
            module, params, obs[t], masks[t], carry=carry, reset=resets[t], critic_obs=world[t]
        )
        logits.append(pi.logits)
        values.append(value)
        if t == 2:
            fresh_pi, fresh_value, fresh_carry = policy_step(
                module, params, obs[t], masks[t], carry=initial_carry(module, 2), critic_obs=world[t]
            )
            _assert_close(pi.logits[0], fresh_pi.logits[0])
            _assert_close(value[0], fresh_value[0])
            _assert_close(jax.tree.map(lambda x: x[0], carry), jax.tree.map(lambda x: x[0], fresh_carry))
            assert not np.allclose(pi.logits[1, [0, 2]], fresh_pi.logits[1, [0, 2]], atol=1e-7)
            assert not np.allclose(value[1], fresh_value[1], atol=1e-7)
    _assert_close(jnp.stack(logits), seq_pi.logits)
    _assert_close(jnp.stack(values), seq_value)
    _assert_close(carry, seq_carry)
    assert (seq_pi.logits[..., 1] < -1e8).all()


@pytest.mark.parametrize("team", ["blue", "red"])
def test_centralized_history_never_enters_actor_and_evaluation_needs_no_critic(team):
    module = policy_from_arch(_arch(team), action_dim=3)
    params = init_policy_params(module, jax.random.PRNGKey(0), 4)
    obs = jax.random.normal(jax.random.PRNGKey(1), (3, 2, 4))
    world = jax.random.normal(jax.random.PRNGKey(2), (3, 2, module.critic_obs_dim))
    carry = initial_carry(module, 2)
    pi, value, trained_carry = policy_sequence(module, params, obs, carry=carry, critic_obs=world)
    other_pi, other_value, other_carry = policy_sequence(module, params, obs, carry=carry, critic_obs=-world)
    _assert_close(pi.logits, other_pi.logits)
    _assert_close(trained_carry["actor"], other_carry["actor"])
    assert not np.allclose(value, other_value)
    actor_params = {"params": {k: v for k, v in params["params"].items() if k.startswith("actor_")}}
    actor_pi, placeholder, actor_carry = policy_sequence(module, actor_params, obs, carry=carry)
    _assert_close(pi.logits, actor_pi.logits)
    _assert_close(trained_carry["actor"], actor_carry["actor"])
    np.testing.assert_array_equal(placeholder, jnp.zeros((3, 2)))
    with pytest.raises(ValueError, match="critic_obs must match"):
        policy_sequence(module, params, obs, carry=carry, critic_obs=world[..., :4])


@pytest.mark.parametrize("team", ["blue", "red"])
def test_checkpoint_actor_preserves_lstm_state_in_existing_evaluator(tmp_path, team):
    arch = _arch(team, critic_input="joint_observations", cage4_enhanced_obs=True)
    obs_dim = blue_obs_size(True) if team == "blue" else RED_OBS_SIZE
    action_dim = TEAM_DIMS[team][1]
    module = policy_from_arch(arch, action_dim=action_dim)
    params = init_policy_params(module, jax.random.PRNGKey(0), obs_dim)
    path = tmp_path / "lstm_mappo.safetensors"
    save_jax_bundle(path, {team: PolicyBundleEntry(params, team, obs_dim, action_dim, arch)})
    loaded = load_matchup_policy(path, team=team, backend="jax")
    assert has_centralized_critic(loaded.module) and is_recurrent(loaded.module)
    assert loaded.module.team == team
    carry = initial_carry(module, 2)
    loaded_carry = initial_carry(loaded.module, 2)
    mask = jnp.ones((2, action_dim), dtype=bool)
    for obs in jax.random.normal(jax.random.PRNGKey(1), (3, 2, obs_dim)):
        expected, _, carry = policy_step(module, params, obs, mask, carry=carry)
        actions, loaded_carry = _jax_actions(
            loaded, obs, mask, jax.random.PRNGKey(2), deterministic=True, carry=loaded_carry
        )
        np.testing.assert_array_equal(actions, jnp.argmax(expected.logits, axis=-1))
        _assert_close(carry, loaded_carry)


@pytest.mark.parametrize("suffix", ["", "_env_diversity"])
def test_experiments_cotrain_both_lstm_mappo_teams_with_baseline_controls(suffix, monkeypatch):
    recipe = load(f"cotraining_mappo_lstm{suffix}")
    control = load(f"cotraining_mappo{suffix}")
    monkeypatch.setattr("jaxborg.recipe._resolve_topology_bank", lambda *_a, **_kw: ())
    assert recipe["algorithm"] == "mappo"
    assert recipe["train"]["teams"] == "both"
    assert recipe["train"]["total_timesteps"] == 50_000_000
    if suffix:
        assert recipe["eval"]["env_diversity"]["baseline_recipe"] == "cotraining_mappo_lstm"
        control["eval"]["env_diversity"]["baseline_recipe"] = "cotraining_mappo_lstm"
    assert recipe["eval"] == control["eval"]
    assert recipe["jax"] == control["jax"]
    for key, value in control["train"].items():
        if key != "team_overrides":
            assert recipe["train"][key] == value
    for team, n in (("blue", 5), ("red", 6)):
        resolved = team_recipe(recipe, team)
        arch = resolved["arch"]
        assert arch["name"] == "recurrent_mappo" and arch["cell"] == "lstm"
        assert arch["cage4_enhanced_obs"] is True
        assert resolved["core"] == team_recipe(control, team)["core"]
        module = policy_from_arch(arch, action_dim=3)
        assert module.team == team
        assert module.critic_obs_dim == critic_obs_size("global_state", team=team, cage4_enhanced_obs=True)
        config = project_jax(recipe, team=team)
        assert config["CLIP_VALUE_LOSS"] is True
        assert config["NUM_ENVS"] * n % config["NUM_MINIBATCHES"] == 0
