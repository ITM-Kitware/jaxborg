"""MAPPO's information boundary, saved-policy contract, and recipe controls."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.checkpoint import PolicyBundleEntry, save_jax_bundle
from jaxborg.constants import BLUE_OBS_SIZE, NUM_BLUE_AGENTS
from jaxborg.critic_observations import blue_critic_obs_size, get_blue_critic_obs
from jaxborg.evaluation.matchup_runner import _jax_actions, load_matchup_policy
from jaxborg.policies import has_centralized_critic, init_policy_params, policy_from_arch, policy_step
from jaxborg.recipe import load, project_jax, team_recipe
from jaxborg.state import create_initial_const, create_initial_state


@pytest.mark.parametrize("mode,width", [("joint_observations", 1055), ("global_state", 4114)])
def test_five_agents_share_world_features_with_distinct_identity(mode, width):
    state, const = create_initial_state(), create_initial_const()
    features = jax.jit(lambda s, c: get_blue_critic_obs(s, c, mode))(state, const)

    assert features.shape == (NUM_BLUE_AGENTS, width)
    assert blue_critic_obs_size(mode) == width
    np.testing.assert_array_equal(features[:, -NUM_BLUE_AGENTS:], np.eye(NUM_BLUE_AGENTS))
    for i in range(1, NUM_BLUE_AGENTS):
        np.testing.assert_array_equal(features[i, :-NUM_BLUE_AGENTS], features[0, :-NUM_BLUE_AGENTS])
    assert np.isfinite(features).all()


def test_hidden_damage_only_reaches_global_critic_input():
    state = create_initial_state()
    const = create_initial_const()
    const = const.replace(host_active=const.host_active.at[0].set(True))
    damaged = state.replace(
        host_compromised=state.host_compromised.at[0].set(2),
        ot_service_stopped=state.ot_service_stopped.at[0].set(True),
    )
    np.testing.assert_array_equal(
        get_blue_critic_obs(state, const, "joint_observations"),
        get_blue_critic_obs(damaged, const, "joint_observations"),
    )
    assert not np.array_equal(
        get_blue_critic_obs(state, const, "global_state"),
        get_blue_critic_obs(damaged, const, "global_state"),
    )
    # Padding and absent services must not masquerade as damaged hosts.
    inactive_damage = state.replace(
        host_compromised=state.host_compromised.at[-1].set(2),
        ot_service_stopped=state.ot_service_stopped.at[-1].set(True),
        host_service_reliability=jnp.zeros_like(state.host_service_reliability),
        host_decoy_reliability=jnp.zeros_like(state.host_decoy_reliability),
    )
    np.testing.assert_array_equal(get_blue_critic_obs(state, const), get_blue_critic_obs(inactive_damage, const))


def test_world_state_changes_values_but_cannot_change_actor_logits():
    module = policy_from_arch({"name": "mappo", "hidden_dim": 8, "hidden_layers": 1}, action_dim=3)
    params = init_policy_params(module, jax.random.PRNGKey(0), 4)
    obs = jnp.ones((5, 4))
    mask = jnp.array([[True, False, True]] * 5)
    world = jax.random.normal(jax.random.PRNGKey(1), (5, module.critic_obs_dim))

    pi, value, _ = policy_step(module, params, obs, mask, critic_obs=world)
    other_pi, other_value, _ = policy_step(module, params, obs, mask, critic_obs=-world)
    np.testing.assert_array_equal(pi.logits, other_pi.logits)
    assert not np.allclose(value, other_value)
    assert (pi.logits[:, 1] < -1e8).all()

    # Evaluation must still work after removing every critic weight.
    actor_params = {"params": {"actor_head": params["params"]["actor_head"]}}
    actor_pi, _, carry = policy_step(module, actor_params, obs, mask)
    np.testing.assert_array_equal(pi.logits, actor_pi.logits)
    assert carry is None
    with pytest.raises(ValueError, match="critic_obs must match"):
        policy_step(module, params, obs, critic_obs=jnp.zeros((5, 4)))


def test_bundle_reloads_for_existing_local_actor_evaluator(tmp_path):
    arch = {"name": "mappo", "hidden_dim": 8, "hidden_layers": 1, "critic_input": "joint_observations"}
    module = policy_from_arch(arch, action_dim=BLUE_ALLOW_TRAFFIC_END)
    params = init_policy_params(module, jax.random.PRNGKey(5), BLUE_OBS_SIZE)
    path = tmp_path / "model_mappo.safetensors"
    save_jax_bundle(
        path,
        {"blue": PolicyBundleEntry(params, "blue", BLUE_OBS_SIZE, BLUE_ALLOW_TRAFFIC_END, arch)},
    )
    loaded = load_matchup_policy(path, team="blue", backend="jax")
    obs = jax.random.normal(jax.random.PRNGKey(6), (5, BLUE_OBS_SIZE))
    mask = jnp.ones((5, BLUE_ALLOW_TRAFFIC_END), dtype=jnp.bool_)
    expected, _, _ = policy_step(module, params, obs, mask)
    actions, carry = _jax_actions(loaded, obs, mask, jax.random.PRNGKey(7), deterministic=True)

    assert has_centralized_critic(loaded.module)
    assert loaded.module.critic_input == "joint_observations"
    np.testing.assert_array_equal(actions, jnp.argmax(expected.logits, axis=-1))
    assert carry is None


@pytest.mark.parametrize(
    "baseline,name,count,critic_input",
    [
        ("cotraining", "cotraining_mappo", 1, "global_state"),
        ("cotraining_env_diversity", "cotraining_mappo_env_diversity", 100, "global_state"),
        ("cotraining", "cotraining_mappo_joint_obs", 1, "joint_observations"),
        ("cotraining_env_diversity", "cotraining_mappo_joint_obs_env_diversity", 100, "joint_observations"),
    ],
)
def test_mappo_recipes_preserve_red_ippo_and_game_controls(baseline, name, count, critic_input, monkeypatch):
    recipe, control = load(name), load(baseline)
    assert recipe["algorithm"] == "mappo"
    assert recipe["train"]["teams"] == "both"
    assert recipe["train"]["topology_generation"]["count"] == count
    blue = team_recipe(recipe, "blue")
    red = team_recipe(recipe, "red")
    assert blue["arch"]["name"] == "mappo"
    assert blue["arch"]["critic_input"] == critic_input
    assert red["arch"] == control["arch"]
    assert red["core"] == control["core"]

    # Projectors may generate topology files: this test checks config plumbing,
    # not the already-covered topology generator or filesystem cache.
    monkeypatch.setattr("jaxborg.recipe._resolve_topology_bank", lambda *_a, **_kw: ())
    blue_config, red_config = project_jax(recipe, team="blue"), project_jax(recipe, team="red")
    assert blue_config["CLIP_VALUE_LOSS"] is True
    assert red_config["CLIP_VALUE_LOSS"] is False
    assert red_config["NETWORK_TYPE"] == "shared"

    stripped = copy.deepcopy(recipe)
    del stripped["train"]["team_overrides"]
    for obj in (stripped, control):
        for key in ("meta", "algorithm", "__source_path__"):
            obj.pop(key)
    assert stripped == control


def test_mappo_launcher_routes_joint_training_and_exports_overrides(tmp_path, monkeypatch):
    from scripts.train.algorithms import ippo_jax

    monkeypatch.setattr("jaxborg.recipe._resolve_topology_bank", lambda *_a, **_kw: ())
    monkeypatch.setattr(ippo_jax, "EXP_DIR", tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["mappo_jax.py", "--recipe", "cotraining_mappo", "--seed", "12", "--total-timesteps", "24000"]
    )
    calls = []
    monkeypatch.setattr(ippo_jax, "_run_joint_training", lambda *args: calls.append(args))
    ippo_jax.main(expected_algorithm="mappo")

    assert len(calls) == 1
    args, recipe, tag, save_dir = calls[0]
    assert args.seed == 12
    assert recipe["train"]["total_timesteps"] == 24000
    assert tag == "cotraining_mappo_seed12"
    assert save_dir == tmp_path / "mappo_jax" / tag
    assert team_recipe(recipe, "blue")["arch"]["name"] == "mappo"
    assert team_recipe(recipe, "red")["arch"]["name"] == "shared"
