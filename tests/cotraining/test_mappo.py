"""MAPPO's information boundary, saved-policy contract, and recipe controls."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.checkpoint import PolicyBundleEntry, save_jax_bundle
from jaxborg.constants import BLUE_OBS_SIZE, NUM_BLUE_AGENTS, NUM_RED_AGENTS, RED_OBS_SIZE
from jaxborg.critic_observations import blue_critic_obs_size, critic_obs_size, get_blue_critic_obs, get_critic_obs
from jaxborg.env import ScenarioEnvState
from jaxborg.evaluation.matchup_runner import _jax_actions, load_matchup_policy
from jaxborg.joint_env import JointPolicyCC4Env
from jaxborg.learned_red import RED_POLICY_ACTION_DIM, get_red_policy_obs
from jaxborg.observations import get_blue_obs
from jaxborg.policies import has_centralized_critic, init_policy_params, policy_from_arch, policy_step
from jaxborg.recipe import load, project_jax, team_recipe
from jaxborg.state import create_initial_const, create_initial_state


@pytest.mark.parametrize("team,num_agents,widths", [("blue", 5, (1055, 4114)), ("red", 6, (4242, 7301))])
@pytest.mark.parametrize("mode", ["joint_observations", "global_state"])
def test_agents_share_world_features_with_distinct_identity(team, num_agents, widths, mode):
    state, const = create_initial_state(), create_initial_const()
    width = widths[mode == "global_state"]
    env = JointPolicyCC4Env()
    env_state = ScenarioEnvState(state=state, const=const)
    features = env.get_critic_obs(env_state, mode, team=team)

    assert features.shape == (num_agents, width)
    assert critic_obs_size(mode, team=team) == width
    np.testing.assert_array_equal(features[:, -num_agents:], np.eye(num_agents))
    for i in range(1, num_agents):
        np.testing.assert_array_equal(features[i, :-num_agents], features[0, :-num_agents])
    assert np.isfinite(features).all()
    observe = get_blue_obs if team == "blue" else get_red_policy_obs
    pooled = np.concatenate([observe(state, const, i) for i in range(num_agents)])
    np.testing.assert_array_equal(features[0, : pooled.size], pooled)
    if mode == "joint_observations":
        assert width == pooled.size + num_agents
    if team == "blue":
        assert blue_critic_obs_size(mode) == width
        np.testing.assert_array_equal(features, get_blue_critic_obs(state, const, mode))


@pytest.mark.parametrize("team", ["blue", "red"])
def test_hidden_damage_only_reaches_global_critic_input(team):
    state = create_initial_state()
    const = create_initial_const()
    const = const.replace(host_active=const.host_active.at[0].set(True))
    damaged = state.replace(
        host_compromised=state.host_compromised.at[0].set(2),
        ot_service_stopped=state.ot_service_stopped.at[0].set(True),
    )
    np.testing.assert_array_equal(
        get_critic_obs(state, const, "joint_observations", team=team),
        get_critic_obs(damaged, const, "joint_observations", team=team),
    )
    assert not np.array_equal(
        get_critic_obs(state, const, "global_state", team=team),
        get_critic_obs(damaged, const, "global_state", team=team),
    )
    # Padding and absent services must not masquerade as damaged hosts.
    inactive_damage = state.replace(
        host_compromised=state.host_compromised.at[-1].set(2),
        ot_service_stopped=state.ot_service_stopped.at[-1].set(True),
        host_service_reliability=jnp.zeros_like(state.host_service_reliability),
        host_decoy_reliability=jnp.zeros_like(state.host_decoy_reliability),
    )
    np.testing.assert_array_equal(
        get_critic_obs(state, const, team=team), get_critic_obs(inactive_damage, const, team=team)
    )


def test_red_critic_pools_teammate_knowledge_without_exposing_it_to_local_actor():
    const = create_initial_const()
    const = const.replace(host_active=const.host_active.at[0].set(True))
    state = create_initial_state()
    state = state.replace(red_agent_active=state.red_agent_active.at[1].set(True))
    discovered = state.replace(
        red_discovered_hosts=state.red_discovered_hosts.at[1, 0].set(True),
        fsm_host_entered=state.fsm_host_entered.at[1, 0].set(True),
    )
    np.testing.assert_array_equal(get_red_policy_obs(state, const, 0), get_red_policy_obs(discovered, const, 0))
    before = get_critic_obs(state, const, "joint_observations", team="red")
    after = get_critic_obs(discovered, const, "joint_observations", team="red")
    assert not np.array_equal(before, after)
    np.testing.assert_array_equal(after[0, RED_OBS_SIZE : 2 * RED_OBS_SIZE], get_red_policy_obs(discovered, const, 1))
    np.testing.assert_array_equal(
        get_blue_critic_obs(state, const, "joint_observations"),
        get_blue_critic_obs(discovered, const, "joint_observations"),
    )


@pytest.mark.parametrize("team,num_agents", [("blue", NUM_BLUE_AGENTS), ("red", NUM_RED_AGENTS)])
def test_world_state_changes_values_but_cannot_change_actor_logits(team, num_agents):
    module = policy_from_arch({"name": "mappo", "team": team, "hidden_dim": 8, "hidden_layers": 1}, action_dim=3)
    params = init_policy_params(module, jax.random.PRNGKey(0), 4)
    obs = jnp.ones((num_agents, 4))
    mask = jnp.array([[True, False, True]] * num_agents)
    world = jax.random.normal(jax.random.PRNGKey(1), (num_agents, module.critic_obs_dim))

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


@pytest.mark.parametrize(
    "team,num_agents,obs_dim,action_dim",
    [("blue", 5, BLUE_OBS_SIZE, BLUE_ALLOW_TRAFFIC_END), ("red", 6, RED_OBS_SIZE, RED_POLICY_ACTION_DIM)],
)
@pytest.mark.parametrize("mode", ["joint_observations", "global_state"])
def test_bundle_reloads_for_existing_local_actor_evaluator(tmp_path, team, num_agents, obs_dim, action_dim, mode):
    arch = {"name": "mappo", "hidden_dim": 8, "hidden_layers": 1, "critic_input": mode}
    # Blue checkpoints predating Red MAPPO have no team field.
    if team == "red":
        arch["team"] = team
    module = policy_from_arch(arch, action_dim=action_dim)
    params = init_policy_params(module, jax.random.PRNGKey(5), obs_dim)
    path = tmp_path / "model_mappo.safetensors"
    save_jax_bundle(
        path,
        {team: PolicyBundleEntry(params, team, obs_dim, action_dim, arch)},
    )
    loaded = load_matchup_policy(path, team=team, backend="jax")
    obs = jax.random.normal(jax.random.PRNGKey(6), (num_agents, obs_dim))
    mask = jnp.ones((num_agents, action_dim), dtype=jnp.bool_)
    expected, _, _ = policy_step(module, params, obs, mask)
    actions, carry = _jax_actions(loaded, obs, mask, jax.random.PRNGKey(7), deterministic=True)

    assert has_centralized_critic(loaded.module)
    assert loaded.module.critic_input == mode
    assert loaded.module.team == team
    assert loaded.module.critic_obs_dim == critic_obs_size(mode, team=team)
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
def test_mappo_recipes_train_both_teams_and_preserve_game_controls(baseline, name, count, critic_input, monkeypatch):
    recipe, control = load(name), load(baseline)
    assert recipe["algorithm"] == "mappo"
    assert recipe["train"]["teams"] == "both"
    assert recipe["train"]["topology_generation"]["count"] == count
    blue = team_recipe(recipe, "blue")
    red = team_recipe(recipe, "red")
    assert blue["arch"]["name"] == "mappo"
    assert blue["arch"]["critic_input"] == critic_input
    assert red["arch"]["name"] == "mappo"
    assert red["arch"]["team"] == "red"
    assert red["arch"]["critic_input"] == critic_input
    for resolved in (blue, red):
        assert resolved["core"] == dict(control["core"], clip_value_loss=True)
        for key in ("hidden_dim", "hidden_layers", "activation"):
            assert resolved["arch"][key] == control["arch"][key]

    # Projectors may generate topology files: this test checks config plumbing,
    # not the already-covered topology generator or filesystem cache.
    monkeypatch.setattr("jaxborg.recipe._resolve_topology_bank", lambda *_a, **_kw: ())
    blue_config, red_config = project_jax(recipe, team="blue"), project_jax(recipe, team="red")
    assert blue_config["CLIP_VALUE_LOSS"] is True
    assert red_config["CLIP_VALUE_LOSS"] is True
    assert red_config["NETWORK_TYPE"] == "mappo"

    stripped = copy.deepcopy(recipe)
    del stripped["train"]["team_overrides"]
    for obj in (stripped, control):
        for key in ("meta", "algorithm", "__source_path__"):
            obj.pop(key)
    assert stripped == control


@pytest.mark.parametrize("trainable", ["both", "red"])
def test_mappo_launcher_routes_joint_training_and_exports_overrides(tmp_path, monkeypatch, trainable):
    from scripts.train.algorithms import ippo_jax

    monkeypatch.setattr("jaxborg.recipe._resolve_topology_bank", lambda *_a, **_kw: ())
    monkeypatch.setattr(ippo_jax, "EXP_DIR", tmp_path)
    configured = load("cotraining_mappo")
    if trainable == "red":
        configured["train"]["teams"] = "red"
        configured["train"]["opponents"] = {"blue": {"path": str(tmp_path / "blue.safetensors")}}
    monkeypatch.setattr(ippo_jax, "load_recipe", lambda _: configured)
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
    assert team_recipe(recipe, "red")["arch"]["name"] == "mappo"
