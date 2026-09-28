from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.encoding import BLUE_RESTORE_START, BLUE_SLEEP, RED_SLEEP
from jaxborg.constants import GLOBAL_MAX_HOSTS
from jaxborg.env import ScenarioEnv
from jaxborg.evaluation.cyborg_reward import CyborgRewardTracker, native_reward_fields
from jaxborg.evaluation.jax_env_factory import make_joint_jax_env
from jaxborg.evaluation.reward_reporting import reward_fields, reward_mlflow_metrics
from jaxborg.recipe import load, project_jax
from jaxborg.reward_config import RewardConfig
from jaxborg.rewards import compute_reward_breakdown
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK, CIA_RESILIENCE
from jaxborg.scenarios.cc4.topology import build_topology, save_topology
from jaxborg.scenarios.cc4.topology_roles import ROLE_AUTH, ROLE_DB, ROLE_WEB
from jaxborg.state import create_initial_state


@pytest.fixture(scope="module")
def const():
    return build_topology(jax.random.PRNGKey(17), op_zone_min_servers=3)


def _state():
    state = create_initial_state()
    return state.replace(host_resilience_role=state.host_resilience_role.at[:3].set(jnp.array([1, 2, 3])))


def _breakdown(state, const, config=RewardConfig()):
    events = jnp.arange(GLOBAL_MAX_HOSTS) % 3 == 0
    return compute_reward_breakdown(
        state,
        const,
        events,
        ~events,
        events,
        blue_actions=jnp.array([BLUE_RESTORE_START, BLUE_SLEEP, BLUE_SLEEP, BLUE_SLEEP, BLUE_SLEEP]),
        reward_config=config,
    )


def test_default_is_bitwise_stock_while_reporting_counterfactual(const):
    state = _state()
    state = state.replace(ot_service_stopped=state.ot_service_stopped.at[0].set(True))
    result = _breakdown(state, const)
    stock = result.ria_reward + result.lwf_reward + result.asf_reward + result.action_cost
    assert np.asarray(result.total).tobytes() == np.asarray(stock).tobytes()
    assert float(result.shaping_reward) == float(stock) - 30
    zero_scale = _breakdown(state, const, RewardConfig("shaping", 0))
    assert np.asarray(zero_scale.total).tobytes() == np.asarray(stock).tobytes()


@pytest.mark.parametrize("role,expected", [(ROLE_AUTH, -10), (ROLE_DB, -10), (ROLE_WEB, 0)])
def test_c_only_weights_and_lambda(const, role, expected):
    state = _state()
    state = state.replace(ot_service_stopped=state.ot_service_stopped.at[role - 1].set(True))
    result = _breakdown(state, const, RewardConfig("shaping", 2, (1, 0, 0)))
    assert float(result.cia_reward) == 2 * expected
    assert float(result.total) == float(result.default_reward) + 2 * expected


def test_uniform_worst_case_is_minus_seventy(const):
    state = _state()
    state = state.replace(ot_service_stopped=state.ot_service_stopped.at[:3].set(True))
    assert float(_breakdown(state, const).cia_reward) == -70


def test_missing_roles_leave_default_valid_and_shaping_unavailable(const):
    result = _breakdown(create_initial_state(), const)
    assert np.isfinite(result.total)
    assert not bool(result.cia_valid)
    assert np.isnan(result.shaping_reward)


@pytest.mark.parametrize("mode", ["zero_sum", "damage"])
@pytest.mark.parametrize("selected", ["default", "shaping"])
def test_env_selects_complete_payoff_and_correct_red_sign(const, monkeypatch, mode, selected):
    import jaxborg.env as module

    # Keep simulation effects fixed to isolate reward selection and Restore cost.
    monkeypatch.setattr(module, "apply_all_actions", lambda state, *args, **kwargs: state)
    env = ScenarioEnv(op_zone_min_servers=3, red_reward=mode, reward_config=RewardConfig(selected, 2, (1, 0, 0)))
    monkeypatch.setattr(env, "get_obs", lambda state: {})
    _, state = env._reset_from_const(const, jax.random.PRNGKey(12))
    auth = jnp.argmax(state.state.host_resilience_role == ROLE_AUTH)
    state = state.replace(
        state=state.state.replace(ot_service_stopped=state.state.ot_service_stopped.at[auth].set(True))
    )
    actions = {a: jnp.int32(BLUE_SLEEP) for a in env.blue_agents} | {a: jnp.int32(RED_SLEEP) for a in env.red_agents}
    actions["blue_0"] = jnp.int32(BLUE_RESTORE_START)
    _, _, rewards, _, info = env.step_env(jax.random.PRNGKey(2), state, actions)
    assert float(info["reward_default"]) == -1
    assert float(info["reward_shaping"]) == -21
    assert float(rewards["blue_0"]) == (-21 if selected == "shaping" else -1)
    if mode == "zero_sum":
        assert float(rewards["blue_0"] + rewards["red_0"]) == 0
    else:
        assert float(rewards["red_0"]) == (20 if selected == "shaping" else 0)
        assert float(rewards["blue_0"] + rewards["red_0"]) == -1


def test_roles_reproducible_rotate_and_preserve_stock_topology(const, monkeypatch):
    default = ScenarioEnv(op_zone_min_servers=3)
    shaped = ScenarioEnv(op_zone_min_servers=3, reward_config=RewardConfig("shaping"))
    monkeypatch.setattr(default, "get_obs", lambda _: {})
    monkeypatch.setattr(shaped, "get_obs", lambda _: {})
    maps = []
    for seed in range(3):
        key = jax.random.PRNGKey(seed)
        a = default._reset_from_const(const, key)[1]
        b = shaped._reset_from_const(const, key)[1]
        reset = shaped._reset_state_from_const(const, key)
        np.testing.assert_array_equal(a.state.host_resilience_role, b.state.host_resilience_role)
        np.testing.assert_array_equal(b.state.host_resilience_role, reset.state.host_resilience_role)
        roles = np.asarray(b.state.host_resilience_role)
        assert sorted(roles[roles > 0]) == [1, 2, 3]
        assert np.all(np.asarray(const.host_active & const.host_is_server)[roles > 0])
        maps.append(roles.tobytes())
    assert len(set(maps)) > 1
    key = jax.random.PRNGKey(8)
    for a, b in zip(
        jax.tree.leaves(default._select_const(key)), jax.tree.leaves(shaped._select_const(key)), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_shaping_rejects_invalid_topology_without_changing_default(tmp_path):
    # One server per operational zone cannot hold three roles.
    invalid = build_topology(jax.random.PRNGKey(7), op_zone_min_servers=1)
    path = tmp_path / "small.snapshot.npz"
    save_topology(invalid, path)
    env = ScenarioEnv(topology_path=path)
    state = env._reset_state_from_const(invalid, jax.random.PRNGKey(1))
    assert not np.any(state.state.host_resilience_role)
    with pytest.raises(ValueError, match="three"):
        ScenarioEnv(topology_path=path, reward_config=RewardConfig("shaping"))
    with pytest.raises(ValueError, match="resilience_roles"):
        make_joint_jax_env(CC4_STOCK, reward_config=RewardConfig("shaping"))


@pytest.mark.parametrize(
    "raw",
    [
        {"name": "typo"},
        {"lambda": -1},
        {"lambda": float("nan")},
        {"lambda": True},
        {"weights": [1, 0, 0]},
        {"weights": {"C": -1}},
        {"weights": {"A": float("inf")}},
        {"weights": {"X": 1}},
        {"scale": 1},
    ],
)
def test_reward_config_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="reward"):
        RewardConfig.from_recipe({"train": {"reward": raw}})


def test_config_defaults_projection_and_alignment_copies(monkeypatch):
    import jaxborg.recipe as module

    monkeypatch.setattr(module, "_resolve_topology_bank", lambda *args, **kwargs: ())
    assert RewardConfig.from_recipe({}) == RewardConfig("default", 1, (1, 1, 1))
    assert project_jax(load("cotraining"))["REWARD_CONFIG"] == RewardConfig()
    originals = list(Path("recipes/cotraining").glob("cotraining*.yaml"))
    copies = list(Path("recipes/cotraining/alignment").glob("*.yaml"))
    assert len(copies) == len(originals)
    for path in copies:
        recipe = load(str(path))
        cfg = project_jax(recipe)
        assert cfg["REWARD_CONFIG"] == RewardConfig("shaping", 1, (1, 0, 0))
        assert cfg["TRAIN_VARIANT"].resilience_roles
        assert recipe["train"]["topology_generation"]["op_zone_servers"] == 3
        assert "alignment" in recipe["train"]["topology_generation"]["cache_dir"]
        baseline = recipe.get("eval", {}).get("env_diversity", {}).get("baseline_recipe")
        if baseline:
            assert baseline.endswith("_alignment_c")
            load(baseline)


def test_paired_evaluation_uses_episode_sums_and_reports_both_modes():
    recipe = {"train": {"reward": {"name": "default", "lambda": 2, "weights": {"C": 1, "I": 0, "A": 0}}}}
    result = reward_fields([-5, -7], [{"c": -1, "i": -9, "a": -9}, {"c": -2, "i": 0, "a": 0}], steps=10, recipe=recipe)
    assert result["reward_team"] == "blue"
    assert result["per_episode_reward_default"] == [-5, -7]
    assert result["per_episode_reward_shaping"] == [-25, -47]
    assert result["mean_reward_default"] == -6
    assert result["mean_reward_shaping"] == -36
    metrics = reward_mlflow_metrics("eval.test", result)
    assert metrics["eval.test.mean_reward_default"] == -6
    assert metrics["eval.test.mean_reward_shaping"] == -36
    missing = reward_fields([-5], [], steps=10, recipe=recipe)
    assert missing["mean_reward_shaping"] is None
    assert not missing["shaping_available"]


def test_native_cyborg_scores_persistent_damage_and_missing_roles():
    service = SimpleNamespace(active=False, get_service_reliability=lambda: 100)
    host = SimpleNamespace(services={"OTSERVICE": service})
    hosts = {"auth": host, "db": SimpleNamespace(services={}), "web": SimpleNamespace(services={})}
    native = SimpleNamespace(environment_controller=SimpleNamespace(state=SimpleNamespace(hosts=hosts)))
    env = SimpleNamespace(unwrapped=native)
    tracker = CyborgRewardTracker(env, 0, {"auth": ROLE_AUTH, "db": ROLE_DB, "web": ROLE_WEB})
    incomplete = CyborgRewardTracker(env, 0, {"auth": ROLE_AUTH})
    incomplete.step()
    assert incomplete.cia_sum is None
    tracker.step()
    tracker.step()
    assert tracker.cia_sum == [-20, -20, -20]
    service.active = True
    tracker.step()
    assert tracker.cia_sum == [-20, -20, -20]
    recipe = {"train": {"reward": {"lambda": 2, "weights": {"C": 1, "I": 0, "A": 0}}}}
    result = native_reward_fields([-5], [tracker.cia_sum], recipe)
    assert result["mean_reward_shaping"] == -45
    missing = native_reward_fields([-5, -6], [tracker.cia_sum, None], recipe)
    assert missing["per_episode_reward_shaping"] == [-45, None]
    assert missing["mean_reward_shaping"] is None


def test_joint_env_real_transition_reports_both_and_stays_zero_sum():
    from jaxborg.actions.red_policy import RED_POLICY_SLEEP

    env = make_joint_jax_env(replace(CIA_RESILIENCE, num_steps=2), reward_config=RewardConfig("shaping", 1, (1, 0, 0)))
    _, state = env.reset(jax.random.PRNGKey(33))
    auth = jnp.argmax(state.state.host_resilience_role == ROLE_AUTH)
    state = state.replace(
        state=state.state.replace(ot_service_stopped=state.state.ot_service_stopped.at[auth].set(True))
    )
    actions = {a: jnp.int32(BLUE_SLEEP) for a in env.blue_agents} | {
        a: jnp.int32(RED_POLICY_SLEEP) for a in env.red_agents
    }
    for seed in (34, 35):
        _, state, rewards, dones, info = env.step_env(jax.random.PRNGKey(seed), state, actions)
        assert float(rewards["blue_0"] + rewards["red_0"]) == 0
        assert float(rewards["blue_0"]) == float(info["reward_shaping"])
        assert float(info["reward_shaping"] - info["reward_default"]) == -10
    assert bool(dones["__all__"])


def test_learned_matchup_rows_preserve_both_scores():
    from jaxborg.evaluation.cross_play import _cell_row
    from jaxborg.evaluation.matchup_runner import MatchupEvaluation
    from jaxborg.evaluation.play_priors import PeriodicCheckpoint, _result_row

    recipe = {"train": {"reward": {"lambda": 2, "weights": {"C": 1, "I": 0, "A": 0}}}}
    evaluation = MatchupEvaluation(
        blue_returns=[-5, -7],
        red_returns=[5, 7],
        episode_seeds=[1, 2],
        policies={},
        per_episode_cia=[{"c": -1, "i": -9, "a": -9}, {"c": -2, "i": 0, "a": 0}],
    )
    checkpoint = PeriodicCheckpoint(Path("model.safetensors"), 100)
    kwargs = dict(
        evaluation=evaluation,
        recipe=recipe,
        backend="jax",
        seeds=(1, 2),
        episodes_per_seed=1,
        deterministic=True,
        wall_time_s=0,
        eval_id="test",
    )
    cross = _cell_row(**kwargs, blue=checkpoint, red=checkpoint, blue_index=0, red_index=0)
    prior = _result_row(**kwargs, focal_team="blue", current=checkpoint, prior=checkpoint, current_index=0)
    for row in (cross, prior):
        assert row["mean_reward_default"] == row["mean_reward"] == -6
        assert row["per_episode_reward_shaping"] == [-1005, -2007]
        assert row["mean_reward_shaping"] == -1506


@pytest.mark.parametrize("batched", [False, True])
def test_scripted_roles_match_reward_state_after_reset_and_autoreset(const, monkeypatch, batched):
    from jaxborg.evaluation.jax_env_factory import make_jax_env

    env = make_jax_env(CIA_RESILIENCE, reward_config=RewardConfig("shaping"))
    obs = {agent: jnp.zeros(1) for agent in env.agents}
    monkeypatch.setattr(env, "_get_blue_obs", lambda state: obs)
    monkeypatch.setattr(env._env, "_reset_state", lambda state, key: env._env._reset_state_from_const(const, key))
    inner = env._env._reset_state_from_const(const, jax.random.PRNGKey(1))
    _, state = env._wrap_reset(obs, inner, jax.random.PRNGKey(2))
    np.testing.assert_array_equal(state.state.host_resilience_role, state.extras["host_resilience_role"])

    def terminal_step(key, state, actions):
        return obs, state, {}, {"__all__": jnp.bool_(True)}, {}

    monkeypatch.setattr(env, "step_env", terminal_step)
    key = jax.random.PRNGKey(4)
    if batched:
        state = jax.tree.map(lambda value: jnp.stack([value, value]), state)
        keys = jax.random.split(key, 2)
        _, result, _, _, _ = env.step_batch.__wrapped__(env, keys, state, {})
        _, repeated, _, _, _ = env.step_batch.__wrapped__(env, keys, state, {})
    else:
        _, result, _, _, _ = env.step.__wrapped__(env, key, state, {})
        _, repeated, _, _, _ = env.step.__wrapped__(env, key, state, {})
    np.testing.assert_array_equal(result.state.host_resilience_role, result.extras["host_resilience_role"])
    np.testing.assert_array_equal(result.state.host_resilience_role, repeated.state.host_resilience_role)
    assert not np.array_equal(result.state.host_resilience_role, state.state.host_resilience_role)


def test_partial_cia_results_keep_episode_pairing_and_serialize_as_null():
    import json

    result = reward_fields(
        [-1, -2, -3],
        [{"c": -1, "i": 0, "a": 0}, None, {"c": float("nan"), "i": 0, "a": 0}],
        steps=2,
        recipe={},
    )
    assert result["per_episode_reward_shaping"] == [-3, None, None]
    assert result["mean_reward_default"] == -2
    assert result["mean_reward_shaping"] is None
    assert not result["shaping_available"]
    json.dumps(result, allow_nan=False)
    unavailable = reward_fields([-1, -2], [], steps=2, recipe={})
    assert unavailable["per_episode_reward_shaping"] == [None, None]


def test_reward_configuration_round_trips_through_checkpoint_sidecar(tmp_path):
    from jaxborg.checkpoint import read_sidecar, write_sidecar

    config = RewardConfig("shaping", 0.25, (1, 0, 0))
    recipe = {"train": {"reward": config.as_dict()}}
    write_sidecar(tmp_path / "recipe_model.yaml", recipe, seed=7, total_steps=10, backend="jax")
    assert RewardConfig.from_recipe(read_sidecar(tmp_path / "model.safetensors")) == config
