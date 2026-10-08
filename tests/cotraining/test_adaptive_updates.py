from __future__ import annotations

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from jaxborg.adaptive_updates import AdaptiveUpdateSettings, adaptive_red_update, initial_adaptive_update_state
from jaxborg.recipe import load, project_cleanrl, project_team_configs


def _schedule(returns, **settings):
    config = AdaptiveUpdateSettings(**settings)

    def step(state, inputs):
        rollout_number, blue_return = inputs
        updated, state, metrics = adaptive_red_update(
            config, state, blue_return=blue_return, rollout_number=rollout_number, warmup_update_every=4
        )
        return state, {**metrics, "updated": updated}

    @jax.jit
    def run():
        return jax.lax.scan(
            step,
            initial_adaptive_update_state(config),
            (jnp.arange(1, len(returns) + 1), jnp.asarray(returns, dtype=jnp.float32)),
        )

    return jax.device_get(run())


def test_phase_one_gives_exactly_50_red_updates_before_adapting_at_201():
    _, metrics = _schedule([-1999.0] * 205)
    expected = np.array([number % 4 == 0 for number in range(1, 201)] + [True] * 5)
    np.testing.assert_array_equal(metrics["updated"], expected)
    assert metrics["updated"][:200].sum() == 50
    np.testing.assert_array_equal(metrics["adaptive_phase"], [1] * 200 + [2] * 5)
    np.testing.assert_array_equal(metrics["gate_open"], [0] * 200 + [1] * 5)


def test_paper_gate_is_closed_at_exact_threshold():
    _, metrics = _schedule([-2000.0] * 205)
    np.testing.assert_array_equal(metrics["updated"][200:], [False] * 5)
    np.testing.assert_array_equal(metrics["reward_window_mean"][200:], [-2000.0] * 5)
    np.testing.assert_array_equal(metrics["gate_open"], [0] * 205)


def test_gate_uses_last_five_rollouts_and_can_close_again():
    returns = [-3000.0] * 4 + [-1000.0] * 4 + [-4000.0] * 5
    _, metrics = _schedule(returns, phase_switch_blue_updates=0)
    means = [np.mean(returns[max(0, i - 4) : i + 1]) for i in range(len(returns))]
    np.testing.assert_allclose(metrics["reward_window_mean"], means)
    # A full window is required. A single good rollout is insufficient; the
    # gate opens on rollout 7 and closes again on rollout 10.
    np.testing.assert_array_equal(metrics["updated"], [False] * 6 + [True] * 3 + [False] * 4)


def test_first_adaptive_decision_includes_warmup_history():
    _, metrics = _schedule([-3000.0] * 196 + [-1000.0] * 4 + [-5000.0])
    assert metrics["reward_window_mean"][-1] == -1800.0
    assert metrics["gate_open"][-1] == 1


def test_force_update_after_50_closed_rollouts_then_reset_counter():
    state, metrics = _schedule([-3000.0] * 350)
    np.testing.assert_array_equal(np.flatnonzero(metrics["forced_update"]) + 1, [250, 300, 350])
    np.testing.assert_array_equal(np.flatnonzero(metrics["updated"]) + 1, list(range(4, 201, 4)) + [250, 300, 350])
    assert metrics["frozen_rollouts"][199] == 0  # Warmup skips do not count.
    assert metrics["frozen_rollouts"][248] == 49
    assert metrics["frozen_rollouts"][249] == 0
    assert metrics["gate_open"].sum() == 0  # Forced updates do not latch the gate open.
    assert state.frozen_rollouts == 0


def test_gate_opening_resets_freeze_counter():
    _, metrics = _schedule(
        [-3000.0, -3000.0, -1000.0, -3000.0, -3000.0, -3000.0],
        phase_switch_blue_updates=0,
        window_blue_updates=1,
        max_frozen_blue_rollouts=3,
    )
    np.testing.assert_array_equal(metrics["updated"], [False, False, True, False, False, True])
    np.testing.assert_array_equal(metrics["forced_update"], [0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(metrics["frozen_rollouts"], [1, 2, 0, 1, 2, 0])


def _write_recipe(tmp_path, config):
    recipe = load("cotraining")
    recipe["train"]["update_every"] = {"blue": 1, "red": 4}
    recipe["train"]["adaptive_updates"] = config
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path, recipe


@pytest.mark.parametrize(
    "config",
    [
        True,
        [],
        5,
        {"unknown": 1},
        {"phase_switch_blue_updates": -1},
        {"phase_switch_blue_updates": True},
        {"phase_switch_blue_updates": 2.5},
        {"window_blue_updates": 0},
        {"window_blue_updates": "5"},
        {"window_blue_updates": False},
        {"max_frozen_blue_rollouts": 0},
        {"max_frozen_blue_rollouts": 1.5},
        {"reward_threshold": "-2000"},
        {"reward_threshold": True},
        {"reward_threshold": float("nan")},
        {"reward_threshold": float("inf")},
    ],
)
def test_invalid_adaptive_configuration_is_rejected(tmp_path, config):
    path, _ = _write_recipe(tmp_path, config)
    with pytest.raises(ValueError, match="train.adaptive_updates"):
        load(str(path))


@pytest.mark.parametrize("incompatible", ["single_team", "blue_interval", "red_annealing"])
def test_adaptive_recipe_rejects_incompatible_training_settings(tmp_path, incompatible):
    path, recipe = _write_recipe(tmp_path, {})
    if incompatible == "single_team":
        recipe["train"].update(teams="blue", update_every={})
    elif incompatible == "blue_interval":
        recipe["train"]["update_every"]["blue"] = 2
    else:
        recipe["train"]["team_overrides"] = {"red": {"core": {"anneal_lr": True}}}
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="train.adaptive_updates"):
        load(str(path))


@pytest.mark.parametrize(
    "base", ["cotraining", "cotraining_env_diversity", "cotraining_lstm", "cotraining_lstm_env_diversity"]
)
def test_adaptive_recipes_preserve_standard_settings_and_project_schedule(base, monkeypatch):
    monkeypatch.setattr("jaxborg.recipe._resolve_topology_bank", lambda *_args, **_kwargs: ())
    baseline = load(base)
    recipe = load(f"cotraining/training_configurations/adaptive_updates/{base}_adaptive_updates")
    assert recipe["meta"]["name"] == f"{base}_adaptive_updates"
    configs = project_team_configs(recipe, "jax")
    for team, interval in (("blue", 1), ("red", 4)):
        assert configs[team]["ADAPTIVE_UPDATES"] == asdict(AdaptiveUpdateSettings())
        assert configs[team]["UPDATE_EVERY"] == interval
        assert configs[team]["LR"] == pytest.approx(baseline["core"]["lr"])
    with pytest.raises(ValueError, match="only by the JAX joint trainer"):
        project_cleanrl(recipe)
    if base.endswith("_env_diversity"):
        control = f"{base.removesuffix('_env_diversity')}_adaptive_updates"
        assert recipe["eval"]["env_diversity"]["baseline_recipe"] == control
        assert load(control)["train"]["topology_generation"]["count"] == 1
        baseline["eval"]["env_diversity"]["baseline_recipe"] = control
    del recipe["train"]["adaptive_updates"], recipe["train"]["update_every"]
    # The adaptive controls already disable cross-seed evaluation; preserve
    # that existing recipe choice independently of their training schedule.
    assert recipe["eval"]["cross_seed_play"]["enabled"] is False
    baseline["eval"]["cross_seed_play"]["enabled"] = False
    for item in (recipe, baseline):
        del item["meta"], item["__source_path__"]
    assert recipe == baseline


def test_population_empty_red_batches_keep_returns_and_defer_forced_update():
    settings = AdaptiveUpdateSettings(
        phase_switch_blue_updates=0, window_blue_updates=1, reward_threshold=-2000, max_frozen_blue_rollouts=2
    )
    state = initial_adaptive_update_state(settings)
    for number, available in enumerate((False, False, False, True), 1):
        updated, state, metrics = adaptive_red_update(
            settings,
            state,
            blue_return=jnp.float32(-3000),
            rollout_number=jnp.int32(number),
            warmup_update_every=4,
            update_available=jnp.asarray(available),
        )
        assert bool(updated) == available
        assert metrics["reward_window_mean"] == -3000
        assert metrics["reward_window_count"] == 1
        assert metrics["frozen_rollouts"] == (0 if available else number)
        assert metrics["forced_update"] == int(available)
