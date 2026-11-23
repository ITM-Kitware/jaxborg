from copy import deepcopy

import jax
import numpy as np
import pytest
import yaml

from jaxborg.opponent_population import OpponentPopulationSettings
from jaxborg.recipe import load, project_cleanrl, project_team_configs


def population_config(preserve=False, learned=75):
    return {
        "enabled": True,
        "preserve_red_batch_size": preserve,
        "blue": [{"opponent": "cotrained", "percentage": learned}, {"opponent": "fsm", "percentage": 100 - learned}],
    }


def test_population_sampling_is_seeded_and_respects_percentages():
    settings = OpponentPopulationSettings.from_config(population_config())
    keys = jax.random.split(jax.random.PRNGKey(13), 10000)
    first = jax.jit(jax.vmap(settings.sample))(keys)
    np.testing.assert_array_equal(first, jax.vmap(settings.sample)(keys))
    assert set(np.asarray(first)) == {0, 1}
    assert np.mean(np.asarray(first) == 0) == pytest.approx(0.75, abs=0.02)
    for percentage, expected in ((0, 1), (100, 0)):
        settings = OpponentPopulationSettings.from_config(population_config(learned=percentage))
        np.testing.assert_array_equal(jax.vmap(settings.sample)(keys[:100]), expected)


@pytest.mark.parametrize(
    "raw",
    [
        [],
        True,
        {"enabled": "true"},
        {"red": []},
        {"blue": []},
        {"blue": "fsm"},
        {"preserve_red_batch_size": 1},
        {"other": True},
        {"blue": [{"opponent": "unknown", "percentage": 100}]},
        {"blue": [{"opponent": [], "percentage": 100}]},
        {"blue": [{"opponent": "fsm", "percentage": 50}, {"opponent": "fsm", "percentage": 50}]},
        {"blue": [{"opponent": "fsm", "percentage": 99}]},
        {"blue": [{"opponent": "fsm", "percentage": -1}]},
        {"blue": [{"opponent": "fsm", "percentage": float("nan")}]},
        {"blue": [{"opponent": "fsm", "percentage": float("inf")}]},
        {"blue": [{"opponent": "fsm", "percentage": True}]},
        {"blue": [{"opponent": "fsm", "percentage": "100"}]},
        {"blue": [{"opponent": "fsm", "percentage": 100, "extra": True}]},
    ],
)
def test_invalid_population_settings_are_rejected(raw):
    with pytest.raises(ValueError, match="opponent_population"):
        OpponentPopulationSettings.from_config(raw)


def test_disabled_population_and_missing_population_are_legacy():
    assert OpponentPopulationSettings.from_config(None) is None
    assert OpponentPopulationSettings.from_config({"enabled": False}) is None
    assert OpponentPopulationSettings.from_config({**population_config(), "enabled": False}) is None


def test_population_requires_joint_jax_training(tmp_path, monkeypatch):
    monkeypatch.setattr("jaxborg.recipe._resolve_topology_bank", lambda *_args, **_kwargs: ())
    recipe = load("cotraining")
    recipe["train"]["opponent_population"] = population_config()
    with pytest.raises(ValueError, match="opponent_population.*JAX joint"):
        project_cleanrl(recipe)
    recipe["train"]["teams"] = "blue"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="opponent_population requires train.teams: both"):
        load(str(path))


@pytest.mark.parametrize(
    "base",
    [
        "cotraining",
        "cotraining_lstm",
        "cotraining_mappo",
        "cotraining_env_diversity",
        "cotraining_lstm_env_diversity",
        "cotraining_mappo_env_diversity",
    ],
)
@pytest.mark.parametrize("split,learned", [("5050", 50), ("7525", 75)])
def test_population_recipe_splits_preserve_training_settings(base, split, learned, monkeypatch):
    monkeypatch.setattr("jaxborg.recipe._resolve_topology_bank", lambda *_args, **_kwargs: ())
    suffix = "pbt" if "mappo" in base else "pbt_adaptive_updates"
    recipe = load(f"{base}_{suffix}_{split}")
    assert recipe["meta"]["name"] == f"{base}_{suffix}_{split}"
    settings = recipe["train"]["opponent_population"]
    assert settings == population_config(learned=learned)
    configs = project_team_configs(recipe, "jax")
    assert configs["blue"]["OPPONENT_POPULATION"] == configs["red"]["OPPONENT_POPULATION"] == settings
    assert configs["blue"]["OPPONENT_POPULATION"] is not settings
    if "lstm" in base or "mappo" in base:
        assert all(cfg["ADAPTIVE_UPDATES"] is None and cfg["UPDATE_EVERY"] == 1 for cfg in configs.values())
    else:
        source = project_team_configs(load(f"{base}_adaptive_updates_separate"), "jax")
        for team in ("blue", "red"):
            for key in ("ADAPTIVE_UPDATES", "UPDATE_EVERY", "LR", "ANNEAL_LR"):
                assert configs[team][key] == source[team][key]
    if base.endswith("_env_diversity"):
        control = f"{base.removesuffix('_env_diversity')}_{suffix}_{split}"
        assert recipe["eval"]["env_diversity"]["baseline_recipe"] == control
        assert load(control)["train"]["topology_generation"]["count"] == 1
    for suite in ("play_priors", "checkpoint_scripted_reds"):
        assert recipe["eval"][suite]["enabled"] is True

    other_split = "7525" if split == "5050" else "5050"
    other = load(f"{base}_{suffix}_{other_split}")
    for item in (recipe, other):
        del item["train"]["opponent_population"]
    # The splits have independently selected evaluation suites, while their
    # training settings differ only in opponent percentages.
    for section in ("algorithm", "cage4_enhanced_obs", "core", "arch", "train", "jax"):
        assert recipe[section] == other[section]


def test_every_scripted_type_is_supported_without_altering_percentages():
    config = deepcopy(population_config())
    config["blue"] = [{"opponent": name, "percentage": 25} for name in ("fsm", "cia_c", "cia_i", "cia_a")]
    assert OpponentPopulationSettings.from_config(config).percentages == (0, 25, 25, 25, 25)
