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
    "base", ["cotraining", "cotraining_lstm", "cotraining_env_diversity", "cotraining_lstm_env_diversity"]
)
def test_population_recipes_preserve_adaptive_sources(base, monkeypatch):
    monkeypatch.setattr("jaxborg.recipe._resolve_topology_bank", lambda *_args, **_kwargs: ())
    source = load(f"{base}_adaptive_updates")
    recipe = load(f"{base}_pbt_adaptive_updates")
    assert recipe["meta"]["name"] == f"{base}_pbt_adaptive_updates"
    settings = recipe["train"]["opponent_population"]
    assert settings == population_config()
    configs = project_team_configs(recipe, "jax")
    assert configs["blue"]["OPPONENT_POPULATION"] == configs["red"]["OPPONENT_POPULATION"] == settings
    assert configs["blue"]["OPPONENT_POPULATION"] is not settings
    if base.endswith("_env_diversity"):
        control = f"{base.removesuffix('_env_diversity')}_pbt_adaptive_updates"
        assert recipe["eval"]["env_diversity"]["baseline_recipe"] == control
        assert load(control)["train"]["topology_generation"]["count"] == 1
        source["eval"]["env_diversity"]["baseline_recipe"] = control
    del recipe["train"]["opponent_population"]
    for item in (source, recipe):
        del item["meta"], item["__source_path__"]
    assert recipe == source


def test_every_scripted_type_is_supported_without_altering_percentages():
    config = deepcopy(population_config())
    config["blue"] = [{"opponent": name, "percentage": 25} for name in ("fsm", "cia_c", "cia_i", "cia_a")]
    assert OpponentPopulationSettings.from_config(config).percentages == (0, 25, 25, 25, 25)
