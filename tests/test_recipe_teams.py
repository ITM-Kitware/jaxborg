from pathlib import Path

import pytest
import yaml

from jaxborg.recipe import (
    load,
    project_cleanrl,
    project_eval,
    project_jax,
    project_team_configs,
    resolve_eval_policies,
    resolve_model_ref,
    resolve_train_opponents,
    team_recipe,
    training_teams,
)


def _recipe(*, teams=None):
    train = {
        "episode_length": 10,
        "buffer_size": 80,
        "total_timesteps": 1000,
        "variant": "cc4_stock",
    }
    if teams is not None:
        train["teams"] = teams
    return {
        "meta": {"name": "test"},
        "algorithm": "ippo",
        "core": {"lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95},
        "arch": {"name": "shared", "hidden_dim": 256, "hidden_layers": 2, "activation": "tanh"},
        "train": train,
        "jax": {"num_envs": 2, "num_minibatches": 2},
        "cleanrl": {"num_envs": 2, "num_rollouts_per_update": 1, "num_minibatches": 2},
    }


def _write_recipe(tmp_path: Path, recipe: dict, name: str = "recipe.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    return path


def test_missing_train_teams_defaults_to_blue(tmp_path):
    recipe = load(str(_write_recipe(tmp_path, _recipe())))
    assert training_teams(recipe) == ("blue",)
    assert project_jax(recipe)["TRAIN_TEAM"] == "blue"
    assert project_cleanrl(recipe)["train_team"] == "blue"


def test_team_overrides_are_deep_merged_and_projected(tmp_path):
    raw = _recipe(teams="both")
    raw["core"]["optimizer"] = {"eps": 1e-5, "nested": {"one": 1, "two": 2}}
    raw["train"]["team_overrides"] = {
        "red": {
            "core": {"lr": 1e-4, "optimizer": {"nested": {"two": 20}}},
            "arch": {"hidden_dim": 512},
        }
    }
    recipe = load(str(_write_recipe(tmp_path, raw)))

    red_recipe = team_recipe(recipe, "red")
    assert red_recipe["core"]["optimizer"] == {"eps": 1e-5, "nested": {"one": 1, "two": 20}}
    assert recipe["core"]["optimizer"]["nested"]["two"] == 2

    jax_configs = project_team_configs(recipe, "jax")
    assert set(jax_configs) == {"blue", "red"}
    assert jax_configs["blue"]["LR"] == pytest.approx(3e-4)
    assert jax_configs["blue"]["HIDDEN_DIM"] == 256
    assert jax_configs["red"]["LR"] == pytest.approx(1e-4)
    assert jax_configs["red"]["HIDDEN_DIM"] == 512
    assert jax_configs["red"]["TRAIN_TEAMS"] == ("blue", "red")

    torch_configs = project_team_configs(recipe, "torch")
    assert torch_configs["red"]["lr"] == pytest.approx(1e-4)
    assert torch_configs["red"]["hidden_dim"] == 512


@pytest.mark.parametrize(
    ("mode", "opponents", "message"),
    [
        ("red", None, "requires train.opponents.blue"),
        ("both", {"red": {"experiment": "red_base"}}, "not allowed"),
        ("blue", {"blue": {"experiment": "blue_base"}}, "may only configure train.opponents.red"),
    ],
)
def test_opponent_rules_are_validated(tmp_path, mode, opponents, message):
    raw = _recipe(teams=mode)
    if opponents is not None:
        raw["train"]["opponents"] = opponents
    with pytest.raises(ValueError, match=message):
        load(str(_write_recipe(tmp_path, raw)))


def test_red_only_requires_and_projects_frozen_blue_ref(tmp_path):
    raw = _recipe(teams="red")
    raw["train"]["opponents"] = {"blue": {"experiment": "blue_seed42"}}
    recipe = load(str(_write_recipe(tmp_path, raw)))
    assert training_teams(recipe) == ("red",)
    assert project_jax(recipe)["TRAIN_TEAM"] == "red"
    assert project_cleanrl(recipe)["train_team"] == "red"

    expected = tmp_path / "exp" / "ippo_jax" / "blue_seed42" / "model_blue_seed42.safetensors"
    assert resolve_train_opponents(recipe, backend="jax", exp_dir=tmp_path / "exp", must_exist=False) == {
        "blue": expected.resolve()
    }


def test_model_ref_path_wins_and_is_recipe_relative(tmp_path):
    recipe_path = _write_recipe(tmp_path, _recipe())
    recipe = load(str(recipe_path))
    ref = {"path": "models/custom.safetensors", "experiment": "ignored"}
    assert (
        resolve_model_ref(ref, backend="jax", recipe=recipe, must_exist=False)
        == (tmp_path / "models/custom.safetensors").resolve()
    )


def test_model_ref_experiment_uses_recipe_algorithm(tmp_path):
    raw = _recipe()
    raw["algorithm"] = "mappo"
    recipe = load(str(_write_recipe(tmp_path, raw)))

    assert (
        resolve_model_ref(
            {"experiment": "baseline"},
            backend="cyborg",
            recipe=recipe,
            exp_dir=tmp_path / "exp",
            must_exist=False,
        )
        == (tmp_path / "exp" / "mappo_cyborg" / "baseline" / "model_baseline.pt").resolve()
    )


def test_model_ref_experiment_without_recipe_keeps_ippo_default(tmp_path):
    assert (
        resolve_model_ref(
            {"experiment": "legacy"},
            backend="jax",
            exp_dir=tmp_path / "exp",
            must_exist=False,
        )
        == (tmp_path / "exp" / "ippo_jax" / "legacy" / "model_legacy.safetensors").resolve()
    )


def test_model_ref_enforces_backend_by_metadata_and_suffix(tmp_path):
    recipe = load(str(_write_recipe(tmp_path, _recipe())))
    with pytest.raises(ValueError, match="does not match selected backend"):
        resolve_model_ref({"experiment": "x", "backend": "cyborg"}, backend="jax", recipe=recipe)
    with pytest.raises(ValueError, match="not compatible"):
        resolve_model_ref("model.pt", backend="jax", recipe=recipe, must_exist=False)


def test_eval_policy_sources_project_and_resolve(tmp_path):
    raw = _recipe()
    raw["eval"] = {
        "variant": "cc4_stock",
        "policy_backend": "jax",
        "policies": {
            "blue": {"experiment": "blue_run"},
            "red": {"path": "models/red.safetensors"},
        },
    }
    recipe = load(str(_write_recipe(tmp_path, raw)))
    projected = project_eval(recipe)
    assert projected["policy_backend"] == "jax"
    assert set(projected["policies"]) == {"blue", "red"}

    paths = resolve_eval_policies(recipe, exp_dir=tmp_path / "exp", must_exist=False)
    assert paths["blue"] == (tmp_path / "exp" / "ippo_jax" / "blue_run" / "model_blue_run.safetensors").resolve()
    assert paths["red"] == (tmp_path / "models/red.safetensors").resolve()


@pytest.mark.parametrize(
    ("eval_cfg", "message"),
    [
        ({"policies": {"blue": {"experiment": "blue"}}}, "policy_backend is required"),
        (
            {"policy_backend": "jax", "policies": {"blue": {"experiment": "blue"}}},
            "requires both blue and red",
        ),
        (
            {
                "policy_backend": "jax",
                "policies": {
                    "blue": {"experiment": "blue", "backend": "cyborg"},
                    "red": {"experiment": "red"},
                },
            },
            "does not match selected backend",
        ),
    ],
)
def test_eval_policy_config_validation(tmp_path, eval_cfg, message):
    raw = _recipe()
    raw["eval"] = eval_cfg
    with pytest.raises(ValueError, match=message):
        load(str(_write_recipe(tmp_path, raw)))
