from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jaxborg.evaluation.cross_seed_play import CrossSeedPlaySettings, validate_cross_seed_models
from jaxborg.evaluation.post_training import run_configured_evaluations_after_training
from jaxborg.recipe import load


def test_config_all_documents_valid_cross_seed_settings_without_enabling_them():
    settings = CrossSeedPlaySettings.from_recipe(load("config_all"))
    assert settings == CrossSeedPlaySettings()


def _model(tmp_path, seed, recipe):
    directory = tmp_path / f"{recipe.get('algorithm', 'mappo')}_jax" / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / f"model_seed{seed}.safetensors"
    model.touch()
    model.with_name(f"recipe_seed{seed}.yaml").write_text(yaml.safe_dump({**recipe, "run": {"seed": seed}}))
    return model


@pytest.mark.parametrize(
    "name",
    [
        f"{family}{suffix}"
        for family in (
            "cotraining",
            "cotraining_rnn",
            "cotraining_lstm",
            "cotraining_mappo",
            "cotraining_mappo_joint_obs",
        )
        for suffix in ("", "_env_diversity")
    ],
)
def test_cotraining_pipelines_match_mappo_and_use_one_final_cross_seed_opponent(tmp_path, monkeypatch, name):
    monkeypatch.delenv("JAXBORG_SKIP_POST_TRAINING_EVAL", raising=False)
    recipe = load(name)
    assert recipe["eval"] == load("cotraining_mappo_joint_obs")["eval"]
    blue = _model(tmp_path, 42, recipe)
    red = _model(tmp_path, 100, recipe)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    manifest = run_configured_evaluations_after_training(blue, recipe, cross_seed_red=red, run_subprocess=fake_run)
    assert [kwargs["env"]["JAXBORG_EVAL_NAME"] for _, kwargs in calls] == [
        "cross-seed-play",
        "cross_play",
        "learned-red-ppo",
        "scripted-reds",
    ]
    cross_calls = [(cmd, kw) for cmd, kw in calls if kw["env"]["JAXBORG_EVAL_NAME"] == "cross-seed-play"]
    assert len(cross_calls) == 1
    command, kwargs = cross_calls[0]
    assert Path(command[1]).name == "eval_matchup.py"
    for flag, expected in {
        "--blue-path": str(blue),
        "--red-path": str(red),
        "--name": "cross-seed-play",
        "--seeds": ",".join(map(str, range(1000, 1010))),
        "--episodes-per-seed": "1",
        "--mlflow-source-team": "blue",
        "--policy-backend": "jax",
    }.items():
        assert command[command.index(flag) + 1] == expected
    assert "--deterministic" not in command
    sidecar = yaml.safe_load(Path(command[command.index("--recipe") + 1]).read_text())
    assert sidecar["eval"]["cia"]["enabled"] is True
    assert sidecar["eval"]["topology_generation"]["count"] == 10
    record = json.loads(manifest.read_text())["evaluations"][0]
    assert record["name"] == "cross-seed-play"
    assert record["status"] == "succeeded"


def test_no_opponent_skips_cross_seed_but_keeps_other_evaluations(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("JAXBORG_SKIP_POST_TRAINING_EVAL", raising=False)
    recipe = load("cotraining_mappo")
    blue = _model(tmp_path, 42, recipe)
    calls = []
    run_configured_evaluations_after_training(blue, recipe, run_subprocess=lambda cmd, **kw: calls.append(cmd))
    assert "Skipping cross-seed-play" in capsys.readouterr().out
    assert calls
    assert all("cross-seed-play" not in cmd for cmd in calls)


@pytest.mark.parametrize("problem", ["same_file", "same_seed", "missing", "checkpoint", "unknown_seed"])
def test_invalid_cross_seed_opponents_are_rejected(tmp_path, problem):
    blue = _model(tmp_path, 42, {})
    red = _model(tmp_path, 100, {})
    if problem == "same_file":
        red = blue
    elif problem in ("same_seed", "unknown_seed"):
        red.with_name("recipe_seed100.yaml").write_text("run: {seed: 42}" if problem == "same_seed" else "{}")
    elif problem == "missing":
        red = red.with_name("model_missing.safetensors")
    else:
        red = red.rename(red.with_name("checkpoint_123.safetensors"))
    with pytest.raises((ValueError, FileNotFoundError), match="cross-seed-play"):
        validate_cross_seed_models(blue, red)


@pytest.mark.parametrize(
    "setting", [{"episodes_per_seed": 0}, {"enabled": "yes"}, {"max_checkpoints": 3}, {"seeds": []}]
)
def test_invalid_cross_seed_settings(setting):
    with pytest.raises(ValueError, match="eval.cross_seed_play"):
        CrossSeedPlaySettings.from_recipe({"eval": {"cross_seed_play": setting}})
