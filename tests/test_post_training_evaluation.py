from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jaxborg.evaluation import post_training
from jaxborg.evaluation.post_training import (
    PostTrainingEvalSettings,
    run_configured_evaluations_after_training,
)
from jaxborg.recipe import load


def _recipe(evaluations=None) -> dict:
    return {
        "meta": {"name": "multi-eval-test"},
        "algorithm": "ippo",
        "core": {"lr": 3e-4},
        "arch": {"name": "shared"},
        "train": {
            "teams": "both",
            "episode_length": 10,
            "buffer_size": 20,
            "total_timesteps": 100,
            "variant": "cc4_stock",
        },
        "eval": {"variant": "cc4_stock", "after_training": evaluations or []},
    }


def _final_model(tmp_path: Path) -> Path:
    run_dir = tmp_path / "exp" / "ippo_jax" / "run"
    run_dir.mkdir(parents=True)
    model = run_dir / "model_run.safetensors"
    model.write_bytes(b"model")
    (run_dir / "recipe_run.yaml").write_text("meta:\n  name: multi-eval-test\n")
    return model


def test_settings_preserve_order_and_accept_numeric_cli_arguments(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.touch()
    second.touch()
    settings = PostTrainingEvalSettings.from_recipe(
        _recipe(
            [
                {"name": "stochastic", "script": str(first), "args": ["--episodes-per-seed", 10]},
                {
                    "name": "deterministic",
                    "script": str(second),
                    "args": ["--output", "{eval_dir}/{name}.jsonl"],
                    "model_arg": "--checkpoint",
                    "required": False,
                },
            ]
        )
    )

    assert [evaluation.name for evaluation in settings.evaluations] == ["stochastic", "deterministic"]
    assert settings.evaluations[0].args == ("--episodes-per-seed", "10")
    assert settings.evaluations[1].model_arg == "--checkpoint"
    assert settings.evaluations[1].required is False


@pytest.mark.parametrize(
    ("evaluations", "message"),
    [
        ({"name": "bad"}, "must be a list"),
        ([{"name": "missing-script"}], "missing required"),
        (
            [
                {"name": "same", "script": "one.py"},
                {"name": "same", "script": "two.py"},
            ],
            "names must be unique",
        ),
        ([{"name": "bad name", "script": "one.py"}], "contain only"),
        ([{"name": "bad", "script": "one.py", "args": ["{unknown}"]}], "unknown placeholders"),
    ],
)
def test_settings_reject_invalid_pipelines(evaluations, message):
    with pytest.raises(ValueError, match=message):
        PostTrainingEvalSettings.from_recipe(_recipe(evaluations))


def test_recipe_load_fails_before_training_when_evaluation_script_is_missing(tmp_path):
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            _recipe([{"name": "missing", "script": str(tmp_path / "does-not-exist.py")}]),
            sort_keys=False,
        )
    )

    with pytest.raises(FileNotFoundError, match="evaluation script not found"):
        load(str(recipe_path))


def test_runs_scripts_in_order_with_exact_model_and_writes_manifest(tmp_path, monkeypatch):
    model = _final_model(tmp_path)
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.touch()
    second.touch()
    recipe = _recipe(
        [
            {"name": "first-way", "script": str(first), "args": ["--episodes-per-seed", 2]},
            {
                "name": "second-way",
                "script": str(second),
                "model_arg": "--checkpoint",
                "args": ["--output", "{eval_dir}/{name}.jsonl", "--recipe", "{recipe}"],
            },
        ]
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(post_training.time, "time_ns", lambda: 123)
    manifest_path = run_configured_evaluations_after_training(model, recipe, run_subprocess=fake_run)

    assert manifest_path is not None
    assert [Path(call[0][1]).name for call in calls] == ["first.py", "second.py"]
    assert calls[0][0][2:] == ["--model", str(model.resolve()), "--episodes-per-seed", "2"]
    assert calls[1][0][2:4] == ["--checkpoint", str(model.resolve())]
    assert calls[1][0][-2:] == ["--recipe", str(model.with_name("recipe_run.yaml").resolve())]
    assert calls[0][1]["check"] is True
    assert calls[0][1]["env"]["JAX_PLATFORMS"] == "cpu"
    assert calls[0][1]["env"]["JAXBORG_EVAL_NAME"] == "first-way"
    assert calls[1][1]["env"]["JAXBORG_EVAL_NAME"] == "second-way"
    assert calls[0][1]["env"]["JAXBORG_TRAINED_BACKEND"] == "jax"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["model"] == str(model.resolve())
    assert manifest["backend"] == "jax"
    assert [entry["name"] for entry in manifest["evaluations"]] == ["first-way", "second-way"]
    assert [entry["status"] for entry in manifest["evaluations"]] == ["succeeded", "succeeded"]


def test_optional_failure_is_recorded_and_does_not_stop_later_scripts(tmp_path):
    model = _final_model(tmp_path)
    script = tmp_path / "eval.py"
    script.touch()
    recipe = _recipe(
        [
            {"name": "allowed-to-fail", "script": str(script), "required": False},
            {"name": "still-runs", "script": str(script)},
        ]
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(7, command)
        return SimpleNamespace(returncode=0)

    manifest_path = run_configured_evaluations_after_training(model, recipe, run_subprocess=fake_run)
    manifest = json.loads(manifest_path.read_text())

    assert len(calls) == 2
    assert manifest["evaluations"][0]["status"] == "failed"
    assert manifest["evaluations"][0]["returncode"] == 7
    assert manifest["evaluations"][1]["status"] == "succeeded"


def test_cotraining_pipeline_uses_joint_bundle_then_scripted_reds(tmp_path):
    model = _final_model(tmp_path)
    recipe = load("cotraining")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    run_configured_evaluations_after_training(model, recipe, run_subprocess=fake_run)

    learned, scripted = calls
    assert Path(learned[1]).name == "eval_matchup.py"
    assert learned[learned.index("--policy-backend") + 1] == "jax"
    assert learned[learned.index("--blue-path") + 1] == str(model.resolve())
    assert learned[learned.index("--red-path") + 1] == str(model.resolve())
    assert Path(scripted[1]).name == "eval_scripted_reds.py"
    assert scripted[scripted.index("--model") + 1] == str(model.resolve())
    assert scripted[scripted.index("--reds") + 1 : scripted.index("--seeds")] == [
        "fsm",
        "cia_c",
        "cia_i",
        "cia_a",
    ]


def test_absent_pipeline_delegates_to_legacy_scripted_red_hook(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "jaxborg.evaluation.scripted_red.run_configured_after_training",
        lambda model, recipe, **kwargs: calls.append((model, recipe, kwargs)),
    )
    recipe = _recipe()

    result = run_configured_evaluations_after_training("model.pt", recipe)

    assert result is None
    assert calls[0][0] == "model.pt"
