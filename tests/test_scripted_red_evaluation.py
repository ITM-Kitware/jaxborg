from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from jaxborg.evaluation import scripted_red
from jaxborg.evaluation.post_training import PostTrainingEvalSettings
from jaxborg.evaluation.scripted_red import (
    DEFAULT_SCRIPTED_REDS,
    ScriptedRedEvalSettings,
    attach_results_to_mlflow,
    evaluate_scripted_reds,
    parse_seeds,
    run_configured_after_training,
    write_results,
)
from jaxborg.recipe import load


def _recipe(*, after_training: bool = True) -> dict:
    return {
        "meta": {"name": "cotrain-test"},
        "algorithm": "ippo",
        "core": {"lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95},
        "arch": {"name": "shared"},
        "train": {
            "teams": "both",
            "episode_length": 10,
            "buffer_size": 20,
            "total_timesteps": 100,
            "variant": "cc4_stock",
        },
        "eval": {
            "variant": "cc4_stock",
            "scripted_red": {
                "after_training": after_training,
                "reds": list(DEFAULT_SCRIPTED_REDS),
                "seeds": "40-42",
                "episodes_per_seed": 1,
                "deterministic": False,
                "workers": 2,
            },
        },
        "run": {"train_run_id": "run-123", "total_steps": 100},
    }


def test_cotraining_recipe_runs_learned_then_scripted_red_evaluations():
    recipe = load("cotraining")
    settings = PostTrainingEvalSettings.from_recipe(recipe)

    assert [evaluation.name for evaluation in settings.evaluations] == ["learned-red-ppo", "scripted-reds"]
    learned, scripted = settings.evaluations
    assert Path(learned.script).name == "eval_matchup.py"
    assert learned.model_arg is None
    assert learned.args[learned.args.index("--blue-path") + 1] == "{model}"
    assert learned.args[learned.args.index("--red-path") + 1] == "{model}"
    assert learned.args[learned.args.index("--policy-backend") + 1] == "{backend}"
    assert "--episodes-per-seed" in learned.args
    assert "--episodes" not in learned.args
    assert Path(scripted.script).name == "eval_scripted_reds.py"
    assert scripted.args[scripted.args.index("--reds") + 1 : scripted.args.index("--seeds")] == DEFAULT_SCRIPTED_REDS
    assert ScriptedRedEvalSettings.from_recipe(recipe).after_training is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12, (12,)),
        ("12,14-16,12", (12, 14, 15, 16)),
        ([8, 7, 8], (7, 8)),
    ],
)
def test_parse_seeds(value, expected):
    assert parse_seeds(value) == expected


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("reds", ["fsm", "unknown"], "unsupported agents"),
        ("seeds", "9-3", "ascending"),
        ("episodes_per_seed", 0, "must be positive"),
        ("workers", 0, "must be positive"),
        ("after_training", "yes", "must be a boolean"),
    ],
)
def test_scripted_red_settings_reject_invalid_recipe(setting, value, message):
    recipe = _recipe()
    recipe["eval"]["scripted_red"][setting] = value
    with pytest.raises(ValueError, match=message):
        ScriptedRedEvalSettings.from_recipe(recipe)


def test_recipe_loader_validates_scripted_red_settings(tmp_path):
    recipe = _recipe()
    recipe["eval"]["scripted_red"]["reds"] = ["random"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))

    with pytest.raises(ValueError, match="unsupported agents"):
        load(str(path))


def test_recipe_loader_rejects_automatic_blue_eval_for_red_only_training(tmp_path):
    recipe = _recipe()
    recipe["train"]["teams"] = "red"
    recipe["train"]["opponents"] = {"blue": "frozen-blue.safetensors"}
    path = tmp_path / "red-only.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))

    with pytest.raises(ValueError, match="requires Blue to be trainable"):
        load(str(path))


def test_sweep_uses_exact_model_and_dispatches_blue_to_all_scripted_reds(monkeypatch, tmp_path):
    model = tmp_path / "model_cotrain.safetensors"
    model.write_bytes(b"placeholder")
    recipe = _recipe()
    calls = []

    monkeypatch.setattr(
        scripted_red,
        "_load_trained_blue_contract",
        lambda path, backend: (
            recipe,
            {
                "bundle_schema_version": 1,
                "bundle_legacy": False,
                "bundle_provenance": {"tag": "cotrain"},
                "blue_policy_trainable": True,
                "blue_policy_source": {"kind": "fresh"},
            },
        ),
    )
    monkeypatch.setattr(scripted_red, "_git_commit", lambda: "abc123")

    def fake_cell(backend, model_path, **kwargs):
        calls.append((backend, model_path, kwargs))
        episode_seeds = [seed + ep for seed in kwargs["seeds"] for ep in range(kwargs["episodes_per_seed"])]
        return [float(i) for i in range(len(episode_seeds))], episode_seeds

    rows = evaluate_scripted_reds(
        model,
        seeds="50-51",
        episodes_per_seed=2,
        deterministic=True,
        workers=3,
        cell_evaluator=fake_cell,
    )

    assert len(calls) == 4
    assert [call[2]["variant"].red_agent for call in calls] == ["finite_state", "c", "i", "a"]
    assert [call[2]["variant"].resilience_roles for call in calls] == [False, True, True, True]
    for backend, called_model, kwargs in calls:
        assert backend == "jax"
        assert called_model == model.resolve()
        assert kwargs["seeds"] == [50, 51]
        assert kwargs["episodes_per_seed"] == 2
        assert kwargs["deterministic"] is True
        assert kwargs["workers"] == 3

    assert [row["eval_red"] for row in rows] == list(DEFAULT_SCRIPTED_REDS)
    assert all(row["model"] == str(model.resolve()) for row in rows)
    assert all(row["policy_team"] == "blue" for row in rows)
    assert all(row["n_episodes"] == 4 for row in rows)
    assert all(row["train_run_id"] == "run-123" for row in rows)
    assert len({row["eval_id"] for row in rows}) == 4


def test_write_and_attach_results_use_one_file_and_red_qualified_metrics(monkeypatch, tmp_path):
    rows = [
        {
            "eval_id": f"20260827_120000_123456789_{red}",
            "recipe_name": "cotrain",
            "model": "/models/model_cotrain.safetensors",
            "eval_red": red,
            "mean_reward": float(index),
            "std_reward": 0.5,
            "n_episodes": 10,
            "train_run_id": "run-123",
        }
        for index, red in enumerate(("fsm", "cia_c"), 1)
    ]
    output = write_results(rows, tmp_path / "sweep.jsonl")
    written = [json.loads(line) for line in output.read_text().splitlines()]

    assert [row["eval_red"] for row in written] == ["fsm", "cia_c"]
    captured = {}
    monkeypatch.setattr(
        "jaxborg.mlflow_setup.attach_eval_metrics",
        lambda run_id, metrics: captured.update(run_id=run_id, metrics=metrics),
    )
    attach_results_to_mlflow(rows)

    assert captured["run_id"] == "run-123"
    assert captured["metrics"] == {
        "eval.scripted_red.fsm.blue.mean_reward": 1.0,
        "eval.scripted_red.fsm.blue.std_reward": 0.5,
        "eval.scripted_red.fsm.blue.episodes": 10.0,
        "eval.scripted_red.cia_c.blue.mean_reward": 2.0,
        "eval.scripted_red.cia_c.blue.std_reward": 0.5,
        "eval.scripted_red.cia_c.blue.episodes": 10.0,
    }


def test_named_results_use_distinct_file_and_mlflow_namespaces(monkeypatch, tmp_path):
    rows = [
        {
            "eval_id": "20260827_120000_123456789_fsm",
            "eval_name": "deterministic-reds",
            "recipe_name": "cotrain",
            "model": "/models/model_cotrain.safetensors",
            "eval_red": "fsm",
            "mean_reward": 3.0,
            "std_reward": 0.25,
            "n_episodes": 10,
            "train_run_id": "run-123",
        }
    ]
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(tmp_path))
    output = write_results(rows)
    captured = {}
    monkeypatch.setattr(
        "jaxborg.mlflow_setup.attach_eval_metrics",
        lambda run_id, metrics: captured.update(run_id=run_id, metrics=metrics),
    )

    attach_results_to_mlflow(rows)

    assert "deterministic-reds" in output.name
    assert captured["metrics"] == {
        "eval.after_training.deterministic-reds.scripted_red.fsm.blue.mean_reward": 3.0,
        "eval.after_training.deterministic-reds.scripted_red.fsm.blue.std_reward": 0.25,
        "eval.after_training.deterministic-reds.scripted_red.fsm.blue.episodes": 10.0,
    }


def test_post_training_hook_passes_exact_final_model_to_cpu_child(tmp_path):
    model = tmp_path / "model_custom.pt"
    model.write_bytes(b"placeholder")
    (tmp_path / "recipe_custom.yaml").write_text(yaml.safe_dump(_recipe(), sort_keys=False))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    assert run_configured_after_training(model, _recipe(), run_subprocess=fake_run) is True
    assert len(calls) == 1
    command, kwargs = calls[0]
    model_flag = command.index("--model")
    assert command[model_flag + 1] == str(model.resolve())
    assert command[command.index("--reds") + 1 : command.index("--seeds")] == list(DEFAULT_SCRIPTED_REDS)
    assert command[command.index("--seeds") + 1] == "40,41,42"
    assert kwargs["check"] is True
    assert kwargs["env"]["JAX_PLATFORMS"] == "cpu"
    assert kwargs["env"]["JAXBORG_EXP_DIR"] == str(tmp_path.parents[1])
    assert Path(kwargs["cwd"]) == Path(__file__).resolve().parents[1]


def test_disabled_post_training_hook_does_not_require_or_launch_model():
    assert run_configured_after_training("missing.pt", _recipe(after_training=False)) is False
