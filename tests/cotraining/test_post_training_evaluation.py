from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jaxborg.evaluation import post_training
from jaxborg.evaluation.checkpoint_scripted_reds import CheckpointScriptedRedsSettings
from jaxborg.evaluation.post_training import (
    PostTrainingEvalSettings,
    run_configured_evaluations_after_training,
)
from jaxborg.recipe import RECIPES_DIR, load


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
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
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
    assert calls[0][1]["env"]["JAX_PLATFORMS"] == "cuda"
    assert calls[0][1]["env"]["JAXBORG_EVAL_NAME"] == "first-way"
    assert calls[1][1]["env"]["JAXBORG_EVAL_NAME"] == "second-way"
    assert calls[0][1]["env"]["JAXBORG_TRAINED_BACKEND"] == "jax"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["model"] == str(model.resolve())
    assert manifest["backend"] == "jax"
    assert manifest["jax_platforms"] == "cuda"
    assert [entry["name"] for entry in manifest["evaluations"]] == ["first-way", "second-way"]
    assert [entry["status"] for entry in manifest["evaluations"]] == ["succeeded", "succeeded"]


def test_explicit_jax_platform_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    model = _final_model(tmp_path)
    script = tmp_path / "eval.py"
    script.touch()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    manifest_path = run_configured_evaluations_after_training(
        model,
        _recipe([{"name": "one", "script": str(script)}]),
        run_subprocess=fake_run,
    )

    assert calls[0][1]["env"]["JAX_PLATFORMS"] == "cpu"
    assert json.loads(manifest_path.read_text())["jax_platforms"] == "cpu"


def test_evaluation_override_is_archived_and_forwarded_to_children(tmp_path):
    model = _final_model(tmp_path)
    original_sidecar = model.with_name("recipe_run.yaml")
    before = original_sidecar.read_bytes()
    recipe = load("cotraining")
    recipe["run"] = {"seed": 42, "train_run_id": "original-run"}
    calls = []
    manifest_path = run_configured_evaluations_after_training(
        model,
        recipe,
        save_evaluation_recipe=True,
        run_subprocess=lambda command, **kwargs: calls.append((command, kwargs)),
    )
    effective = manifest_path.with_suffix(".yaml")
    saved = yaml.safe_load(effective.read_text())
    assert saved["eval"]["topology_generation"]["count"] == 10
    assert saved["eval"]["cross_play"]["max_checkpoints"] == 3
    assert saved["run"]["train_run_id"] == "original-run"
    assert json.loads(manifest_path.read_text())["recipe"] == str(effective)
    for command, kwargs in calls:
        if Path(command[1]).name == "eval_scripted_reds.py":
            # Native benchmark takes its observation contract from the checkpoint
            # sidecar and always uses stock topologies, regardless of eval overrides.
            assert "--recipe" not in command
        else:
            assert command[command.index("--recipe") + 1] == str(effective)
        assert kwargs["env"]["JAXBORG_RECIPE_PATH"] == str(effective)
    assert original_sidecar.read_bytes() == before


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


@pytest.mark.parametrize(
    "recipe_name",
    sorted(path.stem for path in (RECIPES_DIR / "cotraining").glob("cotraining*.yaml")),
)
def test_cotraining_pipeline_uses_cross_play_then_final_checks_without_duplicate_priors(tmp_path, recipe_name):
    model = _final_model(tmp_path)
    recipe = load(recipe_name)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    run_configured_evaluations_after_training(model, recipe, run_subprocess=fake_run)

    # Historical cross-play runs first, then any scripted checkpoint curve; priors are off.
    # Cross-seed play needs an explicit opponent, so it is skipped here.
    cross_play, *calls = calls
    if CheckpointScriptedRedsSettings.from_recipe(recipe).enabled:
        curve, *calls = calls
        assert Path(curve[1]).name == "eval_checkpoint_scripted_reds.py"
    benchmark, learned, scripted, *extra = calls
    if any(job["name"] == "hmarl-reds" for job in recipe["eval"]["after_training"]):
        assert len(extra) == 1 and Path(extra[0][1]).name == "eval_hmarl_reds.py"
    else:
        assert not extra
    assert Path(benchmark[1]).name == "eval_scripted_reds.py"
    assert benchmark[benchmark.index("--seeds") + 1] == "1000-1099"
    assert benchmark[benchmark.index("--reds") + 1] == "fsm"
    assert Path(cross_play[1]).name == "eval_cross_play.py"
    assert cross_play[cross_play.index("--model") + 1] == str(model.resolve())
    assert cross_play[cross_play.index("--recipe") + 1] == str(model.with_name("recipe_run.yaml").resolve())
    assert Path(learned[1]).name == "eval_matchup.py"
    assert learned[learned.index("--policy-backend") + 1] == "jax"
    assert learned[learned.index("--blue-path") + 1] == str(model.resolve())
    assert learned[learned.index("--red-path") + 1] == str(model.resolve())
    assert Path(scripted[1]).name == "eval_scripted_reds_jax.py"
    assert scripted[scripted.index("--model") + 1] == str(model.resolve())
    assert scripted[scripted.index("--recipe") + 1] == str(model.with_name("recipe_run.yaml").resolve())
    assert scripted[scripted.index("--reds") + 1 : scripted.index("--seeds")] == [
        "fsm",
        "cia_c",
        "cia_i",
        "cia_a",
    ]


def test_builtin_history_order_is_independent_of_yaml_key_order(tmp_path):
    model = _final_model(tmp_path)
    recipe = load("cotraining_rnn")
    recipe["eval"]["checkpoint_scripted_reds"]["enabled"] = True
    # Deliberately reverse the configured order; runtime order is canonical.
    checkpoint_suite = recipe["eval"].pop("checkpoint_scripted_reds")
    recipe["eval"] = {"checkpoint_scripted_reds": checkpoint_suite, **recipe["eval"]}
    assert list(recipe["eval"]).index("checkpoint_scripted_reds") < list(recipe["eval"]).index("cross_play")
    calls = []
    run_configured_evaluations_after_training(model, recipe, run_subprocess=lambda cmd, **kwargs: calls.append(cmd))
    assert [Path(cmd[1]).name for cmd in calls] == [
        "eval_cross_play.py",
        "eval_checkpoint_scripted_reds.py",
        "eval_scripted_reds.py",
        "eval_matchup.py",
        "eval_scripted_reds_jax.py",
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


def test_native_benchmark_can_use_cpu_after_gpu_training(tmp_path, monkeypatch):
    model = _final_model(tmp_path)
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.delenv("JAXBORG_SKIP_POST_TRAIN_EVAL", raising=False)
    recipe = _recipe(
        [
            {
                "name": "cage4-benchmark",
                "script": "scripts/eval/eval_scripted_reds.py",
                "jax_platforms": "cpu",
                "args": ["--reds", "fsm", "--seeds", "1000-1099"],
            }
        ]
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["env"]["JAX_PLATFORMS"]))
        return SimpleNamespace(returncode=0)

    run_configured_evaluations_after_training(model, recipe, run_subprocess=fake_run)
    assert len(calls) == 1
    assert calls[0][1] == "cpu"
    assert calls[0][0].count("--model") == 1


@pytest.mark.parametrize("launcher_platform", [None, "cuda"])
def test_comparison_checkpoint_curves_run_on_gpu(tmp_path, monkeypatch, launcher_platform):
    model = _final_model(tmp_path)
    if launcher_platform is None:
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    else:
        monkeypatch.setenv("JAX_PLATFORMS", launcher_platform)
    monkeypatch.delenv("JAXBORG_SKIP_POST_TRAINING_EVAL", raising=False)
    platforms = {}

    def fake_run(command, **kwargs):
        platforms[Path(command[1]).name] = kwargs["env"]["JAX_PLATFORMS"]
        return SimpleNamespace(returncode=0)

    run_configured_evaluations_after_training(model, load("cotraining_lstm"), run_subprocess=fake_run)
    assert platforms["eval_checkpoint_scripted_reds.py"] == "cuda"
    assert platforms["eval_scripted_reds.py"] == "cpu"  # CybORG benchmark stays on CPU.


@pytest.mark.parametrize("base", ["cotraining", "cotraining_lstm", "cotraining_mappo"])
def test_diverse_recipe_evaluates_exact_nondiverse_counterpart(tmp_path, base):
    model = _final_model(tmp_path)
    recipe = load(base + "_env_diversity")
    recipe["run"] = {"seed": 42}
    red = model.with_name("model_baseline.safetensors")
    red.touch()
    saved = load(base)
    saved["run"] = {"seed": 42}
    red.with_name("recipe_baseline.yaml").write_text(yaml.safe_dump(saved))
    calls = []
    manifest_path = run_configured_evaluations_after_training(
        model,
        recipe,
        nondiverse_red=red,
        run_subprocess=lambda cmd, **kw: calls.append(cmd),
    )
    command = next(cmd for cmd in calls if "nondiverse-red" in cmd)
    assert command[command.index("--blue-path") + 1] == str(model)
    assert command[command.index("--red-path") + 1] == str(red)
    assert command[command.index("--seeds") + 1] == "1000-1009"
    manifest = json.loads(manifest_path.read_text())
    assert all(job["status"] == "succeeded" and job["required"] for job in manifest["evaluations"])


@pytest.mark.parametrize("wrong_seed,wrong_recipe", [(100, False), (42, True)])
def test_nondiverse_counterpart_rejects_wrong_seed_or_condition(tmp_path, wrong_seed, wrong_recipe):
    model = _final_model(tmp_path)
    recipe = load("cotraining_env_diversity")
    recipe["run"] = {"seed": 42}
    red = model.with_name("model_baseline.safetensors")
    red.touch()
    saved = load("cotraining_env_diversity" if wrong_recipe else "cotraining")
    saved["run"] = {"seed": wrong_seed}
    red.with_name("recipe_baseline.yaml").write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError, match="configured baseline.*same seed"):
        run_configured_evaluations_after_training(
            model, recipe, nondiverse_red=red, run_subprocess=lambda *a, **kw: pytest.fail("must fail before work")
        )


def test_cli_passes_counterpart_and_preserves_saved_training_settings(tmp_path, monkeypatch):
    model = _final_model(tmp_path)
    saved = load("cotraining_lstm_env_diversity")
    saved["run"] = {"seed": 42, "total_steps": 49_968_000}
    saved["jax"]["num_minibatches"] = 8  # Eval override must never rewrite training provenance.
    saved["eval"] = {}
    model.with_name("recipe_run.yaml").write_text(yaml.safe_dump(saved))
    captured = []
    monkeypatch.setattr(
        post_training,
        "run_configured_evaluations_after_training",
        lambda *args, **kwargs: captured.append((args, kwargs)) or Path("manifest.json"),
    )
    post_training.main(
        ["--model", str(model), "--recipe", "cotraining_lstm_env_diversity", "--nondiverse-red", "baseline.safetensors"]
    )
    (actual_model, recipe), kwargs = captured[0]
    assert actual_model == str(model)
    assert recipe["jax"]["num_minibatches"] == 8
    assert recipe["run"] == saved["run"]
    assert any(job["name"] == "nondiverse-red" for job in recipe["eval"]["after_training"])
    assert kwargs["nondiverse_red"] == "baseline.safetensors"
    assert kwargs["save_evaluation_recipe"] is True
    assert yaml.safe_load(model.with_name("recipe_run.yaml").read_text())["eval"] == {}
