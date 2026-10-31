from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jaxborg.evaluation.cross_seed_play import (
    CrossSeedPlaySettings,
    run_cross_seed_play,
    validate_cross_seed_models,
    validate_cross_seed_pool,
)
from jaxborg.evaluation.matchup_runner import MatchupEvaluation
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
    model.with_name(f"recipe_seed{seed}.yaml").write_text(
        yaml.safe_dump(
            {
                **recipe,
                "run": {"seed": seed, "total_steps": 400, "train_run_id": f"run-{seed}"},
            }
        )
    )
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
    ]
    + [path.stem for path in sorted((Path(__file__).resolve().parents[2] / "recipes/cotraining/pbt").glob("*.yaml"))],
)
def test_cotraining_pipeline_passes_every_held_out_red_to_existing_evaluator(tmp_path, monkeypatch, name):
    monkeypatch.delenv("JAXBORG_SKIP_POST_TRAINING_EVAL", raising=False)
    recipe = load(name)
    assert CrossSeedPlaySettings.from_recipe(recipe).enabled
    blue = _model(tmp_path, 42, recipe)
    red = _model(tmp_path, 100, recipe)
    other_red = _model(tmp_path, 200, recipe)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    manifest = run_configured_evaluations_after_training(
        blue,
        recipe,
        cross_seed_red=[red, other_red],
        run_subprocess=fake_run,
    )
    names = [kwargs["env"]["JAXBORG_EVAL_NAME"] for _, kwargs in calls]
    assert names[0] == "cross-seed-play"
    assert {"cross_play", "learned-red-ppo", "scripted-reds"} <= set(names)
    cross_calls = [(cmd, kw) for cmd, kw in calls if kw["env"]["JAXBORG_EVAL_NAME"] == "cross-seed-play"]
    assert len(cross_calls) == 1
    command, kwargs = cross_calls[0]
    assert Path(command[1]).name == "eval_cross_seed_only.py"
    assert command[command.index("--model") + 1] == str(blue)
    assert [command[i + 1] for i, arg in enumerate(command) if arg == "--red-model"] == [str(red), str(other_red)]
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
    "setting",
    [
        {"episodes_per_seed": 0},
        {"enabled": "yes"},
        {"max_checkpoints": 0},
        {"max_checkpoints": True},
        {"max_checkpoints": 2.5},
        {"seeds": []},
    ],
)
def test_invalid_cross_seed_settings(setting):
    with pytest.raises(ValueError, match="eval.cross_seed_play"):
        CrossSeedPlaySettings.from_recipe({"eval": {"cross_seed_play": setting}})


def test_duplicate_red_seeds_are_rejected(tmp_path):
    blue = _model(tmp_path, 42, {})
    red = _model(tmp_path, 100, {})
    with pytest.raises(ValueError, match="distinct Red training seeds"):
        validate_cross_seed_pool(blue, [red, red])


@pytest.mark.parametrize("maximum,expected_steps", [(1, [400]), (5, [40, 120, 200, 320, 400])])
def test_blue_history_uses_all_fixed_held_out_reds_and_logs_at_checkpoint_steps(
    tmp_path,
    monkeypatch,
    maximum,
    expected_steps,
):
    from jaxborg.checkpoint import read_sidecar

    recipe = {
        "meta": {"name": "curve-test"},
        "train": {"teams": "both", "episode_length": 10, "variant": "cc4_stock"},
        "jax": {"num_envs": 2, "checkpoint_every_updates": 2},
        "eval": {"cross_seed_play": {"enabled": True, "max_checkpoints": maximum}},
    }
    final = _model(tmp_path, 42, recipe)
    reds = [_model(tmp_path, seed, recipe) for seed in (100, 200)]
    recipe = read_sidecar(final)
    for step in range(40, 481, 40):
        checkpoint = final.with_name(f"checkpoint_{step}.safetensors")
        checkpoint.touch()
        checkpoint.with_name(f"recipe_checkpoint_{step}.yaml").write_text(
            yaml.safe_dump(
                {
                    **recipe,
                    "run": {**recipe["run"], "total_steps": step},
                }
            )
        )
    projection = {
        "EVAL_VARIANT": "cc4_stock",
        "TOPOLOGY_BANK": ["held-out-topology"],
        "TOPOLOGY_SAMPLING": "exhaustive",
        "CIA": {"enabled": True},
    }
    monkeypatch.setattr("jaxborg.recipe.project_eval", lambda *_a, **_kw: projection)
    seen, attached = [], []

    def evaluate(blue, red, **kwargs):
        step = read_sidecar(blue)["run"]["total_steps"]
        seed = read_sidecar(red)["run"]["seed"]
        seen.append((step, seed, blue))
        assert kwargs["seeds"] == list(range(1000, 1010))
        assert kwargs["topology_path"] == ["held-out-topology"]
        assert kwargs["cia"]["enabled"]
        payoff = step - seed
        return MatchupEvaluation(
            blue_returns=[payoff - 1, payoff + 1],
            red_returns=[1 - payoff, -1 - payoff],
            episode_seeds=[1000, 1001],
            policies={"blue": {"train_run_id": "run-42"}},
            cia_summary={"n": 2, **{axis: {"mean": 0.5, "std": 0.1} for axis in ("c", "i", "a")}},
        )

    output = run_cross_seed_play(
        final,
        reds,
        recipe,
        output=tmp_path / "curve.jsonl",
        evaluate_fn=evaluate,
        attach_metrics_fn=lambda run_id, metrics, step: attached.append((run_id, metrics, step)),
    )
    assert [(step, seed) for step, seed, _ in seen] == [(step, seed) for step in expected_steps for seed in (100, 200)]
    assert all(blue == final for step, _, blue in seen if step == 400)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    summaries = [row for row in rows if row["eval_name"] == "cross_seed_play_summary"]
    assert [row["blue_step"] for row in summaries] == expected_steps
    assert [row["mean_reward"] for row in summaries] == [step - 150 for step in expected_steps]
    assert [row["worst_reward"] for row in summaries] == [step - 200 for step in expected_steps]
    assert all(row["red_step"] == 400 for row in rows if "red_step" in row)
    final_rows = [json.loads(path.read_text()) for path in tmp_path.glob("*matchup_cross-seed-play*.jsonl")]
    assert len(final_rows) == 2
    assert {row["red_train_seed"] for row in final_rows} == {100, 200}
    assert all(row["eval_name"] == "cross-seed-play" and row["blue_step"] == 400 for row in final_rows)
    assert attached[-1][1]["eval.after_training.cross-seed-play.jax_matchup.blue_mean"] == 250
    assert [step for _, _, step in attached] == expected_steps
    for run_id, metrics, _ in attached:
        assert run_id == "run-42"
        for seed in (100, 200):
            assert f"eval.cross_seed_play.red_seed_{seed}.blue_mean" in metrics
            assert f"eval.cross_seed_play.red_seed_{seed}.cia.c.mean" in metrics


def test_all_pbt_recipes_use_five_cross_seed_and_cross_play_checkpoints():
    paths = sorted((Path(__file__).resolve().parents[2] / "recipes/cotraining/pbt").glob("*.yaml"))
    assert len(paths) == 6
    for path in paths:
        recipe = load(str(path))
        settings = CrossSeedPlaySettings.from_recipe(recipe)
        assert settings.enabled and settings.max_checkpoints == 5
        assert recipe["eval"]["cross_play"]["max_checkpoints"] == 5
        assert recipe["eval"]["cross_play"]["include_final"] is True


def test_discovery_schedules_all_six_seed_pairs_without_self_play(tmp_path, monkeypatch):
    from jaxborg.checkpoint import read_sidecar
    from scripts.eval import eval_cross_seed_only as cli

    recipe = load("cotraining_mappo_pbt")
    models = {}
    for seed in (42, 100, 200):
        path = _model(tmp_path, seed, recipe)
        models[seed] = SimpleNamespace(path=str(path), steps=400, recipe=read_sidecar(path))
    monkeypatch.setattr(cli, "discover_models", lambda *_a, **_kw: (models, []))
    monkeypatch.setattr(cli, "project_eval", lambda *_a, **_kw: {"EVAL_VARIANT": "cia_resilience"})
    calls = []
    output = tmp_path / "results"
    cli.main(
        ["--recipe", "cotraining_mappo_pbt", "--exp-dir", str(tmp_path), "--output-dir", str(output)],
        run_subprocess=lambda command, **kw: calls.append(command) or SimpleNamespace(returncode=0),
    )
    pairs = []
    for command in calls:
        blue = read_sidecar(command[command.index("--model") + 1])["run"]["seed"]
        for i, arg in enumerate(command):
            if arg == "--red-model":
                pairs.append((blue, read_sidecar(command[i + 1])["run"]["seed"]))
    assert set(pairs) == {(blue, red) for blue in models for red in models if blue != red}
    assert len(pairs) == 6
    assert all(
        row["status"] == "succeeded" for row in json.loads((output / "manifest.json").read_text())["evaluations"]
    )


def test_single_run_cli_preserves_training_provenance_and_applies_eval_override(tmp_path, monkeypatch):
    from scripts.eval import eval_cross_seed_only as cli

    model = _model(tmp_path, 42, {"eval": {"cross_seed_play": False}})
    captured = []
    monkeypatch.setattr(cli, "run_cross_seed_play", lambda *args, **kw: captured.append(args))
    cli.main(
        ["--model", str(model), "--red-model", "red100", "--red-model", "red200", "--recipe", "cotraining_mappo_pbt"]
    )
    _, reds, recipe = captured[0]
    assert reds == ["red100", "red200"]
    assert recipe["run"]["seed"] == 42
    assert recipe["run"]["total_steps"] == 400
    assert recipe["eval"]["cross_seed_play"]["max_checkpoints"] == 5


def test_post_training_cli_passes_repeated_cross_seed_reds(tmp_path, monkeypatch):
    from jaxborg.evaluation import post_training

    model = _model(tmp_path, 42, {})
    captured = []
    monkeypatch.setattr(
        post_training, "run_configured_evaluations_after_training", lambda *a, **kw: captured.append(kw) or "manifest"
    )
    post_training.main(["--model", str(model), "--cross-seed-red", "red100", "--cross-seed-red", "red200"])
    assert captured[0]["cross_seed_red"] == ["red100", "red200"]
