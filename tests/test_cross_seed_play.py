from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jaxborg.evaluation.cross_seed_play import CrossSeedPlaySettings, validate_cross_seed_models
from jaxborg.evaluation.post_training import run_configured_evaluations_after_training
from jaxborg.recipe import load

ROOT = Path(__file__).resolve().parents[1]


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


@pytest.mark.parametrize("source_team,expected_runs", [("blue", ["blue-run"]), (None, ["blue-run", "red-run"])])
def test_matchup_records_reward_and_cia_without_overwriting_other_blue_runs(
    tmp_path, monkeypatch, source_team, expected_runs
):
    # This tests CLI reporting; GPU rollout and the MLflow server are replaced.
    # Load a fresh script so these dependencies cannot leak into other tests.
    spec = importlib.util.spec_from_file_location("cross_seed_matchup_cli", ROOT / "scripts/eval/eval_matchup.py")
    cli = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "jaxborg.evaluation.matchup_runner", SimpleNamespace(evaluate_matchup=None))
        imports.setitem(sys.modules, "jaxborg.mlflow_setup", SimpleNamespace(attach_eval_metrics=None))
        spec.loader.exec_module(cli)

    recipe = load("cotraining_mappo")
    output = tmp_path / "result.json"
    args = [
        "eval_matchup.py",
        "--recipe",
        "cotraining_mappo",
        "--policy-backend",
        "jax",
        "--blue-path",
        "blue",
        "--red-path",
        "red",
        "--name",
        "cross-seed-play",
        "--seeds",
        "1000",
        "--episodes-per-seed",
        "1",
        "--output",
        str(output),
    ]
    if source_team:
        args.extend(["--mlflow-source-team", source_team])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(cli, "load", lambda _: recipe)
    monkeypatch.setattr(cli, "resolve_eval_policies", lambda *a, **kw: {"blue": "blue", "red": "red"})
    monkeypatch.setattr(cli, "project_eval", lambda *a, **kw: {"TOPOLOGY_BANK": ["topology"]})
    cia = {"n": 1, **{key: {"mean": -i, "std": 0.0} for i, key in enumerate(("c", "i", "a"), 1)}}
    calls, attached = [], []

    def evaluate(blue, red, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            blue_returns=[-12.0],
            red_returns=[12.0],
            episode_seeds=[1000],
            policies={"blue": {"train_run_id": "blue-run"}, "red": {"train_run_id": "red-run"}},
            topology_sampling="exhaustive",
            topology_paths=["topology"],
            episode_topology_paths=["topology"],
            cia_summary=cia,
            per_episode_cia=[{"c": -1, "i": -2, "a": -3}],
        )

    monkeypatch.setattr(cli, "evaluate_matchup", evaluate)
    monkeypatch.setattr(cli, "attach_eval_metrics", lambda run, metrics: attached.append((run, metrics)))
    cli.main()
    assert calls[0]["cia"]["enabled"] is True
    assert calls[0]["topology_sampling"] == "exhaustive"
    row = json.loads(output.read_text())
    assert row["eval_name"] == "cross-seed-play"
    assert row["blue_mean_return"] == -12
    assert row["red_mean_return"] == 12
    assert row["cia_summary"] == cia
    assert row["per_episode_cia"] == [{"c": -1, "i": -2, "a": -3}]
    assert sorted(run for run, _ in attached) == expected_runs
    prefix = "eval.after_training.cross-seed-play.jax_matchup"
    for _, metrics in attached:
        assert metrics[f"{prefix}.blue_mean"] == -12
        assert metrics[f"{prefix}.blue_std"] == 0
        assert metrics[f"{prefix}.red_mean"] == 12
        assert metrics[f"{prefix}.red_std"] == 0
        for key in ("c", "i", "a"):
            assert metrics[f"{prefix}.cia.{key}.mean"] == cia[key]["mean"]


@pytest.mark.parametrize("failed_seed", ["", "100"])
@pytest.mark.parametrize(
    "launcher,family,algorithm",
    [
        ("run_cotraining.sh", "cotraining_mappo", "mappo"),
        ("run_cotraining_ippo.sh", "cotraining", "ippo"),
        ("run_cotraining_ippo_lstm.sh", "cotraining_lstm", "ippo"),
        ("run_cotraining_mappo_joint_obs.sh", "cotraining_mappo_joint_obs", "mappo"),
    ],
)
def test_launcher_pairs_final_models_cyclically_after_training(tmp_path, failed_seed, launcher, family, algorithm):
    # Exercise the actual Bash launcher without GPU jobs or dependency installs.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if "-c" in args:
    print(os.environ["TEST_ALGORITHM"], "1")
    sys.exit(0)
with open(os.environ["TEST_CALLS"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if "--tag" in args:
    tag = args[args.index("--tag") + 1]
    if args[args.index("--seed") + 1] == os.environ["TEST_FAILED_SEED"]:
        sys.exit(7)
    run = Path(os.environ["JAXBORG_EXP_DIR"]) / (os.environ["TEST_ALGORITHM"] + "_jax") / tag
    run.mkdir(parents=True)
    (run / f"model_{tag}.safetensors").touch()
"""
    )
    uv.chmod(0o755)
    calls_file = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "JAXBORG_REPO_DIR": str(ROOT),
        "JAXBORG_EXP_DIR": str(tmp_path / "exp"),
        "JAXBORG_LOG_DIR": str(tmp_path / "logs"),
        "JAXBORG_SEEDS": "42 100 200",
        "JAXBORG_RECIPES": f"{family} {family}_env_diversity",
        "GPU_COUNT": "2",
        "TEST_CALLS": str(calls_file),
        "TEST_FAILED_SEED": failed_seed,
        "TEST_ALGORITHM": algorithm,
    }
    result = subprocess.run(["bash", str(ROOT / "scripts/train" / launcher)], env=env, capture_output=True, text=True)
    assert result.returncode == (1 if failed_seed else 0), result.stderr
    calls = [json.loads(line) for line in calls_file.read_text().splitlines()]
    assert len(calls[:6]) == 6
    assert all("--tag" in call for call in calls[:6])
    assert {call[call.index("--recipe") + 1] for call in calls[:6]} == {family, f"{family}_env_diversity"}
    evaluations = calls[6:]
    assert len(evaluations) == (4 if failed_seed else 6)
    expected = {"42": "100", "100": "200", "200": "42"}
    for command in evaluations:
        blue = Path(command[command.index("--model") + 1])
        blue_seed = blue.stem.split("_seed")[1].split("-")[0]
        if failed_seed and blue_seed == "42":
            assert "--cross-seed-red" not in command
            continue
        red = Path(command[command.index("--cross-seed-red") + 1])
        assert red.is_file()
        assert red.stem == blue.stem.rsplit("_seed", 1)[0] + f"_seed{expected[blue_seed]}"
    if failed_seed:
        assert "final Red unavailable" in result.stderr
