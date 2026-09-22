"""Exercise the real shell launchers with fake GPU jobs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from jaxborg.recipe import load

ROOT = Path(__file__).resolve().parents[2]
RECIPES = (
    "cotraining_lstm",
    "cotraining_lstm_env_diversity",
    "cotraining_mappo_lstm",
    "cotraining_mappo_lstm_env_diversity",
)


def launch(tmp_path, *, failed_phase="", disabled=False, recipes=None, launcher="run_cotraining_lstm.sh", gpu_count=2):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys
from pathlib import Path
args = sys.argv[sys.argv.index("python") + 1:]
if "-c" in args:
    os.execv(sys.executable, [sys.executable, *args])
record = {
    "args": args,
    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "skip": os.environ.get("JAXBORG_SKIP_POST_TRAINING_EVAL"),
}
with open(os.environ["TEST_CALLS"], "a") as stream:
    stream.write(json.dumps(record) + "\\n")
phase = "train" if "--tag" in args else "diversity" if "eval_env_diversity.py" in args[0] else "eval"
if phase == os.environ["TEST_FAILED_PHASE"]:
    sys.exit(7)
if phase == "train":
    tag = args[args.index("--tag") + 1]
    algorithm = "mappo" if "mappo" in tag else "ippo"
    run = Path(os.environ["JAXBORG_EXP_DIR"]) / f"{algorithm}_jax" / tag
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
        "CUDA_VISIBLE_DEVICES": "3,5" if gpu_count == 2 else "3,5,7",
        "TEST_CALLS": str(calls_file),
        "TEST_FAILED_PHASE": failed_phase,
    }
    env.pop("GPU_COUNT", None)
    if gpu_count is not None:
        env["GPU_COUNT"] = str(gpu_count)
    env.pop("JAXBORG_RECIPES", None)
    if disabled:
        paths = []
        for name in RECIPES:
            recipe = load(name)
            recipe["eval"]["cross_seed_play"] = False
            recipe["eval"]["env_diversity"] = False
            path = tmp_path / f"{name}.yaml"
            path.write_text(yaml.safe_dump(recipe))
            paths.append(str(path))
        env["JAXBORG_RECIPES"] = " ".join(paths)
    elif recipes:
        env["JAXBORG_RECIPES"] = recipes
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train" / launcher)],
        env=env,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
    return result, calls


def test_lstm_wrapper_trains_both_algorithms_then_runs_both_evaluations(tmp_path):
    result, calls = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 26  # 12 training jobs, 12 post-training suites, two comparisons.
    training, evaluations, comparisons = calls[:12], calls[12:24], calls[24:]
    assert {c["gpu"] for c in training} == {"3", "5"}
    for call in training:
        args = call["args"]
        recipe = args[args.index("--recipe") + 1]
        assert recipe in RECIPES
        assert args[0] == f"scripts/train/algorithms/{load(recipe)['algorithm']}_jax.py"
        assert call["skip"] == "1"
        assert "--total-timesteps" not in args
    cycle = {"42": "100", "100": "200", "200": "42"}
    for call in evaluations:
        args = call["args"]
        blue = Path(args[args.index("--model") + 1])
        red = Path(args[args.index("--cross-seed-red") + 1])
        prefix, seed = blue.stem.rsplit("_seed", 1)
        assert red.stem == f"{prefix}_seed{cycle[seed]}"
        assert blue.is_file() and red.is_file()
        assert call["skip"] == "0"
    for call, recipe in zip(comparisons, RECIPES[1::2], strict=True):
        args = call["args"]
        assert args[0] == "scripts/eval/eval_env_diversity.py"
        assert args[args.index("--recipe") + 1] == recipe
        assert args[args.index("--train-seeds") + 1] == "42,100,200"
        assert "--episodes-per-seed" not in args  # The evaluator reads YAML.


def test_yaml_can_disable_cross_seed_and_diversity_evaluations(tmp_path):
    result, calls = launch(tmp_path, disabled=True)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 24
    assert all("--cross-seed-red" not in c["args"] for c in calls)


@pytest.mark.parametrize("algorithm,family", [("ippo", "cotraining_lstm"), ("mappo", "cotraining_mappo_lstm")])
def test_separate_lstm_wrappers_use_three_gpus_and_evaluate_only_their_family(tmp_path, algorithm, family):
    result, calls = launch(tmp_path, launcher=f"run_cotraining_{algorithm}_lstm.sh", gpu_count=None)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 13  # Six training jobs, six post-training suites, one paired comparison.
    training, evaluations, comparison = calls[:6], calls[6:12], calls[12]["args"]
    expected_recipes = {family, f"{family}_env_diversity"}
    assert {call["gpu"] for call in training} == {"3", "5", "7"}
    jobs = set()
    for call in training:
        args = call["args"]
        assert args[0] == f"scripts/train/algorithms/{algorithm}_jax.py"
        jobs.add((args[args.index("--recipe") + 1], args[args.index("--seed") + 1]))
        assert call["skip"] == "1"
    assert jobs == {(recipe, seed) for recipe in expected_recipes for seed in ("42", "100", "200")}
    cycle = {"42": "100", "100": "200", "200": "42"}
    for call in evaluations:
        args = call["args"]
        assert args[0] == "scripts/eval/run_after_training.py"
        blue = Path(args[args.index("--model") + 1])
        red = Path(args[args.index("--cross-seed-red") + 1])
        prefix, seed = blue.stem.rsplit("_seed", 1)
        assert red.stem == f"{prefix}_seed{cycle[seed]}"
        assert blue.is_file() and red.is_file()
        assert call["skip"] == "0"
    assert comparison[0] == "scripts/eval/eval_env_diversity.py"
    assert comparison[comparison.index("--recipe") + 1] == f"{family}_env_diversity"
    assert comparison[comparison.index("--train-seeds") + 1] == "42,100,200"
    assert "--episodes-per-seed" not in comparison


@pytest.mark.parametrize("phase", ["train", "eval", "diversity"])
def test_failures_propagate_and_failed_training_is_never_evaluated(tmp_path, phase):
    result, calls = launch(tmp_path, failed_phase=phase)
    assert result.returncode == 1, result.stderr
    assert "FAIL" in result.stderr
    if phase == "train":
        assert len(calls) == 12


def test_missing_baseline_fails_before_training(tmp_path):
    result, calls = launch(tmp_path, recipes="cotraining_lstm_env_diversity")
    assert result.returncode == 1
    assert "requires cotraining_lstm in JAXBORG_RECIPES" in result.stderr
    assert calls == []
