"""Run the evaluation-only shell launcher with saved models and fake GPU jobs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/temp/run_cotraining_ippo_lstm.sh"
RECIPES = ("cotraining_lstm", "cotraining_lstm_env_diversity")
SEEDS = ("42", "100", "200")


@pytest.fixture
def saved_runs(tmp_path):
    artifacts = []
    for recipe in RECIPES:
        for seed in SEEDS:
            tag = f"{recipe}_seed{seed}"
            run = tmp_path / "exp" / "ippo_jax" / tag
            run.mkdir(parents=True)
            model = run / f"model_{tag}.safetensors"
            sidecar = run / f"recipe_{tag}.yaml"
            model.write_bytes(b"existing checkpoint")
            sidecar.write_text(f"meta:\n  name: {recipe}\n")
            artifacts.extend([model, sidecar])

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys
args = sys.argv[sys.argv.index("python") + 1:]
if args[0] == "-c":
    recipe = args[-1]
    baseline = "cotraining_lstm" if recipe.endswith("_env_diversity") else "-"
    print(f"ippo 1 {baseline} {recipe}")
    sys.exit(0)
assert args[0] in ("scripts/eval/run_after_training.py", "scripts/eval/eval_env_diversity.py"), args
record = {
    "args": args,
    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "platform": os.environ.get("JAX_PLATFORMS"),
    "skip": os.environ.get("JAXBORG_SKIP_POST_TRAINING_EVAL"),
}
with open(os.environ["TEST_CALLS"], "a") as stream:
    stream.write(json.dumps(record) + "\\n")
phase = "eval" if "run_after_training.py" in args[0] else "diversity"
if phase == os.environ.get("TEST_FAILED_PHASE"):
    print("simulated evaluation failure", flush=True)
    sys.exit(7)
"""
    )
    uv.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("JAXBORG_", "SLURM_", "TEST_")) and key != "GPU_COUNT"
    }
    env.update(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        JAXBORG_EXP_DIR=str(tmp_path / "exp"),
        JAXBORG_LOG_DIR=str(tmp_path / "logs"),
        JAXBORG_SKIP_POST_TRAINING_EVAL="1",
        CUDA_VISIBLE_DEVICES="3,5,7",
        TEST_CALLS=str(tmp_path / "calls.jsonl"),
    )
    return env, artifacts


def launch(tmp_path, env):
    # An unrelated working directory exercises repository discovery outside Slurm.
    result = subprocess.run(["bash", str(LAUNCHER)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    calls_file = Path(env["TEST_CALLS"])
    calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
    return result, calls


@pytest.mark.parametrize("gpu_count", [None, 1, 2])
def test_evaluates_saved_models_with_original_pairings_and_no_training(tmp_path, saved_runs, gpu_count):
    env, artifacts = saved_runs
    before = {path: path.read_bytes() for path in artifacts}
    if gpu_count is not None:
        env["GPU_COUNT"] = str(gpu_count)

    result, calls = launch(tmp_path, env)

    assert result.returncode == 0, result.stderr
    assert len(calls) == 7
    evaluations, comparison = calls[:6], calls[6]
    assert all(call["platform"] == "cuda" and call["skip"] == "0" for call in calls)
    cycle = dict(zip(SEEDS, (*SEEDS[1:], SEEDS[0]), strict=True))
    expected_models = {path for path in artifacts if path.suffix == ".safetensors"}
    actual_models = set()
    for call in evaluations:
        args = call["args"]
        assert args[0] == "scripts/eval/run_after_training.py"
        blue = Path(args[args.index("--model") + 1])
        red = Path(args[args.index("--cross-seed-red") + 1])
        prefix, seed = blue.stem.rsplit("_seed", 1)
        assert red.stem == f"{prefix}_seed{cycle[seed]}"
        assert red in expected_models
        actual_models.add(blue)
        assert call["gpu"] == ("3", "5", "7")[SEEDS.index(seed) % (gpu_count or 3)]
        assert "--recipe" not in args  # Each model retains its saved evaluation settings.
    assert actual_models == expected_models
    assert comparison["gpu"] == "3"
    assert comparison["args"] == [
        "scripts/eval/eval_env_diversity.py",
        "--recipe",
        "cotraining_lstm_env_diversity",
        "--train-seeds",
        "42,100,200",
        "--baseline-tag",
        "cotraining_lstm_seed*",
        "--diverse-tag",
        "cotraining_lstm_env_diversity_seed*",
    ]
    assert {path: path.read_bytes() for path in artifacts} == before


@pytest.mark.parametrize("suffix", [".safetensors", ".yaml"])
def test_missing_saved_input_stops_before_any_evaluation(tmp_path, saved_runs, suffix):
    env, artifacts = saved_runs
    missing = next(path for path in reversed(artifacts) if path.suffix == suffix)
    missing.unlink()

    result, calls = launch(tmp_path, env)

    assert result.returncode == 1
    assert "Missing" in result.stderr
    assert missing.parent.name in result.stderr
    assert calls == []


@pytest.mark.parametrize("phase", ["eval", "diversity"])
def test_evaluation_failures_propagate_after_remaining_jobs_finish(tmp_path, saved_runs, phase):
    env, _ = saved_runs
    env["TEST_FAILED_PHASE"] = phase

    result, calls = launch(tmp_path, env)

    assert result.returncode == 1
    assert "FAIL" in result.stderr
    assert "simulated evaluation failure" in result.stderr
    assert len(calls) == 7


def test_standard_recipe_can_be_evaluated_alone(tmp_path, saved_runs):
    env, _ = saved_runs
    env["JAXBORG_RECIPES"] = "cotraining_lstm"

    result, calls = launch(tmp_path, env)

    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert all(call["args"][0] == "scripts/eval/run_after_training.py" for call in calls)
