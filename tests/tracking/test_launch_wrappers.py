"""Run scheduling wrappers with a fake Slurm boundary, never touching GPUs."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def launch_checkout(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    shutil.copytree(REPO / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "recipes", repo / "recipes")
    for name in (
        "scripts/train/run.sh",
        "scripts/train/run_seeds.sh",
        "scripts/train/allocated.sh",
        "scripts/sbatch/run_ippo.sh",
        "uv.lock",
    ):
        dest = repo / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, dest)
        if name.endswith(".sh"):
            dest.chmod(0o755)
    # Ordinary Python program stands in for training after verification.
    trainer = repo / "scripts/train/algorithms/ippo_jax.py"
    trainer.parent.mkdir()
    trainer.write_text("print('trainer executed')\n")
    venv = repo / ".venv/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    (repo / ".gitignore").write_text(".venv/\n__pycache__/\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "launch",
        ],
        check=True,
    )
    bins = tmp_path / "bin"
    bins.mkdir()
    env = {**os.environ, "JAXBORG_EXP_DIR": str(tmp_path / "experiments"), "PATH": f"{bins}:{os.environ['PATH']}"}
    env["PYTHONPATH"] = str(repo / "src")
    env.pop("SLURM_JOB_ID", None)
    env.pop("JAXBORG_ALLOW_CPU", None)
    env.pop("JAXBORG_ALLOW_DIRTY", None)
    return repo, bins, env


def test_batch_pins_at_submission_and_rejects_changed_source_before_execution(launch_checkout, tmp_path):
    repo, bins, env = launch_checkout
    record = tmp_path / "submission.json"
    sbatch = bins / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f'json.dump({{"args":sys.argv[1:],"env":dict(os.environ)}},open({str(record)!r},"w"))\n'
    )
    sbatch.chmod(0o755)
    subprocess.run(["scripts/sbatch/run_ippo.sh", "default", "--seed", "7"], cwd=repo, env=env, check=True)
    submission = json.loads(record.read_text())
    assert "--partition=community" in submission["args"]
    assert (
        submission["env"]["JAXBORG_EXPECTED_SHA"]
        == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    )
    # Simulate a queue delay during which someone modifies the launch checkout.
    (repo / "scripts/train/algorithms/ippo_jax.py").write_text("print('changed trainer')\n")
    job_env = {**submission["env"], "SLURM_JOB_ID": "fake-job"}
    result = subprocess.run(
        ["scripts/sbatch/run_ippo.sh", "default"], cwd=repo, env=job_env, capture_output=True, text=True
    )
    assert result.returncode != 0 and "Dirty source" in result.stderr
    assert "trainer executed" not in result.stdout


def test_interactive_gpu_preflight_is_inside_srun(launch_checkout, tmp_path):
    repo, bins, env = launch_checkout
    log = tmp_path / "srun.json"
    srun = bins / "srun"
    srun.write_text(f'#!{sys.executable}\nimport json, sys\njson.dump(sys.argv[1:],open({str(log)!r},"w"))\n')
    srun.chmod(0o755)
    # No allocation/GPU exists. Submission must get to srun without device discovery.
    subprocess.run(["scripts/train/run.sh", "jax", "default", "42"], cwd=repo, env=env, check=True)
    args = json.loads(log.read_text())
    assert "--partition=community" in args
    assert "--gres=gpu:1" in args
    assert any(a.endswith("/scripts/train/allocated.sh") for a in args)
    assert (Path(env["JAXBORG_EXP_DIR"]) / "launches").is_dir()


def test_wrong_expected_sha_stops_batch_submission(launch_checkout, tmp_path):
    repo, bins, env = launch_checkout
    marker = tmp_path / "submitted"
    sbatch = bins / "sbatch"
    sbatch.write_text(f'#!/usr/bin/env bash\ntouch "{marker}"\n')
    sbatch.chmod(0o755)
    env["JAXBORG_EXPECTED_SHA"] = "0" * 40
    result = subprocess.run(
        ["scripts/sbatch/run_ippo.sh", "default"], cwd=repo, env=env, capture_output=True, text=True
    )
    assert result.returncode != 0 and "Wrong launch SHA" in result.stderr
    assert not marker.exists()


def test_seed_launches_share_submission_identity(launch_checkout, tmp_path):
    repo, bins, env = launch_checkout
    log = tmp_path / "seeds.jsonl"
    srun = bins / "srun"
    srun.write_text(
        f"#!{sys.executable}\nimport json, os\n"
        f'with open({str(log)!r},"a") as f: f.write(json.dumps({{"sha":os.environ["JAXBORG_EXPECTED_SHA"],'
        '"record":os.environ["JAXBORG_LAUNCH_RECORD"]})+"\\n")\n'
    )
    srun.chmod(0o755)
    subprocess.run(["scripts/train/run_seeds.sh", "jax", "default", "2", "42"], cwd=repo, env=env, check=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 2 and rows[0] == rows[1]
