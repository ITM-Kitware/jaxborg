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

from jaxborg.recipe import load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("cross_seed_only_cli", ROOT / "scripts/eval/eval_cross_seed_only.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recipe = load("cotraining_mappo")
    models = {}
    for seed in (42, 200):
        path = tmp_path / f"model_seed{seed}.safetensors"
        path.touch()
        saved = {**recipe, "run": {"seed": seed}}
        path.with_name(f"recipe_seed{seed}.yaml").write_text(yaml.safe_dump(saved))
        models[seed] = SimpleNamespace(path=str(path), recipe=saved, steps=69_984_000)
    monkeypatch.setattr(module, "discover_models", lambda *a, **kw: (models, []))
    return module


@pytest.mark.parametrize("returncode", [0, 7])
def test_only_missing_suite_runs_with_recipe_budget_and_blue_mlflow_attachment(cli, tmp_path, returncode):
    output = tmp_path / "eval"
    exp = tmp_path / "exp"
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=returncode)

    args = ["--output-dir", str(output), "--exp-dir", str(exp)]
    if returncode:
        with pytest.raises(SystemExit) as error:
            cli.main(args, run_subprocess=execute)
        assert error.value.code == 1
    else:
        cli.main(args, run_subprocess=execute)
    assert len(calls) == 2
    for (command, kwargs), (blue, red) in zip(calls, [(42, 200), (200, 42)], strict=True):
        assert command[1] == str(ROOT / "scripts/eval/eval_matchup.py")
        for flag, expected in {
            "--name": "cross-seed-play",
            "--mlflow-source-team": "blue",
            "--policy-backend": "jax",
            "--episodes-per-seed": "1",
            "--seeds": ",".join(map(str, range(1000, 1010))),
            "--blue-path": str(tmp_path / f"model_seed{blue}.safetensors"),
            "--red-path": str(tmp_path / f"model_seed{red}.safetensors"),
        }.items():
            assert command[command.index(flag) + 1] == expected
        assert "--deterministic" not in command
        assert kwargs["env"]["JAXBORG_EXP_DIR"] == str(exp)
        recipe = yaml.safe_load(Path(command[command.index("--recipe") + 1]).read_text())
        assert recipe["eval"]["cia"]["enabled"]
        assert recipe["eval"]["topology_generation"]["count"] == 10
    manifest = json.loads((output / "manifest.json").read_text())
    assert {r["status"] for r in manifest["evaluations"]} == {"failed" if returncode else "succeeded"}
    assert manifest["suite"] == "cross-seed-play"


def test_dry_run_and_missing_requested_seed_never_evaluate(cli, tmp_path, capsys):
    output = tmp_path / "dry"

    def fail(*args, **kwargs):
        pytest.fail("Unexpected evaluation")

    cli.main(["--dry-run", "--output-dir", str(output)], run_subprocess=fail)
    assert not output.exists()
    assert "training seed cycle=[42, 200]" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        cli.main(["--train-seeds", "42,100,200", "--output-dir", str(output)], run_subprocess=fail)
    assert error.value.code == 2
    assert "missing for training seeds [100]" in capsys.readouterr().err
    assert not output.exists()


def test_shell_forwards_only_to_cross_seed_entrypoint(tmp_path):
    binary = tmp_path / "uv"
    binary.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    binary.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/eval/run_mappo_cross_seed.sh"),
            "--recipe",
            "cotraining_mappo_env_diversity",
            "--dry-run",
        ],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "JAXBORG_REPO_DIR": str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    args = json.loads(result.stdout)
    assert args[-4:] == [
        "scripts/eval/eval_cross_seed_only.py",
        "--recipe",
        "cotraining_mappo_env_diversity",
        "--dry-run",
    ]
    assert args[args.index("python") + 1] == "scripts/eval/eval_cross_seed_only.py"
