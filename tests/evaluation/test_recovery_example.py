"""The recovery example accepts only explicit, verified baseline inputs."""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaxborg.evaluation.capture_store import finish_evaluation, record_episode, start_evaluation
from jaxborg.scenarios.cc4.game_variant import GameVariant

spec = importlib.util.spec_from_file_location(
    "recovery_example", Path(__file__).resolve().parents[2] / "experiments/reward-offset/recovery_example.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def baseline(tmp_path, *, offset=0, status="succeeded"):
    model = tmp_path / "model_fixture.safetensors"
    model.write_bytes(b"preflight fixture")
    sidecar = tmp_path / "recipe_fixture.yaml"
    sidecar.write_text("{}\n")
    topology = tmp_path / "topology.npz"
    topology.write_bytes(b"fixture topology")
    capture = start_evaluation(
        tmp_path / "evidence",
        protocol=example.protocol(),
        trial_id="retry",
        requested_episodes=1,
        resolved_config={
            "backend": "jax",
            "builder": {"arguments": {"offset": offset}},
            "variant": asdict(GameVariant("fixture")),
            "recipe": {},
            "seed_roles": {"episode_roots": [1]},
        },
        inputs={
            "blue checkpoint": model,
            "red checkpoint": model,
            "blue sidecar": sidecar,
            "red sidecar": sidecar,
            "topology 0": topology,
        },
        source_files={},
    )
    if status == "succeeded":
        record_episode(capture, episode_index=0, seed=1, reward=0, steps=1)
    finish_evaluation(capture, status=status)
    return capture


@pytest.mark.parametrize("case", ["unknown", "failed", "offset", "corrupt"])
def test_recovery_preflight_rejects_unsuitable_source_before_writing(tmp_path, case):
    capture = baseline(
        tmp_path, offset=1 if case == "offset" else 0, status="failed" if case == "failed" else "succeeded"
    )
    if case == "corrupt":
        capture.inputs["blue checkpoint"].write_bytes(b"modified")
    output = tmp_path / "recovery"
    with pytest.raises(ValueError):
        example.run_example(
            capture.root, "absent" if case == "unknown" else capture.evaluation_id, output, expected_revision="a" * 40
        )
    assert not output.exists()


def test_recovery_resolves_exact_retained_source_bytes(tmp_path):
    capture = baseline(tmp_path)
    config, identities, inputs = example.source_inputs(capture.root, capture.evaluation_id)
    assert config["builder"]["arguments"]["offset"] == 0
    assert identities == capture.input_ids
    assert inputs["blue checkpoint"].read_bytes() == b"preflight fixture"


def test_recovery_report_replays_from_archived_generic_analyzer(tmp_path, monkeypatch):
    source = baseline(tmp_path)

    def fixture_matchup(blue, red, **settings):
        capture = start_evaluation(
            settings["root"],
            protocol=settings["protocol"],
            trial_id=settings["trial_id"],
            requested_episodes=1,
            resolved_config={"fixture": "orchestration test"},
            inputs={"policy": blue},
            source_files={},
            retry_of=settings.get("retry_of"),
            correction=settings.get("correction"),
        )
        if Path(blue).read_bytes().startswith(b"Deliberately invalid"):
            error = example.SafetensorError("deliberately invalid fixture")
            error._evaluation_capture = capture
            finish_evaluation(capture, status="failed", error=error)
            raise error
        record_episode(capture, episode_index=0, seed=1, reward=-3, steps=2)
        finish_evaluation(capture, status="succeeded")
        return capture, None

    monkeypatch.setattr(example, "run_matchup", fixture_matchup)
    monkeypatch.setattr(example, "collect_provenance", lambda **kwargs: SimpleNamespace(verify_sources=lambda: None))
    output = example.run_example(source.root, source.evaluation_id, tmp_path / "recovery", expected_revision="a" * 40)
    before = json.loads((output / "before-retry.json").read_text())
    assert before["rows"] == []
    assert any(record.get("status") == "failed" for record in before["records"])
    report, rebuilt = output / "report", tmp_path / "rebuilt"
    archive = report / "analysis-source"
    result = subprocess.run(
        [
            sys.executable,
            str(archive / "scripts/eval/plot_results.py"),
            "--collation",
            str(report / "collation.json"),
            "--evidence-root",
            str(output / "evidence"),
            "--output",
            str(rebuilt),
        ],
        cwd=tmp_path,
        env=os.environ
        | {
            "PYTHONPATH": os.pathsep.join((str(archive / "src"), str(archive))),
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (rebuilt / "joined.json").read_bytes() == (report / "joined.json").read_bytes()
    assert json.loads((rebuilt / "manifest.json").read_text())["corrections"]
    assert json.loads((output / "before-retry.json").read_text()) == before
