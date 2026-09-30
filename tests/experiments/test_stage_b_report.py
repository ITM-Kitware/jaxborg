"""Check report arithmetic, plot inputs, exports and links against controlled episode records."""

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

from jaxborg.oracle_stage_b import TEST_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS
from jaxborg.tracking import Run, file_hash, git, input_artifact, resolve_artifact

REPO = Path(__file__).resolve().parents[2]


def test_controller_does_not_pass_its_gpu_pool_setting_to_children(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    spec = importlib.util.spec_from_file_location("pilot_controller", REPO / "scripts/experiments/oracle_stage_b.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    assert pilot.os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    observed = []
    monkeypatch.setattr(pilot.subprocess, "run", lambda *args, **kwargs: observed.append(kwargs["env"]))
    pilot.command("scripts/eval/eval_matchup.py")
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in observed[0]


def test_report_matches_episode_records_and_plot_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("JAXBORG_ALLOW_DIRTY", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("JAXBORG_EXPECTED_SHA", raising=False)
    monkeypatch.delenv("JAXBORG_MLFLOW_EXPERIMENT", raising=False)
    spec = importlib.util.spec_from_file_location("pilot_report", REPO / "scripts/experiments/oracle_stage_b.py")
    pilot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pilot
    spec.loader.exec_module(pilot)
    with Run({"meta": {"name": "test-inputs"}}, backend="cpu", kind="input") as owner:
        source = owner.path("source.bin")
        source.write_bytes(b"controlled policy input; rollout is supplied by test")
        source_ref = owner.publish(source, "source.bin")
        topology = owner.path("topology.bin")
        topology.write_bytes(b"controlled shared topology")
        topology_ref = owner.publish(topology, "topology.bin")
    report_dir = tmp_path / "collection/results/stage-b"
    report_dir.mkdir(parents=True)
    collection = report_dir.parent.parent / "README.md"
    collection.write_text("Existing collection\n")
    manifest = {
        "campaign": "test-stage-b",
        "source_revision": git("rev-parse", "HEAD"),
        "source": {"checkpoint": source_ref, "sha256": file_hash(resolve_artifact(source_ref))},
        "game": {"topology_sha256": file_hash(resolve_artifact(topology_ref))},
        "report_dir": str(report_dir),
    }
    state = {
        "training": {},
        "validation": {},
        "test": {},
        "started_epoch": 0,
        "finished_epoch": 3600,
        "slurm_job_ids": ["test"],
        "status": "evaluations_finished",
        "selection": {"candidate": "seed-11001", "checkpoint": source_ref},
    }
    for seed in TRAIN_SEEDS:
        with Run({"meta": {"name": str(seed)}}, backend="cpu") as owner:
            rows = [
                {
                    "env_steps": (i + 1) * 48000,
                    "wall_time_s": i + 1,
                    "team.red.return": 10 + i,
                    "team.blue.return": -10 - i,
                    "throughput_sps": 48000,
                }
                for i in range(40)
            ]
            metrics = owner.path("logs/metrics.jsonl")
            metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
            owner.publish(metrics, "logs/metrics.jsonl", mutable=True)
            state["training"][f"seed-{seed}"] = {"run_id": owner.run_id, "final_checkpoint": source_ref}

    def evaluation(seeds, returns):
        with Run(
            {"meta": {"name": "controlled-evaluation"}},
            backend="cpu",
            kind="evaluation",
            inputs=[input_artifact(source_ref), input_artifact(source_ref), input_artifact(topology_ref)],
        ) as owner:
            return owner.write_json(
                "evaluations/result.json",
                dict(
                    eval_id=owner.run_id,
                    n_episodes=len(seeds),
                    per_episode_seeds=list(seeds),
                    per_episode_blue_returns=list(returns),
                    per_episode_red_returns=[-v for v in returns],
                    blue_mean_return=float(np.mean(returns)),
                    variant="cc4_stock",
                    stochastic=True,
                    wall_time_s=2,
                ),
            )

    for name in ["original", *[f"seed-{seed}" for seed in TRAIN_SEEDS]]:
        value = -12 if name == "seed-11001" else -10
        state["validation"][name] = evaluation(VALIDATION_SEEDS, [value] * 100)
    state["test"]["original"] = evaluation(TEST_SEEDS, [-10] * 600)
    state["test"]["selected"] = evaluation(TEST_SEEDS, [-22] * 600)
    (report_dir / "manifest.json").write_text(json.dumps(manifest))
    (report_dir / "smoke.json").write_text("{}\n")
    bars = []
    import matplotlib.axes

    real_bar = matplotlib.axes.Axes.bar

    def bar(self, labels, values, **kwargs):
        bars.append(list(values))
        return real_bar(self, labels, values, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", bar)
    result = pilot.aggregate(manifest, state)
    assert result["red_improvement"] == 12
    assert result["ci95"] == [12, 12]
    assert bars == [[-10, -22]]
    with (report_dir / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["mean_blue_return"]) for row in rows] == bars[0]
    readme = (report_dir / "README.md").read_text()
    assert "**12.00 points**" in readme
    for link in re.findall(r"\]\(([^)]+)\)", readme):
        assert (report_dir / link).is_file(), link
    assert (report_dir / "comparison.png").stat().st_size > 1000
    assert (report_dir / "handoff.md").is_file()
    assert "Completed Stage B pilot:" in collection.read_text()
