"""Check report arithmetic, plot inputs, exports and links against controlled episode records."""

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jaxborg.research_tracking import parameter_hash
from jaxborg.response_oracle import TEST_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS
from jaxborg.tracking import Run, file_hash, git, input_artifact, resolve_artifact

REPO = Path(__file__).resolve().parents[2]


def test_controller_does_not_pass_its_gpu_pool_setting_to_children(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    spec = importlib.util.spec_from_file_location("pilot_controller", REPO / "scripts/experiments/response_oracle.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    assert pilot.os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    observed = []
    monkeypatch.setattr(pilot.subprocess, "run", lambda *args, **kwargs: observed.append(kwargs["env"]))
    pilot.command("scripts/eval/eval_matchup.py")
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in observed[0]


@pytest.mark.parametrize("source_steps", [9600000, 49968000])
@pytest.mark.parametrize("team", ["blue", "red"])
def test_report_matches_episode_records_and_plot_inputs(tmp_path, monkeypatch, source_steps, team):
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("JAXBORG_ALLOW_DIRTY", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("JAXBORG_EXPECTED_SHA", raising=False)
    monkeypatch.delenv("JAXBORG_MLFLOW_EXPERIMENT", raising=False)
    spec = importlib.util.spec_from_file_location("pilot_report", REPO / "scripts/experiments/response_oracle.py")
    pilot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pilot
    spec.loader.exec_module(pilot)
    with Run({"meta": {"name": "test-inputs"}}, backend="cpu", kind="input") as owner:
        source = owner.path("source.bin")
        source.write_bytes(b"controlled policy input; rollout is supplied by test")
        source_ref = owner.publish(source, "source.bin")
        challenger = owner.path("challenger.bin")
        challenger.write_bytes(b"distinct learned challenger input")
        challenger_ref = owner.publish(challenger, "challenger.bin")
        topology = owner.path("topology.bin")
        topology.write_bytes(b"controlled shared topology")
        topology_ref = owner.publish(topology, "topology.bin")
    report_dir = tmp_path / "collection/results/stage-b"
    report_dir.mkdir(parents=True)
    collection = report_dir.parent.parent / "README.md"
    collection.write_text("Existing collection\n")
    if source_steps == 49968000:
        report_dir = report_dir / "mappo-source-49968000"
        report_dir.mkdir()
    manifest = {
        "trainable_team": team,
        "campaign": "test-stage-b",
        "source_revision": git("rev-parse", "HEAD"),
        "source": {
            "checkpoint": source_ref,
            "sha256": file_hash(resolve_artifact(source_ref)),
            "algorithm": "mappo",
            "original_training_steps": source_steps,
            "original_training_seed": 42,
        },
        "training_seeds": list(TRAIN_SEEDS),
        "randomness": {
            "validation_episode_roots": list(VALIDATION_SEEDS),
            "final_test_episode_roots": list(TEST_SEEDS),
        },
        "confidence_interval": {"resamples": 10000, "seed": 3000001},
        "oracle_budget_per_attempt": {
            "requested_steps": 10000000,
            "completed_steps": 9984000,
            "updates": 208,
            "steps_per_update": 48000,
        },
        "collection_readme": str(collection),
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
        "selection": {"candidate": "seed-11001", "checkpoint": challenger_ref},
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
            state["training"][f"seed-{seed}"] = {"run_id": owner.run_id, "final_checkpoint": challenger_ref}

    def evaluation(seeds, returns, challenger=False):
        blue = challenger_ref if challenger and team == "blue" else source_ref
        red = challenger_ref if challenger and team == "red" else source_ref
        with Run(
            {"meta": {"name": "controlled-evaluation"}},
            backend="cpu",
            kind="evaluation",
            inputs=[input_artifact(blue), input_artifact(red), input_artifact(topology_ref)],
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
        value = (-8 if team == "blue" else -12) if name == "seed-11001" else -10
        state["validation"][name] = evaluation(VALIDATION_SEEDS, [value] * 100, challenger=name != "original")
    state["test"]["original"] = evaluation(TEST_SEEDS, [-10] * 600)
    selected_value = -4 if team == "blue" else -22
    improvement = 6 if team == "blue" else 12
    state["test"]["selected"] = evaluation(TEST_SEEDS, [selected_value] * 600, challenger=True)
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
    assert result[f"{team}_improvement"] == improvement
    assert result["ci95"] == [improvement, improvement]
    assert bars == [[-10, selected_value]]
    with (report_dir / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["mean_blue_return"]) for row in rows] == bars[0]
    readme = (report_dir / "README.md").read_text()
    assert f"**{improvement:.2f} points**" in readme
    assert f"{'Red' if team == 'blue' else 'Blue'} stayed exactly unchanged" in readme
    assert f"The {'highest' if team == 'blue' else 'lowest'} mean Blue return won" in readme
    assert f"{source_steps:,} source steps" in readme
    for link in re.findall(r"\]\(([^)]+)\)", readme):
        assert (report_dir / link).is_file(), link
    assert (report_dir / "comparison.png").stat().st_size > 1000
    assert (report_dir / "handoff.md").is_file()
    assert ("Completed Blue response:" if team == "blue" else "Completed Stage B pilot:") in collection.read_text()
    assert f"[MAPPO {'Red' if team == 'blue' else 'Blue'} at {source_steps:,} source steps]" in collection.read_text()
    with (report_dir / "per-episode.csv").open() as stream:
        assert {float(row[f"{team}_improvement"]) for row in csv.DictReader(stream)} == {improvement}
    with (report_dir / "learning-tail.csv").open() as stream:
        assert {float(row["change"]) for row in csv.DictReader(stream)} == {-20 if team == "blue" else 20}


@pytest.mark.parametrize("team", ["blue", "red"])
def test_evaluate_routes_candidate_to_the_correct_team(tmp_path, monkeypatch, team):
    spec = importlib.util.spec_from_file_location("response_evaluate", REPO / "scripts/experiments/response_oracle.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    observed = []

    def command(*argv):
        observed.extend(argv)
        (tmp_path / "test-challenger.json").write_text(json.dumps({"eval_id": "test"}))

    monkeypatch.setattr(pilot, "command", command)
    monkeypatch.setattr(pilot, "checked_evaluation", lambda reference, seeds: {})
    pilot.evaluate(
        {
            "trainable_team": team,
            "source": {"checkpoint": "original"},
            "study_dir": str(tmp_path),
            "campaign": "test",
            "eval_recipe": "recipe",
        },
        ("challenger", "fresh"),
        "test",
        [1, 2],
    )
    assert observed[observed.index("--blue-path") + 1] == ("fresh" if team == "blue" else "original")
    assert observed[observed.index("--red-path") + 1] == ("original" if team == "blue" else "fresh")


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("corrupt_frozen", [False, True])
def test_train_rejects_changed_saved_frozen_weights(tmp_path, monkeypatch, team, corrupt_frozen):
    spec = importlib.util.spec_from_file_location("response_train", REPO / "scripts/experiments/response_oracle.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    frozen = "red" if team == "blue" else "blue"
    original = {"blue": np.array([1.0]), "red": np.array([2.0])}
    saved = {t: SimpleNamespace(weights=v.copy()) for t, v in original.items()}
    saved[team].weights += 5
    checks = {t: {"changed": t == team, "current_sha256": parameter_hash(saved[t].weights)} for t in original}
    if corrupt_frozen:
        saved[frozen].weights += 1
    result = {"run_id": "test", "final_checkpoint": "saved"}

    def command(*argv):
        (tmp_path / "training-seed-17.json").write_text(json.dumps(result))

    monkeypatch.setattr(pilot, "command", command)
    monkeypatch.setattr(pilot, "read_manifest", lambda _: ({"status": "FINISHED", "parameter_checks": checks}, None))
    monkeypatch.setattr(pilot, "resolve_artifact", lambda _: tmp_path / "saved")
    monkeypatch.setattr(pilot, "load_jax_bundle", lambda _: SimpleNamespace(policies=saved))
    manifest = {
        "trainable_team": team,
        "study_dir": str(tmp_path),
        "campaign": "test",
        "source": {"policies": {t: {"parameter_sha256": parameter_hash(v)} for t, v in original.items()}},
    }
    if corrupt_frozen:
        with pytest.raises(ValueError, match=f"saved {frozen} is not the original frozen source"):
            pilot.train(manifest, 17, "recipe")
    else:
        assert pilot.train(manifest, 17, "recipe") == result


@pytest.mark.parametrize("smoke_fails", [False, True])
def test_combined_pilot_executes_only_after_its_own_smoke(monkeypatch, smoke_fails):
    spec = importlib.util.spec_from_file_location("response_pilot", REPO / "scripts/experiments/response_oracle.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    observed = []

    def smoke(args):
        observed.append("smoke")
        if smoke_fails:
            raise ValueError("failed actual defender smoke")

    monkeypatch.setattr(pilot, "smoke", smoke)
    monkeypatch.setattr(pilot, "execute", lambda args: observed.append("run"))
    monkeypatch.setattr(sys, "argv", ["response_oracle", "pilot", "--manifest", "campaign-manifest.json"])
    if smoke_fails:
        with pytest.raises(ValueError, match="actual defender"):
            pilot.main()
        assert observed == ["smoke"]
    else:
        pilot.main()
        assert observed == ["smoke", "run"]
