"""Exercise the bounded response protocol with controlled paired episode records."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jaxborg.response_campaign import load_campaign
from jaxborg.tracking import Run, file_hash, git, input_artifact, resolve_artifact

REPO = Path(__file__).resolve().parents[2]
MILESTONES = [104, 208, 312, 416]


@pytest.fixture
def campaign(tmp_path):
    config = yaml.safe_load((REPO / "campaigns/response-oracles/ippo-seed42-red-curve.yaml").read_text())
    config["name"] = "red-budget-test"
    config["defenders"] = [config["defenders"][0]]
    config["training"] = {"seeds": [540001], "requested_steps": 20000000}
    config["challenger"]["team"] = "red"
    config["selection"] = {
        "rule": "lowest_validation_blue_mean",
        "checkpoint": "budget_milestones",
        "original_fallback": True,
        "tie_break": "original_then_checkpoint_order",
        "checkpoint_updates": MILESTONES,
        "budget_updates": [208, 416],
    }
    config["resources"].update(time_limit="04:00:00", max_concurrent_jobs=1)
    config["tracking"]["root"] = str(tmp_path / "artifacts")
    config["collection_readme"] = str(tmp_path / "collection/README.md")
    config["defenders"][0]["report_dir"] = str(tmp_path / "report")
    return config


def save_campaign(tmp_path, config):
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_campaign_accepts_one_twenty_million_step_attempt_with_disjoint_evaluation(campaign, tmp_path):
    config, defender, roots = load_campaign(save_campaign(tmp_path, campaign), "ippo-49968000")
    assert defender["steps"] == 49968000
    assert config["training"] == {"seeds": [540001], "requested_steps": 20000000}
    assert config["selection"]["checkpoint_updates"] == MILESTONES
    assert roots["training_environment_roots"] == [540001]
    assert len(roots["validation_episode_roots"]) == 100
    assert len(roots["final_test_episode_roots"]) == 600
    assert set(roots["validation_episode_roots"]).isdisjoint(roots["final_test_episode_roots"])


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "seeds", [540001, 550001]),
        ("training", "seeds", [1000000]),
        ("training", "requested_steps", 10000000),
        ("selection", "checkpoint_updates", [104, 208, 416]),
        ("selection", "budget_updates", [208, 312]),
        ("selection", "rule", "highest_validation_blue_mean"),
        ("resources", "max_concurrent_jobs", 2),
    ],
)
def test_invalid_campaign_is_rejected_before_any_training(campaign, tmp_path, section, key, value):
    campaign[section][key] = value
    with pytest.raises(ValueError):
        load_campaign(save_campaign(tmp_path, campaign), "ippo-49968000")


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    spec = importlib.util.spec_from_file_location(
        "response_budget_test_driver", REPO / "scripts/experiments/response_budget.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def controlled_artifacts(campaign, tmp_path, monkeypatch):
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("JAXBORG_ALLOW_DIRTY", "1")
    monkeypatch.setenv("JAXBORG_ALLOW_CPU", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("JAXBORG_EXPECTED_SHA", raising=False)
    monkeypatch.delenv("JAXBORG_MLFLOW_EXPERIMENT", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    _, _, randomness = load_campaign(save_campaign(tmp_path, campaign), "ippo-49968000")
    study = tmp_path / "study"
    study.mkdir()
    report = tmp_path / "report"
    report.mkdir()
    collection = Path(campaign["collection_readme"])
    collection.parent.mkdir()
    collection.write_text("# Controlled study\n")
    with Run({"meta": {"name": "budget-test-inputs"}}, backend="cpu", kind="input") as owner:
        source = owner.path("source.bin")
        source.write_bytes(b"controlled original policy pair")
        source_ref = owner.publish(source, "source.bin")
        topology = owner.path("topology.bin")
        topology.write_bytes(b"controlled shared topology")
        topology_ref = owner.publish(topology, "topology.bin")
    checkpoints = {}
    with Run({"meta": {"name": "budget-test-training"}}, backend="cpu") as owner:
        for update in MILESTONES:
            path = owner.path(f"checkpoints/checkpoint_{update * 48000}.bin")
            path.write_bytes(f"controlled Red checkpoint at update {update}".encode())
            reference = owner.publish(path, str(path.relative_to(owner.staging)))
            checkpoints[f"update-{update}"] = {
                "reference": reference,
                "sha256": file_hash(path),
                "step": update * 48000,
                "update": update,
            }
        logs = owner.path("logs/metrics.jsonl")
        logs.write_text(
            "".join(
                json.dumps(
                    {
                        "env_steps": update * 48000,
                        "update_idx": update,
                        "wall_time_s": update,
                        "team.red.return": 10 + update,
                        "team.blue.return": -10 - update,
                        "team.red.lr": 0.0003,
                        "throughput_sps": 48000,
                    }
                )
                + "\n"
                for update in range(1, 417)
            )
        )
        owner.publish(logs, "logs/metrics.jsonl")
        training = {
            "run_id": owner.run_id,
            "actual_steps": 19968000,
            "final_checkpoint": checkpoints["update-416"]["reference"],
        }
    checkpoints = {
        "original": {
            "reference": source_ref,
            "sha256": file_hash(resolve_artifact(source_ref)),
            "step": 0,
            "update": 0,
        },
        **checkpoints,
    }
    manifest = {
        "campaign": campaign["name"],
        "campaign_configuration": campaign,
        "trainable_team": "red",
        "source_revision": git("rev-parse", "HEAD"),
        "source": {
            "checkpoint": source_ref,
            "sha256": file_hash(resolve_artifact(source_ref)),
            "algorithm": "ippo",
            "original_training_steps": 49968000,
            "original_training_seed": 42,
        },
        "training_seeds": [540001],
        "randomness": randomness,
        "confidence_interval": {"resamples": 1000, "seed": 3000001},
        "oracle_budget_per_attempt": {
            "requested_steps": 20000000,
            "completed_steps": 19968000,
            "updates": 416,
            "steps_per_update": 48000,
        },
        "selection": campaign["selection"],
        "game": {"topology": topology_ref, "topology_sha256": file_hash(resolve_artifact(topology_ref))},
        "study_dir": str(study),
        "report_dir": str(report),
        "collection_readme": str(collection),
        "recipes": {"540001": str(study / "recipe.yaml")},
    }
    manifest_path = study / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (report / "manifest.json").write_text(json.dumps(manifest))
    (report / "smoke.json").write_text("{}\n")

    def evaluation(name, values, seeds, split):
        red = source_ref if name == "original" else checkpoints[name]["reference"]
        with Run(
            {"meta": {"name": f"controlled-{split}-{name}"}},
            backend="cpu",
            kind="evaluation",
            inputs=[input_artifact(source_ref), input_artifact(red), input_artifact(topology_ref)],
        ) as owner:
            return owner.write_json(
                "evaluations/result.json",
                {
                    "eval_id": owner.run_id,
                    "n_episodes": len(seeds),
                    "per_episode_seeds": list(seeds),
                    "per_episode_blue_returns": list(values),
                    "per_episode_red_returns": [-value for value in values],
                    "blue_mean_return": sum(values) / len(values),
                    "variant": "cc4_stock",
                    "stochastic": True,
                    "wall_time_s": 2,
                },
            )

    return SimpleNamespace(
        manifest=manifest,
        manifest_path=manifest_path,
        training=training,
        checkpoints=checkpoints,
        evaluation=evaluation,
        study=study,
        report=report,
    )


@pytest.mark.parametrize(
    "validation,selected,test_values,added_gain",
    [
        (
            [-10, -12, -11, -14, -13],
            {"208": "update-104", "416": "update-312"},
            {"original": -10, "update-104": -20, "update-312": -23},
            3,
        ),
        (
            [-10, -11, -12, -13, -14],
            {"208": "update-208", "416": "update-416"},
            {"original": -10, "update-208": -20, "update-416": -18},
            -2,
        ),
        (
            [-10, -12, -12, -12, -12],
            {"208": "update-104", "416": "update-104"},
            {"original": -10, "update-104": -20},
            0,
        ),
        (
            [-10, -10, -10, -10, -10],
            {"208": "original", "416": "original"},
            {"original": -10},
            0,
        ),
    ],
    ids=["cumulative-checkpoints", "heldout-reversal", "earliest-tie", "original-tie"],
)
def test_saved_selection_and_paired_report_use_only_validation(
    driver, controlled_artifacts, monkeypatch, validation, selected, test_values, added_gain
):
    data = controlled_artifacts
    manifest = data.manifest
    validation_seeds = manifest["randomness"]["validation_episode_roots"]
    test_seeds = manifest["randomness"]["final_test_episode_roots"]
    names = ["original", *[f"update-{update}" for update in MILESTONES]]
    evaluations = {}
    for name, value in zip(names, validation):
        evaluations["validation", name] = data.evaluation(name, [value] * 100, validation_seeds, "validation")
    for name, value in test_values.items():
        evaluations["confirmation", name] = data.evaluation(name, [value] * 600, test_seeds, "confirmation")
    calls = []
    training_calls = []

    def train(_manifest, seed, recipe):
        training_calls.append((seed, recipe))
        return data.training

    def evaluate(_manifest, candidate, split, seeds, **_kwargs):
        name, reference = candidate
        assert reference == data.checkpoints[name]["reference"]
        if split == "confirmation":
            saved = json.loads((data.study / "state.json").read_text())
            assert saved["selection"]["candidates"] == selected
            assert saved["selection"]["validation_scores"] == dict(zip(names, validation))
            assert saved["selection"]["frozen_before_test_epoch"] > 0
            assert seeds == test_seeds
        else:
            assert seeds == validation_seeds
        calls.append((split, name))
        evaluation_ref = evaluations[split, name]
        return driver.oracle.checked_evaluation(evaluation_ref, seeds), evaluation_ref

    monkeypatch.setattr(driver, "load_protocol", lambda _path: manifest)
    monkeypatch.setattr(driver, "checkpoint_candidates", lambda _manifest, _result: data.checkpoints)
    monkeypatch.setattr(driver.oracle, "require_smoke", lambda *_args: None)
    monkeypatch.setattr(driver.oracle, "assigned_devices", lambda: [{"platform": "cpu"}])
    monkeypatch.setattr(driver.oracle, "train", train)
    monkeypatch.setattr(driver.oracle, "evaluate", evaluate)
    monkeypatch.setenv("SLURM_JOB_ID", "controlled-budget-test")
    driver.execute(SimpleNamespace(manifest=str(data.manifest_path)))

    saved = json.loads((data.study / "state.json").read_text())
    summary = json.loads((data.report / "summary.json").read_text())
    assert saved["status"] == "complete"
    assert saved["selection"]["candidates"] == selected
    assert summary["selection"] == saved["selection"]
    assert len(training_calls) == 1
    assert calls == [("validation", name) for name in names] + [("confirmation", name) for name in test_values]
    assert summary["confirmation_unique_matchups"] == len(test_values)
    assert summary["training_time_s"] == 416
    assert summary["longer_vs_shorter"]["red_improvement"] == added_gain
    assert summary["longer_vs_shorter"]["ci95"] == [added_gain, added_gain]
    original_gain = test_values["original"] - test_values[selected["416"]]
    assert summary["longer_vs_original"]["red_improvement"] == original_gain
    assert summary["longer_vs_original"]["episodes"] == 600
    with (data.report / "confirmation-episodes.csv").open() as stream:
        episodes = list(csv.DictReader(stream))
    assert [int(row["episode_seed"]) for row in episodes] == test_seeds
    assert {float(row["shorter_blue_return"]) - float(row["longer_blue_return"]) for row in episodes} == {added_gain}
    with (data.report / "validation-curve.csv").open() as stream:
        validation_rows = list(csv.DictReader(stream))
    assert [row["candidate"] for row in validation_rows] == names
    assert [float(row["blue_mean_return"]) for row in validation_rows] == validation
    readme = (data.report / "README.md").read_text()
    assert f"Shorter-budget selection: `{selected['208']}`" in readme
    assert f"Longer-budget selection: `{selected['416']}`" in readme
    assert f"Longer versus shorter-budget gain: {added_gain:.6f}" in readme
