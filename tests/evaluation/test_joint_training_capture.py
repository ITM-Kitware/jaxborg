"""Observe trainer initialization failures through the durable store boundary."""

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jaxborg.evaluation.capture_store import load_records
from jaxborg.recipe import load
from scripts.train.algorithms import ippo_jax as trainer

PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def recipe():
    value = load(str(PROJECT / "recipes/cotraining/cotraining.yaml"))
    value["train"] = {"teams": "both", "variant": "cc4_stock", "episode_length": 2, "total_timesteps": 2}
    value["arch"] = {"name": "separate", "hidden_dim": 4, "hidden_layers": 1}
    value["jax"] = {"num_envs": 1, "num_minibatches": 1, "update_epochs": 1}
    value["eval"] = {"variant": "cc4_stock", "policy_backend": "jax"}
    return value


def test_joint_trainer_failure_is_recorded_before_initialization_and_retry(tmp_path, monkeypatch, recipe):
    topology = tmp_path / "topology.npz"
    np.savez(topology, fixture=np.asarray([1]))
    recipe["train"]["topology_bank"] = [str(topology)]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    root = tmp_path / "evidence"
    observed = []

    def fail_initialization(*args, **kwargs):
        records = load_records(root)
        active = [record for record in records if record["kind"] == "attempt" and record["status"] == "running"]
        observed.append(active[0]["id"])
        assert active[0]["budgets"][0]["completed"] == 0
        configuration = next(root.glob("attempts/*/configuration.json"))
        assert json.loads(configuration.read_text())["recipe"]["meta"]["added"] == "2026-08-16"
        raise RuntimeError("deliberate initialization failure")

    monkeypatch.setattr(trainer, "make_joint_train", fail_initialization)
    active = []

    def start_tracking(*args, **kwargs):
        assert not active, "previous failed execution left tracking active"
        run = SimpleNamespace(info=SimpleNamespace(run_id=f"fixture-{len(observed)}"))
        active.append(run)
        return run

    monkeypatch.setattr(trainer, "start_run", start_tracking)
    monkeypatch.setattr(trainer.mlflow, "active_run", lambda: active[0] if active else None)
    monkeypatch.setattr(trainer.mlflow, "end_run", lambda **kwargs: active.clear())
    recording = {
        "root": root,
        "protocol": {
            "schema_version": 1,
            "kind": "protocol",
            "id": "training-protocol",
            "study_id": "training",
            "revision": "1",
            "selection_rule": "explicit",
            "required_facts": {"attempt": [], "evaluation": []},
            "metric_definitions": [],
            "trials": [{"id": "trial", "condition_id": "control", "replicate_id": "1", "matched_unit_id": None}],
        },
        "trial_id": "trial",
        "expected_revision": revision,
        "allow_dirty": True,
        "entrypoint": Path(__file__),
        "supplied_recipe": copy.deepcopy(recipe),
    }
    for _ in range(2):
        with pytest.raises(RuntimeError, match="initialization failure"):
            trainer._run_joint_training(SimpleNamespace(seed=42), recipe, "trial", tmp_path, recording=recording)
        recording["retry_of"] = observed[-1]
    attempts = [record for record in load_records(root) if record["kind"] == "attempt"]
    assert len(attempts) == 2
    assert not active
    assert all(record["status"] == "failed" and record["budgets"][0]["completed"] == 0 for record in attempts)
    assert next(record for record in attempts if record["id"] == observed[1])["retry_of"] == observed[0]


def test_completed_training_remains_successful_when_later_output_delivery_fails(tmp_path, monkeypatch, recipe):
    from jaxborg.evaluation.capture_store import record_training_update

    topology = tmp_path / "topology.npz"
    np.savez(topology, fixture=np.asarray([1]))
    recipe["train"]["topology_bank"] = [str(topology)]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    root = tmp_path / "evidence"

    def execute(*args, capture, **kwargs):
        record_training_update(
            capture, {"update_idx": 1, "env_steps": 2, "team.blue.update_count": 1, "team.red.update_count": 1}
        )
        return {}

    def fail_delivery(*args):
        raise OSError("delivery failed after training completed")

    monkeypatch.setattr(trainer, "_execute_joint_training", execute)
    monkeypatch.setattr(trainer, "_finish_training_outputs", fail_delivery)
    protocol = {
        "schema_version": 1,
        "kind": "protocol",
        "id": "p",
        "study_id": "study",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": [], "evaluation": []},
        "metric_definitions": [],
        "trials": [{"id": "t", "condition_id": "control", "replicate_id": "1", "matched_unit_id": None}],
    }
    with pytest.raises(OSError, match="delivery failed"):
        trainer._run_joint_training(
            SimpleNamespace(seed=42),
            recipe,
            "trial",
            tmp_path,
            recording={
                "root": root,
                "protocol": protocol,
                "trial_id": "t",
                "expected_revision": revision,
                "allow_dirty": True,
                "entrypoint": Path(__file__),
            },
        )
    attempt = next(record for record in load_records(root) if record["kind"] == "attempt")
    assert attempt["status"] == "succeeded"
    assert attempt["budgets"][0]["completed"] == 2
