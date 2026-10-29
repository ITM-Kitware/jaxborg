"""Exercise durable training and tracking delivery without compiling a campaign."""

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jaxborg.evaluation.capture_store import load_records
from jaxborg.experiment_discovery import execution
from jaxborg.mlflow_setup import tracking_client
from jaxborg.recipe import load
from scripts.train.algorithms import ippo_jax as trainer

PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def training(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(tmp_path / "tracking"))
    recipe = load(str(PROJECT / "recipes/cotraining/cotraining.yaml"))
    topology = tmp_path / "topology.npz"
    np.savez(topology, fixture=np.asarray([1]))
    recipe["train"] = {
        "teams": "both",
        "variant": "cc4_stock",
        "episode_length": 2,
        "total_timesteps": 4,
        "topology_bank": [str(topology)],
    }
    recipe["arch"] = {"name": "separate", "hidden_dim": 4, "hidden_layers": 1}
    recipe["jax"] = {"num_envs": 1, "num_minibatches": 1, "update_epochs": 1}
    recipe["eval"] = {"variant": "cc4_stock", "policy_backend": "jax"}
    recording = {
        "root": tmp_path / "evidence",
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
        "expected_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "allow_dirty": True,
        "entrypoint": Path(__file__),
        "supplied_recipe": copy.deepcopy(recipe),
    }
    return recipe, recording


def test_joint_trainer_failure_is_recorded_before_initialization_and_retry(tmp_path, monkeypatch, training):
    recipe, recording = training
    root = recording["root"]
    observed = []

    def fail_initialization(*args, **kwargs):
        records = load_records(root)
        active = [record for record in records if record["kind"] == "attempt" and record["status"] == "running"]
        observed.append(active[0]["id"])
        assert active[0]["budgets"][0]["completed"] == 0
        configuration = next(root.glob("attempts/*/configuration.json"))
        assert json.loads(configuration.read_text())["recipe"]["meta"]["added"] == "2026-08-16"
        raise RuntimeError("deliberate initialization failure")

    def unexpected(*args, **kwargs):
        pytest.fail("Recorded training must not open a fluent MLflow run")

    monkeypatch.setattr(trainer, "make_joint_train", fail_initialization)
    monkeypatch.setattr(trainer, "start_run", unexpected)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="initialization failure"):
            trainer._run_joint_training(SimpleNamespace(seed=42), recipe, "trial", tmp_path, recording=recording)
        recording["retry_of"] = observed[-1]
    attempts = [record for record in load_records(root) if record["kind"] == "attempt"]
    runs = tracking_client().search_runs(["1"])
    assert len(attempts) == len(runs) == 2
    assert trainer.mlflow.active_run() is None
    assert {run.data.tags["jaxborg.discovery.attempt_id"] for run in runs} == set(observed)
    assert all(run.info.status == "FAILED" for run in runs)
    assert all(record["status"] == "failed" and record["budgets"][0]["completed"] == 0 for record in attempts)
    assert next(record for record in attempts if record["id"] == observed[1])["retry_of"] == observed[0]


@pytest.fixture
def synthetic_rollouts(monkeypatch):
    """Replace only scientific computation; use the trainer's actual delivery loop."""
    calls = []

    def make_joint_train(configs, *args, **kwargs):
        for config in configs.values():
            config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"] * config["NUM_STEPS"])
        states = {
            team: SimpleNamespace(params={"weight": np.zeros(1, dtype=np.float32)}, opt_state={"count": 0}, step=0)
            for team in ("blue", "red")
        }
        values = dict.fromkeys(
            [
                "actor_loss",
                "critic_loss",
                "entropy",
                "total_loss",
                "approx_kl",
                "clip_frac",
                "explained_var",
                "grad_norm",
                "pre_clip_grad_norm",
                "raw_rollout_return",
                "mean_rollout_return",
                "actor_fraction",
                "critic_fraction",
                "updated",
                *trainer.joint_reward_components,
            ],
            1.0,
        )

        def collect(states, env_state, obs, rng, normalizers, *, update_idx, **kwargs):
            calls.append(update_idx)
            metrics = {team: {**values, "update_count": update_idx + 1} for team in states}
            metrics["game"] = dict.fromkeys(["blue_return", "red_return", *trainer.joint_game_counters], 1.0)
            return states, env_state, obs, rng, normalizers, metrics

        return None, {"blue": np.zeros(1)}, {"time": 0}, lambda key: states, collect

    monkeypatch.setattr(trainer, "make_joint_train", make_joint_train)
    return calls


@pytest.mark.parametrize("failure", [None, "creation", "metadata", "metrics", "artifacts"])
def test_recorded_training_completes_independently_of_tracking_delivery(
    tmp_path, monkeypatch, training, synthetic_rollouts, failure, capsys
):
    recipe, recording = training
    client = tracking_client()
    deliveries = []

    def unexpected(*args, **kwargs):
        pytest.fail("Recorded training must not use fluent tracking or create a fallback run")

    for name in ("log_metrics", "log_artifact", "end_run"):
        monkeypatch.setattr(trainer.mlflow, name, unexpected)
    monkeypatch.setattr(trainer, "start_run", unexpected)
    if failure == "creation":

        def unavailable():
            raise OSError("tracking unavailable")

        monkeypatch.setattr(execution, "tracking_client", unavailable)
    elif failure in {"metrics", "metadata"}:
        log_batch = type(client).log_batch

        def deliver(self, run_id, *args, metrics=(), params=(), **kwargs):
            if (metrics and failure == "metrics") or (params and failure == "metadata"):
                deliveries.append(run_id)
                raise OSError(f"{failure} delivery unavailable")
            return log_batch(self, run_id, *args, metrics=metrics, params=params, **kwargs)

        monkeypatch.setattr(type(client), "log_batch", deliver)
    elif failure == "artifacts":
        log_artifact = type(client).log_artifact

        def unavailable_artifacts(*args, **kwargs):
            deliveries.append(args)
            raise OSError("artifact delivery unavailable")

        monkeypatch.setattr(type(client), "log_artifact", unavailable_artifacts)

    capture = trainer._run_joint_training(
        SimpleNamespace(seed=42), recipe, "trial", tmp_path / "outputs", recording=recording
    )
    records = load_records(recording["root"])
    attempt = next(record for record in records if record["kind"] == "attempt")
    assert attempt["status"] == "succeeded"
    assert {budget["unit"]: budget["completed"] for budget in attempt["budgets"]} == {
        "environment_steps": 4,
        "joint_rollout_updates": 2,
        "blue_policy_updates": 2,
        "red_policy_updates": 2,
    }
    assert synthetic_rollouts == [0, 1]
    rows = [json.loads(line) for line in (tmp_path / "outputs/metrics.jsonl").read_text().splitlines()]
    assert [row["env_steps"] for row in rows] == [2, 4]
    assert [row["team.blue.update_count"] for row in rows] == [1, 2]
    assert (tmp_path / "outputs/model_trial.safetensors").is_file()
    assert "Done in" in capsys.readouterr().out
    assert trainer.mlflow.active_run() is None
    runs = client.search_runs(["1"]) if failure != "creation" else []
    if failure == "creation":
        assert capture.tracking is None
        monkeypatch.setattr(execution, "tracking_client", tracking_client)
        with pytest.raises(ValueError, match="No existing execution run"):
            execution.sync_execution(recording["root"], attempt["id"])
        assert not client.search_runs([item.experiment_id for item in client.search_experiments()])
        return
    assert len(runs) == 1
    run = runs[0]
    assert run.info.run_id == capture.tracking.run_id
    assert run.data.tags["jaxborg.discovery.attempt_id"] == attempt["id"]
    assert run.data.tags["jaxborg.discovery.scientific_status"] == "succeeded"
    if failure != "metadata":
        assert run.data.tags["recipe.name"] == recipe["meta"]["name"]
        assert run.data.tags["seed"] == "42"
        assert run.data.params["recipe.train.total_timesteps"] == "4"
        assert run.info.run_name == f"{recipe['algorithm']}-jax-{recipe['meta']['name']}-seed42"
    if failure:
        assert deliveries
        assert run.info.status == "FAILED"
        assert run.data.tags["jaxborg.discovery.delivery"] == "failed"
        assert run.data.tags["jaxborg.discovery.data_incomplete"] == "true"
    else:
        assert run.info.status == "FINISHED"
        assert run.data.tags["jaxborg.discovery.delivery"] == "current"
        curve = client.get_metric_history(run.info.run_id, "team.blue.return")
        assert [(metric.step, metric.value) for metric in curve] == [(2, 1.0), (4, 1.0)]
    sidecar = trainer.read_sidecar(tmp_path / "outputs/model_trial.safetensors")
    assert sidecar["run"]["train_run_id"] == run.info.run_id
    if failure:
        if failure == "artifacts":
            monkeypatch.setattr(type(client), "log_artifact", log_artifact)
        else:
            monkeypatch.setattr(type(client), "log_batch", log_batch)
        if failure == "metrics":
            assert client.get_metric_history(run.info.run_id, "team.blue.return") == []
        terminal = capture.directory / "terminal.json"
        before = terminal.read_bytes()
        receipt = execution.sync_execution(recording["root"], attempt["id"])
        assert receipt["run_id"] == run.info.run_id
        assert receipt["delivery"] == "failed"
        assert receipt["data_incomplete"] is True
        assert terminal.read_bytes() == before
        assert len(client.search_runs([run.info.experiment_id])) == 1
        repaired = client.get_run(run.info.run_id)
        assert repaired.info.status == "FAILED"
        assert repaired.data.tags["jaxborg.discovery.scientific_status"] == "succeeded"
        assert repaired.data.tags["jaxborg.discovery.delivery"] == "failed"
        assert repaired.data.tags["jaxborg.discovery.data_incomplete"] == "true"
        if failure == "metrics":
            assert client.get_metric_history(run.info.run_id, "team.blue.return") == []
        elif failure == "metadata":
            assert repaired.data.params == {}
        else:
            assert "model_trial.safetensors" not in {
                artifact.path for artifact in client.list_artifacts(run.info.run_id)
            }


def test_unrecorded_training_keeps_fluent_curves_and_completion(tmp_path, training, synthetic_rollouts):
    recipe, _ = training
    assert trainer._run_joint_training(SimpleNamespace(seed=42), recipe, "trial", tmp_path / "outputs") is None
    client = tracking_client()
    runs = client.search_runs(["1"])
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"
    assert "jaxborg.discovery.role" not in runs[0].data.tags
    curve = client.get_metric_history(runs[0].info.run_id, "team.blue.return")
    assert [(metric.step, metric.value) for metric in curve] == [(2, 1.0), (4, 1.0)]
    assert trainer.mlflow.active_run() is None
