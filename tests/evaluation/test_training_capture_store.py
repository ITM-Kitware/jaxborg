"""Training work, owned policy inputs, corrections and interrupted attempts."""

import copy
import json
import subprocess
import sys

import pytest

from jaxborg.evaluation.capture_store import (
    add_artifact,
    finish_evaluation,
    finish_training,
    load_records,
    record_episode,
    record_training_update,
    start_evaluation,
    start_training,
)
from jaxborg.evaluation.provenance import writer_identity
from jaxborg.evaluation.reconcile import reconcile_interrupted
from jaxborg.experiment_discovery.package import prepare_package, verify_package
from jaxborg.experiment_records import collate, validate_collation, validate_records


def protocol():
    return {
        "schema_version": 1,
        "kind": "protocol",
        "id": "p",
        "study_id": "study",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": [], "evaluation": []},
        "trials": [
            {"id": name, "condition_id": "control", "replicate_id": name, "matched_unit_id": "cohort"}
            for name in ("train", "eval", "unlaunched")
        ],
        "metric_definitions": [
            {
                "name": "blue_return",
                "unit": "score",
                "direction": "higher",
                "aggregation": "mean",
                "sampling_unit": "episode",
                "uncertainty": "none",
            }
        ],
    }


def train(root, **settings):
    return start_training(
        root,
        protocol=protocol(),
        trial_id="train",
        requested_steps=settings.pop("requested_steps", 4),
        requested_updates=settings.pop("requested_updates", 2),
        resolved_config=settings.pop("resolved_config", {}),
        inputs={},
        source_files={},
        **settings,
    )


def update(capture, number):
    record_training_update(
        capture,
        {
            "update_idx": number,
            "env_steps": 2 * number,
            "team.blue.update_count": number,
            "team.red.update_count": number // 2,
        },
    )


def evaluate(root, checkpoint, **settings):
    capture = start_evaluation(
        root,
        protocol=protocol(),
        trial_id="eval",
        requested_episodes=1,
        resolved_config={},
        inputs={},
        retained_inputs={"policy": checkpoint},
        source_files={},
        **settings,
    )
    record_episode(capture, episode_index=0, seed=7, reward=3.0, steps=2)
    finish_evaluation(capture, status="succeeded")
    return capture


def selection(records, training, evaluation):
    return collate(
        records,
        protocol_id="p",
        selected_attempts={"train": training.attempt_id, "eval": evaluation.attempt_id},
        selected_evaluations=[evaluation.evaluation_id],
        analysis={"source": "fixture", "configuration": {}},
    )


def test_training_failure_retry_owned_evaluation_and_correction_keep_original_evidence(tmp_path):
    root = tmp_path / "store"
    failed = train(root)
    update(failed, 1)
    finish_training(failed, status="failed", error=RuntimeError("deliberate fixture failure"))
    retried = train(root, retry_of=failed.attempt_id)
    update(retried, 1)
    with pytest.raises(ValueError, match="all declared updates"):
        finish_training(retried, status="succeeded")
    update(retried, 2)
    weights = tmp_path / "policy.bin"
    weights.write_bytes(b"retained fixture weights")
    checkpoint = add_artifact(retried, weights, role="joint policy checkpoint")
    finish_training(retried, status="succeeded")
    first = evaluate(root, checkpoint)
    before = selection(load_records(root), retried, first)
    original = copy.deepcopy(before)
    correction = {
        "evaluation_id": first.evaluation_id,
        "reason": "fixture procedure correction",
        "changes": {"procedure": "corrected fixture source identity"},
    }
    corrected = evaluate(root, checkpoint, correction=correction)
    records = load_records(root)
    after = selection(records, retried, corrected)
    assert before == original == validate_collation(before)
    assert before["sha256"] != after["sha256"]
    assert after["rows"][-1]["correction"]["values"] == correction
    assert after["missing"][0]["trial_id"] == "unlaunched"
    indexed = validate_records(records)
    assert checkpoint in indexed[corrected.evaluation_id]["inputs"]
    assert indexed[checkpoint]["producer_id"] == retried.attempt_id
    assert indexed[failed.attempt_id]["budgets"][0]["completed"] == 2
    assert indexed[retried.attempt_id]["budgets"][-2:][0]["completed"] == 2
    assert indexed[retried.attempt_id]["budgets"][-1]["completed"] == 1
    package = tmp_path / "package"
    manifest = prepare_package("evaluation-capture", root, package)
    assert verify_package(package) == manifest
    broken = copy.deepcopy(records)
    current = next(record for record in broken if record["id"] == corrected.evaluation_id)
    current["extensions"]["evaluation_correction"]["values"]["evaluation_id"] = "absent"
    with pytest.raises(ValueError, match="dangling"):
        validate_records(broken)


def test_running_writer_blocks_reconciliation_and_dead_writer_retains_interruption(tmp_path):
    root = tmp_path / "live"
    current = train(root, resolved_config={"execution": {"writer": writer_identity()}})
    with pytest.raises(ValueError, match="still alive"):
        reconcile_interrupted(root, current.attempt_id, reason="operator inspected fixture")
    with pytest.raises(ValueError, match="Running capture"):
        prepare_package("evaluation-capture", root, tmp_path / "running-package")
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    assert child.wait() == 0
    root = tmp_path / "interrupted"
    stopped = train(root, resolved_config={"execution": {"writer": {**writer_identity(), "pid": child.pid}}})
    update(stopped, 1)
    records = reconcile_interrupted(root, stopped.attempt_id, reason="writer exited after completed update")
    attempt = next(record for record in records if record["id"] == stopped.attempt_id)
    assert attempt["status"] == "cancelled" and attempt["budgets"][0]["completed"] == 2
    assert attempt["facts"]["interruption"]["value"]["check"] == "local writer PID absent"
    retry = train(root, retry_of=stopped.attempt_id)
    assert retry.attempt_id != stopped.attempt_id


@pytest.mark.parametrize("key", ["hostname", "boot_id", "pid_namespace"])
def test_reconciliation_rejects_unverifiable_writer_environment(tmp_path, key):
    identity = writer_identity()
    identity[key] = "different environment"
    capture = train(tmp_path, resolved_config={"execution": {"writer": identity}})
    with pytest.raises(ValueError, match="original writer environment"):
        reconcile_interrupted(tmp_path, capture.attempt_id, reason="operator inspected fixture")
    attempt = next(record for record in load_records(tmp_path) if record["id"] == capture.attempt_id)
    assert attempt["status"] == "running"


def test_training_failure_seals_corrupt_evidence_without_claiming_verified_success(tmp_path):
    root = tmp_path / "store"
    capture = train(root)
    update(capture, 1)
    source = tmp_path / "checkpoint.bin"
    source.write_bytes(b"checkpoint")
    identity = add_artifact(capture, source, role="joint policy checkpoint")
    artifact = next(record for record in load_records(root) if record["id"] == identity)
    (root / artifact["uri"]).write_bytes(b"corrupt")
    finish_training(capture, status="failed", error=RuntimeError("execution failed"))
    records = load_records(root, verify=False)
    attempt = next(record for record in records if record["id"] == capture.attempt_id)
    assert attempt["status"] == "failed" and "integrity_error" in attempt["facts"]
    assert (capture.directory / "terminal.json").exists()
    with pytest.raises(ValueError, match="differs from its recorded bytes"):
        load_records(root)


@pytest.mark.parametrize(
    "changes",
    [
        {"update_idx": 3},
        {"update_idx": True},
        {"env_steps": 2},
        {"env_steps": 5},
        {"env_steps": True},
        {"team.blue.update_count": 0},
        {"team.blue.update_count": 3},
        {"team.red.update_count": 0.5},
        {"team.red.update_count": True},
        {"env_steps": -1},
        {"team.red.update_count": -1},
        {"team.red.update_count": float("nan")},
        {"team.red.update_count": float("inf")},
    ],
)
def test_invalid_update_can_be_corrected_without_losing_or_claiming_work(tmp_path, changes):
    capture = train(tmp_path)
    update(capture, 1)
    snapshot = (capture.directory / "current.json").read_bytes()
    row = {"update_idx": 2, "env_steps": 4, "team.blue.update_count": 2, "team.red.update_count": 1}
    with pytest.raises(ValueError):
        record_training_update(capture, row | changes)
    assert (capture.directory / "current.json").read_bytes() == snapshot
    assert not (capture.directory / "updates/2.json").exists()
    update(capture, 2)
    with pytest.raises(ValueError, match="exceed the requested budget"):
        update(capture, 3)
    assert not (capture.directory / "updates/3.json").exists()
    finish_training(capture, status="succeeded")
    attempt = next(row for row in load_records(tmp_path) if row["id"] == capture.attempt_id)
    assert [budget["completed"] for budget in attempt["budgets"]] == [4, 2, 2, 1]


def test_larger_training_history_records_partial_failure_and_retry_without_reopening_prior_events(tmp_path):
    capture = train(tmp_path, requested_steps=256, requested_updates=128)
    for number in range(1, 49):
        update(capture, number)
    observation = capture.directory / "updates/1.json"
    original = observation.read_bytes()
    observation.unlink()
    for number in range(49, 97):
        update(capture, number)
    snapshot = json.loads((capture.directory / "current.json").read_text())
    assert [budget["completed"] for budget in snapshot[0]["budgets"]] == [192, 96, 96, 48]
    with pytest.raises(ValueError, match="Cannot verify"):
        finish_training(capture, status="succeeded")
    finish_training(capture, status="failed", error=RuntimeError("interrupted fixture"))
    failed = next(row for row in load_records(tmp_path, verify=False) if row["id"] == capture.attempt_id)
    assert failed["status"] == "failed"
    assert [budget["completed"] for budget in failed["budgets"]] == [192, 96, 96, 48]
    assert "integrity_error" in failed["facts"]
    with pytest.raises(ValueError, match="Cannot verify"):
        load_records(tmp_path)
    observation.write_bytes(original)
    retry = train(tmp_path, retry_of=capture.attempt_id)
    update(retry, 1)
    update(retry, 2)
    finish_training(retry, status="succeeded")
    records = load_records(tmp_path)
    attempts = [row for row in records if row["kind"] == "attempt"]
    assert {row["status"] for row in attempts} == {"failed", "succeeded"}
    assert next(row for row in attempts if row["id"] == retry.attempt_id)["retry_of"] == capture.attempt_id
    assert len([row for row in records if row.get("role") == "training update observation"]) == 98


def test_evaluation_seal_verifies_owned_input_bytes_again(tmp_path):
    training = train(tmp_path)
    update(training, 1)
    update(training, 2)
    source = tmp_path / "policy.bin"
    source.write_bytes(b"checkpoint")
    checkpoint = add_artifact(training, source, role="joint policy checkpoint")
    finish_training(training, status="succeeded")
    capture = start_evaluation(
        tmp_path,
        protocol=protocol(),
        trial_id="eval",
        requested_episodes=1,
        resolved_config={},
        inputs={},
        retained_inputs={"policy": checkpoint},
        source_files={},
    )
    record_episode(capture, episode_index=0, seed=7, reward=3)
    capture.inputs["policy"].write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="differs from its recorded bytes"):
        finish_evaluation(capture, status="succeeded")
    assert not (capture.directory / "terminal.json").exists()
    finish_evaluation(capture, status="failed")
    evaluation = next(row for row in load_records(tmp_path, verify=False) if row["id"] == capture.evaluation_id)
    assert evaluation["status"] == "failed" and "integrity_error" in evaluation["facts"]


def test_numeric_initial_state_is_owned_and_readable_without_pickle(tmp_path):
    import json
    from types import SimpleNamespace

    import jax.numpy as jnp
    import numpy as np

    from jaxborg.evaluation.training_capture import retain_initial_state

    root = tmp_path / "store"
    capture = train(root)
    retain_initial_state(
        capture,
        train_states={"blue": SimpleNamespace(opt_state={"mu": jnp.zeros(2)}, step=0)},
        reward_norm_states={"blue": {"count": jnp.zeros(1)}},
        rng=jnp.asarray([42, 7], dtype=jnp.uint32),
        env_state={"time": jnp.int32(0)},
        obs={"blue": jnp.zeros(3)},
    )
    records = {record["id"]: record for record in load_records(root)}
    value = records[capture.attempt_id]["facts"]["initial_state"]["value"]
    snapshot, descriptor = (records[value[key]] for key in ("numeric_state_artifact", "manifest_artifact"))
    assert snapshot["producer_id"] == descriptor["producer_id"] == capture.attempt_id
    leaves = json.loads((root / descriptor["uri"]).read_text())["leaves"]
    with np.load(root / snapshot["uri"], allow_pickle=False) as arrays:
        by_path = {leaf["path"]: arrays[leaf["key"]] for leaf in leaves}
        np.testing.assert_array_equal(by_path["['rng']"], [42, 7])
        np.testing.assert_array_equal(by_path["['normalizer']['blue']['count']"], [0])
        assert all(list(arrays[leaf["key"]].shape) == leaf["shape"] for leaf in leaves)
    assert value["exact_continuation_supported"] is False


def test_reconciliation_repairs_existing_tracking_and_terminal_sync_is_retryable(tmp_path, monkeypatch):
    import hashlib

    import mlflow

    from jaxborg.experiment_discovery.execution import sync_execution

    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    assert child.wait() == 0
    root = tmp_path / "evidence"
    capture = train(
        root, track=True, resolved_config={"execution": {"writer": {**writer_identity(), "pid": child.pid}}}
    )
    update(capture, 1)
    client = mlflow.MlflowClient(tracking_uri=uri)
    deliver = mlflow.MlflowClient.log_dict

    def failed_delivery(*args, **kwargs):
        raise OSError("offline projection")

    monkeypatch.setattr(mlflow.MlflowClient, "log_dict", failed_delivery)
    records = reconcile_interrupted(root, capture.attempt_id, reason="writer has exited")
    assert next(record for record in records if record["id"] == capture.attempt_id)["status"] == "cancelled"
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(OSError, match="unchanged"):
        sync_execution(root, capture.attempt_id)
    monkeypatch.setattr(mlflow.MlflowClient, "log_dict", deliver)
    receipt = sync_execution(root, capture.attempt_id)
    assert receipt["run_id"] == capture.tracking.run_id
    assert receipt["delivery"] == "current" and receipt["data_incomplete"] is False
    repaired = client.get_run(receipt["run_id"])
    assert repaired.info.status == "KILLED"
    assert repaired.data.tags["jaxborg.discovery.scientific_status"] == "cancelled"
    assert repaired.data.tags["jaxborg.discovery.delivery"] == "current"
    assert repaired.data.tags["jaxborg.discovery.data_incomplete"] == "false"
    assert before == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()
    }
    # Ambiguous publication never chooses one of the matches or creates another.
    client.create_run(repaired.info.experiment_id, tags=dict(repaired.data.tags))
    with pytest.raises(ValueError, match="Multiple execution runs"):
        sync_execution(root, capture.attempt_id)
    assert len(client.search_runs([repaired.info.experiment_id])) == 2
