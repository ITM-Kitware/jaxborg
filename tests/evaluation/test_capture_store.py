import json
import shutil

import pytest

from jaxborg.evaluation.capture_store import (
    add_artifact,
    declare_protocol,
    finish_evaluation,
    load_records,
    record_episode,
    start_evaluation,
)
from jaxborg.experiment_records import collate, validate_collation


@pytest.fixture
def protocol():
    return {
        "schema_version": 1,
        "kind": "protocol",
        "id": "study:protocol:1",
        "study_id": "study",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": ["resolved_config", "source_files"], "evaluation": ["resolved_config"]},
        "trials": [
            {"id": name, "condition_id": "policy", "replicate_id": name, "matched_unit_id": None}
            for name in ("trial:one", "trial:two")
        ],
        "metric_definitions": [
            {
                "name": "blue_return",
                "unit": "score",
                "direction": "higher",
                "aggregation": "arithmetic mean",
                "sampling_unit": "episode conditional on fixed policy",
                "uncertainty": "none",
            }
        ],
    }


@pytest.fixture
def begin(tmp_path, protocol):
    model = tmp_path / "model_demo.pt"
    sidecar = tmp_path / "recipe_demo.yaml"
    model.write_bytes(b"weights")
    sidecar.write_text("arch: example\n")
    source = tmp_path / "runner.py"
    source.write_text("VALUE = 1\n")

    def start(**overrides):
        arguments = {
            "protocol": protocol,
            "trial_id": "trial:one",
            "requested_episodes": 2,
            "resolved_config": {"episodes": 2, "stochastic": False},
            "inputs": {"checkpoint": model, "sidecar": sidecar},
            "source_files": {"src/runner.py": source},
        }
        return start_evaluation(tmp_path / "store", **(arguments | overrides))

    return start


def test_failed_retry_and_unlaunched_cell_retain_explicit_scientific_selection(tmp_path, protocol, begin):
    failed = begin()
    record_episode(failed, episode_index=0, seed=42, reward=-4.0, steps=500)
    finish_evaluation(failed, status="failed", error=RuntimeError("episode failed"))
    retry = begin(retry_of=failed.attempt_id)
    record_episode(retry, episode_index=1, seed=43, reward=3.0, policy_seed=101, steps=500)
    record_episode(retry, episode_index=0, seed=42, reward=1.0, policy_seed=100, steps=500)
    finish_evaluation(retry, status="succeeded")
    records = load_records(retry.root)
    result = collate(
        records,
        protocol_id=protocol["id"],
        selected_attempts={"trial:one": retry.attempt_id},
        selected_evaluations=[retry.evaluation_id],
        analysis={"source": "analysis.py:sha256", "configuration": {}},
    )
    assert result["included"] == sorted([retry.attempt_id, retry.evaluation_id])
    assert result["missing"] == [
        {
            "id": "trial:two",
            "trial_id": "trial:two",
            "condition_id": "policy",
            "replicate_id": "trial:two",
            "matched_unit_id": None,
            "reason": "no_attempt",
        }
    ]
    assert {row["id"] for row in result["excluded"]} == {failed.attempt_id, failed.evaluation_id}
    failed_evaluation = next(row for row in records if row["id"] == failed.evaluation_id)
    assert failed_evaluation["budgets"] == [
        {"unit": "episodes", "requested": 2, "completed": 1},
        {
            "unit": "completed_episode_environment_steps",
            "requested": {
                "value": None,
                "reason": "Only completed episode steps are observed",
                "source": "episode observations",
            },
            "completed": 500,
        },
    ]
    selected = next(row for row in records if row["id"] == retry.evaluation_id)
    assert selected["metrics"] == [{"name": "blue_return", "unit": "score", "value": 2.0}]
    assert validate_collation(result) == result
    # A retained collation remains independent from later store contents.
    begin()
    assert validate_collation(result) == result
    assert len([row for row in load_records(retry.root) if row["kind"] == "attempt"]) == 3


def test_inputs_are_retained_with_adjacent_sidecar_and_store_is_relocatable(tmp_path, begin):
    capture = begin()
    assert capture.inputs["checkpoint"].name == "model_demo.pt"
    assert capture.inputs["sidecar"] == capture.inputs["checkpoint"].with_name("recipe_demo.yaml")
    (tmp_path / "model_demo.pt").write_bytes(b"new weights")
    (tmp_path / "runner.py").write_text("VALUE = 2\n")
    assert capture.inputs["checkpoint"].read_bytes() == b"weights"
    assert (capture.directory / "source/src/runner.py").read_text() == "VALUE = 1\n"
    record_episode(capture, episode_index=0, seed=4, reward=1)
    finish_evaluation(capture, status="cancelled")
    original = load_records(capture.root)
    relocated = tmp_path / "relocated"
    shutil.copytree(capture.root, relocated)
    assert load_records(relocated) == original
    assert all(not row["uri"].startswith("/") for row in original if row["kind"] == "artifact")


def test_attempt_exists_before_copy_failure(tmp_path, begin):
    with pytest.raises(ValueError, match="not a file"):
        begin(inputs={"checkpoint": tmp_path / "missing.pt"})
    records = load_records(tmp_path / "store")
    executions = [row for row in records if row["kind"] in {"attempt", "evaluation"}]
    assert len(executions) == 2
    assert all(row["status"] == "failed" and row["budgets"][0]["completed"] == 0 for row in executions)
    assert any(row["role"] == "evaluation failure" for row in records if row["kind"] == "artifact")


def test_rejects_duplicate_episode_incomplete_success_and_writes_after_terminal(begin):
    capture = begin()
    record_episode(capture, episode_index=0, seed=42, reward=0)
    with pytest.raises(FileExistsError):
        record_episode(capture, episode_index=0, seed=42, reward=100)
    with pytest.raises(ValueError, match="all declared episodes"):
        finish_evaluation(capture, status="succeeded")
    assert next(row for row in load_records(capture.root) if row["id"] == capture.evaluation_id)["status"] == "running"
    record_episode(capture, episode_index=1, seed=43, reward=2)
    finish_evaluation(capture, status="succeeded")
    saved = (capture.directory / "terminal.json").read_bytes()
    for action in (
        lambda: record_episode(capture, episode_index=1, seed=43, reward=2),
        lambda: finish_evaluation(capture, status="failed"),
    ):
        with pytest.raises(ValueError, match="already terminal"):
            action()
    assert (capture.directory / "terminal.json").read_bytes() == saved


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_rewards_do_not_create_observations(begin, reward):
    capture = begin()
    with pytest.raises(ValueError, match="finite"):
        record_episode(capture, episode_index=0, seed=42, reward=reward)
    assert not (capture.directory / "episodes").exists()


def test_protocol_and_retry_membership_are_explicit(tmp_path, protocol, begin):
    capture = begin()
    with pytest.raises(ValueError, match="different protocol"):
        declare_protocol(capture.root, protocol | {"revision": "2"})
    with pytest.raises(ValueError, match="terminal attempt"):
        begin(retry_of=capture.attempt_id)
    finish_evaluation(capture, status="cancelled")
    with pytest.raises(ValueError, match="terminal attempt"):
        begin(trial_id="trial:two", retry_of=capture.attempt_id)
    with pytest.raises(ValueError, match="Unknown trial"):
        begin(trial_id="unknown")
    assert len([row for row in load_records(capture.root) if row["kind"] == "attempt"]) == 1


def test_tampered_artifact_blocks_completion_and_verified_loading(begin):
    capture = begin()
    record_episode(capture, episode_index=0, seed=42, reward=1)
    (capture.directory / "episodes/0.json").write_text("{}")
    finish_evaluation(capture, status="failed")
    with pytest.raises(ValueError, match="recorded bytes"):
        load_records(capture.root)
    diagnostic = load_records(capture.root, verify=False)
    evaluation = next(row for row in diagnostic if row["id"] == capture.evaluation_id)
    assert evaluation["status"] == "failed"
    assert "integrity_error" in evaluation["facts"]
    assert evaluation["budgets"][0]["completed"] == 1
    assert (capture.directory / "terminal.json").exists()


def test_output_copies_are_exclusive_and_escaping_paths_rejected(tmp_path, begin):
    capture = begin()
    output = tmp_path / "topology.json"
    output.write_text('{"hosts": ["one"]}')
    identity = add_artifact(capture, output, role="actual topology")
    output.write_text("changed")
    artifact = next(row for row in load_records(capture.root) if row["id"] == identity)
    assert json.loads((capture.root / artifact["uri"]).read_text()) == {"hosts": ["one"]}
    with pytest.raises(FileExistsError):
        add_artifact(capture, output, role="actual topology")
    with pytest.raises(ValueError, match="relative"):
        add_artifact(capture, output, role="actual topology", name="../../escape")
    outside = tmp_path / "outside"
    outside.mkdir()
    (capture.directory / "artifacts/link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        add_artifact(capture, output, role="actual topology", name="link/stolen")
    assert list(outside.iterdir()) == []


def test_invalid_status_rejected_without_changing_evidence(begin):
    capture = begin()
    saved = (capture.directory / "current.json").read_bytes()
    with pytest.raises(ValueError, match="Completion status"):
        finish_evaluation(capture, status="running")
    assert (capture.directory / "current.json").read_bytes() == saved


def test_running_partial_observation_is_discoverable_and_unselected(begin, protocol):
    capture = begin()
    record_episode(capture, episode_index=1, seed=9, reward=-2)
    records = load_records(capture.root)
    evaluation = next(row for row in records if row["id"] == capture.evaluation_id)
    assert evaluation["status"] == "running"
    assert evaluation["budgets"] == [{"unit": "episodes", "requested": 2, "completed": 1}]
    assert len(evaluation["raw_artifacts"]) == 1
    result = collate(
        records,
        protocol_id=protocol["id"],
        selected_attempts={},
        selected_evaluations=[],
        analysis={"source": "analysis", "configuration": {}},
    )
    assert {row["trial_id"]: row["reason"] for row in result["missing"]} == {
        "trial:one": "no_explicit_selection",
        "trial:two": "no_attempt",
    }


def test_metric_semantics_must_match_and_success_cannot_contain_error(begin, protocol):
    protocol["metric_definitions"][0]["aggregation"] = "median"
    capture = begin()
    record_episode(capture, episode_index=0, seed=42, reward=1)
    record_episode(capture, episode_index=1, seed=43, reward=2)
    with pytest.raises(ValueError, match="execution error"):
        finish_evaluation(capture, status="succeeded", error=RuntimeError("failed"))
    finish_evaluation(capture, status="succeeded")
    evaluation = next(row for row in load_records(capture.root) if row["id"] == capture.evaluation_id)
    assert evaluation["metrics"] == []


def test_invalid_retained_status_is_rejected(begin):
    capture = begin()
    snapshot = capture.directory / "current.json"
    records = json.loads(snapshot.read_text())
    records[0]["status"] = "FINISHED"
    snapshot.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="invalid status"):
        load_records(capture.root)


@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_unreadable_episode_evidence_can_be_sealed_as_failed(begin, damage):
    import hashlib

    capture = begin()
    record_episode(capture, episode_index=0, seed=42, reward=1)
    episode = capture.directory / "episodes/0.json"
    if damage == "missing":
        episode.unlink()
    else:
        episode.write_bytes(b"not json")
        snapshot = capture.directory / "current.json"
        records = json.loads(snapshot.read_text())
        artifact = next(row for row in records if row.get("uri", "").endswith("episodes/0.json"))
        artifact["sha256"] = hashlib.sha256(episode.read_bytes()).hexdigest()
        artifact["size_bytes"] = episode.stat().st_size
        snapshot.write_text(json.dumps(records))
    finish_evaluation(capture, status="failed", error=RuntimeError("rollout failed"))
    diagnostic = load_records(capture.root, verify=False)
    evaluation = next(row for row in diagnostic if row["id"] == capture.evaluation_id)
    assert evaluation["status"] == "failed"
    assert "integrity_error" in evaluation["facts"]
    assert "error" in evaluation["facts"]
    with pytest.raises(ValueError):
        load_records(capture.root)


def test_success_rejects_missing_bytes_without_sealing(begin):
    capture = begin()
    record_episode(capture, episode_index=0, seed=42, reward=1)
    record_episode(capture, episode_index=1, seed=43, reward=2)
    capture.inputs["checkpoint"].unlink()
    with pytest.raises(ValueError, match="Cannot verify"):
        finish_evaluation(capture, status="succeeded")
    assert not (capture.directory / "terminal.json").exists()
    finish_evaluation(capture, status="failed")
    assert (
        next(row for row in load_records(capture.root, verify=False) if row["id"] == capture.attempt_id)["status"]
        == "failed"
    )


def test_unpublished_initial_directory_cannot_make_declared_cells_unreadable(begin, protocol, tmp_path, monkeypatch):
    import jaxborg.evaluation.capture_store as store

    def interrupt_publication(*args):
        raise OSError("interrupted before execution publication")

    monkeypatch.setattr(store.os, "rename", interrupt_publication)
    with pytest.raises(OSError, match="interrupted"):
        begin()
    records = load_records(tmp_path / "store")
    assert records == [protocol]
    result = collate(
        records,
        protocol_id=protocol["id"],
        selected_attempts={},
        selected_evaluations=[],
        analysis={"source": "analysis", "configuration": {}},
    )
    assert {row["reason"] for row in result["missing"]} == {"no_attempt"}


@pytest.mark.parametrize(
    "damage", ["mismatched_status", "terminal_status_in_current", "different_identity", "duplicate_episode"]
)
def test_snapshot_consistency_is_checked(begin, damage):
    capture = begin()
    record_episode(capture, episode_index=0, seed=42, reward=1)
    snapshot = capture.directory / "current.json"
    records = json.loads(snapshot.read_text())
    if damage == "mismatched_status":
        records[0]["status"] = "failed"
    elif damage == "terminal_status_in_current":
        records[0]["status"] = records[1]["status"] = "failed"
    elif damage == "different_identity":
        old = records[0]["id"]
        records[0]["id"] = "another-attempt"
        records[1]["attempt_id"] = "another-attempt"
        for record in records[2:]:
            if record["producer_id"] == old:
                record["producer_id"] = "another-attempt"
    else:
        duplicate = next(row.copy() for row in records if row.get("role") == "evaluation episode observation")
        duplicate["id"] = "another-observation"
        records.append(duplicate)
        records[1]["raw_artifacts"].append(duplicate["id"])
    snapshot.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        load_records(capture.root)


@pytest.mark.parametrize("damage", ["mean", "step_budget", "configuration_fact", "source_fact", "missing_config_fact"])
def test_verified_reads_recompute_derived_evidence(begin, damage):
    capture = begin()
    record_episode(capture, episode_index=0, seed=42, reward=1, steps=500)
    record_episode(capture, episode_index=1, seed=43, reward=3, steps=400)
    finish_evaluation(capture, status="succeeded")
    snapshot = capture.directory / "terminal.json"
    records = json.loads(snapshot.read_text())
    if damage == "mean":
        records[1]["metrics"][0]["value"] = 1000
    elif damage == "step_budget":
        for owner in records[:2]:
            owner["budgets"][1]["completed"] = 1000
    elif damage == "configuration_fact":
        records[0]["facts"]["resolved_config"]["value"]["stochastic"] = 0
    elif damage == "source_fact":
        records[1]["facts"]["source_files"]["value"]["src/runner.py"] = "f" * 64
    else:
        del records[1]["facts"]["resolved_config"]
    snapshot.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        load_records(capture.root)
    assert load_records(capture.root, verify=False)


def test_verified_reads_crosscheck_source_manifest_against_retained_source_files(begin):
    import hashlib

    capture = begin()
    finish_evaluation(capture, status="cancelled")
    snapshot = capture.directory / "terminal.json"
    records = json.loads(snapshot.read_text())
    manifest = capture.directory / "source-files.json"
    forged = {"src/runner.py": "f" * 64}
    manifest.write_text(json.dumps(forged))
    artifact = next(row for row in records if row.get("role") == "evaluation source manifest")
    artifact["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    artifact["size_bytes"] = manifest.stat().st_size
    for owner in records[:2]:
        owner["facts"]["source_files"]["value"] = forged
    snapshot.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Source manifest differs"):
        load_records(capture.root)
    assert load_records(capture.root, verify=False)


def test_verified_reads_reject_aggregate_metrics_on_failed_evaluation(begin):
    capture = begin()
    record_episode(capture, episode_index=0, seed=42, reward=1)
    finish_evaluation(capture, status="failed")
    snapshot = capture.directory / "terminal.json"
    records = json.loads(snapshot.read_text())
    records[1]["metrics"] = [{"name": "blue_return", "unit": "score", "value": 1}]
    snapshot.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Metrics differ"):
        load_records(capture.root)


def test_verified_reads_allow_partial_source_capture_before_setup_failure(begin, tmp_path):
    source = tmp_path / "first.py"
    source.write_text("FIRST = True\n")
    with pytest.raises(ValueError, match="not a file"):
        begin(source_files={"first.py": source, "missing.py": tmp_path / "missing.py"})
    records = load_records(tmp_path / "store")
    evaluation = next(row for row in records if row["kind"] == "evaluation")
    assert evaluation["status"] == "failed"
    assert "resolved_config" in evaluation["facts"]
    assert "source_files" not in evaluation["facts"]
    assert any(row.get("role") == "evaluation source" for row in records)
