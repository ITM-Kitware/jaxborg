"""Native transport failure and immutable publication behavior."""

import json
import runpy
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from jaxborg.evaluation.capture_store import declare_protocol, finish_evaluation, start_evaluation
from jaxborg.experiment_discovery import mlflow as transport
from jaxborg.experiment_discovery.package import prepare_package


@pytest.fixture
def publication(tmp_path):
    protocol = {
        "schema_version": 1,
        "kind": "protocol",
        "id": "study:protocol:1",
        "study_id": "study",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": [], "evaluation": []},
        "trials": [{"id": "trial:unlaunched", "condition_id": "A", "replicate_id": "1", "matched_unit_id": None}],
        "metric_definitions": [],
    }
    declare_protocol(tmp_path / "store", protocol)
    package = tmp_path / "package"
    manifest = prepare_package("evaluation-capture", tmp_path / "store", package)
    uri = "sqlite:///" + str(tmp_path / "tracking.sqlite")
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment("evidence", artifact_location=(tmp_path / "artifacts").as_uri())
    return package, manifest, uri, client


def test_sequential_reuse_fetch_and_unlaunched_protocol_are_explicit(publication, tmp_path):
    package, manifest, uri, client = publication
    original_tracking = mlflow.get_tracking_uri()
    receipt = transport.publish_package(package, uri, "evidence")
    assert receipt["state"] == "verified" and not receipt["reused"]
    repeated = transport.publish_package(package, uri, "evidence")
    assert repeated == receipt | {"reused": True}
    result = transport.search_packages(uri, "evidence", study="study", trial="trial:unlaunched")
    assert len(result["publications"]) == 1
    inspected = transport.inspect_publication(uri, receipt["run_id"], manifest["sha256"])
    assert inspected["attempts"] == [] and inspected["evaluations"] == []
    assert inspected["missing"] == [
        {"protocol_id": "study:protocol:1", "trial_id": "trial:unlaunched", "reason": "no_attempt"}
    ]
    fetched = transport.fetch_package(uri, receipt["run_id"], manifest["sha256"], tmp_path / "consumer")
    assert json.loads((tmp_path / "consumer/manifest.json").read_text()) == manifest
    assert fetched["paths"]["materialization"] == str(tmp_path / "consumer/manifest.json")
    assert len(client.search_runs([client.get_experiment_by_name("evidence").experiment_id])) == 1
    assert mlflow.get_tracking_uri() == original_tracking
    with pytest.raises(ValueError, match="Output already exists"):
        transport.fetch_package(uri, receipt["run_id"], manifest["sha256"], tmp_path / "consumer")


def test_fetch_rejects_checkout_destination_before_creating_it(publication):
    package, manifest, uri, _ = publication
    receipt = transport.publish_package(package, uri, "evidence")
    destination = Path(__file__).resolve().parents[2] / "retrieved-evidence-forbidden"
    assert not destination.exists()
    with pytest.raises(ValueError, match="outside the source checkout"):
        transport.fetch_package(uri, receipt["run_id"], manifest["sha256"], destination)
    assert not destination.exists()


def test_native_indexes_project_real_frozen_producer_backend_and_source(publication, tmp_path):
    _, _, uri, client = publication
    project = Path(__file__).resolve().parents[2]
    fixture = runpy.run_path(str(project / "experiments/frozen-defender-response/test_analyze.py"))["SyntheticBundle"]
    bundle = fixture(tmp_path / "frozen")
    package = tmp_path / "frozen-package"
    manifest = prepare_package("frozen-defender", bundle.root, package)
    receipt = transport.publish_package(package, uri, "evidence")
    inspected = transport.inspect_publication(uri, receipt["run_id"], manifest["sha256"])
    records = {record["id"]: record for record in inspected["records"]}
    for record_id, run_id in inspected["index_runs"].items():
        record = records[record_id]
        tags = client.get_run(run_id).data.tags
        if record["kind"] == "attempt":
            assert tags[transport.TAG_PREFIX + "source_revision"] == record["facts"]["source"]["value"]["git_commit"]
        else:
            assert tags[transport.TAG_PREFIX + "backend"] == record["facts"]["policy_backend"]["value"]
            assert tags[transport.TAG_PREFIX + "policy_backend"] == record["facts"]["policy_backend"]["value"]
            if "eval_env" in record["facts"]:
                assert tags[transport.TAG_PREFIX + "eval_env"] == record["facts"]["eval_env"]["value"]


@pytest.mark.parametrize("known", [False, True])
def test_native_capture_indexes_preserve_known_and_unknown_facts(publication, tmp_path, known):
    _, _, uri, client = publication
    protocol = json.loads((tmp_path / "store/protocol.json").read_text())
    unknown = {"value": None, "reason": "Not retained by producer", "source": "input declaration"}
    facts = (
        {
            "policy_backend": {"value": "torch", "source": "input declaration"},
            "eval_env": {"value": "cyborg", "source": "input declaration"},
            "source": {"value": {"git_commit": "a" * 40}, "source": "input declaration"},
        }
        if known
        else {"policy_backend": unknown, "eval_env": unknown, "source": unknown}
    )
    capture = start_evaluation(
        tmp_path / "store",
        protocol=protocol,
        trial_id="trial:unlaunched",
        requested_episodes=1,
        resolved_config={},
        inputs={},
        source_files={},
        facts=facts,
    )
    finish_evaluation(capture, status="cancelled")
    package = tmp_path / "unknown-package"
    manifest = prepare_package("evaluation-capture", tmp_path / "store", package)
    receipt = transport.publish_package(package, uri, "evidence")
    inspected = transport.inspect_publication(uri, receipt["run_id"], manifest["sha256"])
    for run_id in inspected["index_runs"].values():
        tags = client.get_run(run_id).data.tags
        if known:
            assert all(
                tags[transport.TAG_PREFIX + key] == value
                for key, value in {
                    "backend": "torch",
                    "policy_backend": "torch",
                    "eval_env": "cyborg",
                    "source_revision": "a" * 40,
                }.items()
            )
        else:
            assert all(
                transport.TAG_PREFIX + name not in tags
                for name in ("backend", "policy_backend", "eval_env", "source_revision")
            )
    executions = [record for record in inspected["records"] if record["kind"] in {"attempt", "evaluation"}]
    assert all(
        record["facts"][name] == facts[name]
        for record in executions
        for name in ("policy_backend", "eval_env", "source")
    )


def test_interrupted_publication_reports_candidate_and_never_creates_a_replacement(publication, monkeypatch):
    package, manifest, uri, client = publication
    original = client.log_artifact

    def lose_marker_response(run_id, local_path, artifact_path=None):
        original(run_id, local_path, artifact_path)
        raise RuntimeError("response lost after completion upload")

    with monkeypatch.context() as patch:
        patch.setattr(client, "log_artifact", lose_marker_response)
        patch.setattr(transport, "_client", lambda tracking_uri: client)
        with pytest.raises(transport.PublicationConflict, match="outcome is uncertain") as failure:
            transport.publish_package(package, uri, "evidence")
    candidate = failure.value.candidate_run_ids[0]
    assert transport.inspect_publication(uri, candidate, manifest["sha256"]) == {
        "run_id": candidate,
        "sha256": manifest["sha256"],
        "state": "provisional",
        "reused": False,
        "available": False,
    }
    search = transport.search_packages(uri, "evidence", study="study")
    assert search["publications"][0]["available"] is False
    with pytest.raises(transport.PublicationConflict, match="cannot be reused") as retry:
        transport.publish_package(package, uri, "evidence")
    assert retry.value.candidate_run_ids == [candidate]
    assert len(client.search_runs([client.get_experiment_by_name("evidence").experiment_id])) == 1


def test_changed_completion_or_bytes_cannot_expose_a_fetch_destination(publication, tmp_path):
    package, manifest, uri, client = publication
    receipt = transport.publish_package(package, uri, "evidence")
    marker = tmp_path / "completion.json"
    marker.write_text(
        json.dumps({"schema_version": 1, "run_id": "wrong", "sha256": manifest["sha256"], "index_runs": {}})
    )
    client.log_artifact(receipt["run_id"], str(marker))
    with pytest.raises(ValueError, match="Completion marker"):
        transport.fetch_package(uri, receipt["run_id"], manifest["sha256"], tmp_path / "bad-marker")
    assert not (tmp_path / "bad-marker").exists()
    marker.write_text(
        json.dumps({"schema_version": 1, "run_id": receipt["run_id"], "sha256": manifest["sha256"], "index_runs": {}})
    )
    client.log_artifact(receipt["run_id"], str(marker))
    bad_records = tmp_path / "records.json"
    bad_records.write_text("[]")
    client.log_artifact(receipt["run_id"], str(bad_records), artifact_path="package")
    with pytest.raises(ValueError, match="records.json"):
        transport.fetch_package(uri, receipt["run_id"], manifest["sha256"], tmp_path / "bad-bytes")
    assert not (tmp_path / "bad-bytes").exists()


def test_multiple_candidates_are_conflicts_without_recency_selection(publication):
    package, manifest, uri, client = publication
    receipt = transport.publish_package(package, uri, "evidence")
    original = client.get_run(receipt["run_id"])
    duplicate = client.create_run(original.info.experiment_id, tags=original.data.tags)
    with pytest.raises(transport.PublicationConflict, match="Multiple publication") as failure:
        transport.publish_package(package, uri, "evidence")
    assert set(failure.value.candidate_run_ids) == {receipt["run_id"], duplicate.info.run_id}
    search = transport.search_packages(uri, "evidence", study="study")
    assert len(search["publications"]) == 2
    assert all(row["conflict"]["reason"] == "ambiguous_package_digest" for row in search["publications"])


def test_search_distinguishes_added_attempts_from_changed_record_identity(publication, tmp_path):
    package, _, uri, _ = publication
    original = transport.publish_package(package, uri, "evidence")
    protocol = json.loads((tmp_path / "store/protocol.json").read_text())
    capture = start_evaluation(
        tmp_path / "store",
        protocol=protocol,
        trial_id="trial:unlaunched",
        requested_episodes=1,
        resolved_config={},
        inputs={},
        source_files={},
    )
    finish_evaluation(capture, status="cancelled")
    newer = tmp_path / "newer-package"
    prepare_package("evaluation-capture", tmp_path / "store", newer)
    added = transport.publish_package(newer, uri, "evidence")
    rows = transport.search_packages(uri, "evidence", study="study")["publications"]
    assert {row["run_id"] for row in rows} == {original["run_id"], added["run_id"]}
    assert all("identity_conflicts" not in row for row in rows)
    declare_protocol(tmp_path / "changed-store", protocol | {"revision": "changed-with-same-id"})
    changed_package = tmp_path / "changed-package"
    prepare_package("evaluation-capture", tmp_path / "changed-store", changed_package)
    changed = transport.publish_package(changed_package, uri, "evidence")
    rows = transport.search_packages(uri, "evidence", study="study")["publications"]
    candidates = sorted([original["run_id"], added["run_id"], changed["run_id"]])
    assert len(rows) == 3
    assert all(
        row["identity_conflicts"]
        == [{"record_id": protocol["id"], "reason": "record_identity_conflict", "candidate_run_ids": candidates}]
        for row in rows
    )
