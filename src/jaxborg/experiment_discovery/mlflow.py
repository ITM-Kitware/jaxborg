"""Native MLflow discovery for immutable, verified evidence packages."""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from mlflow import MlflowClient
from mlflow.entities import ViewType

from .package import fresh_output, inspect_package, load_package_records, package_paths, verify_package

TAG_PREFIX = "jaxborg.discovery."
PACKAGE_ARTIFACT = "package"
COMPLETION_ARTIFACT = "completion.json"


class PublicationConflict(ValueError):
    """Publication cannot select one intact, completed package."""

    def __init__(self, message, candidate_run_ids=()):
        super().__init__(message)
        self.candidate_run_ids = list(candidate_run_ids)


def _client(tracking_uri):
    if not isinstance(tracking_uri, str) or not tracking_uri.strip():
        raise ValueError("An explicit tracking URI is required")
    return MlflowClient(tracking_uri=tracking_uri)


def _digest(sha256):
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("sha256 must be an exact lowercase SHA-256 digest")
    return sha256


def _runs(client, experiment_id):
    token = None
    while True:
        page = client.search_runs(
            [experiment_id],
            filter_string=f"tags.`{TAG_PREFIX}role` = 'package'",
            run_view_type=ViewType.ALL,
            max_results=1000,
            page_token=token,
        )
        yield from page
        token = page.token
        if not token:
            break


def _receipt(run, *, reused=False):
    return {
        "run_id": run.info.run_id,
        "sha256": run.data.tags.get(TAG_PREFIX + "sha256"),
        "state": run.data.tags.get(TAG_PREFIX + "state", "provisional"),
        "reused": reused,
    }


def _identity(client, run_id, sha256):
    _digest(sha256)
    run = client.get_run(run_id)
    tags = run.data.tags
    if (
        tags.get(TAG_PREFIX + "schema_version") != "1"
        or tags.get(TAG_PREFIX + "role") != "package"
        or tags.get(TAG_PREFIX + "sha256") != sha256
        or run.info.lifecycle_stage != "active"
    ):
        raise PublicationConflict("Run does not identify the requested active evidence package", [run_id])
    return run


def _download_package(client, run_id, sha256, directory):
    root = Path(client.download_artifacts(run_id, PACKAGE_ARTIFACT, str(directory))).absolute()
    if root.parent != Path(directory).absolute() or root.name != PACKAGE_ARTIFACT:
        raise ValueError("Artifact download did not remain inside the package staging directory")
    verify_package(root, expected_digest=sha256)
    return root


def _completion(client, run_id, sha256, directory):
    marker = Path(client.download_artifacts(run_id, COMPLETION_ARTIFACT, str(directory)))
    if (
        marker.name != COMPLETION_ARTIFACT
        or marker.is_symlink()
        or not marker.is_file()
        or marker.parent.absolute() != Path(directory).absolute()
    ):
        raise ValueError("Invalid completion marker path")
    value = json.loads(marker.read_text())
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "run_id", "sha256", "index_runs"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["run_id"] != run_id
        or value["sha256"] != sha256
        or not isinstance(value["index_runs"], dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in value["index_runs"].items())
    ):
        raise ValueError("Completion marker does not match requested publication")
    return value


def _verify_remote(client, run_id, sha256, directory):
    run = _identity(client, run_id, sha256)
    if run.data.tags.get(TAG_PREFIX + "state") != "verified":
        raise PublicationConflict("Package publication is provisional; no completed evidence is available", [run_id])
    marker = _completion(client, run_id, sha256, directory)
    root = _download_package(client, run_id, sha256, directory)
    records = load_package_records(root)
    executions = {r["id"]: r for r in records if r["kind"] in {"attempt", "evaluation"}}
    if set(marker["index_runs"]) != set(executions) or len(set(marker["index_runs"].values())) != len(executions):
        raise ValueError("Completion marker does not contain the exact attempt/evaluation index")
    for record_id, child_id in marker["index_runs"].items():
        child = client.get_run(child_id)
        expected = _index_tags(executions[record_id], records, run_id, sha256) | {TAG_PREFIX + "state": "verified"}
        if child.info.experiment_id != run.info.experiment_id or child.info.lifecycle_stage != "active":
            raise ValueError(f"Index run has changed experiment or lifecycle: {child_id}")
        if any(child.data.tags.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Index run does not match retained record: {record_id}")
    return root


def _index_tags(record, records, parent_id, sha256):
    index = {r["id"]: r for r in records}
    attempt_id = record.get("attempt_id")
    attempt = record if record["kind"] == "attempt" else index.get(attempt_id) if isinstance(attempt_id, str) else None
    protocol_id = attempt.get("protocol_id") if attempt else None
    protocol = index.get(protocol_id) if isinstance(protocol_id, str) else None
    tags = {
        "mlflow.parentRunId": parent_id,
        TAG_PREFIX + "schema_version": "1",
        TAG_PREFIX + "role": "index",
        TAG_PREFIX + "sha256": sha256,
        TAG_PREFIX + "record_id": record["id"],
        TAG_PREFIX + "record_kind": record["kind"],
        TAG_PREFIX + "scientific_status": record["status"],
    }
    for key, value in {
        "study_id": protocol["study_id"] if protocol else None,
        "protocol_id": protocol_id,
        "trial_id": attempt.get("trial_id") if attempt else None,
        "attempt_id": attempt["id"] if attempt else None,
        "retry_of": attempt.get("retry_of") if attempt else None,
        "execution_kind": attempt.get("execution_kind") if attempt else None,
    }.items():
        if isinstance(value, str):
            tags[TAG_PREFIX + key] = value
    facts = record["facts"]
    for key in ("policy_backend", "eval_env"):
        fact = facts.get(key)
        if fact and isinstance(fact.get("value"), (str, int, float, bool)):
            tags[TAG_PREFIX + key] = str(fact["value"])
    backend = facts.get("backend", facts.get("policy_backend", {})).get("value")
    source = facts.get("source", {}).get("value")
    revision = (
        facts["source_revision"].get("value")
        if "source_revision" in facts
        else source.get("git_commit")
        if isinstance(source, dict)
        else None
    )
    for key, value in (("backend", backend), ("source_revision", revision)):
        if isinstance(value, (str, int, float, bool)):
            tags[TAG_PREFIX + key] = str(value)
    for number, budget in enumerate(record["budgets"]):
        tags[f"{TAG_PREFIX}budget.{number}.unit"] = budget["unit"]
        for key in ("requested", "completed"):
            if isinstance(budget[key], (int, float)):
                tags[f"{TAG_PREFIX}budget.{number}.{key}"] = str(budget[key])
    return tags


def publish_package(package, tracking_uri, experiment):
    """Publish once under a single-writer policy, or verify an exact prior run."""
    client = _client(tracking_uri)
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("An explicit experiment name is required")
    manifest = verify_package(package)
    sha256 = _digest(manifest["sha256"])
    remote_experiment = client.get_experiment_by_name(experiment)
    experiment_id = remote_experiment.experiment_id if remote_experiment else client.create_experiment(experiment)
    candidates = [run for run in _runs(client, experiment_id) if run.data.tags.get(TAG_PREFIX + "sha256") == sha256]
    if candidates:
        ids = [run.info.run_id for run in candidates]
        if len(candidates) != 1:
            raise PublicationConflict("Multiple publication candidates have this package digest", ids)
        try:
            with tempfile.TemporaryDirectory(prefix="evidence-verify-") as stage:
                _verify_remote(client, ids[0], sha256, Path(stage))
        except Exception as error:
            raise PublicationConflict(f"Existing publication cannot be reused: {error}", ids) from error
        return _receipt(client.get_run(ids[0]), reused=True)

    run_id = None
    try:
        with tempfile.TemporaryDirectory(prefix="evidence-publish-") as stage:
            stage = Path(stage)
            snapshot = stage / PACKAGE_ARTIFACT
            shutil.copytree(package, snapshot, symlinks=True)
            verify_package(snapshot, expected_digest=sha256)
            records = load_package_records(snapshot)
            tags = {
                TAG_PREFIX + "schema_version": "1",
                TAG_PREFIX + "role": "package",
                TAG_PREFIX + "sha256": sha256,
                TAG_PREFIX + "state": "provisional",
                TAG_PREFIX + "kind": manifest["kind"],
            }
            protocols = [r for r in records if r["kind"] == "protocol"]
            if len(protocols) == 1:
                tags[TAG_PREFIX + "study_id"] = protocols[0]["study_id"]
                tags[TAG_PREFIX + "protocol_id"] = protocols[0]["id"]
            run_id = client.create_run(experiment_id, tags=tags, run_name=f"evidence {sha256[:12]}").info.run_id
            client.log_artifacts(run_id, str(snapshot), artifact_path=PACKAGE_ARTIFACT)
            with tempfile.TemporaryDirectory(prefix="evidence-readback-") as readback:
                _download_package(client, run_id, sha256, Path(readback))
            index_runs = {}
            for record in records:
                if record["kind"] not in {"attempt", "evaluation"}:
                    continue
                child_tags = _index_tags(record, records, run_id, sha256) | {TAG_PREFIX + "state": "provisional"}
                child_id = client.create_run(experiment_id, tags=child_tags, run_name=record["kind"]).info.run_id
                index_runs[record["id"]] = child_id
                client.set_tag(child_id, TAG_PREFIX + "state", "verified")
                client.set_terminated(child_id, status="FINISHED")
            completion = {
                "schema_version": 1,
                "run_id": run_id,
                "sha256": sha256,
                "index_runs": index_runs,
            }
            marker = stage / COMPLETION_ARTIFACT
            marker.write_text(json.dumps(completion, sort_keys=True) + "\n")
            client.log_artifact(run_id, str(marker))
            with tempfile.TemporaryDirectory(prefix="evidence-marker-") as readback:
                if _completion(client, run_id, sha256, Path(readback)) != completion:
                    raise ValueError("Uploaded completion marker differs from the publication receipt")
            client.set_tag(run_id, TAG_PREFIX + "state", "verified")
            client.set_terminated(run_id, status="FINISHED")
    except Exception as error:
        if run_id:
            raise PublicationConflict(f"Publication outcome is uncertain: {error}", [run_id]) from error
        raise
    return _receipt(client.get_run(run_id))


def inspect_publication(tracking_uri, run_id, sha256):
    """Inspect an exact publication; provisional candidates remain explicit."""
    client = _client(tracking_uri)
    run = _identity(client, run_id, sha256)
    result = _receipt(run)
    if result["state"] != "verified":
        return result | {"available": False}
    with tempfile.TemporaryDirectory(prefix="evidence-inspect-") as stage:
        root = _verify_remote(client, run_id, sha256, Path(stage))
        marker = json.loads((Path(stage) / COMPLETION_ARTIFACT).read_text())
        return inspect_package(root) | result | {"available": True, "index_runs": marker["index_runs"]}


def _matches(description, *, study, protocol, trial, record):
    records = description.get("records", [])
    if study is not None and not any(r["kind"] == "protocol" and r["study_id"] == study for r in records):
        return False
    if protocol is not None and not any(r["kind"] == "protocol" and r["id"] == protocol for r in records):
        return False
    if trial is not None and not any(
        r["kind"] == "protocol" and any(t["id"] == trial for t in r["trials"]) for r in records
    ):
        return False
    return record is None or any(r["id"] == record for r in records)


def search_packages(tracking_uri, experiment, *, study=None, protocol=None, trial=None, record=None):
    """Return all matching revisions, never selecting by a display name or time."""
    client = _client(tracking_uri)
    remote_experiment = client.get_experiment_by_name(experiment)
    if remote_experiment is None:
        return {"publications": []}
    publications = []
    for run in _runs(client, remote_experiment.experiment_id):
        result = _receipt(run)
        try:
            with tempfile.TemporaryDirectory(prefix="evidence-search-") as stage:
                if result["state"] == "verified":
                    root = _verify_remote(client, run.info.run_id, result["sha256"], Path(stage))
                    marker = json.loads((Path(stage) / COMPLETION_ARTIFACT).read_text())
                    result["index_runs"] = marker["index_runs"]
                else:
                    root = _download_package(client, run.info.run_id, result["sha256"], Path(stage))
                result |= inspect_package(root)
                matches = _matches(result, study=study, protocol=protocol, trial=trial, record=record)
        except Exception as error:
            result |= {"available": False, "verification_error": str(error)}
            matches = (
                (study is None or run.data.tags.get(TAG_PREFIX + "study_id") == study)
                and (protocol is None or run.data.tags.get(TAG_PREFIX + "protocol_id") == protocol)
                and trial is None
                and record is None
            )
        else:
            result["available"] = result["state"] == "verified"
        if matches:
            publications.append(result)
    digests = {}
    for row in publications:
        digests.setdefault(row["sha256"], []).append(row["run_id"])
    for row in publications:
        if len(digests[row["sha256"]]) > 1:
            row["conflict"] = {"reason": "ambiguous_package_digest", "candidate_run_ids": digests[row["sha256"]]}
    identities = {}
    for row in publications:
        if row.get("available"):
            for retained in row["records"]:
                payload = json.dumps(retained, sort_keys=True, separators=(",", ":"), allow_nan=False)
                identities.setdefault(retained["id"], {}).setdefault(payload, []).append(row["run_id"])
    for record_id, variants in identities.items():
        if len(variants) > 1:
            candidates = sorted(run_id for run_ids in variants.values() for run_id in run_ids)
            for row in publications:
                if row["run_id"] in candidates:
                    row.setdefault("identity_conflicts", []).append(
                        {"record_id": record_id, "reason": "record_identity_conflict", "candidate_run_ids": candidates}
                    )
    return {"publications": sorted(publications, key=lambda row: row["run_id"])}


def fetch_package(tracking_uri, run_id, sha256, output):
    """Verify native artifact downloads in staging, then expose a fresh directory."""
    client = _client(tracking_uri)
    destination = fresh_output(output)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as stage:
        root = _verify_remote(client, run_id, sha256, Path(stage))
        if os.path.lexists(destination):
            raise FileExistsError(f"Output already exists: {destination}")
        root.rename(destination)
    return {"run_id": run_id, "sha256": sha256, "state": "verified", "paths": package_paths(destination)}
