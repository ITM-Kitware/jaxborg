"""Native MLflow discovery for immutable, verified evidence packages."""

import hashlib
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
INDEX_ARTIFACT = "index.json"


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
        or set(value) != {"schema_version", "run_id", "sha256"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or value["run_id"] != run_id
        or value["sha256"] != sha256
    ):
        raise ValueError("Completion marker does not match requested publication")
    return value


def _verify_remote(client, run_id, sha256, directory):
    run = _identity(client, run_id, sha256)
    if run.data.tags.get(TAG_PREFIX + "state") != "verified":
        raise PublicationConflict("Package publication is provisional; no completed evidence is available", [run_id])
    _completion(client, run_id, sha256, directory)
    return _download_package(client, run_id, sha256, directory)


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
            index = stage / INDEX_ARTIFACT
            index.write_text(json.dumps(_search_index(records), sort_keys=True) + "\n")
            client.log_artifact(run_id, str(index))
            completion = {
                "schema_version": 2,
                "run_id": run_id,
                "sha256": sha256,
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
        return inspect_package(root) | result | {"available": True}


def _search_index(records):
    """Small advisory projection; exact inspection verifies the authoritative package."""
    return {
        "protocols": [
            {"id": r["id"], "study_id": r["study_id"], "trials": [t["id"] for t in r["trials"]]}
            for r in records
            if r["kind"] == "protocol"
        ],
        "record_hashes": {
            r["id"]: hashlib.sha256(
                json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            for r in records
        },
    }


def _matches(index, *, study, protocol, trial, record):
    return any(
        (study is None or p["study_id"] == study)
        and (protocol is None or p["id"] == protocol)
        and (trial is None or trial in p["trials"])
        for p in index["protocols"]
    ) and (record is None or record in index["record_hashes"])


def search_packages(tracking_uri, experiment, *, study=None, protocol=None, trial=None, record=None):
    """List unverified candidates from a small index, without downloading evidence."""
    client = _client(tracking_uri)
    remote_experiment = client.get_experiment_by_name(experiment)
    if remote_experiment is None:
        return {"publications": []}
    publications = []
    digests = {}
    for run in _runs(client, remote_experiment.experiment_id):
        result = _receipt(run)
        digests.setdefault(result["sha256"], []).append(result["run_id"])
        result |= {"verified": False, "available": None}
        try:
            with tempfile.TemporaryDirectory(prefix="evidence-search-") as stage:
                path = Path(client.download_artifacts(run.info.run_id, INDEX_ARTIFACT, stage))
                if path.is_symlink() or path.absolute() != Path(stage) / INDEX_ARTIFACT:
                    raise ValueError("Invalid search index path")
                index = json.loads(path.read_text())
                if (
                    not isinstance(index, dict)
                    or set(index) != {"protocols", "record_hashes"}
                    or not isinstance(index["record_hashes"], dict)
                    or not all(isinstance(k, str) and isinstance(v, str) for k, v in index["record_hashes"].items())
                ):
                    raise ValueError("Invalid search index")
                matches = _matches(index, study=study, protocol=protocol, trial=trial, record=record)
                result |= index
                result["match"] = "candidate"
        except Exception as error:
            result |= {"index_error": str(error), "match": "unknown"}
            matches = (study is None or run.data.tags.get(TAG_PREFIX + "study_id") == study) and (
                protocol is None or run.data.tags.get(TAG_PREFIX + "protocol_id") == protocol
            )
        if matches:
            publications.append(result)
    for row in publications:
        if len(digests[row["sha256"]]) > 1:
            row["conflict"] = {"reason": "ambiguous_package_digest", "candidate_run_ids": digests[row["sha256"]]}
    identities = {}
    for row in publications:
        for record_id, digest in row.get("record_hashes", {}).items():
            identities.setdefault(record_id, {}).setdefault(digest, []).append(row["run_id"])
    for record_id, variants in identities.items():
        if len(variants) > 1:
            candidates = sorted(run_id for run_ids in variants.values() for run_id in run_ids)
            for row in publications:
                if row["run_id"] in candidates:
                    row.setdefault("identity_conflicts", []).append(
                        {
                            "record_id": record_id,
                            "reason": "candidate_record_identity_conflict",
                            "candidate_run_ids": candidates,
                        }
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
