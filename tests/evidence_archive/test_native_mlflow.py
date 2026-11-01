"""Browse native runs and download evidence through MLflow's HTTP artifact API."""

import csv
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

import pytest
from mlflow import MlflowClient

from jaxborg.evaluation.capture_store import finish_evaluation, load_records, record_episode, start_evaluation
from jaxborg.evidence_archive import package_paths, prepare_package, verify_package
from jaxborg.experiment_producers import frozen_bundle
from jaxborg.experiment_producers.source import analysis_source_identity
from jaxborg.experiment_records import collate

PROJECT = Path(__file__).resolve().parents[2]
ANALYZER = PROJECT / "studies/frozen-defender-response/analyze.py"


_HTTP_SERVER = r"""
import json
import socket
import traceback

try:
    import uvicorn
    from mlflow.server.fastapi_app import app
    from mlflow.server.handlers import initialize_backend_stores

    initialize_backend_stores()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if not self.started:
                raise RuntimeError("HTTP server did not start")
            print(json.dumps({"uri": "http://127.0.0.1:" + str(listener.getsockname()[1])}), flush=True)

    ReadyServer(uvicorn.Config(app, log_level="warning", access_log=False)).run(sockets=[listener])
except BaseException:
    print(json.dumps({"error": traceback.format_exc()}), flush=True)
    raise
"""


@pytest.fixture(scope="session")
def mlflow_http(tmp_path_factory):
    root = tmp_path_factory.mktemp("http-artifact-service")
    backend = "sqlite:///" + str(root / "tracking.sqlite")
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("MLFLOW_", "_MLFLOW_"))}
    environment.update(
        {
            "_MLFLOW_SERVER_FILE_STORE": backend,
            "_MLFLOW_SERVER_REGISTRY_STORE": backend,
            "_MLFLOW_SERVER_ARTIFACT_ROOT": "mlflow-artifacts:/",
            "_MLFLOW_SERVER_SERVE_ARTIFACTS": "true",
            "_MLFLOW_SERVER_ARTIFACT_DESTINATION": str(root / "artifacts"),
            "MLFLOW_SERVER_ALLOWED_HOSTS": "127.0.0.1:*",
        }
    )
    with (root / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", _HTTP_SERVER],
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            # EOF is the failure signal if the child exits before sending readiness.
            line = process.stdout.readline()
            if not line:
                process.wait()
                log.seek(0)
                pytest.fail("MLflow exited before readiness: " + log.read())
            handshake = json.loads(line)
            assert "error" not in handshake, handshake.get("error")
            uri = handshake["uri"]
            with urlopen(uri + "/health") as response:
                assert response.status == 200
            yield uri
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()
            process.stdout.close()


def _environment():
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(PROJECT / "src"), str(PROJECT))),
        "MLFLOW_TRACKING_URI": "http://unused.invalid",
        "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR": "false",
    }


def _files(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _analyze(cwd, root, output, selection, materialization=None):
    arguments = [
        sys.executable,
        str(ANALYZER),
        "--input",
        str(root),
        "--output",
        str(output),
        "--selection",
        str(selection),
    ]
    if materialization:
        arguments += ["--materialization", str(materialization)]
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env={**_environment(), "MPLCONFIGDIR": str(cwd / "matplotlib")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _frozen_fixture(root):
    fixture = runpy.run_path(str(ANALYZER.with_name("test_analyze.py")))["SyntheticBundle"](root)
    functions = runpy.run_path(str(ANALYZER))
    records = frozen_bundle(root)
    protocol = next(row for row in records if row["kind"] == "protocol")
    selection = collate(
        records,
        protocol_id=protocol["id"],
        selected_attempts={
            row["trial_id"]: row["id"]
            for row in records
            if row["kind"] == "attempt" and isinstance(row["trial_id"], str)
        },
        selected_evaluations=[row["id"] for row in records if row["kind"] == "evaluation"],
        analysis={
            "source": analysis_source_identity(ANALYZER),
            "configuration": functions["resolve_config"](fixture.protocol, fixture.config, {}),
        },
    )
    selection_path = root.parent / "saved collation.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    return selection_path, records, protocol


def _capture_fixture(root):
    protocol = {
        "schema_version": 1,
        "kind": "protocol",
        "id": "retrieval:protocol:1",
        "study_id": "retrieval",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": ["resolved_config", "source_files"], "evaluation": ["resolved_config"]},
        "trials": [
            {"id": name, "condition_id": condition, "replicate_id": name, "matched_unit_id": None}
            for name, condition in (("trial:a", "policy:a"), ("trial:b", "policy:b"), ("trial:c", "policy:b"))
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
    root.mkdir()
    model = root / "model.pt"
    model.write_bytes(b"retained checkpoint")
    source = root / "runner.py"
    source.write_text("EPISODES = 2\n")

    def begin(trial="trial:b", **overrides):
        return start_evaluation(
            root / "store",
            protocol=protocol,
            trial_id=trial,
            requested_episodes=2,
            resolved_config={"episodes": 2, "stochastic": False},
            inputs={"checkpoint": model},
            source_files={"src/runner.py": source},
            **overrides,
        )

    first = begin("trial:a")
    failed = begin()
    record_episode(failed, episode_index=0, seed=42, reward=-4.0, steps=500)
    finish_evaluation(failed, status="failed", error=RuntimeError("retained partial evaluation"))
    retry = begin(retry_of=failed.attempt_id)
    for capture in (first, retry):
        for index, reward in enumerate((1.0, 3.0)):
            record_episode(capture, episode_index=index, seed=42 + index, reward=reward, steps=500)
        finish_evaluation(capture, status="succeeded")
    records = load_records(root / "store")
    selection = collate(
        records,
        protocol_id=protocol["id"],
        selected_attempts={"trial:a": first.attempt_id, "trial:b": retry.attempt_id},
        selected_evaluations=[first.evaluation_id, retry.evaluation_id],
        analysis={"source": "retrieval-fixture", "configuration": {}},
    )
    return root / "store", records, selection, failed, retry, begin


def _capture_report(tmp_path):
    from scripts.eval.plot_results import REPORT_CONFIGURATION, analysis_identity, write_collation_report

    root, records, selection, *_ = _capture_fixture(tmp_path / "producer")
    selection = collate(
        records,
        protocol_id=selection["protocol_id"],
        selected_attempts=selection["selected_attempts"],
        selected_evaluations=selection["selected_evaluations"],
        analysis={"source": analysis_identity(), "configuration": REPORT_CONFIGURATION},
    )
    selected = root.parent / "selection.json"
    selected.write_text(json.dumps(selection))
    report = root.parent / "report"
    write_collation_report(selected, root, report)
    return root, selected, report


@pytest.mark.parametrize(
    "damage",
    ["source", "evidence", "selection", "extra", "listed-extra", "symlink", "deleted", "correction", "version"],
)
def test_collation_report_rejects_incomplete_or_mismatched_closure(tmp_path, damage):
    from jaxborg.evidence_archive import prepare_package

    root, selection, report = _capture_report(tmp_path)
    manifest = json.loads((report / "manifest.json").read_text())
    name = "analysis-source/scripts/eval/plot_results.py"
    if damage == "evidence":
        target = next(root.rglob("model.pt"))
        target.write_bytes(target.read_bytes() + b"changed")
    target = report / name
    if damage == "source":
        target.write_bytes(target.read_bytes() + b"changed")
        manifest["files"][name] = {
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "size_bytes": target.stat().st_size,
        }
    elif damage == "version":
        manifest["schema_version"] = 1
    elif damage == "selection":
        manifest["collation_id"] = "another-collation"
    elif damage in {"extra", "listed-extra"}:
        (report / "unexpected.txt").write_bytes(b"unrelated")
        if damage == "listed-extra":
            manifest["files"]["unexpected.txt"] = {"sha256": hashlib.sha256(b"unrelated").hexdigest(), "size_bytes": 9}
    elif damage == "symlink":
        external = tmp_path / "external-source"
        target.rename(external)
        target.symlink_to(external)
    elif damage == "deleted":
        target.unlink()
    elif damage == "correction":
        manifest["corrections"] = {"invented-evaluation": {"reason": "invented correction"}}
    (report / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        prepare_package("evaluation-capture", root, tmp_path / "rejected", selection=selection, report=report)
    assert not (tmp_path / "rejected").exists()


def test_collation_report_requires_selection_and_rejects_rehashed_source_change(tmp_path):
    from jaxborg.evidence_archive import _digest, _identity, prepare_package, verify_package

    root, selection, report = _capture_report(tmp_path)
    with pytest.raises(ValueError, match="explicit selection"):
        prepare_package("evaluation-capture", root, tmp_path / "unselected", report=report)
    package = tmp_path / "package"
    manifest = prepare_package("evaluation-capture", root, package, selection=selection, report=report)
    report_manifest_path = package / "report/manifest.json"
    report_manifest = json.loads(report_manifest_path.read_text())
    name = "analysis-source/scripts/eval/plot_results.py"
    path = package / "report" / name
    path.write_bytes(path.read_bytes() + b"changed source")
    report_manifest["files"][name] = _identity(path)
    report_manifest_path.write_text(json.dumps(report_manifest))
    for changed in ("report/" + name, "report/manifest.json"):
        manifest["files"][changed] = _identity(package / changed)
    manifest["sha256"] = _digest(manifest)
    (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="analyzer source differs"):
        verify_package(package)


def test_native_run_download_retains_failure_retry_and_ignores_unrelated_running_work(
    tmp_path, mlflow_http, monkeypatch
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_http)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "native-executions")
    root, _, _, _, _, begin = _capture_fixture(tmp_path / "producer")
    unrelated = begin()
    failed = begin(track=True)
    record_episode(failed, episode_index=0, seed=1, reward=-2, steps=2)
    finish_evaluation(failed, status="failed", error=RuntimeError("partial work"))
    retry = begin(track=True, retry_of=failed.attempt_id)
    for index in range(2):
        record_episode(retry, episode_index=index, seed=index, reward=-2, steps=2)
    unrelated.inputs["checkpoint"].write_bytes(b"unrelated incomplete work")
    finish_evaluation(retry, status="succeeded")

    client = MlflowClient(tracking_uri=mlflow_http)
    experiment = client.get_experiment_by_name("native-executions")
    runs = client.search_runs([experiment.experiment_id], "tags.`jaxborg.discovery.study_id` = 'retrieval'")
    assert {run.info.run_id for run in runs} == {failed.tracking.run_id, retry.tracking.run_id}
    selected = client.get_run(retry.tracking.run_id)
    assert selected.info.status == "FINISHED"
    assert selected.data.tags["jaxborg.discovery.retry_of"] == failed.attempt_id
    assert selected.data.tags["jaxborg.discovery.evidence_path"] == "evidence"
    assert selected.data.tags["jaxborg.discovery.scientific_status"] == "succeeded"

    root.parent.rename(tmp_path / "unavailable-producer")
    destination = tmp_path / "consumer"
    destination.mkdir()
    downloaded = Path(client.download_artifacts(selected.info.run_id, "evidence", str(destination)))
    records = load_records(downloaded)
    attempts = [record for record in records if record["kind"] == "attempt"]
    assert {record["id"] for record in attempts} == {failed.attempt_id, retry.attempt_id}
    assert {record["status"] for record in attempts} == {"failed", "succeeded"}
    assert next(record for record in attempts if record["id"] == failed.attempt_id)["budgets"][0]["completed"] == 1
    original = next(record for record in records if record["id"] == failed.evaluation_id)
    assert original["metrics"] == []
    artifact = next(record for record in records if record.get("role") == "checkpoint")
    (downloaded / artifact["uri"]).write_bytes(b"corrupt download")
    with pytest.raises(ValueError):
        load_records(downloaded)


def test_http_optional_frozen_archive_rebuilds_after_producer_disappears(tmp_path, mlflow_http):
    producer = tmp_path / "producer"
    producer.mkdir()
    selection, records, protocol = _frozen_fixture(producer / "inputs")
    baseline = tmp_path / "baseline"
    _analyze(tmp_path, producer / "inputs", baseline, selection)
    archive = tmp_path / "archive"
    manifest = prepare_package("frozen-defender", producer / "inputs", archive, selection)

    client = MlflowClient(tracking_uri=mlflow_http)
    experiment_id = client.create_experiment("frozen-report")
    run = client.create_run(experiment_id, tags={"study_id": protocol["study_id"]}, run_name="frozen report")
    client.log_artifacts(run.info.run_id, str(archive), artifact_path="report-evidence")
    client.set_terminated(run.info.run_id)
    producer.rename(tmp_path / "unavailable-producer")
    archive.rename(tmp_path / "unavailable-archive")
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    downloaded = Path(client.download_artifacts(run.info.run_id, "report-evidence", str(consumer)))
    assert verify_package(downloaded, manifest["sha256"]) == manifest
    paths = {key: Path(value) if value else None for key, value in package_paths(downloaded).items()}
    assert json.loads((downloaded / "records.json").read_text()) == records
    _analyze(consumer, paths["input"], consumer / "rebuilt", paths["selection"], paths["materialization"])
    for name in ("summary.json", "resolved-config.json", "selection.json", "collation.json", "duration.json"):
        assert json.loads((consumer / "rebuilt" / name).read_text()) == json.loads((baseline / name).read_text())
    for name in ("per-episode.csv", "summary.csv", "validation.csv", "learning-curves.csv", "learning-tail.csv"):
        with (consumer / "rebuilt" / name).open(newline="") as actual, (baseline / name).open(newline="") as expected:
            assert list(csv.DictReader(actual)) == list(csv.DictReader(expected))
    assert len(client.search_runs([experiment_id])) == 1


def test_http_optional_report_archive_replays_downloaded_source(tmp_path, mlflow_http):
    root, selection, report = _capture_report(tmp_path)
    expected = {name: (report / name).read_bytes() for name in ("joined.json", "joined.csv", "selection-status.json")}
    archive = tmp_path / "archive"
    manifest = prepare_package("evaluation-capture", root, archive, selection, report)
    client = MlflowClient(tracking_uri=mlflow_http)
    experiment_id = client.create_experiment("collation-report")
    run = client.create_run(experiment_id, run_name="collation report")
    client.log_artifacts(run.info.run_id, str(archive), artifact_path="report-evidence")
    client.set_terminated(run.info.run_id)
    root.parent.rename(tmp_path / "unavailable-producer")
    archive.rename(tmp_path / "unavailable-archive")
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    downloaded = Path(client.download_artifacts(run.info.run_id, "report-evidence", str(consumer)))
    verify_package(downloaded, manifest["sha256"])
    paths = package_paths(downloaded)
    source = Path(paths["analysis_source"])
    before = _files(downloaded)
    result = subprocess.run(
        [
            sys.executable,
            str(source / "scripts/eval/plot_results.py"),
            "--collation",
            paths["selection"],
            "--evidence-root",
            paths["input"],
            "--output",
            str(consumer / "rebuilt"),
        ],
        cwd=consumer,
        env=os.environ
        | {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join((str(source / "src"), str(source))),
            "MPLCONFIGDIR": str(consumer / "matplotlib"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _files(downloaded) == before
    assert {name: (consumer / "rebuilt" / name).read_bytes() for name in expected} == expected
