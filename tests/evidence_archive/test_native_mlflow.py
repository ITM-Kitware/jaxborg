"""Browse native runs and download evidence through MLflow's HTTP artifact API."""

import csv
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

import pytest
from mlflow import MlflowClient

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
