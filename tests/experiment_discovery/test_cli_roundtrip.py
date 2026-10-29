"""Retrieve native evidence over HTTP and consume it without producer paths."""

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

from jaxborg.evaluation.capture_store import (
    finish_evaluation,
    load_records,
    record_episode,
    start_evaluation,
)
from jaxborg.experiment_producers import frozen_bundle
from jaxborg.experiment_producers.source import analysis_source_identity
from jaxborg.experiment_records import collate

PROJECT = Path(__file__).resolve().parents[2]
ANALYZER = PROJECT / "experiments/frozen-defender-response/analyze.py"

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


_INTERRUPTED_PUBLISH = r"""
import sys
from pathlib import Path
from jaxborg.experiment_discovery import mlflow as transport
from jaxborg.experiment_discovery.__main__ import main

mode = sys.argv.pop(1)
factory = transport._client

def interrupted_client(uri):
    client = factory(uri)

    class InterruptedClient:
        def __getattr__(self, name):
            return getattr(client, name)

        def log_artifact(self, run_id, local_path, *args, **kwargs):
            if mode == "before-marker" and Path(local_path).name == transport.COMPLETION_ARTIFACT:
                raise RuntimeError("interrupted after package upload, before completion")
            return client.log_artifact(run_id, local_path, *args, **kwargs)

        def set_terminated(self, run_id, *args, **kwargs):
            result = client.set_terminated(run_id, *args, **kwargs)
            tags = client.get_run(run_id).data.tags
            if mode == "lost-response" and tags.get(transport.TAG_PREFIX + "role") == "package":
                raise RuntimeError("lost response after verified publication")
            return result

    return InterruptedClient()

transport._client = interrupted_client
sys.exit(main())
"""


def _environment():
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(PROJECT / "src"), str(PROJECT))),
        "MLFLOW_TRACKING_URI": "http://unused.invalid",
        "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR": "false",
    }


def _cli(cwd, *arguments, ok=True, publication_failpoint=None):
    cwd.mkdir(parents=True, exist_ok=True)
    program = (
        ["-c", _INTERRUPTED_PUBLISH, publication_failpoint]
        if publication_failpoint
        else ["-m", "jaxborg.experiment_discovery"]
    )
    result = subprocess.run(
        [sys.executable, *program, *map(str, arguments)],
        cwd=cwd,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert (result.returncode == 0) is ok, result.stdout + result.stderr
    document = json.loads(result.stdout)
    assert ("error" not in document) is ok, document
    return document


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


def _publish(cwd, package, uri, experiment):
    return _cli(cwd, "publish", "--package", package, "--tracking-uri", uri, "--experiment", experiment)


def _fetch(cwd, uri, receipt, output, ok=True):
    return _cli(
        cwd,
        "fetch",
        "--tracking-uri",
        uri,
        "--run-id",
        receipt["run_id"],
        "--sha256",
        receipt["sha256"],
        "--output",
        output,
        ok=ok,
    )


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


def test_http_collation_report_replays_only_fetched_sources_after_producer_disappears(tmp_path, mlflow_http):
    from jaxborg.experiment_discovery.package import _identity, verify_package

    root, selection, report = _capture_report(tmp_path)
    sentinel = tmp_path / "archived-source-executed"
    script = report / "analysis-source/scripts/eval/plot_results.py"
    script.write_text(script.read_text() + f"\nPath({str(sentinel)!r}).write_text('explicit execution')\n")
    manifest_path = report / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["analysis-source/scripts/eval/plot_results.py"] = _identity(script)
    hashes = {
        name.removeprefix("analysis-source/"): identity["sha256"]
        for name, identity in manifest["files"].items()
        if name.startswith("analysis-source/")
    }
    saved = json.loads(selection.read_text())
    saved["analysis"]["source"] = (
        "sha256:" + hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    saved = collate(
        **{
            key: saved[key]
            for key in ("records", "protocol_id", "selected_attempts", "selected_evaluations", "analysis")
        }
    )
    selection.write_text(json.dumps(saved))
    (report / "collation.json").write_text(json.dumps(saved))
    manifest.update(collation_id=saved["id"], analysis=saved["analysis"])
    manifest["files"]["collation.json"] = _identity(report / "collation.json")
    manifest_path.write_text(json.dumps(manifest))
    expected = {name: (report / name).read_bytes() for name in ("joined.json", "joined.csv", "selection-status.json")}
    original_selection = selection.read_bytes()
    package = tmp_path / "prepared"
    _cli(
        tmp_path,
        "prepare",
        "--kind",
        "evaluation-capture",
        "--input",
        root,
        "--selection",
        selection,
        "--report",
        report,
        "--output",
        package,
    )
    receipt = _publish(tmp_path, package, mlflow_http, "collation report retrieval")
    root.parent.rename(tmp_path / "unavailable producer")
    package.rename(tmp_path / "unavailable prepared package")
    consumer = tmp_path / "consumer"
    inspected = _cli(
        consumer, "inspect", "--tracking-uri", mlflow_http, "--run-id", receipt["run_id"], "--sha256", receipt["sha256"]
    )
    assert inspected["report"] == {"path": "report", "collation_id": saved["id"], "analysis": saved["analysis"]}
    fetched = _fetch(consumer, mlflow_http, receipt, consumer / "evidence")
    paths = fetched["paths"]
    retained = Path(paths["report"])
    archive = Path(paths["analysis_source"])
    assert not (retained / "evidence").exists()
    assert json.loads((Path(paths["materialization"])).read_text())["schema_version"] == 2
    assert Path(paths["selection"]).read_bytes() == original_selection
    assert {name: (retained / name).read_bytes() for name in expected} == expected
    assert not sentinel.exists(), "Preparation, publication and fetch must not execute archived source"
    result = subprocess.run(
        [
            sys.executable,
            str(archive / "scripts/eval/plot_results.py"),
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
            "PYTHONPATH": os.pathsep.join((str(archive / "src"), str(archive))),
            "MPLCONFIGDIR": str(consumer / "matplotlib"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.read_text() == "explicit execution"
    assert verify_package(consumer / "evidence", fetched["sha256"])["sha256"] == fetched["sha256"]
    assert {name: (consumer / "rebuilt" / name).read_bytes() for name in expected} == expected
    status = json.loads(expected["selection-status.json"])
    assert status["missing"][0]["trial_id"] == "trial:c"
    assert any(row["status"] == "failed" for row in status["excluded"])


@pytest.mark.parametrize(
    "damage",
    ["source", "evidence", "selection", "extra", "listed-extra", "symlink", "deleted", "correction", "version"],
)
def test_collation_report_rejects_incomplete_or_mismatched_closure(tmp_path, damage):
    from jaxborg.experiment_discovery.package import prepare_package

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
    from jaxborg.experiment_discovery.package import _digest, _identity, prepare_package, verify_package

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


def test_http_frozen_report_rebuild_uses_unchanged_selection_after_producer_disappears(tmp_path, mlflow_http):
    producer = tmp_path / "producer files"
    producer.mkdir()
    inputs = producer / "inputs"
    selection, records, protocol = _frozen_fixture(inputs)
    original = _files(producer)
    baseline = tmp_path / "baseline"
    _analyze(tmp_path, inputs, baseline, selection)
    package = tmp_path / "prepared"
    _cli(
        tmp_path,
        "prepare",
        "--kind",
        "frozen-defender",
        "--input",
        inputs,
        "--selection",
        selection,
        "--output",
        package,
    )
    receipt = _publish(tmp_path, package, mlflow_http, "frozen retrieval")
    repeated = _publish(tmp_path, package, mlflow_http, "frozen retrieval")
    assert repeated["run_id"] == receipt["run_id"]
    assert repeated["sha256"] == receipt["sha256"]
    assert repeated["reused"] is True
    client = MlflowClient(tracking_uri=mlflow_http)
    assert client.get_run(receipt["run_id"]).info.artifact_uri.startswith("mlflow-artifacts:/")
    assert _files(producer) == original
    hidden = producer.with_name("unavailable producer")
    producer.rename(hidden)
    assert not inputs.exists()
    consumer = tmp_path / "unrelated consumer"
    found = _cli(
        consumer,
        "search",
        "--tracking-uri",
        mlflow_http,
        "--experiment",
        "frozen retrieval",
        "--study",
        protocol["study_id"],
    )
    assert {row["run_id"] for row in found["publications"]} == {receipt["run_id"]}
    inspected = _cli(
        consumer, "inspect", "--tracking-uri", mlflow_http, "--run-id", receipt["run_id"], "--sha256", receipt["sha256"]
    )
    assert inspected["state"] == "verified"
    destination = consumer / "retrieved evidence"
    fetched = _fetch(consumer, mlflow_http, receipt, destination)
    paths = {key: Path(value) if value else None for key, value in fetched["paths"].items()}
    assert json.loads(paths["records"].read_text()) == records
    assert paths["selection"].read_bytes() == (hidden / selection.name).read_bytes()
    assert json.loads(paths["selection"].read_text())["id"] == json.loads((hidden / selection.name).read_text())["id"]
    report = consumer / "report"
    _analyze(consumer, paths["input"], report, paths["selection"], paths["materialization"])
    for name in ("summary.json", "resolved-config.json", "selection.json", "collation.json", "duration.json"):
        assert json.loads((report / name).read_text()) == json.loads((baseline / name).read_text())
    for name in ("per-episode.csv", "summary.csv", "validation.csv", "learning-curves.csv", "learning-tail.csv"):
        with (report / name).open(newline="") as actual, (baseline / name).open(newline="") as expected:
            assert list(csv.DictReader(actual)) == list(csv.DictReader(expected))
    before = _files(destination)
    _fetch(consumer, mlflow_http, receipt, destination, ok=False)
    assert _files(destination) == before
    assert _files(hidden) == original


def test_http_capture_snapshot_preserves_failure_retry_missing_cell_and_old_revision(tmp_path, mlflow_http):
    root, records, selection, failed, retry, begin = _capture_fixture(tmp_path / "producer")
    before = _files(root)
    package = tmp_path / "snapshot"
    _cli(tmp_path, "prepare", "--kind", "evaluation-capture", "--input", root, "--output", package)
    receipt = _publish(tmp_path, package, mlflow_http, "capture retrieval")
    assert _files(root) == before
    later = begin()
    finish_evaluation(later, status="cancelled", error=RuntimeError("later attempt"))
    next_package = tmp_path / "later snapshot"
    prepared = _cli(tmp_path, "prepare", "--kind", "evaluation-capture", "--input", root, "--output", next_package)
    assert prepared["sha256"] != receipt["sha256"]
    newer = _publish(tmp_path, next_package, mlflow_http, "capture retrieval")
    client = MlflowClient(tracking_uri=mlflow_http)
    for revision in (receipt, newer):
        client.set_tag(revision["run_id"], "mlflow.runName", "same readable study name")
    root.parent.rename(tmp_path / "unavailable producer")
    consumer = tmp_path / "consumer"
    found = _cli(
        consumer, "search", "--tracking-uri", mlflow_http, "--experiment", "capture retrieval", "--study", "retrieval"
    )
    assert {row["run_id"] for row in found["publications"]} == {receipt["run_id"], newer["run_id"]}
    fetched = _fetch(consumer, mlflow_http, receipt, consumer / "evidence")
    retrieved = json.loads(Path(fetched["paths"]["records"]).read_text())
    assert retrieved == records
    assert load_records(fetched["paths"]["input"]) == records
    repeated = collate(
        retrieved,
        protocol_id=selection["protocol_id"],
        selected_attempts={
            row["trial_id"]: row["id"] for row in retrieved if row["kind"] == "attempt" and row["status"] == "succeeded"
        },
        selected_evaluations=[
            row["id"] for row in retrieved if row["kind"] == "evaluation" and row["status"] == "succeeded"
        ],
        analysis=selection["analysis"],
    )
    assert repeated == selection
    assert len([row for row in retrieved if row["kind"] == "attempt"]) == 3
    failed_record = next(row for row in retrieved if row["id"] == failed.evaluation_id)
    assert failed_record["status"] == "failed"
    assert failed_record["budgets"][0] == {"unit": "episodes", "requested": 2, "completed": 1}
    assert failed_record["metrics"] == []
    assert next(row for row in retrieved if row["id"] == retry.attempt_id)["retry_of"] == failed.attempt_id
    assert selection["missing"][0]["reason"] == "no_attempt"
    inspected = _cli(
        consumer, "inspect", "--tracking-uri", mlflow_http, "--run-id", receipt["run_id"], "--sha256", receipt["sha256"]
    )
    assert failed.evaluation_id in json.dumps(inspected)
    assert "trial:c" in json.dumps(inspected)
    assert inspected["records"] == retrieved
    assert next(row for row in inspected["evaluations"] if row["id"] == failed.evaluation_id) == failed_record
    run = client.get_run(receipt["run_id"])
    assert run.data.tags["jaxborg.discovery.study_id"] == "retrieval"
    assert {r.info.run_id for r in client.search_runs([run.info.experiment_id])} == {receipt["run_id"], newer["run_id"]}


@pytest.mark.parametrize("damage", ["input", "completion"])
def test_http_corruption_fails_without_exposing_destination(tmp_path, mlflow_http, damage):
    root, *_ = _capture_fixture(tmp_path / "producer")
    package = tmp_path / "package"
    _cli(tmp_path, "prepare", "--kind", "evaluation-capture", "--input", root, "--output", package)
    receipt = _publish(tmp_path, package, mlflow_http, "corruption " + damage)
    client = MlflowClient(tracking_uri=mlflow_http)
    from jaxborg.experiment_discovery.mlflow import COMPLETION_ARTIFACT, PACKAGE_ARTIFACT

    if damage == "input":
        manifest = json.loads((package / "manifest.json").read_text())
        relative = next(path for path in manifest["files"] if path.startswith("input/"))
        artifact = Path(PACKAGE_ARTIFACT) / relative
    else:
        artifact = Path(COMPLETION_ARTIFACT)
    corrupted = tmp_path / "corrupt" / artifact.name
    corrupted.parent.mkdir()
    corrupted.write_bytes(b"truncated")
    client.log_artifact(
        receipt["run_id"], str(corrupted), artifact.parent.as_posix() if artifact.parent != Path(".") else None
    )
    output = tmp_path / "consumer" / "evidence"
    rejected = _fetch(tmp_path / "consumer", mlflow_http, receipt, output, ok=False)
    assert rejected["error"]["message"]
    assert not output.exists()


def test_http_partial_and_ambiguous_publications_are_explicit_conflicts(tmp_path, mlflow_http):
    root, *_ = _capture_fixture(tmp_path / "producer")
    package = tmp_path / "package"
    prepared = _cli(tmp_path, "prepare", "--kind", "evaluation-capture", "--input", root, "--output", package)
    from jaxborg.experiment_discovery.mlflow import PACKAGE_ARTIFACT, TAG_PREFIX

    client = MlflowClient(tracking_uri=mlflow_http)
    experiment = client.create_experiment("interrupted publication")
    tags = {
        TAG_PREFIX + "schema_version": "1",
        TAG_PREFIX + "role": "package",
        TAG_PREFIX + "sha256": prepared["sha256"],
        TAG_PREFIX + "state": "provisional",
    }
    uncertain = _cli(
        tmp_path,
        "publish",
        "--package",
        package,
        "--tracking-uri",
        mlflow_http,
        "--experiment",
        "interrupted publication",
        ok=False,
        publication_failpoint="before-marker",
    )
    assert uncertain["error"]["type"] == "PublicationConflict"
    (interrupted,) = uncertain["error"]["candidate_run_ids"]
    assert client.list_artifacts(interrupted, PACKAGE_ARTIFACT)
    receipt = {"run_id": interrupted, "sha256": prepared["sha256"]}
    output = tmp_path / "consumer" / "incomplete"
    _fetch(tmp_path / "consumer", mlflow_http, receipt, output, ok=False)
    assert not output.exists()
    conflict = _cli(
        tmp_path,
        "publish",
        "--package",
        package,
        "--tracking-uri",
        mlflow_http,
        "--experiment",
        "interrupted publication",
        ok=False,
    )
    assert conflict["error"]["type"] == "PublicationConflict"
    assert conflict["error"]["candidate_run_ids"] == [interrupted]
    other = client.create_run(experiment, tags=tags).info.run_id
    conflict = _cli(
        tmp_path,
        "publish",
        "--package",
        package,
        "--tracking-uri",
        mlflow_http,
        "--experiment",
        "interrupted publication",
        ok=False,
    )
    assert conflict["error"]["type"] == "PublicationConflict"
    assert set(conflict["error"]["candidate_run_ids"]) == {interrupted, other}
    runs = client.search_runs([experiment], filter_string=f"tags.`{TAG_PREFIX}role` = 'package'")
    assert {run.info.run_id for run in runs} == {interrupted, other}


def test_http_lost_publication_response_reuses_verified_bytes(tmp_path, mlflow_http):
    root, *_ = _capture_fixture(tmp_path / "producer")
    package = tmp_path / "package"
    prepared = _cli(tmp_path, "prepare", "--kind", "evaluation-capture", "--input", root, "--output", package)
    uncertain = _cli(
        tmp_path,
        "publish",
        "--package",
        package,
        "--tracking-uri",
        mlflow_http,
        "--experiment",
        "lost response",
        ok=False,
        publication_failpoint="lost-response",
    )
    assert uncertain["error"]["type"] == "PublicationConflict"
    (candidate,) = uncertain["error"]["candidate_run_ids"]
    recovered = _publish(tmp_path, package, mlflow_http, "lost response")
    assert recovered["run_id"] == candidate
    assert recovered["sha256"] == prepared["sha256"]
    assert recovered["reused"] is True
    fetched = _fetch(tmp_path / "consumer", mlflow_http, recovered, tmp_path / "consumer/evidence")
    assert Path(fetched["paths"]["records"]).read_bytes() == (package / "records.json").read_bytes()


def test_http_live_execution_navigation_preserves_terminal_package_boundary(tmp_path, mlflow_http, monkeypatch):
    from mlflow import MlflowClient

    from jaxborg.experiment_discovery.mlflow import search_packages
    from jaxborg.experiment_discovery.package import prepare_package

    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_http)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "live-execution")
    root, _, _, _, _, begin = _capture_fixture(tmp_path / "producer")
    capture = begin(track=True)
    client = MlflowClient(tracking_uri=mlflow_http)
    experiment = client.get_experiment_by_name("live-execution")
    runs = client.search_runs(
        [experiment.experiment_id],
        f"tags.`jaxborg.discovery.attempt_id` = '{capture.attempt_id}'",
    )
    assert len(runs) == 1
    run_id = runs[0].info.run_id
    assert runs[0].info.status == "RUNNING"
    assert runs[0].data.tags["jaxborg.discovery.state"] == "unverified"
    assert search_packages(mlflow_http, "live-execution") == {"publications": []}
    with pytest.raises(ValueError, match="Running capture"):
        prepare_package("evaluation-capture", root, tmp_path / "unfinished")
    record_episode(capture, episode_index=0, seed=1, reward=-2, steps=2)
    finish_evaluation(capture, status="failed", error=RuntimeError("stopped after one episode"))
    failed = client.get_run(run_id)
    assert failed.info.status == "FAILED"
    assert failed.data.tags["jaxborg.discovery.budget.0.completed"] == "1"
    retry = begin(track=True, retry_of=capture.attempt_id)
    for index in range(2):
        record_episode(retry, episode_index=index, seed=index, reward=-2, steps=2)
    finish_evaluation(retry, status="succeeded")
    final = client.get_run(retry.tracking.run_id)
    assert final.info.status == "FINISHED"
    assert final.data.tags["jaxborg.discovery.retry_of"] == capture.attempt_id
    assert final.data.tags["jaxborg.discovery.scientific_status"] == "succeeded"
    assert final.data.tags["jaxborg.discovery.budget.0.completed"] == "2"
    downloaded = Path(client.download_artifacts(final.info.run_id, "execution.json", str(tmp_path)))
    snapshot = json.loads(downloaded.read_text())
    assert snapshot["records"][0]["id"] == retry.attempt_id
    assert snapshot["records"][0]["status"] == "succeeded"
    assert search_packages(mlflow_http, "live-execution") == {"publications": []}
