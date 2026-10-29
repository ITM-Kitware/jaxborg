"""Exercise immutable packages and original-provenance relocation on retained files."""

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jaxborg.evaluation.capture_store import finish_evaluation, record_episode, start_evaluation
from jaxborg.experiment_discovery.package import (
    inspect_package,
    load_package_records,
    materialize_package,
    package_paths,
    prepare_package,
    verify_package,
)
from jaxborg.experiment_producers import frozen_bundle
from jaxborg.experiment_producers.selection import verify_frozen_selection
from jaxborg.experiment_producers.source import analysis_source_identity
from jaxborg.experiment_records import collate

PROJECT = Path(__file__).resolve().parents[2]
ANALYZER = PROJECT / "experiments/frozen-defender-response/analyze.py"


def read(path):
    return json.loads(Path(path).read_text())


def frozen_fixture(tmp_path, *, overrides=None):
    fixture = runpy.run_path(str(ANALYZER.with_name("test_analyze.py")))["SyntheticBundle"]
    bundle = fixture(tmp_path / "producer inputs")
    records = frozen_bundle(bundle.root)
    protocol = next(r for r in records if r["kind"] == "protocol")
    analyzer = runpy.run_path(str(ANALYZER))
    selection = collate(
        records,
        protocol_id=protocol["id"],
        selected_attempts={
            r["trial_id"]: r["id"] for r in records if r["kind"] == "attempt" and isinstance(r["trial_id"], str)
        },
        selected_evaluations=[r["id"] for r in records if r["kind"] == "evaluation"],
        analysis={
            "source": analysis_source_identity(ANALYZER),
            "configuration": analyzer["resolve_config"](bundle.protocol, bundle.config, overrides or {}),
        },
    )
    selection_path = tmp_path / "original collation.json"
    selection_path.write_text(json.dumps(selection, indent=1) + "\n")
    return bundle, selection_path


def capture_fixture(tmp_path, *, running=False):
    root = tmp_path / "capture"
    protocol = {
        "schema_version": 1,
        "kind": "protocol",
        "id": "study:protocol",
        "study_id": "study",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": [], "evaluation": []},
        "trials": [
            {"id": name, "condition_id": name, "replicate_id": name, "matched_unit_id": None}
            for name in ("A", "B", "C")
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
    model = tmp_path / "model.pt"
    model.write_bytes(b"external checkpoint")
    source = tmp_path / "procedure.py"
    source.write_text("VALUE = 1\n")

    def begin(trial, retry=None):
        return start_evaluation(
            root,
            protocol=protocol,
            trial_id=trial,
            retry_of=retry,
            requested_episodes=2,
            resolved_config={"seed": 17},
            inputs={"external checkpoint": model},
            source_files={"procedure.py": source},
        )

    first = begin("A")
    record_episode(first, episode_index=0, seed=17, reward=1.0, steps=2)
    record_episode(first, episode_index=1, seed=18, reward=2.0, steps=2)
    finish_evaluation(first, status="succeeded")
    failed = begin("B")
    record_episode(failed, episode_index=0, seed=17, reward=-1.0, steps=2)
    finish_evaluation(failed, status="failed", error=ValueError("observed failure"))
    retry = begin("B", failed.attempt_id)
    if not running:
        record_episode(retry, episode_index=0, seed=17, reward=0.0, steps=2)
        record_episode(retry, episode_index=1, seed=18, reward=0.0, steps=2)
        finish_evaluation(retry, status="succeeded")
    return root, retry


def test_terminal_capture_preserves_failures_retry_missing_cells_and_snapshot(tmp_path):
    root, retry = capture_fixture(tmp_path)
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    package = tmp_path / "package"
    manifest = prepare_package("evaluation-capture", root, package)
    paths = materialize_package(package, tmp_path / "consumer")
    assert verify_package(Path(paths["records"]).parent, manifest["sha256"]) == manifest
    records = load_package_records(package)
    report = inspect_package(package)
    assert report["missing"] == [{"protocol_id": "study:protocol", "trial_id": "C", "reason": "no_attempt"}]
    attempts = [r for r in records if r["kind"] == "attempt"]
    assert len(attempts) == 3
    assert next(r for r in attempts if r["id"] == retry.attempt_id)["retry_of"] in {
        r["id"] for r in attempts if r["status"] == "failed"
    }
    failed = next(r for r in records if r["kind"] == "evaluation" and r["status"] == "failed")
    assert failed["metrics"] == [] and failed["budgets"][0]["completed"] == 1
    assert all("current.json" not in name for name in manifest["files"])
    assert before == {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    (root / "unrelated-secret.txt").write_text("not in closure")
    assert prepare_package("evaluation-capture", root, tmp_path / "same snapshot")["sha256"] == manifest["sha256"]


def test_running_capture_fails_without_completed_output(tmp_path):
    root, retry = capture_fixture(tmp_path, running=True)
    with pytest.raises(ValueError, match=retry.attempt_id):
        prepare_package("evaluation-capture", root, tmp_path / "package")
    assert not (tmp_path / "package").exists()


def test_declared_capture_without_attempts_is_packageable(tmp_path):
    from jaxborg.evaluation.capture_store import declare_protocol

    root, _ = capture_fixture(tmp_path)
    protocol = read(root / "protocol.json")
    empty = tmp_path / "declared only"
    declare_protocol(empty, protocol)
    package = tmp_path / "package"
    prepare_package("evaluation-capture", empty, package)
    assert len(inspect_package(package)["missing"]) == 3


def test_frozen_relocation_keeps_original_collation_and_rebuilds_report(tmp_path):
    bundle, selection = frozen_fixture(tmp_path)
    baseline = tmp_path / "baseline"
    environment = os.environ | {
        "PYTHONPATH": str(PROJECT / "src") + os.pathsep + str(PROJECT),
        "MPLCONFIGDIR": str(tmp_path / "mpl"),
    }

    def analyze(input_root, output, selection, materialization=None):
        command = [
            sys.executable,
            str(ANALYZER),
            "--input",
            str(input_root),
            "--output",
            str(output),
            "--selection",
            str(selection),
        ]
        if materialization:
            command += ["--materialization", str(materialization)]
        result = subprocess.run(command, env=environment, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    analyze(bundle.root, baseline, selection)
    original_bytes = selection.read_bytes()
    package = tmp_path / "published"
    manifest = prepare_package("frozen-defender", bundle.root, package, selection)
    shutil.rmtree(bundle.root)
    paths = materialize_package(package, tmp_path / "consumer with spaces")
    assert Path(paths["selection"]).read_bytes() == original_bytes
    verify_frozen_selection(read(paths["selection"]), paths["input"], materialization=paths["materialization"])
    with pytest.raises(ValueError, match="Saved record differs"):
        verify_frozen_selection(read(paths["selection"]), paths["input"])
    analyze(paths["input"], tmp_path / "rebuilt", paths["selection"], paths["materialization"])
    for name in ("summary.json", "per-episode.csv", "validation.csv", "resolved-config.json", "collation.json"):
        assert (baseline / name).read_bytes() == (tmp_path / "rebuilt" / name).read_bytes(), name
    assert verify_package(package)["sha256"] == manifest["sha256"]
    assert manifest["unretained_references"]


@pytest.mark.parametrize(
    "damage", ["truncate", "delete", "extra", "symlink", "traversal", "duplicate", "map", "version"]
)
def test_package_rejects_corruption_and_unsafe_manifest(tmp_path, damage):
    from jaxborg.experiment_discovery.package import _digest

    root, _ = capture_fixture(tmp_path)
    package = tmp_path / "package"
    manifest = prepare_package("evaluation-capture", root, package)
    name = next(n for n in manifest["files"] if n.endswith("model.pt"))
    path = package / name
    if damage == "version":
        manifest["schema_version"] = 1
    elif damage == "truncate":
        path.write_bytes(b"bad")
    elif damage == "delete":
        path.unlink()
    elif damage == "extra":
        (package / "extra.txt").write_text("unlisted")
    elif damage == "symlink":
        path.unlink()
        path.symlink_to(root / name.removeprefix("input/"))
    elif damage == "traversal":
        manifest["files"]["input/../outside"] = manifest["files"].pop(name)
    elif damage == "map":
        next(iter(manifest["artifacts"].values()))["uri"] = "file:///invented"
    if damage == "duplicate":
        payload = (
            (package / "manifest.json")
            .read_text()
            .replace('"schema_version": 2', '"schema_version": 2, "schema_version": 2')
        )
        (package / "manifest.json").write_text(payload)
    else:
        manifest["sha256"] = _digest(manifest)
        (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_package(package)


def test_prepare_does_not_copy_links_or_accept_old_analysis_source(tmp_path):
    bundle, selection = frozen_fixture(tmp_path)
    saved = read(selection)
    saved["analysis"]["source"] = "sha256:" + "0" * 64
    # Rebuild the content-identified selection with the changed source claim.
    from jaxborg.experiment_records import collate

    attempts = {
        r["trial_id"]: r["id"] for r in saved["records"] if r["kind"] == "attempt" and isinstance(r["trial_id"], str)
    }
    changed = collate(
        saved["records"],
        protocol_id=saved["protocol_id"],
        selected_attempts=attempts,
        selected_evaluations=[r["id"] for r in saved["records"] if r["kind"] == "evaluation"],
        analysis=saved["analysis"],
    )
    selection.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Analyzer source differs"):
        prepare_package("frozen-defender", bundle.root, tmp_path / "old source", selection)
    path = bundle.root / next(iter(read(bundle.root / "index.json")["files"]))
    external = tmp_path / "external"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(ValueError, match="Symbolic link"):
        prepare_package("frozen-defender", bundle.root, tmp_path / "linked")
    assert not (tmp_path / "old source").exists() and not (tmp_path / "linked").exists()


def test_changed_input_during_copy_never_exposes_package(tmp_path, monkeypatch):
    from jaxborg.experiment_discovery import package as module

    root, _ = capture_fixture(tmp_path)
    original = module.shutil.copyfile

    def mutate(source, destination):
        result = original(source, destination)
        if Path(source).name == "model.pt":
            Path(source).write_bytes(b"mutated during preparation")
        return result

    monkeypatch.setattr(module.shutil, "copyfile", mutate)
    with pytest.raises(ValueError):
        prepare_package("evaluation-capture", root, tmp_path / "package")
    assert not (tmp_path / "package").exists()


def test_existing_output_is_preserved(tmp_path):
    root, _ = capture_fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        prepare_package("evaluation-capture", root, output)
    assert (output / "sentinel").read_text() == "keep"


def test_frozen_locator_changes_cannot_hide_behind_recomputed_transport_hash(tmp_path):
    from jaxborg.experiment_discovery.package import _digest

    bundle, selection = frozen_fixture(tmp_path)
    package = tmp_path / "package"
    manifest = prepare_package("frozen-defender", bundle.root, package, selection)
    manifest["provenance_root"] = "/different/original/root"
    manifest["sha256"] = _digest(manifest)
    (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="original locators"):
        verify_package(package)


def test_package_digest_is_required_and_changed_native_record_is_rejected(tmp_path):
    from jaxborg.experiment_discovery.package import _digest

    root, _ = capture_fixture(tmp_path)
    package = tmp_path / "package"
    manifest = prepare_package("evaluation-capture", root, package)
    with pytest.raises(ValueError, match="digest"):
        verify_package(package, expected_digest="0" * 64)
    record_path = package / "records.json"
    records = read(record_path)
    next(r for r in records if r["kind"] == "attempt")["execution_kind"] = "other scientific procedure"
    record_path.write_text(json.dumps(records))
    payload = record_path.read_bytes()
    manifest["files"]["records.json"] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    manifest["sha256"] = _digest(manifest)
    (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="authoritative evidence"):
        verify_package(package)


def test_saved_analysis_override_survives_relocation_and_requires_matching_runtime(tmp_path):
    bundle, selection = frozen_fixture(tmp_path, overrides={"bootstrap": {"seed": 789}})
    package = tmp_path / "package"
    prepare_package("frozen-defender", bundle.root, package, selection)
    shutil.rmtree(bundle.root)
    paths = package_paths(package)
    command = [
        sys.executable,
        str(ANALYZER),
        "--input",
        paths["input"],
        "--selection",
        paths["selection"],
        "--materialization",
        paths["materialization"],
        "--output",
        str(tmp_path / "mismatch"),
    ]
    environment = os.environ | {
        "PYTHONPATH": str(PROJECT / "src") + os.pathsep + str(PROJECT),
        "MPLCONFIGDIR": str(tmp_path / "mpl"),
    }
    rejected = subprocess.run(command, cwd=tmp_path, env=environment, text=True, capture_output=True)
    assert rejected.returncode != 0 and "Resolved analysis configuration differs" in rejected.stderr
    assert not (tmp_path / "mismatch").exists()
    accepted = subprocess.run(
        command + ["--bootstrap-seed", "789"], cwd=tmp_path, env=environment, text=True, capture_output=True
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert read(tmp_path / "mismatch" / "resolved-config.json")["bootstrap"]["seed"] == 789


def test_later_attempt_creates_another_package_without_changing_prior_snapshot(tmp_path):
    root, _ = capture_fixture(tmp_path)
    original = tmp_path / "original snapshot"
    manifest = prepare_package("evaluation-capture", root, original)
    capture = start_evaluation(
        root,
        protocol=read(root / "protocol.json"),
        trial_id="C",
        requested_episodes=1,
        resolved_config={},
        inputs={},
        source_files={},
    )
    record_episode(capture, episode_index=0, seed=19, reward=3.0)
    finish_evaluation(capture, status="succeeded")
    updated = tmp_path / "updated snapshot"
    assert prepare_package("evaluation-capture", root, updated)["sha256"] != manifest["sha256"]
    assert verify_package(original) == manifest
    assert len(inspect_package(original)["missing"]) == 1
    assert inspect_package(updated)["missing"] == []


def test_capture_selection_must_only_identify_its_authoritative_store(tmp_path):
    from jaxborg.evaluation.capture_store import declare_protocol, load_records
    from jaxborg.experiment_discovery.package import _digest

    root, _ = capture_fixture(tmp_path)
    records = load_records(root)
    selection = collate(
        records,
        protocol_id="study:protocol",
        selected_attempts={
            r["trial_id"]: r["id"] for r in records if r["kind"] == "attempt" and r["status"] == "succeeded"
        },
        selected_evaluations=[r["id"] for r in records if r["kind"] == "evaluation" and r["status"] == "succeeded"],
        analysis={"source": "analysis.py:sha256", "configuration": {}},
    )
    selected_path = tmp_path / "selection.json"
    selected_path.write_text(json.dumps(selection))
    valid = tmp_path / "valid"
    prepare_package("evaluation-capture", root, valid, selected_path)
    assert read(valid / "selection.json") == selection

    declared = tmp_path / "declared"
    declare_protocol(declared, read(root / "protocol.json"))
    output = tmp_path / "foreign selected evidence"
    with pytest.raises(ValueError, match="exactly the authoritative store records"):
        prepare_package("evaluation-capture", declared, output, selected_path)
    assert not output.exists()

    bad = tmp_path / "bad manually rehashed"
    manifest = prepare_package("evaluation-capture", declared, bad)
    (bad / "selection.json").write_bytes(selected_path.read_bytes())
    payload = selected_path.read_bytes()
    manifest["selection"] = "selection.json"
    manifest["files"]["selection.json"] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    manifest["sha256"] = _digest(manifest)
    (bad / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly the authoritative store records"):
        verify_package(bad)
