"""Saved selections produce reproducible plots without consulting new attempts."""

import csv
import json
import shutil
from pathlib import Path

import pytest

from jaxborg.evaluation.capture_store import finish_evaluation, load_records, record_episode, start_evaluation
from jaxborg.experiment_records import collate
from scripts.eval.plot_results import REPORT_CONFIGURATION, analysis_identity, main, write_collation_report


def evidence(tmp_path, rewards=(2.0, 4.0), inputs=None):
    protocol = {
        "schema_version": 1,
        "kind": "protocol",
        "id": "comparison:1",
        "study_id": "comparison",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": [], "evaluation": []},
        "trials": [
            {"id": name, "condition_id": name, "replicate_id": "cohort-1", "matched_unit_id": "cohort-1"}
            for name in ("control", "unlaunched")
        ],
        "metric_definitions": [
            {
                "name": "blue_return",
                "unit": "score",
                "direction": "higher",
                "aggregation": "arithmetic mean",
                "sampling_unit": "episode conditional on fixed policies",
                "uncertainty": "none",
            }
        ],
    }
    root = tmp_path / "evidence"
    capture = start_evaluation(
        root,
        protocol=protocol,
        trial_id="control",
        requested_episodes=len(rewards),
        resolved_config={"seed": 17},
        inputs=inputs or {},
        source_files={},
    )
    for index, reward in enumerate(rewards):
        record_episode(capture, episode_index=index, seed=17 + index, reward=reward, steps=3)
    finish_evaluation(capture, status="succeeded")
    selection = collate(
        load_records(root),
        protocol_id=protocol["id"],
        selected_attempts={"control": capture.attempt_id},
        selected_evaluations=[capture.evaluation_id],
        analysis={"source": analysis_identity(), "configuration": REPORT_CONFIGURATION},
    )
    path = tmp_path / "collation.json"
    path.write_text(json.dumps(selection))
    return root, path, protocol, capture


def test_single_episode_report_keeps_observation_without_estimated_deviation(tmp_path):
    root, selection, _, _ = evidence(tmp_path, rewards=(2.0,))
    report = tmp_path / "report"
    main(["--collation", str(selection), "--evidence-root", str(root), "--output", str(report)])
    rows = json.loads((report / "joined.json").read_text())
    assert len(rows) == 1
    assert rows[0]["mean_reward"] == 2.0
    assert rows[0]["n_episodes"] == 1
    assert rows[0]["std_reward"] is None
    with (report / "joined.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["std_reward"] == ""
    assert row["n_episodes"] == "1"
    assert (report / "evaluation.png").read_bytes().startswith(b"\x89PNG")
    manifest = json.loads((report / "manifest.json").read_text())
    assert "unavailable for n = 1" in manifest["analysis"]["configuration"]["uncertainty"]


def test_cli_report_relocates_inputs_and_keeps_old_selection_after_retry(tmp_path):
    root, selection, protocol, original = evidence(tmp_path)
    report = tmp_path / "report"
    main(["--collation", str(selection), "--evidence-root", str(root), "--output", str(report)])
    row = json.loads((report / "joined.json").read_text())[0]
    assert row["mean_reward"] == 3
    assert row["std_reward"] == pytest.approx(2**0.5)
    assert row["n_episodes"] == 2
    status = json.loads((report / "selection-status.json").read_text())
    assert status["missing"][0]["trial_id"] == "unlaunched"
    assert status["missing"][0]["reason"] == "no_attempt"
    assert (report / "evaluation.png").read_bytes().startswith(b"\x89PNG")
    retry = start_evaluation(
        root,
        protocol=protocol,
        trial_id="control",
        retry_of=original.attempt_id,
        requested_episodes=1,
        resolved_config={},
        inputs={},
        source_files={},
    )
    record_episode(retry, episode_index=0, seed=19, reward=99.0, steps=3)
    finish_evaluation(retry, status="succeeded")
    second = write_collation_report(selection, root, tmp_path / "second")
    assert (second / "joined.json").read_bytes() == (report / "joined.json").read_bytes()
    root.rename(tmp_path / "producer-unavailable")
    rebuilt = write_collation_report(report / "collation.json", tmp_path / "producer-unavailable", tmp_path / "rebuilt")
    assert (rebuilt / "joined.csv").read_bytes() == (report / "joined.csv").read_bytes()
    manifest = json.loads((rebuilt / "manifest.json").read_text())
    assert manifest["collation_id"] == json.loads(selection.read_text())["id"]
    assert "not a confidence interval" in manifest["analysis"]["configuration"]["uncertainty"]
    with pytest.raises(FileExistsError):
        write_collation_report(selection, tmp_path / "producer-unavailable", second)


@pytest.mark.parametrize("damage", ["bytes", "source", "budget", "metric", "indices"])
def test_invalid_evidence_never_exposes_report(tmp_path, damage):
    root, selection, _, capture = evidence(tmp_path)
    saved = json.loads(selection.read_text())
    if damage == "bytes":
        (capture.directory / "episodes/0.json").write_text("{}")
    else:
        if damage == "source":
            saved["analysis"]["source"] = "wrong-source"
        else:
            evaluation = next(r for r in saved["records"] if r["id"] == capture.evaluation_id)
            if damage == "budget":
                evaluation["budgets"][0]["requested"] = 3
            elif damage == "metric":
                evaluation["metrics"][0]["value"] = 99
            else:
                artifact = next(r for r in saved["records"] if r.get("role") == "evaluation episode observation")
                import hashlib

                observation_path = root / artifact["uri"]
                observation = json.loads(observation_path.read_text())
                observation["episode_index"] = 7
                observation_path.write_text(json.dumps(observation))
                artifact.update(
                    sha256=hashlib.sha256(observation_path.read_bytes()).hexdigest(),
                    size_bytes=observation_path.stat().st_size,
                )
        saved = collate(
            saved["records"],
            **{key: saved[key] for key in ("protocol_id", "selected_attempts", "selected_evaluations", "analysis")},
        )
        selection.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        write_collation_report(selection, root, tmp_path / "invalid-report")
    assert not (tmp_path / "invalid-report").exists()


def test_report_checks_verified_snapshot_without_retaining_duplicate_evidence(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"policy weights\0" * 200_000)
    root, selection, _, _ = evidence(tmp_path, inputs={"policy": weights})
    retained = tmp_path / "original-store-unavailable"

    def remove_source(collation, rows):
        root.rename(retained)

    report = write_collation_report(selection, root, tmp_path / "report", check=remove_source)
    assert not (report / "evidence").exists()
    manifest = json.loads((report / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert not any(name.startswith("evidence/") for name in manifest["files"])
    rebuilt = write_collation_report(report / "collation.json", retained, tmp_path / "rebuilt")
    assert (rebuilt / "joined.json").read_bytes() == (report / "joined.json").read_bytes()
    assert not list(tmp_path.glob(".report-*"))


def test_changed_artifact_during_copy_leaves_no_report(tmp_path, monkeypatch):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"original weights")
    root, selection, _, _ = evidence(tmp_path, inputs={"policy": weights})
    copy = shutil.copyfileobj

    def change_input(reader, writer, length):
        if Path(reader.name).name == weights.name:
            Path(reader.name).write_bytes(b"changed weights")
        copy(reader, writer, length)

    monkeypatch.setattr(shutil, "copyfileobj", change_input)
    with pytest.raises(ValueError, match="Artifact bytes differ"):
        write_collation_report(selection, root, tmp_path / "report")
    assert not (tmp_path / "report").exists()
    assert not list(tmp_path.glob(".report-*"))
