import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from jaxborg.experiment_records import RecordError, collate, validate_collation, validate_records


def records():
    return [
        {
            "schema_version": 1,
            "kind": "protocol",
            "id": "p",
            "study_id": "study",
            "revision": "1",
            "selection_rule": "explicit",
            "required_facts": {"attempt": [], "evaluation": []},
            "trials": [
                {"id": t, "condition_id": "control", "replicate_id": t, "matched_unit_id": None} for t in ["t1", "t2"]
            ],
            "metric_definitions": [],
        },
        {
            "schema_version": 1,
            "kind": "attempt",
            "id": "a",
            "protocol_id": "p",
            "trial_id": "t1",
            "status": "succeeded",
            "execution_kind": "training",
            "retry_of": None,
            "facts": {"seed": {"value": None, "reason": "not retained", "source": "legacy manifest"}},
            "budgets": [{"unit": "environment_steps", "requested": 10, "completed": 10}],
            "inputs": [],
        },
    ]


def selection():
    return {
        "protocol_id": "p",
        "selected_attempts": {"t1": "a"},
        "selected_evaluations": [],
        "analysis": {"source": "retained-analysis", "configuration": {}},
    }


def test_cli_freezes_selection_and_missing_cell(tmp_path):
    source, chosen, output = (tmp_path / name for name in ["records.json", "selection.json", "collation.json"])
    source.write_text(json.dumps(records()))
    chosen.write_text(json.dumps(selection()))
    subprocess.run(
        [sys.executable, "-m", "jaxborg.experiment_records", str(source), str(chosen), str(output)],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    result = json.loads(output.read_text())
    assert result["included"] == ["a"]
    assert result["missing"][0]["id"] == "t2"
    assert result["missing"][0]["reason"] == "no_attempt"
    assert result == collate(list(reversed(records())), **selection())


@pytest.mark.parametrize(
    "problem", ["duplicate", "dangling", "wrong_kind", "cycle", "missing_reason", "unit", "unsuccessful"]
)
def test_invalid_evidence_is_rejected(problem):
    data = records()
    if problem == "duplicate":
        data.append(data[1].copy())
    elif problem == "dangling":
        data[1]["retry_of"] = "absent"
    elif problem == "wrong_kind":
        data[1]["retry_of"] = "p"
    elif problem == "cycle":
        data[1]["retry_of"] = "a"
    elif problem == "missing_reason":
        data[1]["facts"]["seed"] = {"value": None}
    elif problem == "unit":
        data[1]["budgets"][0]["unit"] = ""
    elif problem == "unsuccessful":
        data[1]["status"] = "failed"
    with pytest.raises(RecordError):
        collate(data, **selection())


def test_retries_require_selection_and_stay_excluded():
    data = records()
    old = {**data[1], "id": "old", "status": "failed"}
    data[1]["retry_of"] = "old"
    data.append(old)
    result = collate(data, **selection())
    assert result["excluded"] == [{"id": "old", "reason": "not_selected", "status": "failed"}]
    assert collate(data, **{**selection(), "selected_attempts": {}})["missing"][0]["reason"] == "no_explicit_selection"
    assert len(validate_records(data)) == 3


def test_evaluation_retains_metric_meaning_and_artifact_origin():
    data = records()
    data[0]["metric_definitions"] = [
        {
            "name": "return",
            "unit": "score",
            "direction": "higher",
            "aggregation": "mean",
            "sampling_unit": "episode",
            "uncertainty": "none",
        }
    ]
    data += [
        {
            "schema_version": 1,
            "kind": "artifact",
            "id": "weights",
            "producer_id": None,
            "uri": "bundle/weights",
            "sha256": "a" * 64,
            "size_bytes": 20,
            "role": "policy",
        },
        {
            "schema_version": 1,
            "kind": "artifact",
            "id": "raw",
            "producer_id": "e",
            "uri": "bundle/raw",
            "sha256": "b" * 64,
            "size_bytes": 30,
            "role": "measurements",
        },
        {
            "schema_version": 1,
            "kind": "evaluation",
            "id": "e",
            "attempt_id": "a",
            "inputs": ["weights"],
            "status": "succeeded",
            "facts": {},
            "budgets": [],
            "metrics": [{"name": "return", "value": 3.0, "unit": "score"}],
            "raw_artifacts": ["raw"],
        },
    ]
    result = collate(data, **{**selection(), "selected_evaluations": ["e"]})
    assert result["included"] == ["a", "e"]
    assert result["rows"][1]["metrics"] == [{"name": "return", "value": 3.0, "unit": "score"}]
    assert result["metric_definitions"][0]["sampling_unit"] == "episode"
    data[-1]["metrics"].append({"name": "return", "value": 999.0, "unit": "score"})
    with pytest.raises(RecordError, match="duplicate evaluation metric"):
        collate(data, **{**selection(), "selected_evaluations": ["e"]})
    data[-1]["metrics"].pop()
    data[-1]["metrics"][0]["unit"] = "shaped_score"
    with pytest.raises(RecordError, match="metric unit mismatch"):
        validate_records(data)


def test_required_facts_and_unresolved_discovery():
    data = records()
    data[0]["required_facts"]["attempt"] = ["seed"]
    with pytest.raises(RecordError, match="required known fact"):
        collate(data, **selection())
    data[0]["required_facts"]["attempt"] = []
    data[1]["trial_id"] = {"value": None, "reason": "unretained membership", "source": "manifest"}
    assert validate_records(data)["a"]["trial_id"]["value"] is None
    discovered = collate(data, **{**selection(), "selected_attempts": {}})
    assert discovered["excluded"][0]["reason"] == "unresolved_membership"
    with pytest.raises(RecordError, match="another trial"):
        collate(data, **selection())


def test_saved_collation_is_recomputed_even_when_digest_is_forged():
    result = collate(records(), **selection())
    assert validate_collation(result) == result
    result["included"] = []
    import hashlib

    payload = {k: v for k, v in result.items() if k not in {"id", "sha256"}}
    result["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result["id"] = "collation:sha256:" + result["sha256"]
    with pytest.raises(RecordError, match="recomputed"):
        validate_collation(result)


def test_extensions_require_version_and_source():
    data = records()
    data[1]["extensions"] = {"mechanism": {"schema_version": 1, "source": "rules.py:sha256", "values": {}}}
    validate_records(data)
    del data[1]["extensions"]["mechanism"]["source"]
    with pytest.raises(RecordError):
        validate_records(data)


def test_two_successful_attempts_do_not_inflate_replication():
    data = records()
    data.append({**data[1], "id": "retry", "retry_of": "a"})
    result = collate(data, **selection())
    assert len(result["rows"]) == 1
    assert result["excluded"][0]["id"] == "retry"


def test_cli_refuses_overwrite(tmp_path):
    source, chosen, output = (tmp_path / name for name in ["records.json", "selection.json", "collation.json"])
    source.write_text(json.dumps(records()))
    chosen.write_text(json.dumps(selection()))
    output.write_text("retained")
    completed = subprocess.run(
        [sys.executable, "-m", "jaxborg.experiment_records", str(source), str(chosen), str(output)],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert completed.returncode != 0
    assert output.read_text() == "retained"


def test_distinct_ids_cannot_duplicate_one_scientific_cell():
    data = records()
    data[0]["trials"].append({**data[0]["trials"][0], "id": "duplicate-cell"})
    with pytest.raises(RecordError, match="duplicate scientific trial cell"):
        validate_records(data)


def test_unknown_retry_history_does_not_create_an_extra_replicate():
    data = records()
    unknown = {"value": None, "reason": "not retained", "source": "legacy manifest"}
    data.append({**data[1], "id": "original", "retry_of": unknown})
    data[1]["retry_of"] = "original"
    result = collate(data, **selection())
    assert len(result["rows"]) == 1
    assert result["included"] == ["a"]
    assert result["excluded"][0]["id"] == "original"
    assert validate_collation(result) == result
    data[0]["required_facts"]["attempt"] = ["retry_history"]
    data[1]["facts"]["retry_history"] = unknown
    with pytest.raises(RecordError, match="required known fact"):
        collate(data, **selection())
