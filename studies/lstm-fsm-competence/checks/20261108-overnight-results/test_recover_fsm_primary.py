"""Focused checks for immutable primary receipt reuse and recovery launcher syntax."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery = load_module("recover_fsm_primary")


@pytest.fixture
def recorded_cell(tmp_path):
    case = {
        "red_id": "red-diverse-42-final",
        "variant": {"name": "cc4_stock"},
        "topologies": ["topology-sha"],
        "seeds": [21000, 21001],
        "source_revision": "pinned-revision",
        "checkpoint": "blue-model",
        "checkpoint_sha256": "blue-sha",
        "sidecar_sha256": "blue-sidecar-sha",
        "red_sha256": "red-sha",
        "red_sidecar_sha256": "red-sidecar-sha",
    }
    config = {
        "variant": case["variant"],
        "topology": {"sha256": case["topologies"]},
        "seed_roles": {"episode_roots": case["seeds"]},
        "execution": {"source_revision": case["source_revision"]},
        "deterministic": False,
        "episodes_per_seed": 1,
    }
    inputs = {
        "blue checkpoint": "blue-model",
        "blue sidecar": "blue-sidecar",
        "red checkpoint": "red-model",
        "red sidecar": "red-sidecar",
    }
    directory = tmp_path / "attempts" / "attempt1"
    recovery.write(directory / "configuration.json", config)
    recovery.write(directory / "input-artifacts.json", inputs)
    records = [
        {"id": "evaluation1", "kind": "evaluation", "status": "succeeded", "attempt_id": "attempt1"},
        {"id": "attempt1", "kind": "attempt", "status": "succeeded"},
        *[
            {"id": identifier, "kind": "artifact", "sha256": digest}
            for identifier, digest in [
                ("blue-model", "blue-sha"),
                ("blue-sidecar", "blue-sidecar-sha"),
                ("red-model", "red-sha"),
                ("red-sidecar", "red-sidecar-sha"),
            ]
        ],
    ]
    for index, reward in enumerate((-1000, -3000)):
        path = tmp_path / f"episode{index}.json"
        recovery.write(path, {"reward": reward})
        records.append(
            {
                "id": f"observation{index}",
                "kind": "artifact",
                "producer_id": "evaluation1",
                "role": "evaluation episode observation",
                "uri": path.name,
                "sha256": recovery.sha(path),
            }
        )
    receipt = {
        "case": copy.deepcopy(case),
        "evaluation_id": "evaluation1",
        "checkpoint_steps": 49968000,
        "episodes": 2,
        "mean_return": -2000,
    }
    return receipt, case, records, tmp_path


def test_reuses_primary_nonzero_checkpoint(recorded_cell):
    assert recovery.validate_receipt(*recorded_cell)["checkpoint_steps"] == 49968000


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_revision", "wrong-source"),
        ("topologies", ["wrong-topology"]),
        ("checkpoint_sha256", "wrong-model"),
        ("red_sha256", "wrong-red"),
    ],
)
def test_rejects_incorrect_identity(recorded_cell, field, value):
    receipt, case, records, evidence = recorded_cell
    case[field] = value
    receipt["case"] = copy.deepcopy(case)
    with pytest.raises(ValueError, match="inputs or procedure|opponent"):
        recovery.validate_receipt(receipt, case, records, evidence)


def test_rejects_false_summary(recorded_cell):
    receipt, case, records, evidence = recorded_cell
    receipt["mean_return"] = 123
    with pytest.raises(ValueError, match="summary"):
        recovery.validate_receipt(receipt, case, records, evidence)


def test_rejects_missing_episode(recorded_cell):
    receipt, case, records, evidence = recorded_cell
    with pytest.raises(ValueError, match="summary"):
        recovery.validate_receipt(receipt, case, records[:-1], evidence)


def test_rejects_changed_episode_bytes(recorded_cell):
    receipt, case, records, evidence = recorded_cell
    (evidence / "episode0.json").write_text(json.dumps({"reward": -42}))
    with pytest.raises(ValueError, match="artifact changed"):
        recovery.validate_receipt(receipt, case, records, evidence)


def test_launcher_remote_script_compiles():
    launcher = load_module("launch_fsm_primary_recovery")
    compile("payload={}\n" + launcher.REMOTE, "remote-launcher", "exec")
    assert "--mem=64G" in launcher.REMOTE
    assert "--partition=community" in launcher.REMOTE
