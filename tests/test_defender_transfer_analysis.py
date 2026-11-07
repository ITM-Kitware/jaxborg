"""Exercise the analysis CLI with synthetic measurements in real capture stores."""

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jaxborg.evaluation.capture_store import (
    add_artifact,
    finish_evaluation,
    load_records,
    record_episode,
    start_evaluation,
)

ANALYZER = Path(__file__).resolve().parents[1] / "studies/defender-transfer/analyze.py"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def captured_study(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    root = tmp_path / "evidence"

    def put(name, text):
        path = inputs / name
        path.write_text(text)
        return path

    def policy(identity, **metadata):
        model = put(f"{identity}.safetensors", f"synthetic policy bytes: {identity}")
        sidecar = put(f"{identity}.yaml", f"name: {identity}\n")
        return {
            "id": identity,
            "model": str(model),
            "sha256": sha256(model),
            "sidecar": str(sidecar),
            "sidecar_sha256": sha256(sidecar),
            **metadata,
        }

    blues = [
        policy(f"{condition}-{stage}", condition=condition, training_seed=42, stage=stage)
        for condition in ("single", "diverse")
        for stage in ("early", "final")
    ]
    red = policy("red")
    topologies = [put(f"topology-{index}.npz", f"synthetic topology bytes: {index}") for index in range(2)]
    recipe = put("effective.yaml", "name: screening-evaluation\n")
    variant = {"name": "cc4_stock", "num_steps": 500}
    cells = [
        {
            "trial_id": f"{kind}:{blue['id']}",
            "blue_id": blue["id"],
            "red_id": None if kind == "scripted" else red["id"],
            "kind": kind,
            "bank_id": "anchor",
            "reds": ["fsm"] if kind == "scripted" else [],
            "variant": variant,
            "recipe_sha256": sha256(recipe),
        }
        for blue in blues
        for kind in ("scripted", "learned")
    ]
    cells.append({**cells[0], "trial_id": "no-attempt"})
    manifest = {
        "source_revision": "a" * 40,
        "policies": {"blue": blues, "red": [red]},
        "topology_banks": {
            "anchor": {"snapshots": [{"path": str(path), "sha256": sha256(path)} for path in topologies]}
        },
        "screening": {"horizon": 500, "seeds": [17, 18]},
        "cells": cells,
    }
    protocol = {
        "schema_version": 1,
        "kind": "protocol",
        "id": "defender-screening-test",
        "study_id": "defender-screening-test",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": ["resolved_config"], "evaluation": ["resolved_config"]},
        "trials": [
            {"id": cell["trial_id"], "condition_id": cell["trial_id"], "replicate_id": "42", "matched_unit_id": None}
            for cell in cells
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
        "extensions": {
            "defender_transfer": {
                "schema_version": 1,
                "source": "manifest.json",
                "values": {
                    "manifest_sha256": hashlib.sha256(
                        json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
                    ).hexdigest()
                },
            }
        },
    }
    selection = {}
    for cell in cells[:-1]:
        blue = next(item for item in blues if item["id"] == cell["blue_id"])
        configuration = {
            "topology": {"sha256": [sha256(path) for path in topologies]},
            "episodes_per_seed": 1,
            "deterministic": False,
            "execution": {"source_revision": manifest["source_revision"]},
            "base_variant" if cell["kind"] == "scripted" else "variant": variant,
        }
        if cell["kind"] == "scripted":
            configuration["seeds"] = [17, 18]
        else:
            configuration["seed_roles"] = {"episode_roots": [17, 18]}
        retained_inputs = {
            "blue checkpoint": Path(blue["model"]),
            "blue sidecar": Path(blue["sidecar"]),
            "effective recipe": recipe,
            **{f"topology {index}": path for index, path in enumerate(topologies)},
        }
        if cell["kind"] == "learned":
            retained_inputs.update({"red checkpoint": Path(red["model"]), "red sidecar": Path(red["sidecar"])})
        capture = start_evaluation(
            root,
            protocol=protocol,
            trial_id=cell["trial_id"],
            requested_episodes=4,
            resolved_config=configuration,
            inputs=retained_inputs,
            source_files={"studies/defender-transfer/analyze.py": ANALYZER},
        )
        offset = (5 if blue["stage"] == "final" else 0) + (2 if blue["condition"] == "diverse" else 0)
        rewards = [offset + index for index in range(4)]
        seeds = [17, 18, 17, 18]
        paths = [str(capture.inputs[f"topology {index}"]) for index in range(2) for _ in range(2)]
        for index, (seed, reward) in enumerate(zip(seeds, rewards, strict=True)):
            record_episode(capture, episode_index=index, seed=seed, reward=reward)
        if cell["kind"] == "scripted":
            result = {
                "eval_red": "fsm",
                "per_episode": rewards,
                "per_episode_seeds": seeds,
                "per_episode_topology_paths": paths,
                "episode_role_map_ids": ["role-0"] * 2 + ["role-1"] * 2,
                "per_episode_topology_fingerprints": ["fingerprint-0"] * 2 + ["fingerprint-1"] * 2,
                "per_episode_cia": [{"c": 0, "i": 0, "a": 0}] * 4,
            }
            filename, role = "scripted-fsm.jsonl", "scripted opponent episode results"
        else:
            result = {
                "blue_returns": rewards,
                "red_returns": [-reward for reward in rewards],
                "episode_seeds": seeds,
                "episode_topology_paths": paths,
                "episode_role_map_ids": [],
                "episode_topology_fingerprints": [],
                "per_episode_cia": [],
            }
            filename, role = "result.json", "joint matchup result"
        result_path = capture.directory / filename
        result_path.write_text(json.dumps(result) + "\n")
        add_artifact(capture, result_path, role=role)
        finish_evaluation(capture, status="succeeded")
        selection[cell["trial_id"]] = capture.evaluation_id

    records = load_records(root)
    assert len([record for record in records if record["kind"] == "evaluation"]) == 8
    manifest_path, selection_path = tmp_path / "manifest.json", tmp_path / "selection.json"
    manifest_path.write_text(json.dumps(manifest))
    selection_path.write_text(json.dumps(selection))
    return {"root": root, "manifest": manifest_path, "selection": selection_path, "selections": selection}


def run_analysis(study, output, *, root=None, check=True):
    return subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--manifest",
            str(study["manifest"]),
            "--root",
            str(root or study["root"]),
            "--selection",
            str(study["selection"]),
            "--output",
            str(output),
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def test_selected_capture_store_produces_exact_screening_tables(captured_study, tmp_path):
    output = tmp_path / "analysis"
    run_analysis(captured_study, output)

    episodes = read_csv(output / "episodes.csv")
    assert len(episodes) == 32
    assert {row["kind"] for row in episodes} == {"scripted", "learned"}
    assert all(row["c"] == "0.0" for row in episodes if row["kind"] == "scripted")
    assert all(row["c"] == "" for row in episodes if row["kind"] == "learned")
    assert len(read_csv(output / "summary.csv")) == 16
    contrasts = read_csv(output / "paired-contrasts.csv")
    assert len(contrasts) == 24
    for row in contrasts:
        expected = 5 if row["contrast"] == "final_minus_early" else 2
        assert float(row["reward_mean_difference"]) == expected
        assert int(row["n_paired_episodes"]) == (4 if row["topology_sha256"] == "ALL_EQUAL_TOPOLOGY_WEIGHT" else 2)
    coverage = read_csv(output / "coverage.csv")
    assert len(coverage) == 9
    assert [row["trial_id"] for row in coverage if row["status"] == "no_attempt"] == ["no-attempt"]
    assert sum(row["status"] == "succeeded" for row in coverage) == 8
    assert json.loads((output / "selection.json").read_text()) == captured_study["selections"]
    assert (output / "analyze.py").read_bytes() == ANALYZER.read_bytes()
    for filename, expected in json.loads((output / "hashes.json").read_text()).items():
        assert sha256(output / filename) == expected


def test_completed_evaluations_require_explicit_selection(captured_study, tmp_path):
    captured_study["selection"].write_text("{}")
    output = tmp_path / "unselected"
    run_analysis(captured_study, output)

    coverage = read_csv(output / "coverage.csv")
    assert sum(row["status"] == "unselected" for row in coverage) == 8
    assert all(row["selected_episode_count"] == "0" for row in coverage)
    assert len(json.loads((output / "gaps.json").read_text())) == 9
    assert read_csv(output / "episodes.csv") == []
    assert read_csv(output / "paired-contrasts.csv") == []


def test_relocated_store_preserves_pairing_and_tables(captured_study, tmp_path):
    original_output = tmp_path / "original-analysis"
    run_analysis(captured_study, original_output)
    relocated = tmp_path / "relocated-evidence"
    shutil.move(captured_study["root"], relocated)
    relocated_output = tmp_path / "relocated-analysis"
    run_analysis(captured_study, relocated_output, root=relocated)

    for filename in ("episodes.csv", "summary.csv", "paired-contrasts.csv", "coverage.csv"):
        assert (relocated_output / filename).read_bytes() == (original_output / filename).read_bytes()


def test_retained_result_tampering_rejects_analysis(captured_study, tmp_path):
    result = next((captured_study["root"] / "attempts").glob("*/artifacts/result.json"))
    result.write_text("{}")
    output = tmp_path / "tampered-analysis"
    completed = run_analysis(captured_study, output, check=False)

    assert completed.returncode != 0
    assert "Artifact byte identity differs" in completed.stderr
    assert not output.exists()


def test_manifest_relabeling_rejects_analysis(captured_study, tmp_path):
    manifest = json.loads(captured_study["manifest"].read_text())
    manifest["policies"]["blue"][0]["condition"] = "diverse"
    captured_study["manifest"].write_text(json.dumps(manifest))
    output = tmp_path / "relabeled-analysis"
    completed = run_analysis(captured_study, output, check=False)

    assert completed.returncode != 0
    assert "Manifest differs from the evidence store's declared protocol" in completed.stderr
    assert not output.exists()


def test_selected_attempt_from_another_protocol_rejects_analysis(captured_study, tmp_path):
    terminal = next((captured_study["root"] / "attempts").glob("*/terminal.json"))
    records = json.loads(terminal.read_text())
    attempt = next(record for record in records if record["kind"] == "attempt")
    attempt["protocol_id"] = "another-protocol"
    terminal.write_text(json.dumps(records))
    output = tmp_path / "wrong-protocol-analysis"
    completed = run_analysis(captured_study, output, check=False)

    assert completed.returncode != 0
    assert "Selected attempt belongs to a different protocol" in completed.stderr
    assert not output.exists()
