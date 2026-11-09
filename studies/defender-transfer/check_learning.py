"""Reconstruct the recorded LSTM learning check from verified study-analysis outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

STUDY = Path(__file__).resolve().parent
DEFAULT_RECORD = STUDY / "checks" / "lstm-learning-20261107" / "record.json"
AGGREGATE = "ALL_EQUAL_TOPOLOGY_WEIGHT"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_files(root, expected):
    root = Path(root).resolve()
    for name, identity in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != identity:
            raise ValueError(f"Recorded artifact bytes differ: {name}")


def select_opponent_contrasts(rows, settings):
    """Require the common other-lineage opponents for both compared Blues."""
    selected = []
    for seed, opponents in settings["opponents_by_blue_seed"].items():
        for bank in settings["banks"]:
            for opponent in opponents:
                matches = [
                    row
                    for row in rows
                    if row["contrast"] == "diverse_minus_single"
                    and row["kind"] == "learned"
                    and row["bank_id"] == bank
                    and row["opponent"] == opponent
                    and row["training_seed"] == seed
                    and row["candidate_blue_id"] == f"blue-diverse-{seed}-final"
                    and row["reference_blue_id"] == f"blue-single-{seed}-final"
                    and row["topology_sha256"] == AGGREGATE
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one complete contrast for Blue seed {seed}, {bank}, {opponent}")
                row = matches[0]
                selected.append(
                    {
                        "blue_seed": int(seed),
                        "bank": bank,
                        "opponent": opponent,
                        "candidate_blue_id": row["candidate_blue_id"],
                        "reference_blue_id": row["reference_blue_id"],
                        "n_paired_episodes": int(row["n_paired_episodes"]),
                        "reward_difference": float(row["reward_mean_difference"]),
                    }
                )
    return selected


def summarize(analysis, record):
    means = read_csv(analysis / "policy-means.csv")
    contrasts = read_csv(analysis / "paired-contrasts.csv")
    episodes = read_csv(analysis / "episodes.csv")
    settings = record["comparisons"]
    learning = []
    for condition in settings["conditions"]:
        for seed in settings["training_seeds"]:
            values = {}
            for stage in ("early", "final"):
                matches = [
                    row
                    for row in means
                    if row["blue_id"] == f"blue-{condition}-{seed}-{stage}"
                    and row["kind"] == "scripted"
                    and row["bank_id"] == settings["learning_bank"]
                    and row["opponent"] == settings["learning_opponent"]
                ]
                if len(matches) != 1:
                    raise ValueError(f"Missing or repeated learning reference: {condition}, {seed}, {stage}")
                values[stage] = float(matches[0]["reward"])
            learning.append(
                {
                    "condition": condition,
                    "training_seed": seed,
                    "early_reward": values["early"],
                    "final_reward": values["final"],
                    "reward_difference": values["final"] - values["early"],
                }
            )
    scripted = [
        row
        for row in contrasts
        if row["contrast"] == "final_minus_early" and row["kind"] == "scripted" and row["topology_sha256"] == AGGREGATE
    ]
    if len(scripted) != record["expected_coverage"]["scripted_learning_contrasts"]:
        raise ValueError("Incomplete scripted learning comparison")
    differences = [float(row["reward_mean_difference"]) for row in scripted]
    selection = read_json(analysis / "selection.json")
    if len(selection) != record["expected_coverage"]["selected_cells"]:
        raise ValueError("Selected-cell count differs from the recorded check")
    if len(episodes) != record["expected_coverage"]["episodes"]:
        raise ValueError("Episode count differs from the recorded check")
    return {
        "selected_cells": len(selection),
        "episode_count": len(episodes),
        "reserved_fsm_learning": learning,
        "scripted_learning": {
            "comparisons": len(scripted),
            "positive_reward_differences": sum(value > 0 for value in differences),
            "minimum_reward_difference": min(differences),
            "maximum_reward_difference": max(differences),
            "comparisons_with_any_cia_proxy_worse": sum(
                any(float(row[f"{metric}_mean_difference"]) < 0 for metric in ("c", "i", "a")) for row in scripted
            ),
        },
        "other_lineage_opponents": select_opponent_contrasts(contrasts, settings["opponent_exposure"]),
    }


def run(record_path, analysis, output, exact_snapshot=False):
    record_path, analysis, output = (Path(path).resolve() for path in (record_path, analysis, output))
    if output.exists():
        raise FileExistsError("Check output must be a new directory")
    if output.is_relative_to(STUDY.parents[1]):
        raise ValueError("Generated check outputs belong outside the source checkout")
    record = read_json(record_path)
    if record["schema_version"] != 1:
        raise ValueError("Unsupported check record")
    selection_path = (record_path.parent / record["selection"]["file"]).resolve()
    if not selection_path.is_relative_to(record_path.parent):
        raise ValueError("Recorded selection must remain beside the check definition")
    if digest(selection_path) != record["selection"]["sha256"]:
        raise ValueError("Checked-in selection differs from the recorded check")
    hashes = record["recorded_analysis_sha256"]
    names = hashes if exact_snapshot else record["stable_rebuild_files"]
    verified = {name: hashes[name] for name in names}
    verify_files(analysis, verified)
    manifest = read_json(analysis / "manifest-input.json")
    provenance = read_json(analysis / "provenance.json")
    if manifest["source_revision"] != record["producing_source_revision"]:
        raise ValueError("Evaluator source differs from the recorded check")
    if provenance["protocol"]["sha256"] != record["protocol_sha256"]:
        raise ValueError("Evaluation protocol differs from the recorded check")
    if provenance["manifest"]["sha256"] != hashes["manifest-input.json"]:
        raise ValueError("Manifest provenance differs from the recorded check")
    if provenance["selection"]["sha256"] != record["selection"]["sha256"]:
        raise ValueError("Selection provenance differs from the recorded check")
    if read_json(selection_path) != read_json(analysis / "selection.json"):
        raise ValueError("Analysis does not use the checked-in selection")
    result = {
        "check_id": record["id"],
        "question": record["question"],
        "record_sha256": digest(record_path),
        "check_source_sha256": digest(__file__),
        "producing_source_revision": record["producing_source_revision"],
        "selection_sha256": record["selection"]["sha256"],
        "verification": "exact retained snapshot" if exact_snapshot else "stable numerical reconstruction",
        "verified_analysis_sha256": verified,
        "findings": summarize(analysis, record),
        "limitations": record["limitations"],
    }
    output.mkdir(parents=True)
    (output / "check.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    shutil.copyfile(__file__, output / "check_learning.py")
    shutil.copyfile(record_path, output / "record.json")
    shutil.copyfile(selection_path, output / "selection.json")
    print(json.dumps({"output": str(output), "findings": result["findings"]}, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--analysis", type=Path, required=True, help="Existing or rebuilt defender-transfer analysis")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the source checkout")
    parser.add_argument("--exact-snapshot", action="store_true", help="Also verify original inventory/provenance bytes")
    args = parser.parse_args()
    run(args.record, args.analysis, args.output, args.exact_snapshot)
