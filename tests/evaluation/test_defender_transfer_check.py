"""Focused safeguards for the recorded defender-transfer learning check."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "defender_transfer_check", ROOT / "studies" / "defender-transfer" / "check_learning.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_changed_recorded_bytes_are_rejected(tmp_path):
    artifact = tmp_path / "episodes.csv"
    artifact.write_bytes(b"original episode observations")
    expected = {"episodes.csv": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    CHECK.verify_files(tmp_path, expected)
    artifact.write_bytes(b"changed observations")
    with pytest.raises(ValueError, match="Recorded artifact bytes differ"):
        CHECK.verify_files(tmp_path, expected)


def test_artifact_cannot_escape_analysis_directory(tmp_path):
    artifact = tmp_path / "outside.csv"
    artifact.write_bytes(b"outside")
    inside = tmp_path / "analysis"
    inside.mkdir()
    with pytest.raises(ValueError, match="Recorded artifact bytes differ"):
        CHECK.verify_files(inside, {"../outside.csv": hashlib.sha256(artifact.read_bytes()).hexdigest()})


def contrast(opponent, stage="final"):
    return {
        "contrast": "diverse_minus_single",
        "kind": "learned",
        "bank_id": "unseen",
        "opponent": opponent,
        "training_seed": "42",
        "candidate_blue_id": f"blue-diverse-42-{stage}",
        "reference_blue_id": f"blue-single-42-{stage}",
        "topology_sha256": CHECK.AGGREGATE,
        "n_paired_episodes": "64",
        "reward_mean_difference": "5.109375",
    }


SETTINGS = {
    "banks": ["unseen"],
    "opponents_by_blue_seed": {"42": ["red-single-100-final", "red-diverse-100-final"]},
}


def test_opponent_comparison_excludes_own_lineages_and_early_blues():
    rows = [
        contrast("red-single-100-final"),
        contrast("red-diverse-100-final"),
        contrast("red-single-42-final"),
        contrast("red-diverse-42-final"),
        contrast("red-single-100-final", stage="early"),
    ]
    selected = CHECK.select_opponent_contrasts(rows, SETTINGS)
    assert [row["opponent"] for row in selected] == SETTINGS["opponents_by_blue_seed"]["42"]
    assert all(row["candidate_blue_id"] == "blue-diverse-42-final" for row in selected)


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_ambiguous_common_opponent_cannot_be_silently_averaged(duplicate):
    rows = [contrast("red-single-100-final")]
    if duplicate:
        rows.extend([contrast("red-diverse-100-final"), contrast("red-diverse-100-final")])
    with pytest.raises(ValueError, match="Expected one complete contrast"):
        CHECK.select_opponent_contrasts(rows, SETTINGS)
