"""Deadline choices must preserve complete, equal-budget comparison blocks."""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("adaptive_queue", HERE / "submit_jobs.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def test_remaining_window_selects_whole_blocks_before_gpu_deadline():
    now = datetime(2026, 11, 8, 21, tzinfo=timezone.utc)
    assert queue.block_count(3 * 3600, now) == 2
    assert queue.block_count(4 * 3600, now) == 1
    with pytest.raises(ValueError, match="first complete paired block"):
        queue.block_count(4 * 3600, datetime(2026, 11, 9, 6, tzinfo=timezone.utc))


def test_projection_keeps_compilation_cost_without_extrapolating_it_each_update():
    rows = [{"wall_time_s": 600 + index * 30} for index in range(10)]
    projection = queue.projected_run_seconds(rows)
    assert projection["seconds_per_update"] == 30
    assert projection["expected_seconds"] == 600 + 207 * 30 + 300
    assert projection["reserved_seconds"] > projection["expected_seconds"]
    longer = queue.projected_run_seconds(rows, updates=624)
    assert longer["expected_seconds"] - projection["expected_seconds"] == 416 * 30
    with pytest.raises(ValueError, match="increasing"):
        queue.projected_run_seconds([{"wall_time_s": 600}] * 10)


def test_supplement_waits_for_both_main_lanes_and_keeps_paired_waves(tmp_path, monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 8, 19, tzinfo=timezone.utc)

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "manifest.json").write_text(json.dumps({"production_updates": 624}))
    lanes = {
        "17": [
            "br-blue42-primary-red17",
            "br-blue42-final-red17",
            "br-blue100-primary-red17",
            "br-blue100-final-red17",
        ],
        "83": [
            "br-blue42-final-red83",
            "br-blue42-primary-red83",
            "br-blue100-final-red83",
            "br-blue100-primary-red83",
        ],
    }
    (inputs / "protocol.json").write_text(
        json.dumps({"extensions": {"lstm_adaptive_robustness": {"values": {"lanes": lanes}}}})
    )
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks/production-ready.json").write_text(
        json.dumps(
            {
                "passed": True,
                "source_revision": "a" * 40,
                "manifest_sha256": queue.sha256(inputs / "manifest.json"),
                "fresh_compilation_equivalence": {"passed": True},
            }
        )
    )
    calls = []

    def submit(root, revision, kind, trial, minutes, dependencies):
        entry = {"job_id": len(calls) + 1, "kind": kind, "trial": trial, "dependencies": list(dependencies)}
        calls.append(entry)
        return entry

    monkeypatch.setattr(queue, "datetime", FixedDatetime)
    monkeypatch.setattr(queue, "verify", lambda *args: None)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        queue, "timing_projection", lambda root: {"reserved_seconds": 3 * 3600, "primary_reserved_seconds": 3600}
    )
    monkeypatch.setattr(queue, "submit", submit)
    args = Namespace(study_root=tmp_path, expected_revision="a" * 40, phase="production")
    queue.queue(args)
    trains = {call["trial"]: call for call in calls if call["kind"] == "train"}
    assert len(trains) == 12
    main_ends = {trains[lanes[lane][-1]]["job_id"] for lane in ("17", "83")}
    extra_first = [trains[f"br-blue42-{age}-red149"] for age in ("primary", "final")]
    assert all(set(call["dependencies"]) == main_ends for call in extra_first)
    first_ids = {call["job_id"] for call in extra_first}
    assert all(set(trains[f"br-blue100-{age}-red149"]["dependencies"]) == first_ids for age in ("primary", "final"))
    count = len(calls)
    with pytest.raises(FileExistsError):
        queue.queue(args)
    assert len(calls) == count
