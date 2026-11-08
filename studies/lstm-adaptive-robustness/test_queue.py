"""Deadline choices must preserve complete, equal-budget comparison blocks."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("adaptive_queue", HERE / "submit_jobs.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def test_remaining_window_selects_whole_blocks_and_keeps_evaluation_time():
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
    with pytest.raises(ValueError, match="increasing"):
        queue.projected_run_seconds([{"wall_time_s": 600}] * 10)
