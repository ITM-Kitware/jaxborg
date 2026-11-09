import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("behavior_analysis", Path(__file__).with_name("analyze.py"))
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def trace():
    rows = []
    for step in range(3):
        masks = [[i == 0 for i in range(242)] for _ in range(5)]
        if step == 0:
            masks[0][98] = True
        rows.append(
            {
                "blue_reward": -1 if step == 0 else 0,
                "reward_ria": 0,
                "reward_lwf": 0,
                "reward_asf": 0,
                "action_cost": -1 if step == 0 else 0,
                "green_lwf_count": 0,
                "green_asf_count": 0,
                "impact_count": 0,
                "post_blocked_zones": [[False]],
                "blue_actions": [98 if step == 0 else 0, 0, 0, 0, 0],
                "blue_action_masks": masks,
                "blue_busy": [step > 0, False, False, False, False],
                "blue_effective_type": [4, 0, 0, 0, 0],
                "blue_effective_host": [0, -1, -1, -1, -1],
                "blue_effective_src_subnet": [-1] * 5,
                "blue_effective_dst_subnet": [-1] * 5,
                "blue_effective_completes": [step == 2, True, True, True, True],
                "pre_host_compromised": [1],
                "post_host_compromised": [int(step != 2)],
                "blue_owned_hosts": [[True]] * 5,
                "blue_observed_subnets": [[0, -1, -1]] * 5,
                "observation_host_map": [list(range(16))],
                "host_subnet": [0],
            }
        )
    return {key: [row[key] for row in rows] for key in rows[0]}


def test_pending_restore_is_not_three_choices_or_three_completions():
    result = ANALYZER.summarize(trace())
    assert result["totals"]["idle_decisions"] == 13
    assert result["totals"]["busy_agent_ticks"] == 2
    assert result["totals"]["forced_sleep_ticks"] == 2
    assert result["chosen_counts"]["Restore"] == 1
    assert result["completed_counts"]["Restore"] == 1
    assert result["chosen_counts"]["Sleep"] == 12
    assert result["idle_decisions_with_available_type"]["Restore"] == 1
    assert result["chosen_per_available_type_decision"]["Restore"] == 1
    assert result["host_coverage_of_idle_opportunities"][0] == 1
    assert len(result["events"]) == 1


def test_optional_terminal_padding_is_excluded():
    data = trace()
    data["valid"] = [True, True, False]
    assert ANALYZER.summarize(data)["totals"]["steps"] == 2


def test_reward_mismatch_is_rejected():
    data = trace()
    data["reward_asf"][0] = -3
    with pytest.raises(ValueError, match="Reward components"):
        ANALYZER.summarize(data)


def test_busy_non_sleep_is_rejected():
    data = copy.deepcopy(trace())
    data["blue_actions"][1][0] = 98
    with pytest.raises(ValueError, match="Busy Blue"):
        ANALYZER.summarize(data)


def test_analysis_reads_evaluation_owned_unpadded_traces(tmp_path, monkeypatch):
    from jaxborg.evaluation import capture_store

    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    study = tmp_path / "study"
    evidence = study / "evidence"
    plan_path = study / "check/plan.json"
    case = {"topology_index": 0, "episode_root": 21000, "canonical_return": -1}
    plan = {
        "study_root": str(study),
        "source_revision": "revision",
        "topologies": [{"sha256": "topology"}],
        "protocol": {"episode_roots": [21000], "opponents": ["fsm"], "horizon": 3},
        "cells": [],
    }
    records = []
    for index, steps in enumerate((49968000, 124992000)):
        identity = str(index)
        cell = {
            "id": identity,
            "opponent": "fsm",
            "checkpoint": {"steps": steps, "model_artifact": "model"},
            "reference": {"case": {"variant": {}}, "episodes": [case]},
        }
        plan["cells"].append(cell)
        attempt = f"attempt:{identity}"
        evaluation = f"evaluation:{identity}"
        records.extend(
            [
                {"id": attempt, "kind": "attempt", "status": "succeeded"},
                {
                    "id": evaluation,
                    "kind": "evaluation",
                    "status": "succeeded",
                    "attempt_id": attempt,
                    "inputs": ["model"],
                },
            ]
        )
        directory = evidence / "attempts" / identity
        put(
            directory / "configuration.json",
            {
                "execution": {"source_revision": "revision"},
                "seeds": [21000],
                "topology": {"sha256": ["topology"]},
                "base_variant": {},
                "deterministic": False,
                "episodes_per_seed": 1,
                "trajectories": True,
            },
        )
        trace_path = directory / "artifacts/trajectories/0.json"
        put(trace_path, {"episode_index": 0, "steps": trace()})
        records.append(
            {
                "id": f"trace:{identity}",
                "kind": "artifact",
                "producer_id": evaluation,
                "role": "JAX policy trajectory",
                "uri": str(trace_path.relative_to(evidence)),
                "sha256": ANALYZER.digest(trace_path),
            }
        )
    put(plan_path, plan)
    for cell in plan["cells"]:
        put(
            plan_path.parent / "cells" / cell["id"] / "receipt.json",
            {
                "cell": cell["id"],
                "evaluation_id": f"evaluation:{cell['id']}",
                "canonical_parity": True,
                "plan_sha256": ANALYZER.digest(plan_path),
                "episodes": [{**case, "matches_canonical": True, "traced_return": -1}],
            },
        )
    monkeypatch.setattr(capture_store, "load_records", lambda _: records)
    ANALYZER.analyze(plan_path, study / "summary")
    result = json.loads((study / "summary/summary.json").read_text())
    assert len(result["episodes"]) == 2
    assert len(result["paired_cases"]) == 1
    assert result["missing_cells"] == []
