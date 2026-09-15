from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "eval"
sys.path.insert(0, str(SCRIPT_DIR))

from export_trajectory import _build_trajectory_dict  # noqa: E402
from generate_cynex_trajectories import (  # noqa: E402
    _cotrained_episode_variant,
    _cyborg_blue_indices,
    _resilience_step,
    infer_policy_backend,
)

from jaxborg.scenarios.cc4.game_variants import CC4_STOCK  # noqa: E402


def _action(name: str, host: str, status: str = "TRUE") -> dict:
    return {"step": 0, "Action": name, "Status": status, "Host": host, "Params": {}}


def test_policy_backend_is_inferred_and_mixed_backends_are_rejected():
    assert infer_policy_backend("blue.safetensors", "red.safetensors") == "jax"
    assert infer_policy_backend("blue.pt", "red.pt") == "cyborg"
    with pytest.raises(ValueError, match="same backend"):
        infer_policy_backend("blue.safetensors", "red.pt")


def test_cotrained_episode_variant_compensates_for_cyborg_early_termination():
    assert _cotrained_episode_variant(CC4_STOCK, 500).num_steps == 501


def test_canonical_blue_actions_translate_to_cyborg_indices():
    lookups = [np.asarray([4, 8, -1]), np.asarray([9, 3, 7])]
    translated = _cyborg_blue_indices(np.asarray([8, 7]), lookups)
    np.testing.assert_array_equal(translated, np.asarray([1, 2]))

    with pytest.raises(RuntimeError, match="no unique CybORG translation"):
        _cyborg_blue_indices(np.asarray([2]), [lookups[0]])


def test_resilience_scores_follow_restore_then_red_impact_order():
    role_map = {"auth": 1, "db": 2, "web": 3}
    actions = {
        "blue_agent_0": [_action("Sleep", ""), _action("Restore", "auth")],
        "red_agent_0": [_action("Impact", "auth"), _action("Sleep", "")],
    }

    impacted, score = _resilience_step(frozenset(), role_map, actions, ["blue_agent_0"], ["red_agent_0"], 0)
    assert impacted == {"auth"}
    assert score == {"C": -10.0, "I": -10.0, "A": -10.0, "Resilience": -10.0}

    impacted, score = _resilience_step(impacted, role_map, actions, ["blue_agent_0"], ["red_agent_0"], 1)
    assert impacted == set()
    assert score == {"C": 0.0, "I": 0.0, "A": 0.0, "Resilience": 0.0}


def test_v2_builder_includes_cross_play_metrics_and_provenance():
    actions = {
        "blue_agent_0": [_action("Monitor", "host")],
        "red_agent_0": [_action("Impact", "host")],
    }
    metrics = [{"C": -10.0, "I": 0.0, "A": -10.0, "Resilience": -20 / 3}]
    policies = {
        "blue": {"path": "/runs/a.safetensors", "team": "blue"},
        "red": {"path": "/runs/b.safetensors", "team": "red"},
    }

    trajectory = _build_trajectory_dict(
        0,
        42,
        1,
        "LearnedBlue",
        ["blue_agent_0"],
        ["red_agent_0"],
        [],
        {},
        {},
        actions,
        [],
        metric_scores=metrics,
        host_resilience_roles={"host": 2},
        policies=policies,
    )

    assert trajectory["metric_scores"] == metrics
    assert trajectory["host_resilience_roles"] == {"host": 2}
    assert trajectory["policies"] == policies
    assert trajectory["agent_actions"] == actions
