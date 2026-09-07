from __future__ import annotations

import jax
import numpy as np
import pytest

from jaxborg.evaluation import matchup_runner
from jaxborg.evaluation.cia.fixed_topology import build_evaluation_cases
from jaxborg.evaluation.matchup_runner import LoadedMatchupPolicy
from jaxborg.scenarios.cc4.game_variants import CIA_RESILIENCE
from jaxborg.scenarios.cc4.topology import build_topology, save_topology


def _bank(tmp_path):
    paths = []
    for seed in (21, 22):
        path = tmp_path / f"topology-{seed}.snapshot.npz"
        save_topology(
            build_topology(jax.random.PRNGKey(seed), op_zone_min_servers=3),
            path,
        )
        paths.append(path)
    return paths


def test_matchup_uses_shared_fixed_cases_and_returns_structured_cia(tmp_path, monkeypatch):
    bank = _bank(tmp_path)
    sentinel_env = object()
    observed_roles = []

    monkeypatch.setattr(
        matchup_runner,
        "load_matchup_policy",
        lambda path, *, team, backend: LoadedMatchupPolicy(team, "jax", None, None, {"path": str(path)}),
    )
    monkeypatch.setattr(matchup_runner, "make_joint_jax_env", lambda *args, **kwargs: sentinel_env)

    def fake_episode(
        policies,
        *,
        variant,
        seed,
        deterministic,
        env,
        topology_index,
        host_resilience_role,
    ):
        del policies, variant, deterministic
        assert env is sentinel_env
        observed_roles.append((topology_index, np.asarray(host_resilience_role)))
        return float(seed + topology_index), [-10.0 * topology_index, -2.0 * seed, -30.0]

    monkeypatch.setattr(matchup_runner, "run_matchup_episode", fake_episode)
    cia = {"enabled": True, "metric": "resilience", "role_assignment": "fixed_per_topology"}

    result = matchup_runner.evaluate_matchup(
        "blue.safetensors",
        "red.safetensors",
        backend="jax",
        variant=CIA_RESILIENCE,
        seeds=[7, 9],
        episodes_per_seed=1,
        deterministic=True,
        progress=False,
        topology_path=bank,
        topology_sampling="exhaustive",
        cia=cia,
    )

    cases = build_evaluation_cases(bank, [7, 9], 1)
    assert result.episode_role_map_ids == [case.role_map_id for case in cases]
    assert result.episode_topology_fingerprints == [case.topology_fingerprint for case in cases]
    assert result.cia_config == cia
    assert result.cia_metric == "resilience"
    assert result.per_episode_cia == [
        {"c": 0.0, "i": -14.0, "a": -30.0},
        {"c": 0.0, "i": -18.0, "a": -30.0},
        {"c": -10.0, "i": -14.0, "a": -30.0},
        {"c": -10.0, "i": -18.0, "a": -30.0},
    ]
    assert result.cia_summary["n"] == 4
    assert result.cia_summary["c"] == {"mean": -5.0, "std": pytest.approx(5.773502691896258)}
    assert result.cia_summary["i"] == {"mean": -16.0, "std": pytest.approx(2.309401076758503)}
    assert result.cia_summary["a"] == {"mean": -30.0, "std": 0.0}
    assert [entry[0] for entry in observed_roles] == [0, 0, 1, 1]
    assert len(result.topology_role_maps) == 2
    assert [item["role_map_id"] for item in result.topology_role_maps] == [
        cases[0].role_map_id,
        cases[2].role_map_id,
    ]
