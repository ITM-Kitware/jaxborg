from __future__ import annotations

import json

import jax
import numpy as np
import pytest

from jaxborg.evaluation.cia.fixed_topology import (
    build_evaluation_cases,
    canonical_topology_fingerprint,
    fixed_role_assignment,
    validate_fixed_role_eligibility,
)
from jaxborg.scenarios.cc4.topology import build_topology, save_topology
from jaxborg.scenarios.cc4.topology_roles import ROLE_AUTH, ROLE_DB, ROLE_WEB


def _save_topology(path, seed, *, metadata=None):
    const = build_topology(
        jax.random.PRNGKey(seed),
        op_zone_min_servers=3,
    )
    save_topology(const, path, metadata=metadata)
    return const


def test_fixed_roles_follow_topology_contents_across_paths_and_bank_order(tmp_path):
    first = tmp_path / "first.snapshot.npz"
    copied = tmp_path / "copied.snapshot.npz"
    other = tmp_path / "other.snapshot.npz"
    const = _save_topology(first, 11, metadata={"label": "original"})
    save_topology(const, copied, metadata={"label": "different provenance"})
    _save_topology(other, 12)

    first_assignment = fixed_role_assignment(first)
    copied_assignment = fixed_role_assignment(copied)
    assert canonical_topology_fingerprint(first) == canonical_topology_fingerprint(copied)
    assert first_assignment.host_roles == copied_assignment.host_roles
    assert first_assignment.role_map_id == copied_assignment.role_map_id
    assert sorted(first_assignment.role_by_host_index.values()) == [ROLE_AUTH, ROLE_DB, ROLE_WEB]
    assert set(first_assignment.role_by_host_index) <= set(first_assignment.eligible_host_indices)

    forward = build_evaluation_cases([first, other], [20, 30], 2)
    reordered = build_evaluation_cases([other, copied], [20, 30], 2)
    forward_by_fingerprint = {case.topology_fingerprint: (case.role_map_id, case.host_roles) for case in forward}
    reordered_by_fingerprint = {case.topology_fingerprint: (case.role_map_id, case.host_roles) for case in reordered}
    assert forward_by_fingerprint == reordered_by_fingerprint


def test_evaluation_cases_are_ordered_and_include_auditable_roles(tmp_path):
    first = tmp_path / "first.snapshot.npz"
    second = tmp_path / "second.snapshot.npz"
    _save_topology(first, 21)
    _save_topology(second, 22)

    cases = build_evaluation_cases([first, second], [100, 200], 2)

    assert [(case.topology_index, case.base_seed, case.replicate_index, case.episode_seed) for case in cases] == [
        (0, 100, 0, 100),
        (0, 100, 1, 101),
        (0, 200, 0, 200),
        (0, 200, 1, 201),
        (1, 100, 0, 100),
        (1, 100, 1, 101),
        (1, 200, 0, 200),
        (1, 200, 1, 201),
    ]
    assert all(case.topology_path.is_absolute() for case in cases)
    assert np.asarray(cases[0].role_array).dtype == np.int32
    audit = cases[0].audit_role_map()
    assert [entry["role"] for entry in audit["roles"]] == ["auth", "db", "web"]
    assert audit["role_map_id"] == cases[0].role_map_id
    json.dumps(audit)


def test_fixed_roles_reject_a_topology_with_too_few_candidates(tmp_path):
    path = tmp_path / "insufficient.snapshot.npz"
    const = build_topology(
        jax.random.PRNGKey(31),
        op_zone_min_servers=3,
    )
    candidates = validate_fixed_role_eligibility(const)
    host_is_server = const.host_is_server.at[np.asarray(candidates[2:])].set(False)
    save_topology(const.replace(host_is_server=host_is_server), path)

    with pytest.raises(ValueError, match="only 2 op-zone server candidates"):
        fixed_role_assignment(path)


@pytest.mark.parametrize(
    ("paths", "seeds", "episodes", "message"),
    [
        ([], [1], 1, "non-empty topology bank"),
        (["unused"], [], 1, "at least one episode seed"),
        (["unused"], [True], 1, "non-negative integers"),
        (["unused"], [-1], 1, "non-negative integers"),
        (["unused"], [1], 0, "must be positive"),
    ],
)
def test_evaluation_case_validation_fails_before_loading(paths, seeds, episodes, message):
    with pytest.raises(ValueError, match=message):
        build_evaluation_cases(paths, seeds, episodes)
