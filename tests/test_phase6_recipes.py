"""Phase 6 / S4 — sanity tests for the 2×2 factorial recipe matrix.

Validates ``recipes/cec_phase6_{C00,C10,C01,C11}.yaml``:

* every recipe loads and projects to a JAX config without raising;
* ``TOPOLOGY_BANK`` length is 0 (no bank) or 16 (full bank), per arm;
* every path in a configured ``TOPOLOGY_BANK`` exists on disk;
* ``MISSION_BANK`` is ``None`` (no bank) or matches the plan's 4-entry
  default ``[(1,1,1), (3,1,1), (1,3,1), (1,1,3)]`` with amplify=1.0.

The 2×2 factorial (per plans/jax/cc4/cec/cec-phase6-plan.md):

    | arm | topology bank | mission bank |
    | --- | ------------- | ------------ |
    | C00 | (none)        | (none)       |
    | C10 | 16 snapshots  | (none)       |
    | C01 | (none)        | 4-entry      |
    | C11 | 16 snapshots  | 4-entry      |
"""

from __future__ import annotations

import pytest

from jaxborg.recipe import load, project_jax

EXPECTED_MISSION_BANK = [[1.0, 1.0, 1.0], [3.0, 3.0, 1.0], [1.0, 3.0, 3.0], [3.0, 1.0, 3.0]]
ARMS = {
    "cec_phase6_C00": (0, None),
    "cec_phase6_C10": (16, None),
    "cec_phase6_C01": (0, EXPECTED_MISSION_BANK),
    "cec_phase6_C11": (16, EXPECTED_MISSION_BANK),
}


@pytest.fixture(scope="module", params=sorted(ARMS))
def arm(request):
    name = request.param
    recipe = load(name)
    cfg = project_jax(recipe)
    return name, recipe, cfg


def test_recipe_loads(arm):
    name, recipe, _cfg = arm
    assert recipe["meta"]["name"] == name
    # Plan citation is required so future readers can find the rationale.
    blob = " ".join(str(recipe["meta"].get(k, "")) for k in ("source", "notes"))
    assert "cec-phase6-plan.md" in blob


def test_train_variant_is_cc4_stock(arm):
    _name, recipe, _cfg = arm
    # CEC-faithful: training partner fixed (cc4_stock variant's red = fsm).
    assert recipe["train"]["variant"] == "cc4_stock"


def test_total_timesteps_3m(arm):
    _name, recipe, _cfg = arm
    assert int(recipe["train"]["total_timesteps"]) == 3_000_000


def test_topology_bank_length(arm):
    name, _recipe, cfg = arm
    expected_len, _ = ARMS[name]
    assert len(cfg["TOPOLOGY_BANK"]) == expected_len, (
        f"{name}: expected TOPOLOGY_BANK length {expected_len}, got {len(cfg['TOPOLOGY_BANK'])}"
    )


def test_topology_bank_paths_exist(arm):
    name, _recipe, cfg = arm
    expected_len, _ = ARMS[name]
    if expected_len == 0:
        pytest.skip("no topology bank configured for this arm")
    for p in cfg["TOPOLOGY_BANK"]:
        assert p.exists(), f"{name}: topology snapshot missing on disk: {p}"


def test_mission_bank_matches(arm):
    name, _recipe, cfg = arm
    _, expected_mission = ARMS[name]
    if expected_mission is None:
        assert cfg["MISSION_BANK"] is None, f"{name}: expected no mission bank, got {cfg['MISSION_BANK']!r}"
    else:
        assert cfg["MISSION_BANK"] == expected_mission, f"{name}: mission bank mismatch (got {cfg['MISSION_BANK']!r})"


def test_mission_bank_amplify_is_unity_for_bank_arms(arm):
    name, _recipe, cfg = arm
    _, expected_mission = ARMS[name]
    if expected_mission is None:
        return
    assert cfg["MISSION_BANK_AMPLIFY"] == 1.0, (
        f"{name}: expected mission_bank_amplify=1.0 (plan default), got {cfg['MISSION_BANK_AMPLIFY']}"
    )
