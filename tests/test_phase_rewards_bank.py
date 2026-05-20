"""Tests for the phase_rewards crown-jewel rotation bank (Phase 6 / P3)."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from jaxborg.constants import SUBNET_IDS
from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
from jaxborg.scenarios.cc4.topology_numpy import (
    _build_phase_rewards,
    get_phase_rewards_bank,
)


def test_bank_metadata():
    bank = get_phase_rewards_bank()
    # (N, MISSION_PHASES, NUM_SUBNETS, 3)
    assert bank.ndim == 4
    assert bank.shape[1:] == (3, 9, 3)  # MISSION_PHASES=3, NUM_SUBNETS=9, components=3
    assert bank.dtype == np.float32
    assert bank.shape[0] >= 2, "bank needs at least the canonical entry plus one rotation"


def test_bank_entry_0_matches_canonical():
    """bank[0] is the canonical phase_rewards table — legacy reproducibility."""
    bank = get_phase_rewards_bank()
    canonical = _build_phase_rewards()
    np.testing.assert_array_equal(bank[0], canonical)


def test_bank_entries_are_distinct():
    """No two bank entries should be identical — otherwise sampling is wasted."""
    bank = get_phase_rewards_bank()
    n = bank.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            assert not np.array_equal(bank[i], bank[j]), f"bank[{i}] == bank[{j}]"


def test_swap_entry_rotates_ops_a_ops_b():
    """Bank entry 1 should swap OPS_A ↔ OPS_B in phases 1 and 2."""
    bank = get_phase_rewards_bank()
    canonical = bank[0]
    swapped = bank[1]
    OA, OB = SUBNET_IDS["OPERATIONAL_ZONE_A"], SUBNET_IDS["OPERATIONAL_ZONE_B"]
    np.testing.assert_array_equal(swapped[1, OA], canonical[1, OB])
    np.testing.assert_array_equal(swapped[1, OB], canonical[1, OA])
    np.testing.assert_array_equal(swapped[2, OA], canonical[2, OB])
    np.testing.assert_array_equal(swapped[2, OB], canonical[2, OA])


def test_env_uses_bank_when_supplied():
    """Single-entry custom bank → const.phase_rewards == that entry."""
    canonical = _build_phase_rewards()
    custom = canonical.copy()
    custom[0, 0, 0] = 42.0  # arbitrary marker
    env = make_jax_env(CC4_STOCK, phase_rewards_bank=[custom])
    _, state = env.reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(state.const.phase_rewards), custom)


def test_no_bank_preserves_canonical_phase_rewards():
    env = make_jax_env(CC4_STOCK)
    _, state = env.reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(state.const.phase_rewards), _build_phase_rewards())


def test_determinism_same_key_same_entry():
    env = make_jax_env(CC4_STOCK, phase_rewards_bank=get_phase_rewards_bank())
    key = jax.random.PRNGKey(11)
    _, state_a = env.reset(key)
    _, state_b = env.reset(key)
    np.testing.assert_array_equal(
        np.asarray(state_a.const.phase_rewards),
        np.asarray(state_b.const.phase_rewards),
    )


def test_distribution_samples_multiple_entries():
    """Across many resets, ≥3 distinct bank entries should be observed."""
    bank = get_phase_rewards_bank()
    env = make_jax_env(CC4_STOCK, phase_rewards_bank=bank)
    keys = jax.random.split(jax.random.PRNGKey(99), 256)
    _, state = jax.vmap(env.reset)(keys)
    pr = np.asarray(state.const.phase_rewards)  # (256, MP, NS, 3)
    signatures = {tuple(slab.flatten().tolist()) for slab in pr}
    assert len(signatures) >= 3, f"only {len(signatures)} distinct bank entries hit in 256 resets"


def test_composes_with_mission_bank():
    """Phase-rewards bank applies BEFORE mission multiplier — both transform."""
    bank = get_phase_rewards_bank()
    env = make_jax_env(
        CC4_STOCK,
        phase_rewards_bank=[bank[1]],  # single non-canonical entry
        mission_bank=[[3.0, 1.0, 1.0]],  # single LWF=3 multiplier
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    expected = bank[1].copy()
    expected[..., 0] *= 3.0  # LWF column
    np.testing.assert_allclose(np.asarray(state.const.phase_rewards), expected, rtol=1e-5)


def test_invalid_shape_raises():
    bad = np.zeros((3, 9), dtype=np.float32)  # missing the (3,) component axis
    with pytest.raises(ValueError, match="phase_rewards_bank"):
        make_jax_env(CC4_STOCK, phase_rewards_bank=[bad])


def test_recipe_projection_bool_form():
    from jaxborg.recipe import project_jax

    recipe = {
        "meta": {"name": "test"},
        "algorithm": "ippo",
        "core": {"lr": 1e-3, "gamma": 0.99, "gae_lambda": 0.95},
        "arch": {"name": "shared"},
        "train": {
            "episode_length": 500,
            "total_timesteps": 1000,
            "phase_rewards_bank": True,
        },
    }
    cfg = project_jax(recipe)
    assert cfg["PHASE_REWARDS_BANK"] is not None
    assert cfg["PHASE_REWARDS_BANK"].ndim == 4


def test_recipe_projection_false_form():
    from jaxborg.recipe import project_jax

    recipe = {
        "meta": {"name": "test"},
        "algorithm": "ippo",
        "core": {"lr": 1e-3, "gamma": 0.99, "gae_lambda": 0.95},
        "arch": {"name": "shared"},
        "train": {
            "episode_length": 500,
            "total_timesteps": 1000,
            "phase_rewards_bank": False,
        },
    }
    cfg = project_jax(recipe)
    assert cfg["PHASE_REWARDS_BANK"] is None
