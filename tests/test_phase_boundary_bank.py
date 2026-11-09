"""Tests for the phase-boundary jitter bank (Phase 6 / P2)."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
from jaxborg.scenarios.cc4.topology_numpy import (
    PHASE_BOUNDARIES_BANK,
    get_phase_boundaries_bank,
)

DEFAULT_BANK = [list(t) for t in PHASE_BOUNDARIES_BANK]


def test_default_bank_metadata():
    bank = get_phase_boundaries_bank()
    assert bank.shape == (4, 3)
    assert bank.dtype == np.int32
    # phase 0 always starts at step 0
    assert (bank[:, 0] == 0).all()
    # boundaries strictly increasing within each entry
    for entry in bank:
        assert entry[0] < entry[1] < entry[2]


def test_bank_overrides_const_phase_boundaries():
    """Sampling from a 1-entry bank produces that exact triple in const."""
    target = [(0, 50, 100)]
    env = make_jax_env(CC4_STOCK, phase_boundary_bank=target)
    _, state = env.reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(state.const.phase_boundaries), [0, 50, 100])


def test_no_bank_preserves_canonical_boundaries():
    env = make_jax_env(CC4_STOCK)
    _, state = env.reset(jax.random.PRNGKey(0))
    pb = np.asarray(state.const.phase_boundaries)
    # canonical CC4 split for 500-step episodes (3 ~equal phases)
    assert pb[0] == 0
    assert pb[1] > 0 and pb[2] > pb[1]


def test_determinism_same_key_same_index():
    env = make_jax_env(CC4_STOCK, phase_boundary_bank=DEFAULT_BANK)
    key = jax.random.PRNGKey(7)
    _, state_a = env.reset(key)
    _, state_b = env.reset(key)
    np.testing.assert_array_equal(
        np.asarray(state_a.const.phase_boundaries),
        np.asarray(state_b.const.phase_boundaries),
    )


def test_distribution_uniform_chi_square():
    """Across 10000 keys, each of 4 entries is sampled within ~5% of uniform."""
    env = make_jax_env(CC4_STOCK, phase_boundary_bank=DEFAULT_BANK)
    keys = jax.random.split(jax.random.PRNGKey(123), 10000)
    _, state = jax.vmap(env.reset)(keys)
    # Use phase_boundaries[1] (the unique phase-1 start) as the bucket key —
    # the four bank entries have phase 1 starts {167, 100, 200, 150}.
    p1_starts = np.asarray(state.const.phase_boundaries[:, 1])
    targets = np.array([t[1] for t in PHASE_BOUNDARIES_BANK])
    counts = np.array([(p1_starts == t).sum() for t in targets])
    expected = 10000 / len(targets)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    # df=3 chi-square critical at α=0.001 is 16.27. 0.001 picks up real bias
    # while leaving room for natural sampling jitter.
    assert chi2 < 16.27, f"phase-boundary bank under-uniform: counts={counts.tolist()}, chi2={chi2:.2f}"


def test_empty_bank_fast_path():
    env = make_jax_env(CC4_STOCK, phase_boundary_bank=[])
    _, state = env.reset(jax.random.PRNGKey(0))
    pb = np.asarray(state.const.phase_boundaries)
    # Identical to no-bank canonical
    env_ref = make_jax_env(CC4_STOCK)
    _, state_ref = env_ref.reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(pb, np.asarray(state_ref.const.phase_boundaries))


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="phase_boundary_bank"):
        make_jax_env(CC4_STOCK, phase_boundary_bank=[[0, 100], [0, 200]])


def test_recipe_projection():
    from jaxborg.recipe import project_jax

    recipe = {
        "meta": {"name": "test"},
        "algorithm": "ippo",
        "core": {"lr": 1e-3, "gamma": 0.99, "gae_lambda": 0.95},
        "arch": {"name": "shared"},
        "train": {
            "episode_length": 500,
            "total_timesteps": 1000,
            "phase_boundary_bank": [[0, 100, 300], [0, 200, 400]],
        },
    }
    cfg = project_jax(recipe)
    assert cfg["PHASE_BOUNDARY_BANK"] == [[0, 100, 300], [0, 200, 400]]
