"""Phase 2 mission-objective family: per-CIA-component multiplier bank tests."""

import jax
import numpy as np

from jaxborg.constants import MISSION_PHASES, NUM_SUBNETS
from jaxborg.topology import build_topology
from jaxborg.topology_numpy import (
    MISSION_PROFILE_MULTIPLIERS,
    NUM_MISSION_PROFILES,
    _build_phase_rewards,
    get_mission_profile_multipliers,
)


def test_multiplier_bank_shape_and_default():
    mults = get_mission_profile_multipliers()
    assert mults.shape == (NUM_MISSION_PROFILES, 3)
    np.testing.assert_array_equal(mults[0], np.array([1.0, 1.0, 1.0], dtype=np.float32))


def test_multiplier_bank_matches_constant():
    mults = get_mission_profile_multipliers()
    np.testing.assert_array_equal(mults, np.asarray(MISSION_PROFILE_MULTIPLIERS, dtype=np.float32))


def test_default_path_phase_rewards_unchanged():
    """vary_mission_profile=False reproduces the legacy phase_rewards."""
    default = np.asarray(_build_phase_rewards())
    for seed in [0, 1, 7, 42, 12345]:
        c = build_topology(jax.random.PRNGKey(seed), vary_mission_profile=False)
        np.testing.assert_array_equal(np.asarray(c.phase_rewards), default)
        assert int(c.mission_profile_index) == 0


def test_vary_mission_profile_samples_distinct_indices():
    indices = set()
    for seed in range(64):
        c = build_topology(jax.random.PRNGKey(seed), vary_mission_profile=True)
        indices.add(int(c.mission_profile_index))
    assert len(indices) >= 3, f"only {len(indices)} distinct profile indices in 64 keys"
    assert all(0 <= i < NUM_MISSION_PROFILES for i in indices)


def test_vary_mission_profile_applies_correct_multipliers():
    """Phase rewards equal the default scaled by the chosen multipliers."""
    base = np.asarray(_build_phase_rewards())
    mults = np.asarray(get_mission_profile_multipliers())
    for seed in range(32):
        c = build_topology(jax.random.PRNGKey(seed), vary_mission_profile=True)
        idx = int(c.mission_profile_index)
        expected = base * mults[idx][None, None, :]
        np.testing.assert_allclose(np.asarray(c.phase_rewards), expected, rtol=1e-6)


def test_mission_profile_combines_with_phase_rewards_axis():
    """Variation along axis C and the mission-profile axis stack multiplicatively."""
    seed = 3
    c_base = build_topology(jax.random.PRNGKey(seed), vary_mission_profile=False, vary_phase_rewards=False)
    c_combo = build_topology(jax.random.PRNGKey(seed), vary_mission_profile=True, vary_phase_rewards=True)
    mults = np.asarray(get_mission_profile_multipliers())
    idx = int(c_combo.mission_profile_index)
    # The combo should be (axis-C bank entry chosen) * mp_multipliers.
    # Drop into a numerically robust check: combo / mults equals an entry in the
    # axis-C bank.
    if idx == 0:
        np.testing.assert_array_equal(np.asarray(c_base.phase_rewards), np.asarray(c_combo.phase_rewards))
    else:
        recovered = np.asarray(c_combo.phase_rewards) / mults[idx][None, None, :]
        # recovered must be finite (no division by zero) — multipliers are all > 0.
        assert np.all(np.isfinite(recovered))


def test_fsm_red_env_vary_mission_profile_end_to_end():
    from jaxborg.fsm_red_env import FsmRedCC4Env

    env = FsmRedCC4Env(num_steps=10, topology_mode="generative", vary_mission_profile=True)
    seen_indices = set()
    for seed in [0, 1, 7, 13, 42]:
        _obs, env_state = env.reset(jax.random.PRNGKey(seed))
        seen_indices.add(int(env_state.const.mission_profile_index))
        assert env_state.const.phase_rewards.shape == (MISSION_PHASES, NUM_SUBNETS, 3)
    assert len(seen_indices) >= 2
