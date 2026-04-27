"""Axis C (CEC env-diversity) phase-reward landscape variation tests.

Covers:
1. Bank shape ``(31, 3, 9, 3)`` with float32 dtype.
2. ``bank[0]`` matches :func:`_build_phase_rewards` exactly.
3. Each variant has exactly one ``[-10, 0, -10]`` row in phase 1 and exactly
   one in phase 2.
4. ``build_topology(key, vary_phase_rewards=False)`` always reproduces the
   default phase_rewards (parity with legacy across multiple seeds).
5. ``build_topology(key, vary_phase_rewards=True)`` produces multiple distinct
   phase_rewards across keys.
6. End-to-end env-level test: ``FsmRedCC4Env(..., vary_phase_rewards=True)``
   resets cleanly and ``state.const.phase_rewards`` matches a bank entry.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import MISSION_PHASES, NUM_SUBNETS
from jaxborg.topology import build_topology
from jaxborg.topology_numpy import (
    PHASE_REWARDS_PRIMARY_TARGET_NAMES,
    _build_phase_rewards,
    _build_phase_rewards_variant,
    get_phase_rewards_bank,
)

HIGH_VALUE_ROW = np.array([-10.0, 0.0, -10.0], dtype=np.float32)


def test_bank_shape_and_dtype():
    bank = get_phase_rewards_bank()
    # 6 candidates → 6 × 5 ordered pairs = 30, plus default = 31.
    assert bank.shape == (31, MISSION_PHASES, NUM_SUBNETS, 3)
    assert bank.dtype == np.float32


def test_bank_zero_is_default():
    bank = get_phase_rewards_bank()
    default = _build_phase_rewards()
    assert np.array_equal(bank[0], default)


def test_primary_target_candidates_are_alpha_sorted():
    assert PHASE_REWARDS_PRIMARY_TARGET_NAMES == tuple(sorted(PHASE_REWARDS_PRIMARY_TARGET_NAMES))


def _count_high_value_rows(matrix: np.ndarray, phase: int) -> int:
    """Return the number of subnets whose phase-`phase` row is exactly [-10,0,-10]."""
    return int(np.sum(np.all(matrix[phase] == HIGH_VALUE_ROW, axis=1)))


def test_each_variant_has_one_high_value_target_per_active_phase():
    bank = get_phase_rewards_bank()
    for i in range(bank.shape[0]):
        assert _count_high_value_rows(bank[i], 1) == 1, f"bank[{i}] phase1 mismatch"
        assert _count_high_value_rows(bank[i], 2) == 1, f"bank[{i}] phase2 mismatch"
        # Phase 0 (preplanning) never has a high-value target.
        assert _count_high_value_rows(bank[i], 0) == 0, f"bank[{i}] phase0 leak"


def test_variant_targets_differ_in_each_entry():
    """For ordered-pair entries 1..30, the two phase targets must differ."""
    bank = get_phase_rewards_bank()
    for i in range(1, bank.shape[0]):
        p1_targets = np.where(np.all(bank[i, 1] == HIGH_VALUE_ROW, axis=1))[0]
        p2_targets = np.where(np.all(bank[i, 2] == HIGH_VALUE_ROW, axis=1))[0]
        assert len(p1_targets) == 1 and len(p2_targets) == 1
        assert int(p1_targets[0]) != int(p2_targets[0]), f"bank[{i}] has equal targets"


def test_build_phase_rewards_variant_default_pair_matches_default():
    """Passing the default op-zones reproduces the default exactly."""
    from jaxborg.constants import SUBNET_IDS

    pr = _build_phase_rewards_variant(SUBNET_IDS["OPERATIONAL_ZONE_A"], SUBNET_IDS["OPERATIONAL_ZONE_B"])
    assert np.array_equal(pr, _build_phase_rewards())


def test_default_path_phase_rewards_unchanged():
    """vary_phase_rewards=False reproduces the legacy phase_rewards exactly."""
    default = np.array(_build_phase_rewards())
    for seed in [0, 1, 7, 42, 12345]:
        c = build_topology(jax.random.PRNGKey(seed), vary_phase_rewards=False)
        assert np.array_equal(np.array(c.phase_rewards), default), f"seed={seed}"


def test_vary_phase_rewards_produces_distinct_matrices():
    sums = set()
    for seed in range(64):
        c = build_topology(jax.random.PRNGKey(seed), vary_phase_rewards=True)
        sums.add(float(jnp.asarray(c.phase_rewards).sum()))
    # 31-entry bank; with 64 keys we expect several distinct entries (variants
    # mostly differ in totals because targets sit on rows with different
    # default values).
    assert len(sums) >= 5, f"only {len(sums)} distinct phase_reward sums in 64 keys"


def test_vary_phase_rewards_outputs_are_in_bank():
    bank = np.asarray(get_phase_rewards_bank())
    for seed in range(32):
        c = build_topology(jax.random.PRNGKey(seed), vary_phase_rewards=True)
        pr = np.asarray(c.phase_rewards)
        match = np.any(np.all(bank == pr[None, ...], axis=(1, 2, 3)))
        assert match, f"seed={seed} phase_rewards not found in bank"


def test_fsm_red_env_vary_phase_rewards_end_to_end():
    """Env-level smoke test: FsmRedCC4Env resets cleanly and phase_rewards in bank."""
    from jaxborg.fsm_red_env import FsmRedCC4Env

    env = FsmRedCC4Env(num_steps=10, topology_mode="generative", vary_phase_rewards=True)
    bank = np.asarray(get_phase_rewards_bank())
    seen_sums = set()
    for seed in [0, 1, 7, 13]:
        _obs, env_state = env.reset(jax.random.PRNGKey(seed))
        pr = np.asarray(env_state.const.phase_rewards)
        assert pr.shape == (MISSION_PHASES, NUM_SUBNETS, 3)
        match = np.any(np.all(bank == pr[None, ...], axis=(1, 2, 3)))
        assert match, f"seed={seed} env phase_rewards not in bank"
        seen_sums.add(float(pr.sum()))
    # Different seeds should typically pick different entries.
    assert len(seen_sums) >= 2


def test_fsm_red_env_default_phase_rewards_match_legacy():
    """Without vary_phase_rewards, env state's phase_rewards equals legacy."""
    from jaxborg.fsm_red_env import FsmRedCC4Env

    default = np.array(_build_phase_rewards())
    env = FsmRedCC4Env(num_steps=10, topology_mode="generative")
    for seed in [0, 1, 99]:
        _obs, env_state = env.reset(jax.random.PRNGKey(seed))
        assert np.array_equal(np.array(env_state.const.phase_rewards), default)
