"""Phase 6 stream S2 — per-reset mission-profile multiplier bank.

Tests the ``mission_bank`` / ``mission_bank_amplify`` plumbing on
:class:`jaxborg.env.ScenarioEnv`.  Per the Phase 6 plan (axis B) the bank is
sampled at reset and post-multiplies ``const.phase_rewards`` — there is no
``state.mission_multipliers`` field, and ``rewards.py`` is unchanged.

Asserts:

1. **Determinism** — same PRNG key → same multiplier triple sampled.
2. **Uniformity** — across 10000 keys each of the 4 default-bank entries is
   sampled within 5% of uniform (chi-square at α=0.01).
3. **Single-entry bank applies the triple** — with ``mission_bank=[(1, 3, 1)]``
   the resulting ``const.phase_rewards`` equals 3× the baseline ASF channel
   exactly (rewards.py would then produce 3× ASF reward at any step where ASF
   triggers).
4. **Amplify scales the entire triple** — ``mission_bank_amplify=10.0`` with
   bank ``[(1, 3, 1)]`` yields a (10, 30, 10) effective multiplier, not
   (1, 30, 1).  Documented this way for simplicity (no special-casing).
5. **Empty / None bank is the legacy fast path** — ``const.phase_rewards``
   is byte-identical to the baseline (no bank configured).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.env import ScenarioEnv
from jaxborg.scenarios.cc4.topology_numpy import (
    MISSION_PROFILE_MULTIPLIERS,
    NUM_MISSION_PROFILES,
    get_mission_profile_multipliers,
)

# Use a short num_steps so reset/topology build is cheap.
NUM_STEPS = 10


def _make_env(mission_bank=None, mission_bank_amplify=1.0):
    return ScenarioEnv(
        num_steps=NUM_STEPS,
        mission_bank=mission_bank,
        mission_bank_amplify=mission_bank_amplify,
    )


def _sample_const(env, key):
    return env._select_const(key)


# ---------------------------------------------------------------------------
# Default bank metadata


class TestMissionProfileTable:
    def test_default_bank_has_four_entries(self):
        assert NUM_MISSION_PROFILES == 4
        assert len(MISSION_PROFILE_MULTIPLIERS) == 4

    def test_default_bank_first_entry_is_unit(self):
        assert MISSION_PROFILE_MULTIPLIERS[0] == (1.0, 1.0, 1.0)

    def test_get_mission_profile_multipliers_shape(self):
        arr = get_mission_profile_multipliers()
        assert arr.shape == (NUM_MISSION_PROFILES, 3)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr[0], np.array([1.0, 1.0, 1.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# Determinism + uniformity of the sampled index


class TestSamplingDeterminism:
    def test_same_key_same_multiplier(self):
        bank = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]
        env = _make_env(mission_bank=bank)
        key = jax.random.PRNGKey(7)
        c1 = _sample_const(env, key)
        c2 = _sample_const(env, key)
        np.testing.assert_array_equal(np.asarray(c1.phase_rewards), np.asarray(c2.phase_rewards))

    def test_different_keys_can_yield_different_multipliers(self):
        bank = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]
        env = _make_env(mission_bank=bank)

        # Build a baseline (no-bank) const so we can recover the per-key
        # multiplier triple by dividing through the LWF/ASF/RIA channels at any
        # nonzero (phase, subnet) cell.
        env_baseline = _make_env(mission_bank=None)

        seen = set()
        for s in range(20):
            key = jax.random.PRNGKey(s)
            c_bank = _sample_const(env, key)
            c_base = _sample_const(env_baseline, key)
            triple = _recover_multiplier(c_base.phase_rewards, c_bank.phase_rewards)
            seen.add(tuple(round(float(x), 4) for x in triple))
        # With 4 entries × 20 draws we expect to hit at least 2 distinct entries.
        assert len(seen) >= 2


def _recover_multiplier(baseline_pr: jnp.ndarray, bank_pr: jnp.ndarray) -> tuple[float, float, float]:
    """Recover the (LWF, ASF, RIA) triple applied to baseline_pr.

    For each component we pick a (phase, subnet) cell where the baseline weight
    is nonzero, then ratio.  Returns floats.
    """
    base = np.asarray(baseline_pr)
    bank = np.asarray(bank_pr)
    triple = []
    for c in range(3):
        base_c = base[..., c]
        bank_c = bank[..., c]
        nz = base_c != 0.0
        if not nz.any():
            triple.append(1.0)
            continue
        ratios = bank_c[nz] / base_c[nz]
        # All ratios for a given component must be identical (we multiply by
        # a scalar) — assert that for safety, then take the first.
        assert np.allclose(ratios, ratios[0], atol=1e-5), f"non-uniform ratio for component {c}: {ratios}"
        triple.append(float(ratios[0]))
    return tuple(triple)  # type: ignore[return-value]


class TestSamplingUniformity:
    """Across 10k keys each of 4 default-bank entries sampled within 5% of uniform."""

    def test_uniform_chisquare(self):
        bank = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]
        # Build env to exercise the construction path (validates bank shape).
        _make_env(mission_bank=bank)
        # Sampling the *index* directly is much cheaper than building the whole
        # const for each key; mirror the env's exact per-key derivation.
        n = 10_000

        @jax.jit
        def sample_idx(key):
            _, key_mission = jax.random.split(key)
            return jax.random.randint(key_mission, (), 0, len(bank))

        keys = jax.random.split(jax.random.PRNGKey(0), n)
        idxs = jax.vmap(sample_idx)(keys)
        idxs_np = np.asarray(idxs)
        counts = np.bincount(idxs_np, minlength=len(bank))
        expected = n / len(bank)

        # 5% relative tolerance per entry.
        rel_err = np.abs(counts - expected) / expected
        assert (rel_err < 0.05).all(), f"counts={counts.tolist()} expected={expected}, rel_err={rel_err.tolist()}"

        # Chi-square at α=0.01 — 3 dof, critical value = 11.345.
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        assert chi2 < 11.345, f"chi2={chi2} ≥ 11.345 (4-bin uniform, α=0.01)"


# ---------------------------------------------------------------------------
# phase_rewards multiplication semantics


class TestPhaseRewardsScaling:
    """Verify that the bank entry actually scales const.phase_rewards.

    rewards.py reads ``const.phase_rewards[phase, subnet, channel]`` directly,
    so scaling a channel here is equivalent to scaling that channel's per-step
    contribution to the total reward (proportional to the number of triggered
    hosts on that subnet, but the *per-host weight* scales linearly).
    """

    def test_single_entry_bank_scales_asf_channel(self):
        """bank=[(1, 3, 1)] → ASF channel exactly 3× baseline; LWF & RIA unchanged."""
        env_base = _make_env(mission_bank=None)
        env_bank = _make_env(mission_bank=[[1.0, 3.0, 1.0]])

        key = jax.random.PRNGKey(123)
        c_base = _sample_const(env_base, key)
        c_bank = _sample_const(env_bank, key)

        base_pr = np.asarray(c_base.phase_rewards)
        bank_pr = np.asarray(c_bank.phase_rewards)

        np.testing.assert_allclose(bank_pr[..., 0], base_pr[..., 0] * 1.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(bank_pr[..., 1], base_pr[..., 1] * 3.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(bank_pr[..., 2], base_pr[..., 2] * 1.0, rtol=0, atol=1e-6)

    def test_amplify_multiplies_entire_triple(self):
        """``mission_bank_amplify=10`` × bank ``[(1, 3, 1)]`` → effective (10, 30, 10).

        Documented behavior: amplify scales the entire sampled triple element-wise.
        amplify=10 with (1, 3, 1) yields (10, 30, 10), NOT (1, 30, 1).
        """
        env_base = _make_env(mission_bank=None)
        env_bank = _make_env(mission_bank=[[1.0, 3.0, 1.0]], mission_bank_amplify=10.0)

        key = jax.random.PRNGKey(99)
        c_base = _sample_const(env_base, key)
        c_bank = _sample_const(env_bank, key)

        base_pr = np.asarray(c_base.phase_rewards)
        bank_pr = np.asarray(c_bank.phase_rewards)

        np.testing.assert_allclose(bank_pr[..., 0], base_pr[..., 0] * 10.0, rtol=0, atol=1e-5)
        np.testing.assert_allclose(bank_pr[..., 1], base_pr[..., 1] * 30.0, rtol=0, atol=1e-5)
        np.testing.assert_allclose(bank_pr[..., 2], base_pr[..., 2] * 10.0, rtol=0, atol=1e-5)


class TestEmptyBankIsLegacyFastPath:
    """When ``mission_bank`` is None or empty, ``const.phase_rewards`` is unchanged."""

    @pytest.mark.parametrize("bank", [None, []])
    def test_no_bank_no_change(self, bank):
        env_base = _make_env(mission_bank=None)
        env_other = _make_env(mission_bank=bank)
        # Internal flags should reflect "disabled."
        assert env_other._mission_bank is None
        assert env_other._mission_bank_size == 0

        key = jax.random.PRNGKey(2026)
        c_base = _sample_const(env_base, key)
        c_other = _sample_const(env_other, key)

        np.testing.assert_array_equal(np.asarray(c_base.phase_rewards), np.asarray(c_other.phase_rewards))

    def test_amplify_alone_without_bank_is_noop(self):
        """``mission_bank=None`` ignores ``mission_bank_amplify`` (no scaling)."""
        env_base = _make_env(mission_bank=None)
        env_amp = _make_env(mission_bank=None, mission_bank_amplify=10.0)

        key = jax.random.PRNGKey(2027)
        c_base = _sample_const(env_base, key)
        c_amp = _sample_const(env_amp, key)

        np.testing.assert_array_equal(np.asarray(c_base.phase_rewards), np.asarray(c_amp.phase_rewards))


# ---------------------------------------------------------------------------
# Construction validation


class TestBankShapeValidation:
    def test_rejects_non_triple_entries(self):
        with pytest.raises(ValueError, match="3-tuples"):
            _make_env(mission_bank=[[1.0, 1.0]])

    def test_accepts_default_4_entry_bank(self):
        bank = [list(t) for t in MISSION_PROFILE_MULTIPLIERS]
        env = _make_env(mission_bank=bank)
        assert env._mission_bank_size == 4
        assert env._mission_bank.shape == (4, 3)


# ---------------------------------------------------------------------------
# Recipe projection wiring


class TestRecipeProjection:
    """``project_jax`` reads ``train.mission_bank`` and ``train.mission_bank_amplify``."""

    def _recipe(self, **train_extra):
        train = {
            "variant": "cc4_stock",
            "episode_length": 10,
            "total_timesteps": 100,
        }
        train.update(train_extra)
        return {
            "meta": {"name": "test"},
            "algorithm": {"name": "ippo_jax"},
            "core": {"lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95},
            "arch": {"name": "mlp"},
            "train": train,
        }

    def test_default_no_bank(self):
        from jaxborg.recipe import project_jax

        cfg = project_jax(self._recipe())
        assert cfg["MISSION_BANK"] is None
        assert cfg["MISSION_BANK_AMPLIFY"] == 1.0

    def test_projects_bank_and_amplify(self):
        from jaxborg.recipe import project_jax

        cfg = project_jax(
            self._recipe(
                mission_bank=[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
                mission_bank_amplify=10.0,
            )
        )
        assert cfg["MISSION_BANK"] == [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]
        assert cfg["MISSION_BANK_AMPLIFY"] == 10.0

    def test_empty_bank_projects_to_none(self):
        from jaxborg.recipe import project_jax

        cfg = project_jax(self._recipe(mission_bank=[]))
        assert cfg["MISSION_BANK"] is None

    def test_bad_triple_rejected(self):
        from jaxborg.recipe import project_jax

        with pytest.raises(ValueError, match="LWF, ASF, RIA"):
            project_jax(self._recipe(mission_bank=[[1.0, 2.0]]))
