"""Observation hash fingerprinting.

At every step, hash the full blue observation vector from both CybORG and JAX.
Compares hash sequences across episodes. Any hash mismatch reveals an
observation encoding gap that state-level comparison can't see.

Also tests action mask parity at every step.
"""

import hashlib

import numpy as np
import pytest

from tests.differential.harness import CC4DifferentialHarness


def _obs_hash(obs_array):
    """Deterministic hash of a float32 observation vector."""
    return hashlib.sha256(np.asarray(obs_array, dtype=np.float32).tobytes()).hexdigest()[:16]


class TestObservationFingerprinting:
    """Compare observation hashes between CybORG and JAX at every step."""

    @pytest.mark.parametrize("seed", range(5))
    def test_obs_hashes_match(self, seed):
        harness = CC4DifferentialHarness(
            seed=seed,
            max_steps=50,
            sync_green_rng=True,
            check_obs=True,
            check_masks=True,
            strip_inactive_knowledge=True,
        )
        harness.reset()

        obs_mismatches = []
        for t in range(50):
            result = harness.full_step()
            obs_diffs = [d for d in result.diffs if d.field_name == "observation"]
            if obs_diffs:
                for d in obs_diffs:
                    obs_mismatches.append(f"step {t}: {d.host_or_agent}")

        if obs_mismatches:
            msg = f"Observation hash mismatches in {len(obs_mismatches)} step/agents:\n"
            msg += "\n".join(f"  {m}" for m in obs_mismatches[:20])
            if len(obs_mismatches) > 20:
                msg += f"\n  ... and {len(obs_mismatches) - 20} more"
            pytest.fail(msg)


class TestActionMaskParity:
    """Compare blue action masks between CybORG and JAX at every step."""

    @pytest.mark.parametrize("seed", range(5))
    def test_masks_match(self, seed):
        harness = CC4DifferentialHarness(
            seed=seed,
            max_steps=50,
            sync_green_rng=True,
            check_obs=False,
            check_masks=True,
            strip_inactive_knowledge=True,
        )
        harness.reset()

        mask_mismatches = []
        for t in range(50):
            result = harness.full_step()
            mask_diffs = [d for d in result.diffs if d.field_name == "action_mask"]
            if mask_diffs:
                for d in mask_diffs:
                    mask_mismatches.append(f"step {t}: {d.host_or_agent}")

        if mask_mismatches:
            msg = f"Action mask mismatches in {len(mask_mismatches)} step/agents:\n"
            msg += "\n".join(f"  {m}" for m in mask_mismatches[:20])
            if len(mask_mismatches) > 20:
                msg += f"\n  ... and {len(mask_mismatches) - 20} more"
            pytest.fail(msg)


class TestObservationDeterminism:
    """Same seed, same actions → same observation sequence."""

    def test_deterministic_obs_across_runs(self):
        def _collect_obs_hashes(seed=42, steps=20):
            harness = CC4DifferentialHarness(
                seed=seed, max_steps=steps, sync_green_rng=True, strip_inactive_knowledge=True
            )
            harness.reset()
            hashes = []
            for _ in range(steps):
                harness.full_step()
                from jaxborg.observations import get_blue_obs

                obs = get_blue_obs(harness.jax_state, harness.jax_const, 0)
                hashes.append(_obs_hash(obs))
            return hashes

        run1 = _collect_obs_hashes()
        run2 = _collect_obs_hashes()
        assert run1 == run2, "Observation sequence not deterministic across runs with same seed"
