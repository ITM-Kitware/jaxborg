"""Matched-RNG per-step reward parity regression over long episodes.

Under `sync_green_rng=True`, JAX consumes CybORG's captured np_random stream
each step, so both backends should compute identical per-step rewards. This
test runs a multi-seed, multi-hundred-step sweep asserting zero cumulative
reward diff — the regression gate for the "matched-RNG → byte-perfect
reward" claim from A3 / A1-fup² that underpins every downstream parity
conclusion.

Shorter-horizon versions of this are implicit in `test_random_sync.py` and
`test_fsm_parity.py` (which run 20-30 steps); this one extends it to 200
steps × 5 seeds to catch regressions that only surface under longer
compounding.
"""

from __future__ import annotations

import pytest

from jaxborg.actions.encoding import BLUE_SLEEP
from jaxborg.constants import NUM_BLUE_AGENTS
from tests.differential.harness import CC4DifferentialHarness

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("seed", [1000, 1001, 1002, 1003, 1004])
def test_matched_rng_per_step_reward_matches_cyborg(seed: int) -> None:
    """Under matched RNG, per-step reward diff is zero across 200 steps of sleep blue."""
    harness = CC4DifferentialHarness(
        seed=seed,
        max_steps=200,
        sync_green_rng=True,
        strict_random_sync=False,
        check_rewards=True,
        check_obs=False,
        check_masks=False,
    )
    harness.reset()

    sleep_blue = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}
    per_step_jax = 0.0
    per_step_cy = 0.0
    n_per_step_diffs = 0
    worst_abs_diff = 0.0

    for _ in range(200):
        result = harness.full_step(sleep_blue)
        jr = float(result.jax_rewards["total"])
        cr = float(result.cyborg_rewards["total"])
        per_step_jax += jr
        per_step_cy += cr
        diff = abs(jr - cr)
        if diff > 1e-6:
            n_per_step_diffs += 1
            worst_abs_diff = max(worst_abs_diff, diff)

    assert n_per_step_diffs == 0, (
        f"seed={seed}: {n_per_step_diffs} per-step reward diffs across 200 steps "
        f"(worst |Δ|={worst_abs_diff:.2e}). Matched-RNG byte parity regressed."
    )
    assert abs(per_step_jax - per_step_cy) < 1e-6, (
        f"seed={seed}: cumulative reward diff {per_step_jax - per_step_cy:+.4f} "
        f"(JAX={per_step_jax:.2f}, CybORG={per_step_cy:.2f})"
    )
