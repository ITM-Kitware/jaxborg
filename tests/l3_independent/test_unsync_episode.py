"""L3-independent tests: full episode rollouts with Category A syncs removed.

The differential harness on this branch has had all deterministic outcome
syncs (Category A) permanently removed:
- red_impact_attempted — JAX computes from its own state
- green_lwf/asf_this_step — JAX computes from its own green logic
- forced_primary_hosts/pids — JAX tracks session identity internally
- red_abstract_host_rank — JAX tracks abstract ranks internally

Category B syncs (RNG replay: green randoms, detection, privesc/session-check
choices) remain active — these are deliberate test infrastructure for
synchronizing CybORG's np_random with JAX's jax.random.

These tests run the same harness as the existing L3 synced tests, but now
the harness itself no longer papers over deterministic logic gaps.
"""

import pytest
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent

from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import _ERROR_FIELDS, format_diffs

L3I_SEEDS = range(20)
L3I_STEPS = 500


def _run_episode(seed, max_steps):
    """Run a single episode and fail on first ERROR-level mismatch."""
    harness = CC4DifferentialHarness(
        seed=seed,
        max_steps=max_steps,
        blue_cls=SleepAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        sync_green_rng=True,
        strict_random_sync=False,
    )
    harness.reset()

    for t in range(max_steps):
        result = harness.full_step()

        error_diffs = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]
        if error_diffs:
            d = error_diffs[0]
            detail = format_diffs(result.diffs)
            pytest.fail(
                f"Mismatch at seed={seed}, step={t}: "
                f"{d.field_name} [{d.host_or_agent}] "
                f"cyborg={d.cyborg_value} jax={d.jax_value}\n"
                f"All diffs:\n{detail}"
            )


class TestL3Independent:
    """Full episode rollout with no Category A syncs.

    If these fail, the Karten loop agent should:
    1. Identify which field diverged and at which step
    2. Write a targeted L1/L2 regression test
    3. Fix the JAX logic
    """

    @pytest.mark.parametrize("seed", L3I_SEEDS)
    def test_episode(self, seed):
        _run_episode(seed=seed, max_steps=L3I_STEPS)
