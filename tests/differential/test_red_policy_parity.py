"""Red-policy parity regression: JAX's `fsm_red_select_actions` must match
CybORG's `FiniteStateRedAgent.get_action` given matched RNG.

Uses `CC4DifferentialHarness(record_red_policy=True)`: the harness pre-calls
CybORG's red `get_action` with np_random save/restore so the shared RNG stream
isn't disturbed; `RedPolicyRecorder` maps CybORG's host/action/subnet choices
into JAX index space, and `IndexedRNGTape` replays those indices through
`sample_red_policy_choice`. Per-step JAX picks are compared against CybORG's
pre-captured picks.

Also checks symmetric eligibility (the set of red agents the harness considers
"eligible to act" matches CybORG's `host_states` keys minus FSM_F).

Any mismatch → env-level parity bug in red-policy action selection.
"""

from __future__ import annotations

import pytest

from jaxborg.actions.encoding import BLUE_SLEEP
from jaxborg.constants import NUM_BLUE_AGENTS
from tests.differential.harness import CC4DifferentialHarness

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("seed", [1000, 1001, 1002, 1003, 1004])
def test_red_policy_matches_cyborg_multistep(seed: int) -> None:
    """JAX red picks identical to CybORG red picks across 200 steps under matched RNG."""
    harness = CC4DifferentialHarness(
        seed=seed,
        max_steps=200,
        sync_green_rng=True,
        strict_random_sync=False,
        check_rewards=False,
        check_obs=False,
        check_masks=False,
        record_red_policy=True,
    )
    harness.reset()

    sleep_blue = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}
    for _ in range(200):
        harness.full_step(sleep_blue)

    if harness.red_policy_mismatches:
        lines = [
            (
                f"  step {m['step']}: red_{m['red_agent']} "
                f"cyborg=({m['cyborg_action']} fsm={m['cyborg_fsm']} host={m['cyborg_host_idx']} "
                f"subnet={m['cyborg_subnet_idx']}) "
                f"jax=(fsm={m['jax_fsm']} host={m['jax_host_idx']} subnet={m['jax_subnet_idx']} "
                f"eligible={m['jax_eligible']})"
            )
            for m in harness.red_policy_mismatches[:10]
        ]
        pytest.fail(
            f"seed={seed}: {len(harness.red_policy_mismatches)} red-policy pick "
            f"mismatches across {harness.red_policy_compared} comparisons\n" + "\n".join(lines)
        )

    if harness.red_eligibility_mismatches:
        lines = [
            f"  step {m['step']}: cyborg_eligible={m['cyborg_eligible']} jax_eligible={m['jax_eligible']}"
            for m in harness.red_eligibility_mismatches[:10]
        ]
        pytest.fail(
            f"seed={seed}: {len(harness.red_eligibility_mismatches)} eligibility "
            f"mismatches across {harness.red_eligibility_compared} checks\n" + "\n".join(lines)
        )

    assert harness.red_policy_compared > 0, (
        f"seed={seed}: no red-policy comparisons were made; harness wiring regressed"
    )
