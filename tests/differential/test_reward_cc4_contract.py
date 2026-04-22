"""Per-step parity of the full CC4 reward contract (BRM + action_cost).

CybORG sums `actions.get(agent, Action()).cost` over the caller-submitted
action dict every step (`SimulationController._step:310`), so a policy that
re-submits Restore during the 4 follow-up busy ticks is charged -1 each
step. The headline CC4 scorer (`BlueFixedActionWrapper.step:171-175`,
`Evaluation/evaluation.py`) inherits this via `sum(reward.values())`.

JAX must mirror the same caller-submission accounting in
`compute_reward_breakdown` so headline rewards match.
"""

import jax.numpy as jnp
import numpy as np
from CybORG.Agents import SleepAgent

from jaxborg.actions.encoding import BLUE_SLEEP, encode_blue_action
from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.rewards import compute_reward_breakdown
from tests.differential.harness import CC4DifferentialHarness


def test_action_cost_matches_cc4_contract_during_restore_busy_ticks():
    """JAX action_cost must match CybORG's per-submission charge across busy ticks.

    Restore.duration == 5: step 0 initiates, steps 1-4 are busy ticks. A
    policy that re-submits Restore each step incurs -1 per step in CybORG
    via `controller.step(actions)`'s caller-submission sum. JAX with the
    `is_initiating` gate only charges -1 at step 0, leaving a +4/Restore
    lenience that compounds across an episode.
    """
    target_hostname = "restricted_zone_a_subnet_user_host_0"

    harness = CC4DifferentialHarness(
        seed=0,
        max_steps=20,
        blue_cls=SleepAgent,
        green_cls=SleepAgent,
        red_cls=SleepAgent,
        sync_green_rng=False,
        use_cyborg_blue_policy=False,
    )
    harness.reset()
    controller = harness.cyborg_env.environment_controller

    host_idx = harness.mappings.hostname_to_idx[target_hostname]
    restore_action_idx = int(encode_blue_action("Restore", host_idx, 0, const=harness.jax_const))
    assert restore_action_idx != BLUE_SLEEP, f"Restore for {target_hostname} did not encode for blue_agent_0"

    # Re-submit Restore on agent 0 every step — the harness forwards this to
    # `controller.step(cyborg_actions)`, mirroring `BlueFixedActionWrapper.step`.
    blue_actions = {0: restore_action_idx}
    for b in range(1, NUM_BLUE_AGENTS):
        blue_actions[b] = BLUE_SLEEP

    per_step_jax_ac = []
    per_step_cy_ac = []

    for step in range(5):
        harness.full_step(blue_actions=blue_actions)

        cyborg_action_cost = float(controller.reward["Blue"]["action_cost"])

        # Reconstruct JAX-side reward under the CC4 contract: pass the
        # caller-submitted action — NOT the busy-masked action — to mirror
        # CybORG's `actions.get(agent, Action()).cost` sum.
        blue_actions_submitted = jnp.zeros(NUM_BLUE_AGENTS, dtype=jnp.int32)
        blue_actions_submitted = blue_actions_submitted.at[0].set(restore_action_idx)
        breakdown = compute_reward_breakdown(
            harness.jax_state,
            harness.jax_const,
            harness.jax_state.red_impact_attempted,
            harness.jax_state.green_lwf_this_step,
            harness.jax_state.green_asf_this_step,
            blue_actions=blue_actions_submitted,
        )
        jax_action_cost = float(breakdown.action_cost)

        per_step_jax_ac.append(jax_action_cost)
        per_step_cy_ac.append(cyborg_action_cost)

    # Sanity: CybORG charges -1 every step the caller submits Restore.
    np.testing.assert_array_equal(
        per_step_cy_ac,
        [-1.0, -1.0, -1.0, -1.0, -1.0],
        err_msg=f"CybORG action_cost (sanity check): {per_step_cy_ac}",
    )
    # JAX must match per-step under the CC4 contract.
    np.testing.assert_array_equal(
        per_step_jax_ac,
        per_step_cy_ac,
        err_msg=(
            f"JAX action_cost diverges from CC4 contract on busy ticks. "
            f"JAX={per_step_jax_ac}, CybORG={per_step_cy_ac}. "
            f"Fix `is_initiating` gate in src/jaxborg/rewards.py."
        ),
    )
