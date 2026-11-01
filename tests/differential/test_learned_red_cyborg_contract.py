from __future__ import annotations

import numpy as np

from jaxborg.cyborg_joint import RED_AGENT_IDS, CyborgJointAdapter
from jaxborg.learned_red import compute_red_policy_action_mask, get_red_policy_obs
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
from tests.differential.harness import CC4DifferentialHarness


def test_cyborg_red_policy_projection_matches_jax_at_reset():
    harness = CC4DifferentialHarness(
        seed=42,
        max_steps=500,
        check_rewards=False,
        check_obs=True,
        check_masks=True,
        sync_green_rng=False,
    )
    harness.reset()

    # Project the harness's matched CybORG state through the production joint
    # adapter without constructing a second, unrelated random topology.
    adapter = object.__new__(CyborgJointAdapter)
    adapter.variant = CC4_STOCK
    adapter.raw_env = harness.cyborg_env
    adapter.blue_wrapper = harness._blue_wrapper
    adapter.mappings = harness.mappings
    adapter._discovered = [set() for _ in RED_AGENT_IDS]
    adapter._scanned_by_primary = [set() for _ in RED_AGENT_IDS]
    adapter._primary_identity = [None for _ in RED_AGENT_IDS]
    red_observations = {agent: adapter.raw_env.get_observation(agent) for agent in RED_AGENT_IDS}
    adapter._update_discovery_memory(red_observations)
    adapter._sync_primary_identities()

    for agent_idx in range(len(RED_AGENT_IDS)):
        cyborg_obs = adapter.red_observation(agent_idx)
        jax_obs = np.asarray(get_red_policy_obs(harness.jax_state, harness.jax_const, agent_idx))
        np.testing.assert_array_equal(cyborg_obs, jax_obs)

        cyborg_mask = adapter.red_action_mask(agent_idx).astype(bool)
        jax_mask = np.asarray(compute_red_policy_action_mask(harness.jax_state, harness.jax_const, agent_idx))
        np.testing.assert_array_equal(cyborg_mask, jax_mask)
