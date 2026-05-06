"""Matched-state transfer diagnostics for debugging parity."""

# ruff: noqa: E402,I001

from __future__ import annotations

from scripts.dev.parity.bootstrap import configure_runtime

configure_runtime()

import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import COMPROMISE_PRIVILEGED, COMPROMISE_USER, NUM_BLUE_AGENTS, NUM_RED_AGENTS
from jaxborg.observations import get_blue_obs
from tests.differential.harness import CC4DifferentialHarness
from scripts.dev.parity.cyborg_bridge import _build_cyborg_mask_cache, _live_blue_wrapper_mask_in_jax_space_cached
from scripts.dev.parity.jax_rollout import _all_blue_masks
from scripts.dev.parity.policy import make_batched_inference_fn
from scripts.dev.parity.rollout_types import EpisodeResult, StepSnapshot, TransferComparison


def rollout_matched_transfer(policy, params, num_episodes=3, deterministic=False, seed=0):
    """Compare policy outputs on matched JAX/CybORG states.

    JAX-selected actions drive the synced rollout so the underlying episode stays
    matched. CybORG-selected actions are recorded from the same synced states for
    transfer diagnostics, not applied.

    Optimized: batched policy inference across agents.
    """

    batched_step = make_batched_inference_fn(policy, params, deterministic=deterministic)
    rng = jax.random.PRNGKey(seed + 9999)  # separate stream for action sampling

    all_jax_actions = []
    all_cyborg_actions = []
    episode_rewards = []
    episode_results = []
    all_cyborg_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        harness = CC4DifferentialHarness(seed=seed + ep * 100, check_obs=True, sync_green_rng=True)
        harness.reset()

        cyborg_agent_names = [f"blue_agent_{i}" for i in range(NUM_BLUE_AGENTS)]
        mask_cache = _build_cyborg_mask_cache(harness._blue_wrapper, harness.mappings, harness.jax_const)
        ep_jax_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_cyborg_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_step_rewards = []
        ep_trajectory = []
        cum_reward = 0.0

        for _ in range(500):
            # --- JAX side: batched obs + masks + policy ---
            jax_obs_stack = jnp.stack(
                [get_blue_obs(harness.jax_state, harness.jax_const, i) for i in range(NUM_BLUE_AGENTS)]
            )
            jax_masks = _all_blue_masks(harness.jax_const, harness.jax_state)
            if deterministic:
                step_keys = jnp.zeros((NUM_BLUE_AGENTS, 2), dtype=jnp.uint32)
            else:
                rng, _sub = jax.random.split(rng)
                step_keys = jax.random.split(_sub, NUM_BLUE_AGENTS)
            jax_actions_arr, _ = batched_step(jax_obs_stack, jax_masks, step_keys)

            # --- CybORG side: cached mask translation + training-time traffic filter ---
            cyborg_obs_list = []
            cyborg_mask_list = []
            for i, name in enumerate(cyborg_agent_names):
                cyborg_obs_dict = harness.cyborg_env.get_observation(name)
                cyborg_obs_list.append(
                    jnp.array(harness._blue_wrapper.observation_change(name, cyborg_obs_dict), dtype=jnp.float32)
                )
                raw_mask = _live_blue_wrapper_mask_in_jax_space_cached(
                    harness._blue_wrapper, name, harness.mappings, harness.jax_const, mask_cache
                )
                cyborg_mask_list.append(jnp.array(raw_mask))
            cyborg_obs_stack = jnp.stack(cyborg_obs_list)
            cyborg_masks = jnp.stack(cyborg_mask_list)
            cyborg_actions_arr, _ = batched_step(cyborg_obs_stack, cyborg_masks, step_keys)

            # Single device-to-host sync
            jax_actions_np = np.asarray(jax_actions_arr)
            cyborg_actions_np = np.asarray(cyborg_actions_arr)

            jax_actions = {}
            for i in range(NUM_BLUE_AGENTS):
                jax_act = int(jax_actions_np[i])
                cyborg_act = int(cyborg_actions_np[i])
                jax_actions[i] = jax_act
                ep_jax_actions_by_agent[i].append(jax_act)
                ep_cyborg_actions_by_agent[i].append(cyborg_act)
                all_cyborg_actions_by_agent[i].append(cyborg_act)

            result = harness.full_step(blue_actions=jax_actions)
            if result.diffs:
                details = ", ".join(f"{d.field_name}:{d.host_or_agent}" for d in result.diffs[:5])
                raise RuntimeError(f"Matched transfer replay diverged at step {result.step}: {details}")

            step_reward = float(
                harness.cyborg_env.environment_controller.reward.get("Blue", {}).get("BlueRewardMachine", 0.0)
            )
            cum_reward += step_reward
            ep_step_rewards.append(step_reward)

            st = harness.jax_state
            active = np.array(harness.jax_const.host_active, dtype=bool)
            compromised = np.array(st.host_compromised)
            ep_trajectory.append(
                StepSnapshot(
                    reward=step_reward,
                    cumulative_reward=cum_reward,
                    hosts_compromised_user=int(np.sum((compromised == COMPROMISE_USER) & active)),
                    hosts_compromised_priv=int(np.sum((compromised == COMPROMISE_PRIVILEGED) & active)),
                    red_sessions_total=int(np.sum(np.array(st.red_sessions)[:NUM_RED_AGENTS])),
                    mission_phase=int(st.mission_phase),
                )
            )

        elapsed = time.perf_counter() - t0
        print(f"  Matched ep {ep + 1}: reward={cum_reward:.1f} ({elapsed:.1f}s)")

        flat_jax_actions = [a for step_actions in zip(*ep_jax_actions_by_agent) for a in step_actions]
        flat_cyborg_actions = [a for step_actions in zip(*ep_cyborg_actions_by_agent) for a in step_actions]
        all_jax_actions.extend(flat_jax_actions)
        all_cyborg_actions.extend(flat_cyborg_actions)
        episode_rewards.append(cum_reward)
        episode_results.append(
            EpisodeResult(
                actions_by_agent=ep_jax_actions_by_agent,
                rewards=ep_step_rewards,
                cumulative_reward=cum_reward,
                trajectory=ep_trajectory,
            )
        )

    rewards = np.array(episode_rewards)
    return TransferComparison(
        jax_actions=np.array(all_jax_actions),
        jax_rewards=rewards,
        jax_episodes=episode_results,
        cyborg_actions=np.array(all_cyborg_actions),
        cyborg_rewards=rewards.copy(),
        cyborg_actions_by_agent=all_cyborg_actions_by_agent,
    )
