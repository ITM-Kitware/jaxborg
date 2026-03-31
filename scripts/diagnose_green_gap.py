"""Diagnose green agent reward divergence between JAX and CybORG.

Runs forced-same-actions and compares per-step LWF/ASF/RIA reward components.

Usage:
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
    uv run python scripts/diagnose_green_gap.py \
        --checkpoint jaxborg-exp/ippo_cc4_20260327_1042/checkpoint_final.pkl
"""

# ruff: noqa: E402

import os

os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jaxborg/xla"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from eval_transfer import (
    DEFAULT_BANK_SIZE,
    NUM_BLUE_AGENTS,
    _all_blue_masks,
    _inject_live_red_policy_step,
    _make_jax_eval_env,
    load_checkpoint,
    make_batched_inference_fn,
    make_cyborg_env,
)

from jaxborg.cyborg_red_policy_recorder import RedPolicyRecorder
from jaxborg.topology import build_const_from_cyborg
from jaxborg.translate import build_mappings_from_cyborg, jax_blue_to_cyborg


def _cyborg_step_with_reward_breakdown(wrapper, actions):
    """Step CybORG and return per-component reward breakdown."""
    controller = wrapper.env.environment_controller

    # Step
    obs, rews, dones, info = wrapper.env.parallel_step(actions, skip_valid_action_check=True)

    # Extract BlueRewardMachine value
    brm = float(controller.reward.get("Blue", {}).get("BlueRewardMachine", 0.0))

    # Count green failures from action_dict
    from CybORG.Simulator.Actions.AbstractActions.Impact import Impact
    from CybORG.Simulator.Actions.GreenActions import GreenAccessService, GreenLocalWork

    lwf_count = 0
    asf_count = 0
    ria_count = 0

    action_dict = controller.action
    state = controller.state
    phase_rewards_map = {
        0: {
            "public_access_zone_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
            "admin_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
            "office_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
            "contractor_network_subnet": {"LWF": 0, "ASF": -5, "RIA": -5},
            "restricted_zone_a_subnet": {"LWF": -1, "ASF": -3, "RIA": -1},
            "operational_zone_a_subnet": {"LWF": -1, "ASF": -1, "RIA": -1},
            "restricted_zone_b_subnet": {"LWF": -1, "ASF": -3, "RIA": -1},
            "operational_zone_b_subnet": {"LWF": -1, "ASF": -1, "RIA": -1},
            "internet_subnet": {"LWF": 0, "ASF": 0, "RIA": -1},
        },
        1: {
            "public_access_zone_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
            "admin_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
            "office_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
            "contractor_network_subnet": {"LWF": 0, "ASF": 0, "RIA": 0},
            "restricted_zone_a_subnet": {"LWF": -2, "ASF": -1, "RIA": -3},
            "operational_zone_a_subnet": {"LWF": -10, "ASF": 0, "RIA": -10},
            "restricted_zone_b_subnet": {"LWF": -1, "ASF": -1, "RIA": -1},
            "operational_zone_b_subnet": {"LWF": -1, "ASF": -1, "RIA": -1},
            "internet_subnet": {"LWF": 0, "ASF": 0, "RIA": 0},
        },
        2: {
            "public_access_zone_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
            "admin_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
            "office_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
            "contractor_network_subnet": {"LWF": 0, "ASF": 0, "RIA": 0},
            "restricted_zone_a_subnet": {"LWF": -1, "ASF": -3, "RIA": -3},
            "operational_zone_a_subnet": {"LWF": -1, "ASF": -1, "RIA": -1},
            "restricted_zone_b_subnet": {"LWF": -2, "ASF": -1, "RIA": -3},
            "operational_zone_b_subnet": {"LWF": -10, "ASF": 0, "RIA": -10},
            "internet_subnet": {"LWF": 0, "ASF": 0, "RIA": 0},
        },
    }

    lwf_reward = 0.0
    asf_reward = 0.0
    ria_reward = 0.0

    for agent_name, action_list in action_dict.items():
        if not action_list:
            continue
        action = action_list[0] if isinstance(action_list, list) else action_list

        if isinstance(action, Impact):
            hostname = action.hostname
        elif isinstance(action, (GreenAccessService, GreenLocalWork)):
            hostname = state.ip_addresses.get(action.ip_address, None)
            if hostname is None:
                continue
        else:
            continue

        subnet_name = state.hostname_subnet_map[hostname].value
        sessions = state.sessions.get(agent_name, {}).values()

        if len([s.ident for s in sessions if s.active]) > 0:
            obs_set = controller.observation.get(agent_name)
            if obs_set is None:
                continue
            try:
                success = obs_set.observations[0].data.get("success")
            except (IndexError, AttributeError):
                continue

            phase_rewards = phase_rewards_map.get(state.mission_phase, phase_rewards_map[0])
            rewards_for_zone = phase_rewards.get(subnet_name, {})

            if "green" in agent_name and success == False:  # noqa: E712  (TernaryEnum)
                if isinstance(action, GreenLocalWork):
                    lwf_count += 1
                    lwf_reward += rewards_for_zone.get("LWF", 0)
                elif isinstance(action, GreenAccessService):
                    asf_count += 1
                    asf_reward += rewards_for_zone.get("ASF", 0)

            elif "red" in agent_name and success and isinstance(action, Impact):
                ria_count += 1
                ria_reward += rewards_for_zone.get("RIA", 0)

    observations = {
        agent: wrapper.observation_change(agent, obs[agent]) for agent in wrapper.possible_agents if agent in obs
    }
    wrapper.agents = [agent for agent in wrapper.possible_agents if not dones.get(agent, False)]

    return observations, brm, lwf_count, asf_count, ria_count, lwf_reward, asf_reward, ria_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology-bank-size", type=int, default=DEFAULT_BANK_SIZE)
    args = parser.parse_args()

    policy, params, policy_kind = load_checkpoint(args.checkpoint)
    batched_step = make_batched_inference_fn(policy, params, policy_kind, deterministic=False)

    for ep in range(args.episodes):
        ep_seed = args.seed + ep * 100

        jax_env = _make_jax_eval_env("cyborg_bank", args.topology_bank_size)
        key = jax.random.PRNGKey(ep_seed)
        jax_obs, jax_state = jax_env.reset(key)

        cyborg_env = make_cyborg_env(seed=ep_seed, bank_match_size=args.topology_bank_size)
        cyborg_obs, _ = cyborg_env.reset()
        inner = cyborg_env.env
        const = build_const_from_cyborg(inner)
        mappings = build_mappings_from_cyborg(inner)
        red_recorder = RedPolicyRecorder()
        red_recorder.install(inner, mappings)
        cyborg_agent_names = [f"blue_agent_{i}" for i in range(NUM_BLUE_AGENTS)]

        totals = {
            "jax_lwf": 0,
            "jax_asf": 0,
            "jax_ria": 0,
            "cy_lwf": 0,
            "cy_asf": 0,
            "cy_ria": 0,
            "jax_lwf_r": 0.0,
            "jax_asf_r": 0.0,
            "jax_ria_r": 0.0,
            "cy_lwf_r": 0.0,
            "cy_asf_r": 0.0,
            "cy_ria_r": 0.0,
            "jax_total": 0.0,
            "cy_total": 0.0,
        }

        for step in range(500):
            key, step_key = jax.random.split(key)
            act_keys = jax.random.split(key, NUM_BLUE_AGENTS)

            jax_masks = _all_blue_masks(jax_state.const, jax_state.state)
            jax_obs_stack = jnp.stack([jax_obs[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)])
            jax_actions_arr, _ = batched_step(jax_obs_stack, jax_masks, act_keys)
            jax_np = np.asarray(jax_actions_arr)

            jax_blue_actions = {f"blue_{i}": jax_actions_arr[i] for i in range(NUM_BLUE_AGENTS)}
            cyborg_actions = {}
            for i in range(NUM_BLUE_AGENTS):
                jax_act = int(jax_np[i])
                cyborg_actions[cyborg_agent_names[i]] = jax_blue_to_cyborg(jax_act, i, mappings, const=const)

            # Step CybORG with breakdown
            cyborg_obs, cy_brm, cy_lwf, cy_asf, cy_ria, cy_lwf_r, cy_asf_r, cy_ria_r = (
                _cyborg_step_with_reward_breakdown(cyborg_env, cyborg_actions)
            )
            totals["cy_lwf"] += cy_lwf
            totals["cy_asf"] += cy_asf
            totals["cy_ria"] += cy_ria
            totals["cy_lwf_r"] += cy_lwf_r
            totals["cy_asf_r"] += cy_asf_r
            totals["cy_ria_r"] += cy_ria_r
            totals["cy_total"] += cy_brm

            # Step JAX
            jax_state = _inject_live_red_policy_step(jax_state, red_recorder, step_idx=step)
            jax_obs, jax_state, jax_step_rewards, _, jax_info = jax_env.step(step_key, jax_state, jax_blue_actions)
            jax_reward = float(jnp.stack([jax_step_rewards[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)]).mean())
            totals["jax_total"] += jax_reward

            if isinstance(jax_info, dict):
                totals["jax_lwf"] += int(jax_info.get("green_lwf_count", 0))
                totals["jax_asf"] += int(jax_info.get("green_asf_count", 0))
                totals["jax_ria"] += int(jax_info.get("impact_count", 0))
                totals["jax_lwf_r"] += float(jax_info.get("reward_lwf", 0))
                totals["jax_asf_r"] += float(jax_info.get("reward_asf", 0))
                totals["jax_ria_r"] += float(jax_info.get("reward_ria", 0))

        gap = totals["jax_total"] - totals["cy_total"]
        print(f"\nEpisode {ep + 1} (seed={ep_seed}):")
        hdr = f"  {'':20s} {'JAX':>10s} {'CybORG':>10s} {'Diff':>10s}"
        print(hdr)

        def _row(label, jk, ck, fmt="d"):
            jv, cv = totals[jk], totals[ck]
            d = jv - cv
            if fmt == "d":
                print(f"  {label:20s} {jv:10d} {cv:10d} {d:10d}")
            else:
                print(f"  {label:20s} {jv:10.1f} {cv:10.1f} {d:10.1f}")

        _row("LWF events", "jax_lwf", "cy_lwf")
        _row("ASF events", "jax_asf", "cy_asf")
        _row("RIA events", "jax_ria", "cy_ria")
        _row("LWF reward", "jax_lwf_r", "cy_lwf_r", "f")
        _row("ASF reward", "jax_asf_r", "cy_asf_r", "f")
        _row("RIA reward", "jax_ria_r", "cy_ria_r", "f")
        print(f"  {'Total reward':20s} {totals['jax_total']:10.1f} {totals['cy_total']:10.1f} {gap:10.1f}")


if __name__ == "__main__":
    main()
