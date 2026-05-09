"""Baselines and verbose diagnostics for dev transfer checks."""

from __future__ import annotations

from statistics import mean

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.actions.encoding import BLUE_ANALYSE_START
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS, OBS_VECTOR_HOSTS_PER_SUBNET
from jaxborg.parity.fsm_red_env import FsmRedCC4Env
from jaxborg.parity.translate import build_mappings_from_cyborg, describe_blue_action, jax_blue_to_cyborg
from jaxborg.scenarios.cc4.cyborg_resilience_agents import inject_role_map
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg
from scripts.dev.parity.cyborg_bridge import (
    _live_blue_wrapper_mask_in_jax_space,
    _raw_cyborg_step_with_flat_obs,
    make_cyborg_env,
)
from scripts.dev.parity.policy import policy_dist
from scripts.dev.parity.stats import ACTION_TYPE_NAMES, ACTION_TYPE_RANGES


def run_sleep_baseline(episodes=5, *, variant=None):
    from CybORG.Simulator.Actions import Sleep

    totals = []
    for ep in range(episodes):
        env = make_cyborg_env(seed=ep, variant=variant)
        env.reset()
        if variant is not None and variant.resilience_roles:
            inject_role_map(env, ep_seed=ep)
        total = 0.0
        for _ in range(500):
            actions = {a: Sleep() for a in env.agents}
            _, rewards, _, _, _ = env.step(actions=actions)
            total += mean(rewards.values())
        totals.append(total)
    return mean(totals)


def run_random_baseline(episodes=5, seed=42, *, variant=None):
    rng = np.random.default_rng(seed)
    totals = []
    for ep in range(episodes):
        ep_seed = seed + ep
        env = make_cyborg_env(seed=ep_seed, variant=variant)
        _, _ = env.reset()
        if variant is not None and variant.resilience_roles:
            inject_role_map(env, ep_seed=ep_seed)
        inner = env.env
        const = build_const_from_cyborg(inner)
        mappings = build_mappings_from_cyborg(inner)
        total = 0.0
        for _ in range(500):
            if not env.agents:
                break
            actions = {}
            for agent_idx, agent_name in enumerate(env.agents):
                mask = np.array(_live_blue_wrapper_mask_in_jax_space(env, agent_name, mappings, const), dtype=bool)
                valid = np.where(mask)[0]
                action_idx = int(rng.choice(valid))
                actions[agent_name] = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
            _, rewards, _, _, _ = _raw_cyborg_step_with_flat_obs(env, actions=actions)
            total += mean(rewards.values())
        totals.append(total)
    return mean(totals)


def run_verbose_trace(policy, params, steps=20, seed=42):
    env = make_cyborg_env(seed=seed)
    observations, _ = env.reset()
    inner = env.env
    const = build_const_from_cyborg(inner)
    mappings = build_mappings_from_cyborg(inner)

    from jaxborg.actions.encoding import BLUE_ANALYSE_END
    from jaxborg.scenarios.cc4.topology import BLUE_AGENT_SUBNETS, SUBNET_IDS

    print("\nMASK VALIDATION (step 0):")
    for agent_idx, agent_name in enumerate(env.agents):
        mask = np.array(_live_blue_wrapper_mask_in_jax_space(env, agent_name, mappings, const), dtype=bool)
        valid_indices = np.where(mask)[0]
        agent_subnets = BLUE_AGENT_SUBNETS[agent_idx]
        agent_subnet_ids = [SUBNET_IDS[s] for s in agent_subnets]

        valid_analyse = valid_indices[(valid_indices >= BLUE_ANALYSE_START) & (valid_indices < BLUE_ANALYSE_END)]
        valid_slots = valid_analyse - BLUE_ANALYSE_START

        wrong_subnet_hosts = []
        for slot in valid_slots:
            subnet_id = int(slot // OBS_VECTOR_HOSTS_PER_SUBNET)
            slot_within = int(slot % OBS_VECTOR_HOSTS_PER_SUBNET)
            hidx = int(const.obs_host_map[subnet_id, slot_within])
            if hidx >= GLOBAL_MAX_HOSTS:
                continue
            h_subnet = int(const.host_subnet[hidx])
            if h_subnet not in agent_subnet_ids:
                hostname = mappings.idx_to_hostname.get(int(hidx), f"host_{hidx}")
                wrong_subnet_hosts.append((int(hidx), hostname, h_subnet))

        print(
            f"  {agent_name}: subnets={agent_subnets}, "
            f"valid_analyse_hosts={len(valid_analyse)}, "
            f"wrong_subnet={len(wrong_subnet_hosts)}"
        )
        if wrong_subnet_hosts:
            for hidx, hname, hsub in wrong_subnet_hosts[:5]:
                print(f"    BUG: host_idx={hidx} {hname} in subnet {hsub} allowed!")

    total = 0.0
    for step in range(steps):
        actions = {}
        step_actions_desc = []
        for agent_idx, agent_name in enumerate(env.agents):
            obs_jax = jnp.array(observations[agent_name], dtype=jnp.float32)
            mask = _live_blue_wrapper_mask_in_jax_space(env, agent_name, mappings, const)
            mask_np = np.array(mask, dtype=bool)

            pi = policy_dist(policy, params, obs_jax, mask)
            action_idx = int(jnp.argmax(pi.logits))
            is_valid = bool(mask_np[action_idx])
            cyborg_action = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
            actions[agent_name] = cyborg_action

            desc = describe_blue_action(action_idx, mappings, const=const, agent_id=agent_idx)
            cyborg_cls = type(cyborg_action).__name__
            valid_str = "OK" if is_valid else "MASKED!"
            step_actions_desc.append(
                f"  {agent_name}: idx={action_idx:4d} [{valid_str:7s}] -> {desc:45s} -> CybORG:{cyborg_cls}"
            )

        observations, rewards, _, _, _ = _raw_cyborg_step_with_flat_obs(env, actions=actions)
        step_reward = mean(rewards.values())
        total += step_reward

        if step < 10 or step % 5 == 0:
            print(f"\nStep {step}: reward={step_reward:.2f}  cumulative={total:.2f}")
            for desc in step_actions_desc:
                print(desc)

    print(f"\nVerbose trace total reward ({steps} steps): {total:.2f}")


def print_mask_summary():
    print("\n--- Action Mask Summary ---")
    env = FsmRedCC4Env(num_steps=100, topology_mode="generative")
    key = jax.random.PRNGKey(42)
    _, env_state = env.reset(key)

    for agent_idx in range(NUM_BLUE_AGENTS):
        mask = np.array(compute_blue_action_mask(env_state.const, agent_idx, env_state.state))
        total_valid = mask.sum()
        by_type = []
        for name, (start, end) in zip(ACTION_TYPE_NAMES, ACTION_TYPE_RANGES):
            count = mask[start:end].sum()
            if count > 0:
                by_type.append(f"{name}={count}")
        print(f"  blue_{agent_idx}: {total_valid} valid actions: {', '.join(by_type)}")
