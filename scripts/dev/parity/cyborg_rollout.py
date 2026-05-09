"""CybORG rollout implementation for transfer parity checks."""

# ruff: noqa: E402,I001

from __future__ import annotations

from scripts.dev.parity.bootstrap import configure_runtime

configure_runtime()

import os
import time
from statistics import mean

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.parity.translate import build_mappings_from_cyborg, jax_blue_to_cyborg
from jaxborg.scenarios.cc4.cyborg_resilience_agents import inject_role_map
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg
from scripts.dev.parity.cyborg_bridge import (
    _build_cyborg_mask_cache,
    _live_blue_wrapper_mask_in_jax_space_cached,
    _raw_cyborg_step_with_flat_obs,
    make_cyborg_env,
)
from scripts.dev.parity.policy import load_checkpoint, make_batched_inference_fn
from scripts.dev.parity.rollout_types import CyborgRollout


def _rollout_cyborg_single_episode(args_tuple):
    """Run a single CybORG episode in its own process. Returns (ep, reward, actions_by_agent)."""
    ep, checkpoint_path, deterministic, seed, variant = args_tuple

    policy, params = load_checkpoint(checkpoint_path)
    batched_step = make_batched_inference_fn(policy, params, deterministic)

    ep_seed = seed + ep
    env = make_cyborg_env(seed=ep_seed, variant=variant)
    observations, _ = env.reset()
    if variant is not None and variant.resilience_roles:
        inject_role_map(env, ep_seed=ep_seed)
    inner = env.env
    const = build_const_from_cyborg(inner)
    mappings = build_mappings_from_cyborg(inner)

    rng = jax.random.PRNGKey(seed + ep)
    mask_cache = _build_cyborg_mask_cache(env, mappings, const)
    total = 0.0
    ep_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
    ep_busy_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
    ep_phase_per_step: list = []

    # Monkeypatch BlueRewardMachine to track per-component rewards
    import types as _types

    from CybORG.Simulator.Actions.AbstractActions.Impact import Impact as _Impact
    from CybORG.Simulator.Actions.GreenActions import GreenAccessService as _GAS
    from CybORG.Simulator.Actions.GreenActions import GreenLocalWork as _GLW

    ec = inner.environment_controller
    brm = ec.team_reward_calculators["Blue"]["BlueRewardMachine"]
    _component_log = {"ria": 0.0, "lwf": 0.0, "asf": 0.0}

    def _tracked_calculate(self, current_state, action_dict, agent_observations, done, state):
        self.phase_rewards = self.get_phase_rewards(state.mission_phase)
        reward_list = []
        for agent_name, action in action_dict.items():
            if not action:
                continue
            act = action[0]
            if isinstance(act, _Impact):
                hostname = act.hostname
            elif isinstance(act, (_GAS, _GLW)):
                hostname = state.ip_addresses[act.ip_address]
            else:
                continue
            subnet_name = state.hostname_subnet_map[hostname].value
            sessions = state.sessions[agent_name].values()
            if len([s.ident for s in sessions if s.active]) > 0:
                success = agent_observations[agent_name].observations[0].data["success"]
                rz = self.phase_rewards[subnet_name]
                if "green" in agent_name and success == False:  # noqa: E712
                    if isinstance(act, _GLW):
                        r = rz["LWF"]
                        reward_list.append(r)
                        _component_log["lwf"] += r
                    elif isinstance(act, _GAS):
                        r = rz["ASF"]
                        reward_list.append(r)
                        _component_log["asf"] += r
                elif "red" in agent_name and success and isinstance(act, _Impact):
                    r = rz["RIA"]
                    reward_list.append(r)
                    _component_log["ria"] += r
        return sum(reward_list)

    brm.calculate_reward = _types.MethodType(_tracked_calculate, brm)

    for _ in range(500):
        phase = int(ec.state.mission_phase)
        ep_phase_per_step.append(phase)
        if env.agents:
            rng, *_rngs = jax.random.split(rng, NUM_BLUE_AGENTS + 1)
            act_keys = jnp.stack(_rngs)

            masks = jnp.stack(
                [
                    jnp.array(_live_blue_wrapper_mask_in_jax_space_cached(env, agent_name, mappings, const, mask_cache))
                    for agent_name in env.agents
                ]
            )
            obs_stack = jnp.stack([jnp.array(observations[a], dtype=jnp.float32) for a in env.agents])

            actions_arr, _ = batched_step(obs_stack, masks, act_keys)
            actions_np = np.asarray(actions_arr)

            actions = {}
            for agent_idx, agent_name in enumerate(env.agents):
                action_idx = int(actions_np[agent_idx])
                cyborg_action = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
                actions[agent_name] = cyborg_action
                ep_actions_by_agent[agent_idx].append(action_idx)
                pending = ec.actions_in_progress.get(agent_name)
                ep_busy_by_agent[agent_idx].append(1 if pending and pending["remaining_ticks"] > 0 else 0)
        else:
            from CybORG.Simulator.Actions import Sleep

            actions = {a: Sleep() for a in env.possible_agents}

        observations, rewards, _, _, _ = _raw_cyborg_step_with_flat_obs(env, actions=actions)
        total += mean(rewards.values())

    return (
        ep,
        total,
        ep_actions_by_agent,
        _component_log["ria"],
        _component_log["lwf"],
        _component_log["asf"],
        ep_busy_by_agent,
        ep_phase_per_step,
    )


def rollout_cyborg(
    policy,
    params,
    num_episodes=3,
    deterministic=False,
    seed=0,
    checkpoint_path=None,
    parallel=True,
    max_workers=None,
    variant=None,
):
    if parallel and checkpoint_path:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        if max_workers is None:
            env_cap = int(os.environ.get("JAXBORG_TRANSFER_WORKERS", "10"))
            max_workers = min(num_episodes, multiprocessing.cpu_count(), env_cap)

        print(f"  Running {num_episodes} CybORG episodes in parallel ({max_workers} workers)...", flush=True)
        t0 = time.perf_counter()
        args_list = [(ep, checkpoint_path, deterministic, seed, variant) for ep in range(num_episodes)]
        all_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        all_busy_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        all_phase_per_step: list = [[] for _ in range(num_episodes)]
        episode_rewards = [0.0] * num_episodes
        episode_ria = [0.0] * num_episodes
        episode_lwf = [0.0] * num_episodes
        episode_asf = [0.0] * num_episodes

        # Use "spawn" to avoid fork() + CUDA deadlock; sentinel tells
        # spawned children to set JAX_PLATFORMS=cpu before module-level imports
        os.environ["_JAXBORG_CYBORG_WORKER"] = "1"
        ctx = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
                for ep, reward, ep_actions_by_agent, ria, lwf, asf, ep_busy_by_agent, ep_phase in pool.map(
                    _rollout_cyborg_single_episode, args_list
                ):
                    episode_rewards[ep] = reward
                    episode_ria[ep] = ria
                    episode_lwf[ep] = lwf
                    episode_asf[ep] = asf
                    all_phase_per_step[ep] = ep_phase
                    for i in range(NUM_BLUE_AGENTS):
                        all_actions_by_agent[i].extend(ep_actions_by_agent[i])
                        all_busy_by_agent[i].extend(ep_busy_by_agent[i])
                    print(f"  CybORG  ep {ep + 1}: reward={reward:.1f}", flush=True)
        finally:
            os.environ.pop("_JAXBORG_CYBORG_WORKER", None)

        elapsed = time.perf_counter() - t0
        print(f"  {num_episodes} CybORG episodes done in {elapsed:.0f}s ({elapsed / num_episodes:.1f}s/ep effective)")
        flat_actions = [a for agent_actions in all_actions_by_agent for a in agent_actions]
        return CyborgRollout(
            actions=np.array(flat_actions),
            rewards=np.array(episode_rewards),
            actions_by_agent=all_actions_by_agent,
            ria=np.array(episode_ria),
            lwf=np.array(episode_lwf),
            asf=np.array(episode_asf),
            busy_by_agent=all_busy_by_agent,
            phase_per_step=all_phase_per_step,
        )

    # Fallback: sequential (no checkpoint path or parallel=False)
    batched_step = make_batched_inference_fn(policy, params, deterministic)
    all_actions = []
    episode_rewards = []
    all_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
    all_busy_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        ep_seed = seed + ep
        env = make_cyborg_env(seed=ep_seed, variant=variant)
        observations, _ = env.reset()
        if variant is not None and variant.resilience_roles:
            inject_role_map(env, ep_seed=ep_seed)
        inner = env.env
        const = build_const_from_cyborg(inner)
        mappings = build_mappings_from_cyborg(inner)
        ec = inner.environment_controller

        rng = jax.random.PRNGKey(seed + ep)
        mask_cache = _build_cyborg_mask_cache(env, mappings, const)
        total = 0.0
        ep_actions = []

        for _ in range(500):
            if env.agents:
                rng, *_rngs = jax.random.split(rng, NUM_BLUE_AGENTS + 1)
                act_keys = jnp.stack(_rngs)

                masks = jnp.stack(
                    [
                        jnp.array(
                            _live_blue_wrapper_mask_in_jax_space_cached(env, agent_name, mappings, const, mask_cache)
                        )
                        for agent_name in env.agents
                    ]
                )
                obs_stack = jnp.stack([jnp.array(observations[a], dtype=jnp.float32) for a in env.agents])

                # Batched policy inference (1 forward pass instead of 5)
                actions_arr, _ = batched_step(obs_stack, masks, act_keys)
                actions_np = np.asarray(actions_arr)

                actions = {}
                for agent_idx, agent_name in enumerate(env.agents):
                    action_idx = int(actions_np[agent_idx])
                    cyborg_action = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
                    actions[agent_name] = cyborg_action
                    ep_actions.append(action_idx)
                    all_actions_by_agent[agent_idx].append(action_idx)
                    pending = ec.actions_in_progress.get(agent_name)
                    all_busy_by_agent[agent_idx].append(1 if pending and pending["remaining_ticks"] > 0 else 0)
            else:
                # Episode done but continue stepping to match JAXborg step count.
                # CybORG still processes green/red actions and returns rewards.
                from CybORG.Simulator.Actions import Sleep

                actions = {a: Sleep() for a in env.possible_agents}

            observations, rewards, _, _, _ = _raw_cyborg_step_with_flat_obs(env, actions=actions)
            total += mean(rewards.values())

        elapsed = time.perf_counter() - t0
        print(f"  CybORG  ep {ep + 1}: reward={total:.1f} ({elapsed:.1f}s)")
        all_actions.extend(ep_actions)
        episode_rewards.append(total)

    return CyborgRollout(
        actions=np.array(all_actions),
        rewards=np.array(episode_rewards),
        actions_by_agent=all_actions_by_agent,
        ria=np.zeros(num_episodes),
        lwf=np.zeros(num_episodes),
        asf=np.zeros(num_episodes),
        busy_by_agent=all_busy_by_agent,
        phase_per_step=[[] for _ in range(num_episodes)],
    )
