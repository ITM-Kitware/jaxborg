"""JAXborg rollout implementations used by dev transfer checks."""

# ruff: noqa: E402,I001

from __future__ import annotations

from scripts.dev.parity.bootstrap import DEFAULT_NUM_STEPS, configure_runtime

configure_runtime()

import time

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import COMPROMISE_PRIVILEGED, COMPROMISE_USER, NUM_BLUE_AGENTS, NUM_RED_AGENTS
from jaxborg.parity.fsm_red_env import FsmRedCC4Env
from scripts.dev.parity.policy import make_batched_inference_fn
from scripts.dev.parity.rollout_types import EpisodeResult, JaxRollout, StepSnapshot


def _make_jax_eval_env():
    return FsmRedCC4Env(num_steps=DEFAULT_NUM_STEPS)


def _all_blue_masks(const, state):
    """Compute action masks for all blue agents."""
    return jnp.stack([compute_blue_action_mask(const, i, state) for i in range(NUM_BLUE_AGENTS)])


def make_scan_eval_fn(env, policy, deterministic):
    """Build a fully JIT'd scan-based eval rollout.

    Returns a function: (params, key, env_state, obs) -> (final_state, step_data)
    where step_data contains per-step actions, rewards, and trajectory info.

    Params are passed as a dynamic argument (not captured in closure) so the
    XLA compilation cache can be reused across runs with different checkpoints.

    First call triggers XLA compilation (~5-10 min for CC4, cached to disk).
    Subsequent calls (same or future runs) load from cache and execute in seconds.
    """
    _agent_ids = jnp.arange(NUM_BLUE_AGENTS)
    _mask_single = jax.vmap(compute_blue_action_mask, in_axes=(None, 0, None))

    def _fwd(params, obs_flat, mask_flat):
        pi, _ = policy.apply(params, obs_flat, mask_flat)
        return pi.logits

    def _env_step(carry, _):
        params, key, env_state, obs = carry

        # Compute masks for all agents: (NUM_BLUE_AGENTS, action_dim)
        masks = _mask_single(env_state.const, _agent_ids, env_state.state)

        # Stack obs: (NUM_BLUE_AGENTS, obs_dim)
        obs_stack = jnp.stack([obs[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)])

        # Batched policy inference (params passed through, not closed over)
        logits = jax.vmap(_fwd, in_axes=(None, 0, 0))(params, obs_stack, masks)

        key, _rng = jax.random.split(key)
        if deterministic:
            actions_arr = jnp.argmax(logits, axis=-1)
        else:
            act_keys = jax.random.split(_rng, NUM_BLUE_AGENTS)
            actions_arr = jax.vmap(lambda lg, k: distrax.Categorical(logits=lg).sample(seed=k))(logits, act_keys)

        actions = {f"blue_{i}": actions_arr[i] for i in range(NUM_BLUE_AGENTS)}

        key, step_key = jax.random.split(key)
        new_obs, new_env_state, rewards, dones, info = env.step(step_key, env_state, actions)

        # Collect per-step data
        reward_arr = jnp.stack([rewards[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)])
        st = new_env_state.state
        active = new_env_state.const.host_active
        compromised = st.host_compromised

        step_data = {
            "actions": actions_arr,
            "blue_busy": (env_state.state.blue_pending_ticks > 0).astype(jnp.int32),
            "reward_mean": reward_arr.mean(),
            "hosts_user": jnp.sum((compromised == COMPROMISE_USER) & active),
            "hosts_priv": jnp.sum((compromised == COMPROMISE_PRIVILEGED) & active),
            "red_sessions": jnp.sum(st.red_sessions[:NUM_RED_AGENTS]),
            "mission_phase": st.mission_phase,
            "reward_ria": info["reward_ria"],
            "reward_lwf": info["reward_lwf"],
            "reward_asf": info["reward_asf"],
        }

        return (params, key, new_env_state, new_obs), step_data

    @jax.jit
    def scan_eval(params, key, env_state, obs):
        (_, final_key, final_state, final_obs), step_data = jax.lax.scan(
            _env_step, (params, key, env_state, obs), None, length=500
        )
        return final_state, step_data

    return scan_eval


def rollout_jaxborg_scan(
    policy,
    params,
    num_episodes=3,
    deterministic=False,
    seed=0,
):
    """JAXborg-only eval using jax.lax.scan + jax.vmap — all episodes in parallel.

    Runs all episodes simultaneously on GPU via vmap over seeds.
    First call triggers XLA compilation (~5-10 min, cached to disk).
    Subsequent runs load from XLA cache and execute all episodes in one GPU pass.
    """
    env = _make_jax_eval_env()
    scan_fn = make_scan_eval_fn(env, policy, deterministic)

    # Build keys for all episodes
    keys = jnp.stack([jax.random.PRNGKey(seed + ep) for ep in range(num_episodes)])

    # Reset all episodes in parallel: vmap over seeds
    print(f"  Resetting {num_episodes} episodes in parallel...", flush=True)
    all_obs, all_env_states = jax.vmap(env.reset)(keys)

    # Run all episodes in parallel: vmap(scan) over episodes
    print("  (first run includes XLA compilation — cached for future runs)", flush=True)
    t0 = time.perf_counter()
    _, all_step_data = jax.vmap(scan_fn, in_axes=(None, 0, 0, 0))(params, keys, all_env_states, all_obs)
    # all_step_data shapes: each field is (num_episodes, 500, ...) or (num_episodes, 500)

    # Single device-to-host transfer for all episodes
    actions_np = np.asarray(all_step_data["actions"])  # (num_episodes, 500, NUM_BLUE_AGENTS)
    blue_busy_np = np.asarray(all_step_data["blue_busy"])  # (num_episodes, 500, NUM_BLUE_AGENTS)
    rewards_np = np.asarray(all_step_data["reward_mean"])  # (num_episodes, 500)
    hosts_user_np = np.asarray(all_step_data["hosts_user"])
    hosts_priv_np = np.asarray(all_step_data["hosts_priv"])
    red_sess_np = np.asarray(all_step_data["red_sessions"])
    phase_np = np.asarray(all_step_data["mission_phase"])
    ria_np = np.asarray(all_step_data["reward_ria"])  # (num_episodes, 500)
    lwf_np = np.asarray(all_step_data["reward_lwf"])
    asf_np = np.asarray(all_step_data["reward_asf"])

    elapsed = time.perf_counter() - t0
    ep_totals = rewards_np.sum(axis=1)
    print(f"  {num_episodes} episodes completed in {elapsed:.1f}s ({elapsed / num_episodes:.1f}s/ep)")
    for ep in range(num_episodes):
        print(f"    ep {ep + 1}: reward={ep_totals[ep]:.1f}")

    # Build result objects
    all_actions_flat = []
    episode_rewards = []
    episode_results = []

    for ep in range(num_episodes):
        cum_rewards = np.cumsum(rewards_np[ep])
        ep_trajectory = [
            StepSnapshot(
                reward=float(rewards_np[ep, s]),
                cumulative_reward=float(cum_rewards[s]),
                hosts_compromised_user=int(hosts_user_np[ep, s]),
                hosts_compromised_priv=int(hosts_priv_np[ep, s]),
                red_sessions_total=int(red_sess_np[ep, s]),
                mission_phase=int(phase_np[ep, s]),
            )
            for s in range(500)
        ]
        ep_actions_by_agent = [actions_np[ep, :, i].tolist() for i in range(NUM_BLUE_AGENTS)]
        ep_busy_by_agent = [blue_busy_np[ep, :, i].tolist() for i in range(NUM_BLUE_AGENTS)]
        all_actions_flat.extend(actions_np[ep].ravel().tolist())
        episode_rewards.append(float(ep_totals[ep]))
        episode_results.append(
            EpisodeResult(
                actions_by_agent=ep_actions_by_agent,
                blue_busy_by_agent=ep_busy_by_agent,
                phase_per_step=phase_np[ep].tolist(),
                rewards=rewards_np[ep].tolist(),
                cumulative_reward=float(ep_totals[ep]),
                trajectory=ep_trajectory,
                ria_total=float(ria_np[ep].sum()),
                lwf_total=float(lwf_np[ep].sum()),
                asf_total=float(asf_np[ep].sum()),
            )
        )

    return JaxRollout(
        actions=np.array(all_actions_flat),
        rewards=np.array(episode_rewards),
        episodes=episode_results,
    )


def rollout_jaxborg(
    policy,
    params,
    num_episodes=3,
    deterministic=False,
    seed=0,
):
    env = _make_jax_eval_env()
    batched_step = make_batched_inference_fn(policy, params, deterministic)
    all_actions = []
    episode_rewards = []
    episode_results = []

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        key = jax.random.PRNGKey(seed + ep)
        obs, env_state = env.reset(key)

        ep_reward = np.zeros(NUM_BLUE_AGENTS)
        ep_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_busy_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_step_rewards = []
        ep_trajectory = []
        cum_reward = 0.0

        for step in range(500):
            key, step_key = jax.random.split(key)
            act_keys = jax.random.split(key, NUM_BLUE_AGENTS)

            # Record busy state before stepping
            busy_np = np.asarray(env_state.state.blue_pending_ticks > 0)

            # Batched mask computation + policy inference (1 JIT call instead of 5)
            masks = _all_blue_masks(env_state.const, env_state.state)
            obs_stack = jnp.stack([obs[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)])
            actions_arr, _ = batched_step(obs_stack, masks, act_keys)

            # Single device-to-host transfer for all actions
            actions_np = np.asarray(actions_arr)
            actions = {f"blue_{i}": actions_arr[i] for i in range(NUM_BLUE_AGENTS)}
            for i in range(NUM_BLUE_AGENTS):
                ep_actions_by_agent[i].append(int(actions_np[i]))
                ep_busy_by_agent[i].append(int(busy_np[i]))

            obs, env_state, rewards, dones, _ = env.step(step_key, env_state, actions)

            # Batch reward extraction (single transfer)
            reward_arr = jnp.stack([rewards[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)])
            reward_np = np.asarray(reward_arr)
            step_reward = float(reward_np.mean())
            cum_reward += step_reward
            ep_step_rewards.append(step_reward)
            ep_reward += reward_np

            # Extract trajectory snapshot — defer int() conversions
            st = env_state.state
            active = np.array(env_state.const.host_active, dtype=bool)
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

            if dones["__all__"]:
                break

        elapsed = time.perf_counter() - t0
        total = ep_reward.mean()
        print(f"  JAXborg ep {ep + 1}: reward={total:.1f} ({elapsed:.1f}s)")

        # Flatten per-agent actions for backward compat
        flat_actions = [a for step_actions in zip(*ep_actions_by_agent) for a in step_actions]
        all_actions.extend(flat_actions)
        episode_rewards.append(total)
        episode_results.append(
            EpisodeResult(
                actions_by_agent=ep_actions_by_agent,
                blue_busy_by_agent=ep_busy_by_agent,
                rewards=ep_step_rewards,
                cumulative_reward=cum_reward,
                trajectory=ep_trajectory,
            )
        )

    return JaxRollout(
        actions=np.array(all_actions),
        rewards=np.array(episode_rewards),
        episodes=episode_results,
    )
