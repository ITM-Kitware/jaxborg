"""L4 cross-backend policy transfer test using TOST equivalence.

Loads a trained JAXborg checkpoint, evaluates it in both JaxBorg (FsmRedCC4Env)
and CybORG (BlueFlatWrapper), and applies the Two One-Sided Tests (TOST)
procedure to confirm statistical equivalence of episode rewards.

Set JAXBORG_POLICY_CHECKPOINT to the exported model_<tag>.pkl path to run.
"""

import os
from pathlib import Path
from statistics import mean, stdev

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import ttest_1samp

CHECKPOINT_ENV = "JAXBORG_POLICY_CHECKPOINT"
CKPT_PATH = Path(os.environ[CHECKPOINT_ENV]).expanduser() if os.environ.get(CHECKPOINT_ENV) else None
NUM_EPISODES = int(os.environ.get("JAXBORG_TOST_EPISODES", "50"))
TOST_MARGIN = float(os.environ.get("JAXBORG_TOST_MARGIN", "5.0"))
TOST_ALPHA = float(os.environ.get("JAXBORG_TOST_ALPHA", "0.05"))
skip_reason = None
if CKPT_PATH is None:
    skip_reason = f"Set {CHECKPOINT_ENV}=/path/to/model_<tag>.pkl"
elif not CKPT_PATH.exists():
    skip_reason = f"{CHECKPOINT_ENV} does not exist: {CKPT_PATH}"


def _load_checkpoint(path):
    """Load checkpoint via jax_runner.load_jax_checkpoint (sidecar required)."""
    from jaxborg.evaluation.jax_runner import load_jax_checkpoint

    policy, params, _recipe = load_jax_checkpoint(path)
    return policy, params


def _policy_action(policy, params, obs_jax, mask_jax):
    pi, _ = policy.apply(params, obs_jax, mask_jax)
    return int(pi.mode()[0])


def _rollout_jaxborg(policy, params, seed, num_episodes, num_steps=500):
    """Run episodes in JaxBorg and return per-episode cumulative rewards."""
    from jaxborg.actions.masking import compute_blue_action_mask
    from jaxborg.constants import NUM_BLUE_AGENTS
    from jaxborg.parity.fsm_red_env import FsmRedCC4Env

    env = FsmRedCC4Env(num_steps=num_steps)
    rewards = []

    for ep in range(num_episodes):
        key = jax.random.PRNGKey(seed + ep)
        key, reset_key = jax.random.split(key)
        obs_dict, state = env.reset(reset_key)
        ep_reward = 0.0

        for _step in range(num_steps):
            actions = {}
            for b in range(NUM_BLUE_AGENTS):
                agent_name = f"blue_{b}"
                obs = obs_dict[agent_name]
                mask = compute_blue_action_mask(state.const, b, state.state)
                obs_jax = jnp.array(obs).reshape(1, -1)
                mask_jax = jnp.array(mask).reshape(1, -1)
                actions[agent_name] = jnp.int32(_policy_action(policy, params, obs_jax, mask_jax))

            key, step_key = jax.random.split(key)
            obs_dict, state, reward_dict, done_dict, info = env.step(step_key, state, actions)
            ep_reward += mean(float(v) for v in reward_dict.values())

            if done_dict.get("__all__", False):
                break

        rewards.append(ep_reward)

    return np.array(rewards)


def _rollout_cyborg(policy, params, seed, num_episodes, num_steps=500):
    """Run episodes in CybORG using JAX policy actions translated into CybORG space."""
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.evaluation.jax_runner import _build_action_lookup, _live_cyborg_mask_in_jax_space, _raw_step
    from jaxborg.parity.translate import build_mappings_from_cyborg, jax_blue_to_cyborg
    from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

    rewards = []

    for ep in range(num_episodes):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=num_steps,
        )
        cyborg = CybORG(sg, "sim", seed=seed + ep)
        env = BlueFlatWrapper(env=cyborg, pad_spaces=True)

        observations, _ = env.reset()
        const = build_const_from_cyborg(cyborg)
        mappings = build_mappings_from_cyborg(cyborg)
        lookups = {agent_name: _build_action_lookup(env, agent_name, mappings, const) for agent_name in env.agents}
        ep_reward = 0.0

        for _step in range(num_steps):
            actions = {}
            for agent_idx, agent_name in enumerate(env.agents):
                obs = observations.get(agent_name)
                if obs is None:
                    continue

                obs_jax = jnp.array(obs).reshape(1, -1)
                mask_jax = _live_cyborg_mask_in_jax_space(env, agent_name, lookups[agent_name]).reshape(1, -1)
                action_idx = _policy_action(policy, params, obs_jax, mask_jax)
                actions[agent_name] = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)

            observations, rews, terminated, truncated = _raw_step(env, actions)
            ep_reward += mean(rews.values())

            if any(terminated.values()) or any(truncated.values()):
                break

        rewards.append(ep_reward)

    return np.array(rewards)


def tost_equivalence(diffs, margin, alpha=0.05):
    """Two One-Sided Tests for equivalence within +/- margin.

    Returns (equivalent: bool, p_value: float, mean_diff: float, std_diff: float).
    """
    if len(diffs) < 2:
        mean_diff = float(np.mean(diffs))
        return abs(mean_diff) <= margin, float("nan"), mean_diff, 0.0
    _, p_lower = ttest_1samp(diffs, -margin, alternative="greater")
    _, p_upper = ttest_1samp(diffs, margin, alternative="less")
    p_value = max(p_lower, p_upper)
    return p_value < alpha, p_value, float(np.mean(diffs)), float(np.std(diffs))


@pytest.mark.slow
def test_policy_transfer_tost():
    """Train in JaxBorg, evaluate in CybORG, confirm equivalence via TOST."""
    if skip_reason is not None:
        pytest.skip(skip_reason)

    policy, params = _load_checkpoint(CKPT_PATH)

    jax_rewards = _rollout_jaxborg(policy, params, seed=42, num_episodes=NUM_EPISODES)
    cyborg_rewards = _rollout_cyborg(policy, params, seed=42, num_episodes=NUM_EPISODES)

    diffs = jax_rewards - cyborg_rewards
    equivalent, p_value, mean_diff, std_diff = tost_equivalence(diffs, TOST_MARGIN, TOST_ALPHA)

    print("\n=== TOST Policy Transfer Results ===")
    jax_std = stdev(jax_rewards) if len(jax_rewards) > 1 else 0.0
    cyborg_std = stdev(cyborg_rewards) if len(cyborg_rewards) > 1 else 0.0
    print(f"JaxBorg:  mean={mean(jax_rewards):.2f} std={jax_std:.2f}")
    print(f"CybORG:   mean={mean(cyborg_rewards):.2f} std={cyborg_std:.2f}")
    print(f"Diff:     mean={mean_diff:.2f} std={std_diff:.2f}")
    print(f"TOST:     margin={TOST_MARGIN}, alpha={TOST_ALPHA}, p={p_value:.4f}")
    print(f"Result:   {'EQUIVALENT' if equivalent else 'NOT EQUIVALENT'}")

    assert equivalent, (
        f"TOST failed: mean_diff={mean_diff:.2f}, std={std_diff:.2f}, "
        f"p={p_value:.4f} >= alpha={TOST_ALPHA} (margin={TOST_MARGIN})"
    )
