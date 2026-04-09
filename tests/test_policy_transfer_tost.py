"""L4 cross-backend policy transfer test using TOST equivalence.

Loads a trained JAXborg checkpoint, evaluates it in both JaxBorg (FsmRedCC4Env)
and CybORG (BlueFlatWrapper), and applies the Two One-Sided Tests (TOST)
procedure to confirm statistical equivalence of episode rewards.

Set JAXBORG_CKPT_PATH to the checkpoint path to run.
"""

import os
import pickle
from pathlib import Path
from statistics import mean, stdev

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import ttest_1samp

CKPT_PATH = os.environ.get("JAXBORG_CKPT_PATH", "")
NUM_EPISODES = int(os.environ.get("JAXBORG_TOST_EPISODES", "50"))
TOST_MARGIN = float(os.environ.get("JAXBORG_TOST_MARGIN", "5.0"))
TOST_ALPHA = float(os.environ.get("JAXBORG_TOST_ALPHA", "0.05"))


def _load_checkpoint(path):
    """Load checkpoint and return (policy, params, kind)."""
    from jaxborg.policy import ActorCritic, LegacyActor

    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)

    nested_params = ckpt["params"].get("params", {})
    if "actor_head" in nested_params:
        policy = ActorCritic(
            action_dim=ckpt["action_dim"],
            hidden_dim=ckpt["hidden_dim"],
            activation=ckpt["activation"],
        )
        return policy, ckpt["params"], "current"

    if "Dense_0" in nested_params:
        policy = LegacyActor(
            action_dim=ckpt["action_dim"],
            hidden_dim=ckpt["hidden_dim"],
            activation=ckpt["activation"],
        )
        return policy, ckpt["params"], "legacy"

    raise ValueError(f"Unknown checkpoint format: {sorted(nested_params.keys())}")


def _rollout_jaxborg(policy, params, kind, seed, num_episodes, num_steps=500):
    """Run episodes in JaxBorg and return per-episode cumulative rewards."""
    from jaxborg.actions.masking import compute_blue_action_mask
    from jaxborg.constants import NUM_BLUE_AGENTS
    from jaxborg.fsm_red_env import FsmRedCC4Env
    from jaxborg.policy import ActorCritic

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
                mask = compute_blue_action_mask(state, env.const, b)
                obs_jax = jnp.array(obs).reshape(1, -1)
                mask_jax = jnp.array(mask).reshape(1, -1)

                if kind == "current":
                    dist = policy.apply(params, obs_jax, mask_jax, method=ActorCritic.actor)
                else:
                    dist = policy.apply(params, obs_jax, mask_jax)

                action = int(dist.mode()[0])
                actions[agent_name] = action

            key, step_key = jax.random.split(key)
            obs_dict, state, reward_dict, done_dict, info = env.step(step_key, state, actions)
            ep_reward += float(reward_dict.get("blue_0", 0.0))

            if done_dict.get("__all__", False):
                break

        rewards.append(ep_reward)

    return np.array(rewards)


def _rollout_cyborg(policy, params, kind, seed, num_episodes, num_steps=500):
    """Run episodes in CybORG and return per-episode cumulative rewards."""
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
    from jaxborg.policy import ActorCritic

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
        ep_reward = 0.0

        for _step in range(num_steps):
            actions = {}
            for agent_name in env.possible_agents:
                obs = observations.get(agent_name)
                if obs is None:
                    actions[agent_name] = env.action_space(agent_name).sample()
                    continue

                obs_jax = jnp.array(obs).reshape(1, -1)
                mask = env.get_action_space(agent_name).get("mask")
                if mask is None:
                    mask_jax = jnp.ones((1, BLUE_ALLOW_TRAFFIC_END))
                else:
                    mask_jax = jnp.array(mask).reshape(1, -1)

                if kind == "current":
                    dist = policy.apply(params, obs_jax, mask_jax, method=ActorCritic.actor)
                else:
                    dist = policy.apply(params, obs_jax, mask_jax)

                cyborg_action = int(dist.mode()[0])
                actions[agent_name] = cyborg_action

            observations, rews, terminated, truncated, info = env.step(actions)
            ep_reward += sum(rews.values()) / max(len(rews), 1)

            if any(terminated.values()):
                break

        rewards.append(ep_reward)

    return np.array(rewards)


def tost_equivalence(diffs, margin, alpha=0.05):
    """Two One-Sided Tests for equivalence within +/- margin.

    Returns (equivalent: bool, p_value: float, mean_diff: float, std_diff: float).
    """
    _, p_lower = ttest_1samp(diffs, -margin, alternative="greater")
    _, p_upper = ttest_1samp(diffs, margin, alternative="less")
    p_value = max(p_lower, p_upper)
    return p_value < alpha, p_value, float(np.mean(diffs)), float(np.std(diffs))


@pytest.mark.slow
def test_policy_transfer_tost():
    """Train in JaxBorg, evaluate in CybORG, confirm equivalence via TOST."""
    if not CKPT_PATH:
        pytest.skip("No checkpoint: set JAXBORG_CKPT_PATH env var")

    policy, params, kind = _load_checkpoint(CKPT_PATH)

    jax_rewards = _rollout_jaxborg(policy, params, kind, seed=42, num_episodes=NUM_EPISODES)
    cyborg_rewards = _rollout_cyborg(policy, params, kind, seed=42, num_episodes=NUM_EPISODES)

    diffs = jax_rewards - cyborg_rewards
    equivalent, p_value, mean_diff, std_diff = tost_equivalence(diffs, TOST_MARGIN, TOST_ALPHA)

    print("\n=== TOST Policy Transfer Results ===")
    print(f"JaxBorg:  mean={mean(jax_rewards):.2f} std={stdev(jax_rewards):.2f}")
    print(f"CybORG:   mean={mean(cyborg_rewards):.2f} std={stdev(cyborg_rewards):.2f}")
    print(f"Diff:     mean={mean_diff:.2f} std={std_diff:.2f}")
    print(f"TOST:     margin={TOST_MARGIN}, alpha={TOST_ALPHA}, p={p_value:.4f}")
    print(f"Result:   {'EQUIVALENT' if equivalent else 'NOT EQUIVALENT'}")

    assert equivalent, (
        f"TOST failed: mean_diff={mean_diff:.2f}, std={std_diff:.2f}, "
        f"p={p_value:.4f} >= alpha={TOST_ALPHA} (margin={TOST_MARGIN})"
    )
