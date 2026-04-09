"""Evaluate JAX baselines (sleep and random blue) on FsmRedCC4Env."""

import argparse
from statistics import mean, stdev

import jax
import jax.numpy as jnp

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.fsm_red_env import FsmRedCC4Env

EPISODE_LENGTH = 500


def run_sleep_episode(env, key):
    obs, state = env.reset(key)
    actions = {f"blue_{b}": jnp.int32(0) for b in range(NUM_BLUE_AGENTS)}
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(subkey, state, actions)
        total += float(rewards["blue_0"])
    return total


def run_random_episode(env, key):
    obs, state = env.reset(key)
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        key, act_key, step_key = jax.random.split(key, 3)
        actions = {
            f"blue_{b}": jax.random.randint(jax.random.fold_in(act_key, b), (), 0, BLUE_ALLOW_TRAFFIC_END)
            for b in range(NUM_BLUE_AGENTS)
        }
        obs, state, rewards, dones, info = env.step(step_key, state, actions)
        total += float(rewards["blue_0"])
    return total


def evaluate(policy, seed, max_eps):
    env = FsmRedCC4Env(num_steps=EPISODE_LENGTH)
    run_fn = run_sleep_episode if policy == "sleep" else run_random_episode

    episode_rewards = []
    for ep in range(max_eps):
        key = jax.random.PRNGKey(seed + ep if seed is not None else ep)
        episode_rewards.append(run_fn(env, key))

    print(f"policy:    {policy}")
    print(f"episodes:  {max_eps}")
    print(f"mean:      {mean(episode_rewards):.4f}")
    if len(episode_rewards) > 1:
        print(f"stdev:     {stdev(episode_rewards):.4f}")
    print(f"min:       {min(episode_rewards):.4f}")
    print(f"max:       {max(episode_rewards):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate JAX baselines on FsmRedCC4Env")
    parser.add_argument("--policy", choices=["sleep", "random"], default="sleep")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eps", type=int, default=10)
    args = parser.parse_args()
    evaluate(args.policy, args.seed, args.max_eps)
