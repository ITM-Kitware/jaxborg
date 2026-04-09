"""Evaluate sleep and random baselines on CybORG."""

import argparse
from statistics import mean, stdev

import numpy as np
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

EPISODE_LENGTH = 500


def make_env(seed=None):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    return BlueFlatWrapper(env=cyborg)


def run_sleep_episode(env, _rng):
    env.reset()
    actions = {agent: 0 for agent in env.agents}
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        _, rewards, _, _, _ = env.step(actions)
        total += mean(rewards.values())
    return total


def run_random_episode(env, rng):
    _, info = env.reset()
    masks = {agent: info[agent]["action_mask"] for agent in env.agents}
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        actions = {agent: int(rng.choice(np.flatnonzero(masks[agent]))) for agent in env.agents}
        _, rewards, _, _, info = env.step(actions)
        masks = {agent: info[agent]["action_mask"] for agent in env.agents}
        total += mean(rewards.values())
    return total


def evaluate(policy, seed, max_eps):
    env = make_env(seed)
    rng = np.random.default_rng(seed)
    run_fn = run_sleep_episode if policy == "sleep" else run_random_episode

    episode_rewards = [run_fn(env, rng) for _ in range(max_eps)]
    print(f"policy:    {policy}")
    print(f"episodes:  {max_eps}")
    print(f"mean:      {mean(episode_rewards):.4f}")
    if len(episode_rewards) > 1:
        print(f"stdev:     {stdev(episode_rewards):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sleep/random baselines on CybORG")
    parser.add_argument("--policy", choices=["sleep", "random"], default="sleep")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eps", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.policy, args.seed, args.max_eps)
