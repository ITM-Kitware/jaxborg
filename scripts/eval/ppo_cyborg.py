"""Evaluate an SB3 MaskablePPO blue policy in pure CybORG."""

import argparse
from pathlib import Path
from statistics import mean, stdev

from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from sb3_contrib import MaskablePPO

EPISODE_LENGTH = 500


def make_env(seed=None):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    return BlueFlatWrapper(env=cyborg, pad_spaces=True)


def run_episode(model, env, deterministic):
    observations, info = env.reset()
    total = 0.0

    for _ in range(EPISODE_LENGTH):
        actions = {}
        for agent_name in env.agents:
            action, _ = model.predict(
                observations[agent_name],
                action_masks=info[agent_name]["action_mask"],
                deterministic=deterministic,
            )
            actions[agent_name] = int(action)

        observations, rewards, terminations, truncations, info = env.step(actions=actions)
        total += mean(rewards.values())
        if terminations.get("__all__", False) or truncations.get("__all__", False):
            break

    return total


def evaluate(model_path, episodes, seed, deterministic):
    model = MaskablePPO.load(Path(model_path))

    rewards = []
    for ep in range(episodes):
        episode_seed = seed + ep if seed is not None else ep
        env = make_env(episode_seed)
        reward = run_episode(model, env, deterministic)
        rewards.append(reward)
        print(f"Episode {ep + 1}: {reward:.4f}")

    print(f"\nepisodes:  {episodes}")
    print(f"mean:      {mean(rewards):.4f}")
    if len(rewards) > 1:
        print(f"stdev:     {stdev(rewards):.4f}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate SB3 MaskablePPO policy in CybORG")
    parser.add_argument("--model", required=True, help="Path to SB3 zip model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    evaluate(args.model, args.episodes, args.seed, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()
