import argparse
from statistics import mean, stdev

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


def run_episode(env):
    env.reset()
    actions = {agent: 0 for agent in env.agents}
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        _, rewards, _, _, _ = env.step(actions)
        total += mean(rewards.values())
    return total


def evaluate(seed, max_eps):
    env = make_env(seed)
    episode_rewards = [run_episode(env) for _ in range(max_eps)]
    print(f"episodes:  {max_eps}")
    print(f"mean:      {mean(episode_rewards):.4f}")
    if len(episode_rewards) > 1:
        print(f"stdev:     {stdev(episode_rewards):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sleep baseline on CybORG")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eps", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.seed, args.max_eps)
