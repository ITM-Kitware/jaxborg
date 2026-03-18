"""Evaluate a trained CleanRL PPO model on CC4.

Loads the saved model weights and runs N episodes to get mean/std reward.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()
NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]

SMALL_OBS_DIM = 92
SMALL_ACT_DIM = 82
LARGE_OBS_DIM = 210
LARGE_ACT_DIM = 242


class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.features = nn.Sequential(*layers)
        self.actor = nn.Linear(in_dim, act_dim)
        self.critic = nn.Linear(in_dim, 1)

    def get_action(self, obs, action_mask, deterministic=False):
        features = self.features(obs)
        logits = self.actor(features)
        logits = logits + (action_mask.float() - 1.0) * 1e8
        if deterministic:
            return logits.argmax(dim=-1)
        dist = Categorical(logits=logits)
        return dist.sample()


def make_cyborg_env():
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import EnterpriseMAE
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg)
    return EnterpriseMAE(cyborg)


def evaluate(model_dir, num_episodes=50, deterministic=False, tag="default"):
    agent_small = PPOAgent(SMALL_OBS_DIM, SMALL_ACT_DIM)
    agent_large = PPOAgent(LARGE_OBS_DIM, LARGE_ACT_DIM)

    small_path = model_dir / f"model_small_{tag}.pt"
    large_path = model_dir / f"model_large_{tag}.pt"
    agent_small.load_state_dict(torch.load(small_path, weights_only=True))
    agent_large.load_state_dict(torch.load(large_path, weights_only=True))
    agent_small.eval()
    agent_large.eval()

    env = make_cyborg_env()
    episode_rewards = []
    episode_lengths = []

    t0 = time.time()
    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0
        for step in range(500):
            with torch.no_grad():
                # Small agents
                obs_s = torch.zeros(4, SMALL_OBS_DIM)
                mask_s = torch.zeros(4, SMALL_ACT_DIM)
                for i in range(4):
                    obs_s[i] = torch.from_numpy(obs[AGENT_IDS[i]].astype(np.float32))
                    mask_s[i] = torch.from_numpy(np.array(info[AGENT_IDS[i]]["action_mask"], dtype=np.float32))
                acts_s = agent_small.get_action(obs_s, mask_s, deterministic=deterministic)

                obs_l = torch.from_numpy(obs[AGENT_IDS[4]].astype(np.float32)).unsqueeze(0)
                mask_l = torch.from_numpy(np.array(info[AGENT_IDS[4]]["action_mask"], dtype=np.float32)).unsqueeze(0)
                act_l = agent_large.get_action(obs_l, mask_l, deterministic=deterministic)

            actions = {}
            for i in range(4):
                actions[AGENT_IDS[i]] = int(acts_s[i].item())
            actions[AGENT_IDS[4]] = int(act_l[0].item())

            obs, rew, term, trunc, info = env.step(actions)
            ep_reward += rew[AGENT_IDS[0]]

            if any(term.values()) or any(trunc.values()):
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(step + 1)

        if (ep + 1) % 10 == 0:
            print(
                f"  Episode {ep + 1}/{num_episodes}: "
                f"mean={np.mean(episode_rewards):.1f} +/- {np.std(episode_rewards):.1f}, "
                f"last={ep_reward:.1f}"
            )

    elapsed = time.time() - t0
    mean_rew = np.mean(episode_rewards)
    std_rew = np.std(episode_rewards)
    mean_len = np.mean(episode_lengths)

    print(f"\n{'=' * 60}")
    print(f"Evaluation Results ({num_episodes} episodes)")
    print(f"{'=' * 60}")
    print(f"  Mean reward: {mean_rew:.1f} +/- {std_rew:.1f}")
    print(f"  Min reward:  {np.min(episode_rewards):.1f}")
    print(f"  Max reward:  {np.max(episode_rewards):.1f}")
    print(f"  Mean length: {mean_len:.1f}")
    print(f"  Wall time:   {elapsed:.1f}s")
    print(f"  Mode:        {'deterministic' if deterministic else 'stochastic'}")
    print(f"{'=' * 60}")

    # Save results
    results = {
        "num_episodes": num_episodes,
        "mean_reward": float(mean_rew),
        "std_reward": float(std_rew),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(mean_len),
        "deterministic": deterministic,
        "all_rewards": [float(r) for r in episode_rewards],
    }
    results_path = model_dir / f"eval_results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")

    return mean_rew, std_rew


def main():
    parser = argparse.ArgumentParser(description="Evaluate CleanRL PPO on CC4")
    parser.add_argument("--model-dir", type=str, default=str(EXP_DIR / "cleanrl_ppo"))
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()

    evaluate(
        model_dir=Path(args.model_dir),
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
