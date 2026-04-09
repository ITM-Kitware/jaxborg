"""Evaluate CleanRL PPO on CC4 with action distributions and trajectory logging."""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from jaxborg.constants import BLUE_OBS_SIZE

NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]

SMALL_OBS_DIM = 92
SMALL_ACT_DIM = 82
LARGE_OBS_DIM = BLUE_OBS_SIZE
LARGE_ACT_DIM = 242

# Action classification built from wrapper labels at runtime
_ACTION_TYPE_CACHE = {}


def _build_action_type_map(env, agent_name):
    """Build action index → type mapping from wrapper labels."""
    labels = env.action_labels(agent_name)
    type_map = {}
    for idx, label in enumerate(labels):
        name = label.split("(")[0].strip() if "(" in label else label.strip()
        # Normalize to standard names
        if "Sleep" in name:
            type_map[idx] = "Sleep"
        elif "Monitor" in name:
            type_map[idx] = "Monitor"
        elif "Analyse" in name or "Analyze" in name:
            type_map[idx] = "Analyse"
        elif "Remove" in name:
            type_map[idx] = "Remove"
        elif "Restore" in name:
            type_map[idx] = "Restore"
        elif "Decoy" in name or "DecoyApache" in name or "DecoySSH" in name or "DeployDecoy" in name:
            type_map[idx] = "Decoy"
        elif "AllowTraffic" in name:
            type_map[idx] = "AllowTraffic"
        elif "BlockTraffic" in name:
            type_map[idx] = "BlockTraffic"
        else:
            type_map[idx] = name
    return type_map


def classify_action(env, agent_idx, act_idx):
    agent_name = AGENT_IDS[agent_idx]
    if agent_name not in _ACTION_TYPE_CACHE:
        _ACTION_TYPE_CACHE[agent_name] = _build_action_type_map(env, agent_name)
    return _ACTION_TYPE_CACHE[agent_name].get(act_idx, f"Unknown({act_idx})")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    agent_small = PPOAgent(SMALL_OBS_DIM, SMALL_ACT_DIM)
    agent_large = PPOAgent(LARGE_OBS_DIM, LARGE_ACT_DIM)
    agent_small.load_state_dict(torch.load(model_dir / f"model_small_{args.tag}.pt", weights_only=True))
    agent_large.load_state_dict(torch.load(model_dir / f"model_large_{args.tag}.pt", weights_only=True))
    agent_small.eval()
    agent_large.eval()

    env = make_cyborg_env()

    # Track per-agent action counts
    action_counts = {i: defaultdict(int) for i in range(NUM_AGENTS)}
    total_steps = {i: 0 for i in range(NUM_AGENTS)}
    episode_rewards = []
    # Track one trajectory for printing
    traj_ep = args.num_episodes - 1  # last episode

    t0 = time.time()
    for ep in range(args.num_episodes):
        obs, info = env.reset()
        ep_reward = 0
        traj_data = []

        for step in range(500):
            with torch.no_grad():
                obs_s = torch.zeros(4, SMALL_OBS_DIM)
                mask_s = torch.zeros(4, SMALL_ACT_DIM)
                for i in range(4):
                    obs_s[i] = torch.from_numpy(obs[AGENT_IDS[i]].astype(np.float32))
                    mask_s[i] = torch.from_numpy(np.array(info[AGENT_IDS[i]]["action_mask"], dtype=np.float32))
                acts_s = agent_small.get_action(obs_s, mask_s, deterministic=args.deterministic)

                obs_l = torch.from_numpy(obs[AGENT_IDS[4]].astype(np.float32)).unsqueeze(0)
                mask_l = torch.from_numpy(np.array(info[AGENT_IDS[4]]["action_mask"], dtype=np.float32)).unsqueeze(0)
                act_l = agent_large.get_action(obs_l, mask_l, deterministic=args.deterministic)

            actions = {}
            for i in range(4):
                act_idx = int(acts_s[i].item())
                actions[AGENT_IDS[i]] = act_idx
                act_type = classify_action(env, i, act_idx)
                action_counts[i][act_type] += 1
                total_steps[i] += 1

            act_idx_l = int(act_l[0].item())
            actions[AGENT_IDS[4]] = act_idx_l
            act_type_l = classify_action(env, 4, act_idx_l)
            action_counts[4][act_type_l] += 1
            total_steps[4] += 1

            obs, rew, term, trunc, info = env.step(actions)
            step_rew = rew[AGENT_IDS[0]]
            ep_reward += step_rew

            if ep == traj_ep and step % 50 == 0:
                traj_data.append((step, ep_reward, step_rew))

            if any(term.values()) or any(trunc.values()):
                break

        episode_rewards.append(ep_reward)
        print(f"  Episode {ep + 1}: reward={ep_reward:.1f}")

        if ep == traj_ep:
            traj_data.append((step, ep_reward, step_rew))

    elapsed = time.time() - t0
    mean_rew = np.mean(episode_rewards)
    std_rew = np.std(episode_rewards)

    print(f"\n{'=' * 70}")
    print(f"CybORG v9 Evaluation ({args.num_episodes} episodes, {'deterministic' if args.deterministic else 'stochastic'})")
    print(f"{'=' * 70}")
    print(f"  Mean reward: {mean_rew:.1f} +/- {std_rew:.1f}")
    print(f"  Min: {np.min(episode_rewards):.1f}  Max: {np.max(episode_rewards):.1f}")
    print(f"  Wall time: {elapsed:.1f}s")

    # Action distribution
    action_types = ["Sleep", "Monitor", "Analyse", "Remove", "Restore", "Decoy", "BlockTraffic", "AllowTraffic"]
    print(f"\nPer-Agent Action Distribution:")
    print(f"{'Agent':14s}", end="")
    for at in action_types:
        print(f"{at:>10s}", end="")
    print()
    print("-" * 94)
    for i in range(NUM_AGENTS):
        print(f"{'blue_' + str(i):14s}", end="")
        for at in action_types:
            pct = 100.0 * action_counts[i][at] / max(total_steps[i], 1)
            print(f"{pct:9.1f}%", end="")
        print()

    # Print last trajectory
    print(f"\nTrajectory (episode {traj_ep + 1}):")
    print(f"{'Step':>6s}  {'CumRew':>8s}  {'StepRew':>8s}")
    for step, cum, sr in traj_data:
        print(f"{step:6d}  {cum:8.1f}  {sr:8.1f}")

    # Per-episode rewards
    print(f"\nPer-episode rewards:")
    for i, r in enumerate(episode_rewards):
        print(f"  Episode {i + 1:3d}: {r:.1f}")


if __name__ == "__main__":
    main()
