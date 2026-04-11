"""Evaluate a trained CleanRL PPO model on CC4.

Loads the saved model weights and runs N episodes to get mean/std reward,
action distributions, per-phase breakdowns, and trajectory summaries.
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

from jaxborg.constants import BLUE_OBS_SIZE

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()
NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]

OBS_DIM = BLUE_OBS_SIZE  # 210
ACT_DIM = 242

ACTION_TYPE_NAMES = [
    "Sleep",
    "Monitor",
    "Analyse",
    "Remove",
    "Restore",
    "Decoy",
    "BlockTraf",
    "AllowTraf",
]

# CybORG action label prefixes -> action type
CYBORG_ACTION_PREFIX_MAP = {
    "Sleep": "Sleep",
    "Monitor": "Monitor",
    "Analyse": "Analyse",
    "Remove": "Remove",
    "Restore": "Restore",
    "DeployDecoy": "Decoy",
    "BlockTraffic": "BlockTraf",
    "AllowTraffic": "AllowTraf",
}


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


def classify_action_label(label):
    """Classify a CybORG action label string into an action type index."""
    for prefix, name in CYBORG_ACTION_PREFIX_MAP.items():
        if prefix in label:
            return ACTION_TYPE_NAMES.index(name)
    return 0  # Unknown -> Sleep


def action_distribution(type_indices):
    """Compute action type distribution from a list of type indices."""
    counts = np.zeros(len(ACTION_TYPE_NAMES))
    for idx in type_indices:
        counts[idx] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


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


def run_episode(agent, env, deterministic):
    """Run one episode, collecting actions, rewards, and trajectory data."""
    obs, info = env.reset()

    ep_reward = 0.0
    # Per-agent action type indices: [agent_idx][step] = type_idx
    per_agent_types = [[] for _ in range(NUM_AGENTS)]
    # Per-phase action type indices: phase -> list of type_idx
    per_phase_types = {0: [], 1: [], 2: []}
    # Trajectory snapshots
    trajectory = []
    cum_reward = 0.0

    # Get mission phase from underlying CybORG state
    controller = env.env.environment_controller

    for step in range(500):
        phase = controller.state.mission_phase
        actions = {}
        for i in range(NUM_AGENTS):
            agent_id = AGENT_IDS[i]
            with torch.no_grad():
                raw_obs = obs[agent_id].astype(np.float32)
                raw_mask = np.array(info[agent_id]["action_mask"], dtype=np.float32)
                # Zero-pad to unified dims
                o = torch.zeros(1, OBS_DIM)
                o[0, : len(raw_obs)] = torch.from_numpy(raw_obs)
                m = torch.zeros(1, ACT_DIM)
                m[0, : len(raw_mask)] = torch.from_numpy(raw_mask)
                act = agent.get_action(o, m, deterministic=deterministic).item()

            actions[agent_id] = act

            # Classify action
            labels = env.action_labels(agent_id)
            label = labels[act] if act < len(labels) else "Sleep"
            type_idx = classify_action_label(label)
            per_agent_types[i].append(type_idx)
            per_phase_types[phase].append(type_idx)

        obs, rew, term, trunc, info = env.step(actions)
        step_reward = rew[AGENT_IDS[0]]
        ep_reward += step_reward
        cum_reward += step_reward

        trajectory.append(
            {
                "step": step,
                "reward": step_reward,
                "cum_reward": cum_reward,
                "phase": phase,
            }
        )

        if any(term.values()) or any(trunc.values()):
            break

    return ep_reward, per_agent_types, per_phase_types, trajectory


def print_action_dist_table(per_agent_types, label="CybORG"):
    """Print per-agent action distribution table."""
    header = f"{'Agent':<10}"
    for name in ACTION_TYPE_NAMES:
        header += f" {name:>8}"
    print(f"\nPer-Agent Action Distribution ({label}):")
    print(header)
    print("-" * len(header))
    for i in range(NUM_AGENTS):
        dist = action_distribution(per_agent_types[i])
        row = f"{'blue_' + str(i):<10}"
        for pct in dist:
            row += f" {pct * 100:7.1f}%"
        print(row)

    # Pooled
    all_types = [t for agent in per_agent_types for t in agent]
    dist = action_distribution(all_types)
    print("-" * len(header))
    row = f"{'POOLED':<10}"
    for pct in dist:
        row += f" {pct * 100:7.1f}%"
    print(row)


def print_phase_dist_table(per_phase_types):
    """Print action distribution broken down by mission phase."""
    phase_names = {0: "Phase0", 1: "MissionA", 2: "MissionB"}
    header = f"{'Phase':<10}"
    for name in ACTION_TYPE_NAMES:
        header += f" {name:>8}"
    header += f" {'N':>8}"
    print("\nPer-Phase Action Distribution:")
    print(header)
    print("-" * len(header))
    for phase in [0, 1, 2]:
        types = per_phase_types[phase]
        if not types:
            continue
        dist = action_distribution(types)
        row = f"{phase_names[phase]:<10}"
        for pct in dist:
            row += f" {pct * 100:7.1f}%"
        row += f" {len(types):>8}"
        print(row)


def print_trajectory_summary(trajectory, label="CybORG ep"):
    """Print trajectory sampled at key steps and phase boundaries."""
    if not trajectory:
        return
    shown = set()
    prev_phase = -1
    for i, snap in enumerate(trajectory):
        if i % 50 == 0 or i == len(trajectory) - 1:
            shown.add(i)
        if snap["phase"] != prev_phase:
            shown.add(i)
            prev_phase = snap["phase"]

    phase_labels = {1: "MissionA", 2: "MissionB"}
    print(f"\nTrajectory ({label}):")
    print(f" {'Step':>4}  {'Phase':>5}  {'Reward':>7}  {'CumRew':>8}")
    for i in sorted(shown):
        s = trajectory[i]
        marker = ""
        if i > 0 and trajectory[i]["phase"] != trajectory[i - 1]["phase"]:
            phase_name = phase_labels.get(s["phase"], "Phase" + str(s["phase"]))
            marker = f"  <- {phase_name}"
        print(f" {i:>4}  {s['phase']:>5}  {s['reward']:>7.1f}  {s['cum_reward']:>8.1f}{marker}")


def evaluate(model_dir, num_episodes=50, deterministic=False, tag="default", verbose=False):
    agent = PPOAgent(OBS_DIM, ACT_DIM)

    model_path = model_dir / f"model_{tag}.pt"
    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    env = make_cyborg_env()
    episode_rewards = []
    episode_lengths = []
    # Accumulate across episodes
    all_per_agent_types = [[] for _ in range(NUM_AGENTS)]
    all_per_phase_types = {0: [], 1: [], 2: []}

    t0 = time.time()
    last_trajectory = None
    for ep in range(num_episodes):
        ep_reward, per_agent_types, per_phase_types, trajectory = run_episode(agent, env, deterministic)
        episode_rewards.append(ep_reward)
        episode_lengths.append(len(trajectory))
        last_trajectory = trajectory

        for i in range(NUM_AGENTS):
            all_per_agent_types[i].extend(per_agent_types[i])
        for phase in [0, 1, 2]:
            all_per_phase_types[phase].extend(per_phase_types[phase])

        if verbose:
            print(f"  Episode {ep + 1}: reward={ep_reward:.1f}")
        elif (ep + 1) % 10 == 0 or ep == num_episodes - 1:
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

    # Action distributions
    print_action_dist_table(all_per_agent_types, label="CybORG")
    print_phase_dist_table(all_per_phase_types)
    print_trajectory_summary(last_trajectory, label=f"CybORG ep {num_episodes}")

    if verbose:
        print("\nPer-episode rewards:")
        for i, r in enumerate(episode_rewards):
            print(f"  Episode {i + 1:3d}: {r:.1f}")

    # Save results
    # Build serializable action dist
    pooled = [t for agent in all_per_agent_types for t in agent]
    pooled_dist = action_distribution(pooled).tolist()
    per_agent_dists = [action_distribution(all_per_agent_types[i]).tolist() for i in range(NUM_AGENTS)]
    per_phase_dists = {}
    for phase in [0, 1, 2]:
        if all_per_phase_types[phase]:
            per_phase_dists[str(phase)] = action_distribution(all_per_phase_types[phase]).tolist()

    results = {
        "num_episodes": num_episodes,
        "mean_reward": float(mean_rew),
        "std_reward": float(std_rew),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(mean_len),
        "deterministic": deterministic,
        "all_rewards": [float(r) for r in episode_rewards],
        "action_type_names": ACTION_TYPE_NAMES,
        "pooled_action_dist": pooled_dist,
        "per_agent_action_dist": per_agent_dists,
        "per_phase_action_dist": per_phase_dists,
    }
    results_path = model_dir / f"eval_results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    return mean_rew, std_rew


def main():
    parser = argparse.ArgumentParser(description="Evaluate CleanRL PPO on CC4")
    parser.add_argument("--model-dir", type=str, default=str(EXP_DIR / "cleanrl_ppo"))
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print per-episode rewards")
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()

    evaluate(
        model_dir=Path(args.model_dir),
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        tag=args.tag,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
