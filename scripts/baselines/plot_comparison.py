"""Plot reward curves from SB3 TensorBoard logs and JAX IPPO metrics."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jax_metrics(path: Path):
    steps, rewards = [], []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            steps.append(record["steps"])
            rewards.append(record["episode_reward_mean"])
    return np.array(steps), np.array(rewards)


def load_tensorboard_logs(log_dir: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed, skipping SB3 logs")
        return None, None

    run_dirs = sorted(log_dir.glob("PPO_*"))
    if not run_dirs:
        run_dirs = sorted(log_dir.glob("*"))
    if not run_dirs:
        print(f"No TensorBoard runs found in {log_dir}")
        return None, None

    ea = EventAccumulator(str(run_dirs[0]))
    ea.Reload()

    tag = "rollout/ep_rew_mean"
    if tag not in ea.Tags().get("scalars", []):
        available = ea.Tags().get("scalars", [])
        print(f"Tag '{tag}' not found. Available: {available}")
        return None, None

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    rewards = np.array([e.value for e in events])
    return steps, rewards


def plot(jax_path, sb3_log_dir, output, num_agents=5):
    fig, ax = plt.subplots(figsize=(10, 6))

    if jax_path and jax_path.exists():
        steps, rewards = load_jax_metrics(jax_path)
        ax.plot(steps, rewards, label="JAX IPPO (per agent)", linewidth=2)

    if sb3_log_dir and sb3_log_dir.exists():
        steps, rewards = load_tensorboard_logs(sb3_log_dir)
        if steps is not None:
            rewards_per_agent = rewards / num_agents
            ax.plot(steps, rewards_per_agent, label="SB3 PPO (per agent)", linewidth=2, linestyle="--")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Per-Agent Episode Return")
    ax.set_title("CC4 Blue Agent Training: JAX IPPO vs SB3 PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    exp_dir = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()

    parser = argparse.ArgumentParser(description="Plot reward curve comparison")
    parser.add_argument("--jax-metrics", type=Path, default=exp_dir / "ippo_cc4" / "metrics.jsonl")
    parser.add_argument("--sb3-logs", type=Path, default=Path("logs/ppo"))
    parser.add_argument("--output", type=Path, default=exp_dir / "comparison.png")
    args = parser.parse_args()

    plot(args.jax_metrics, args.sb3_logs, args.output)
