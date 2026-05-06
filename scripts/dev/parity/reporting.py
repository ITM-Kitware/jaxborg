"""Console reports and plots for parity transfer checks."""

from __future__ import annotations

import json
from statistics import stdev

import numpy as np

from scripts.dev.parity.bootstrap import EXP_DIR
from scripts.dev.parity.stats import (
    ACTION_TYPE_NAMES,
    action_distribution,
    classify_action,
    l1_distribution_distance,
    tost_equivalence,
)


def print_per_agent_action_dist(all_actions_by_agent, label="JAXborg"):
    """Print per-agent action distribution table."""
    n_agents = len(all_actions_by_agent)
    if n_agents == 0:
        return
    header = f"{'Agent':<10}"
    for name in ACTION_TYPE_NAMES:
        header += f" {name:>8}"
    print(f"\nPer-Agent Action Distribution ({label}):")
    print(header)
    print("-" * len(header))
    for agent_idx in range(n_agents):
        dist = action_distribution(all_actions_by_agent[agent_idx])
        row = f"{'blue_' + str(agent_idx):<10}"
        for pct in dist:
            row += f" {pct * 100:7.1f}%"
        print(row)


def print_trajectory_summary(trajectory, label="JAXborg ep"):
    """Print compact trajectory table sampled at key steps and phase boundaries."""
    if not trajectory:
        return
    # Collect indices to show: every 50 steps + phase transitions + last step
    shown = set()
    prev_phase = -1
    for i, snap in enumerate(trajectory):
        if i % 50 == 0 or i == len(trajectory) - 1:
            shown.add(i)
        if snap.mission_phase != prev_phase:
            shown.add(i)
            prev_phase = snap.mission_phase

    phase_labels = {1: "MissionA", 2: "MissionB"}
    print(f"\nTrajectory ({label}):")
    print(f" {'Step':>4}  {'Phase':>5}  {'Reward':>7}  {'CumRew':>8}  {'Compromised(U/P)':>17}  {'RedSessions':>11}")
    for i in sorted(shown):
        s = trajectory[i]
        marker = ""
        if i > 0 and trajectory[i].mission_phase != trajectory[i - 1].mission_phase:
            marker = f"  <- {phase_labels.get(s.mission_phase, f'Phase{s.mission_phase}')}"
        print(
            f" {i:>4}  {s.mission_phase:>5}  {s.reward:>7.1f}  {s.cumulative_reward:>8.1f}"
            f"  {s.hosts_compromised_user:>8}/{s.hosts_compromised_priv:<8}"
            f"  {s.red_sessions_total:>11}{marker}"
        )


def print_comparison_report(jax_actions, jax_rewards, cyborg_actions, cyborg_rewards):
    jax_dist = action_distribution(jax_actions)
    cyborg_dist = action_distribution(cyborg_actions)

    print("\n" + "=" * 70)
    print("ACTION DISTRIBUTION COMPARISON")
    print("=" * 70)
    print(f"{'Type':<14} {'JAXborg':>8} {'CybORG':>8} {'Delta':>8}")
    print("-" * 40)
    for name, jp, cp in zip(ACTION_TYPE_NAMES, jax_dist, cyborg_dist):
        delta = jp - cp
        print(f"{name:<14} {jp * 100:7.1f}% {cp * 100:7.1f}% {delta * 100:+7.1f}%")
    l1 = l1_distribution_distance(jax_dist, cyborg_dist)
    print(f"\nL1 distribution distance (pooled): {l1:.3f}  (0 = identical, 2 = disjoint)")

    print("\n" + "=" * 70)
    print("EPISODE REWARD COMPARISON")
    print("=" * 70)
    print(f"{'':14} {'JAXborg':>10} {'CybORG':>10} {'Gap':>10}")
    print("-" * 46)
    for i, (jr, cr) in enumerate(zip(jax_rewards, cyborg_rewards)):
        print(f"Episode {i + 1:5d} {jr:10.1f} {cr:10.1f} {jr - cr:+10.1f}")

    jm, cm = jax_rewards.mean(), cyborg_rewards.mean()
    print("-" * 46)
    print(f"{'Mean':14} {jm:10.1f} {cm:10.1f} {jm - cm:+10.1f}")
    if len(jax_rewards) > 1:
        js, cs = stdev(jax_rewards.tolist()), stdev(cyborg_rewards.tolist())
        print(f"{'Stdev':14} {js:10.1f} {cs:10.1f}")


def save_reward_plot(jax_rewards, cyborg_rewards):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, len(jax_rewards) + 1)
    ax.bar(x - 0.2, jax_rewards, 0.35, label="JAXborg")
    ax.bar(x + 0.2, cyborg_rewards, 0.35, label="CybORG")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward (mean across agents)")
    ax.set_title("JAXborg vs CybORG Transfer Comparison")
    ax.legend()
    ax.set_xticks(x)
    fig.tight_layout()

    out = EXP_DIR / "transfer_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {out}")


def plot_action_distribution(actions, title, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.zeros(len(ACTION_TYPE_NAMES))
    for a in actions:
        counts[classify_action(a)] += 1
    counts = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ACTION_TYPE_NAMES, counts)
    ax.set_ylabel("Fraction of Actions")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    for i, v in enumerate(counts):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_training_curves(metrics_path, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, rewards, entropies = [], [], []
    with open(metrics_path) as f:
        for line in f:
            record = json.loads(line)
            steps.append(record["steps"])
            rewards.append(record["episode_reward_mean"])
            entropies.append(record["entropy"])
    steps, rewards, entropies = np.array(steps), np.array(rewards), np.array(entropies)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, rewards, label="JAX IPPO (masked)", linewidth=2)
    axes[0].set_xlabel("Environment Steps")
    axes[0].set_ylabel("Mean Per-Agent Episode Return")
    axes[0].set_title("Reward Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, entropies, label="JAX IPPO", linewidth=2)
    axes[1].set_xlabel("Environment Steps")
    axes[1].set_ylabel("Policy Entropy")
    axes[1].set_title("Entropy Over Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# --- TOST equivalence report (L4 verification, per Karten et al. 2026) ---


def print_tost_report(
    jax_rewards: np.ndarray,
    cyborg_rewards: np.ndarray,
    margin: float = 200.0,
    alpha: float = 0.05,
    paired: bool = False,
):
    """Print TOST equivalence report for cross-backend transfer validation."""
    result = tost_equivalence(jax_rewards, cyborg_rewards, margin, alpha, paired=paired)

    test_type = "PAIRED" if paired else "INDEPENDENT"
    print("\n" + "=" * 70)
    print(f"L4 CROSS-BACKEND EQUIVALENCE (TOST, {test_type})")
    print("=" * 70)
    print(f"  Test type:            {test_type.lower()} samples")
    print(f"  Margin (Delta):       +/-{result['margin']:.1f}")
    print(f"  Significance level:   alpha={alpha}")
    print(f"  JAXborg episodes:     {len(jax_rewards)}")
    print(f"  CybORG episodes:      {len(cyborg_rewards)}")
    print(f"  Mean diff (perf-ref): {result['mean_diff']:+.2f}")
    print(f"  {int((1 - alpha) * 100)}% CI for diff:      [{result['ci_lower']:+.2f}, {result['ci_upper']:+.2f}]")
    print(f"  p_upper (diff < +D):  {result['p_upper']:.4f}")
    print(f"  p_lower (diff > -D):  {result['p_lower']:.4f}")
    verdict = "EQUIVALENT" if result["equivalent"] else "NOT EQUIVALENT"
    print(f"  Verdict:              {verdict}")
    if not result["equivalent"]:
        print("  -> Policy transfer gap detected. Compare baselines or component rewards")
        print("     to distinguish simulation bugs from policy-environment interaction.")
    print("=" * 70)
    return result
