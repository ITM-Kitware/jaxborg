"""Evaluate JAXborg-trained policy: JAXborg rollout, CybORG transfer, baselines, diagnostics."""

import argparse
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

import jax
import jax.numpy as jnp
import numpy as np
from train_ippo_cc4 import ActorCritic

from jaxborg.actions.encoding import (
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_ANALYSE_START,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_DECOY_START,
    BLUE_MONITOR,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
)
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import (
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
)
from jaxborg.fsm_red_env import FsmRedCC4Env
from jaxborg.topology import build_const_from_cyborg
from jaxborg.translate import (
    build_mappings_from_cyborg,
    describe_blue_action,
    jax_blue_to_cyborg,
)

EXP_DIR = Path(__file__).resolve().parent.parent / "jaxborg-exp"

ACTION_TYPE_NAMES = [
    "Sleep",
    "Monitor",
    "Analyse",
    "Remove",
    "Restore",
    "Decoy",
    "BlockTraffic",
    "AllowTraffic",
]

ACTION_TYPE_RANGES = [
    (BLUE_SLEEP, BLUE_SLEEP + 1),
    (BLUE_MONITOR, BLUE_MONITOR + 1),
    (BLUE_ANALYSE_START, BLUE_ANALYSE_START + GLOBAL_MAX_HOSTS),
    (BLUE_REMOVE_START, BLUE_REMOVE_START + GLOBAL_MAX_HOSTS),
    (BLUE_RESTORE_START, BLUE_RESTORE_START + GLOBAL_MAX_HOSTS),
    (BLUE_DECOY_START, BLUE_BLOCK_TRAFFIC_START),
    (BLUE_BLOCK_TRAFFIC_START, BLUE_ALLOW_TRAFFIC_START),
    (BLUE_ALLOW_TRAFFIC_START, BLUE_ALLOW_TRAFFIC_END),
]


def classify_action(action_idx: int) -> int:
    for i, (start, end) in enumerate(ACTION_TYPE_RANGES):
        if start <= action_idx < end:
            return i
    return 0


def action_distribution(actions):
    counts = np.zeros(len(ACTION_TYPE_NAMES))
    for a in actions:
        counts[classify_action(int(a))] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


@dataclass
class StepSnapshot:
    reward: float
    cumulative_reward: float
    hosts_compromised_user: int
    hosts_compromised_priv: int
    red_sessions_total: int
    mission_phase: int


@dataclass
class EpisodeResult:
    actions_by_agent: list = field(default_factory=list)  # [agent_idx][step] = action_id
    rewards: list = field(default_factory=list)  # per-step rewards
    cumulative_reward: float = 0.0
    trajectory: list = field(default_factory=list)  # list[StepSnapshot]


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


def load_checkpoint(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    network = ActorCritic(
        action_dim=ckpt["action_dim"],
        hidden_dim=ckpt["hidden_dim"],
        activation=ckpt["activation"],
    )
    return network, ckpt["params"]


def make_cyborg_env(seed=42):
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    return BlueFlatWrapper(env=cyborg, pad_spaces=True)


# --- Core rollout functions ---


def rollout_jaxborg(network, params, num_episodes=3, deterministic=False):
    env = FsmRedCC4Env(num_steps=500)
    all_actions = []
    episode_rewards = []
    episode_results = []

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        key = jax.random.PRNGKey(ep * 100)
        obs, env_state = env.reset(key)

        ep_reward = np.zeros(NUM_BLUE_AGENTS)
        ep_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]
        ep_step_rewards = []
        ep_trajectory = []
        cum_reward = 0.0

        for step in range(500):
            key, step_key = jax.random.split(key)
            act_keys = jax.random.split(key, NUM_BLUE_AGENTS)

            actions = {}
            for agent_idx in range(NUM_BLUE_AGENTS):
                agent = f"blue_{agent_idx}"
                avail = compute_blue_action_mask(env_state.const, agent_idx, env_state.state)
                pi, _ = network.apply(params, obs[agent], avail)

                if deterministic:
                    action = jnp.argmax(pi.logits)
                else:
                    action = pi.sample(seed=act_keys[agent_idx])

                actions[agent] = action
                ep_actions_by_agent[agent_idx].append(int(action))

            obs, env_state, rewards, dones, _ = env.step(step_key, env_state, actions)
            step_reward = float(np.mean([float(rewards[f"blue_{i}"]) for i in range(NUM_BLUE_AGENTS)]))
            cum_reward += step_reward
            ep_step_rewards.append(step_reward)
            for i in range(NUM_BLUE_AGENTS):
                ep_reward[i] += float(rewards[f"blue_{i}"])

            # Extract trajectory snapshot from state
            st = env_state.state
            active = np.array(env_state.const.host_active, dtype=bool)
            compromised = np.array(st.host_compromised)
            ep_trajectory.append(
                StepSnapshot(
                    reward=step_reward,
                    cumulative_reward=cum_reward,
                    hosts_compromised_user=int(np.sum((compromised == COMPROMISE_USER) & active)),
                    hosts_compromised_priv=int(np.sum((compromised == COMPROMISE_PRIVILEGED) & active)),
                    red_sessions_total=int(np.sum(np.array(st.red_sessions)[:NUM_RED_AGENTS])),
                    mission_phase=int(st.mission_phase),
                )
            )

            if dones["__all__"]:
                break

        elapsed = time.perf_counter() - t0
        total = ep_reward.mean()
        print(f"  JAXborg ep {ep + 1}: reward={total:.1f} ({elapsed:.1f}s)")

        # Flatten per-agent actions for backward compat
        flat_actions = [a for step_actions in zip(*ep_actions_by_agent) for a in step_actions]
        all_actions.extend(flat_actions)
        episode_rewards.append(total)
        episode_results.append(
            EpisodeResult(
                actions_by_agent=ep_actions_by_agent,
                rewards=ep_step_rewards,
                cumulative_reward=cum_reward,
                trajectory=ep_trajectory,
            )
        )

    return np.array(all_actions), np.array(episode_rewards), episode_results


def rollout_cyborg(network, params, num_episodes=3, deterministic=False, seed=0):
    all_actions = []
    episode_rewards = []
    all_actions_by_agent = [[] for _ in range(NUM_BLUE_AGENTS)]

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        env = make_cyborg_env(seed=seed + ep * 100)
        observations, _ = env.reset()
        inner = env.env
        const = build_const_from_cyborg(inner)
        mappings = build_mappings_from_cyborg(inner)

        rng = jax.random.PRNGKey(seed + ep * 100)
        total = 0.0
        ep_actions = []

        for _ in range(500):
            actions = {}
            for agent_idx, agent_name in enumerate(env.agents):
                obs_jax = jnp.array(observations[agent_name], dtype=jnp.float32)
                mask = compute_blue_action_mask(const, agent_idx)
                pi, _ = network.apply(params, obs_jax, mask)

                if deterministic:
                    action_idx = int(jnp.argmax(pi.logits))
                else:
                    rng, _rng = jax.random.split(rng)
                    action_idx = int(pi.sample(seed=_rng))

                cyborg_action = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
                actions[agent_name] = cyborg_action
                ep_actions.append(action_idx)
                all_actions_by_agent[agent_idx].append(action_idx)

            observations, rewards, _, _, _ = env.step(actions=actions)
            total += mean(rewards.values())

        elapsed = time.perf_counter() - t0
        print(f"  CybORG  ep {ep + 1}: reward={total:.1f} ({elapsed:.1f}s)")
        all_actions.extend(ep_actions)
        episode_rewards.append(total)

    return np.array(all_actions), np.array(episode_rewards), all_actions_by_agent


# --- Report / plotting ---


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


# --- Baseline / diagnostic functions ---


def run_sleep_baseline(episodes=5):
    from CybORG.Simulator.Actions import Sleep

    totals = []
    for ep in range(episodes):
        env = make_cyborg_env(seed=ep)
        env.reset()
        total = 0.0
        for _ in range(500):
            actions = {a: Sleep() for a in env.agents}
            _, rewards, _, _, _ = env.step(actions=actions)
            total += mean(rewards.values())
        totals.append(total)
    return mean(totals)


def run_random_baseline(episodes=5, seed=42):
    rng = np.random.default_rng(seed)
    totals = []
    for ep in range(episodes):
        env = make_cyborg_env(seed=seed + ep)
        env.reset()
        inner = env.env
        const = build_const_from_cyborg(inner)
        mappings = build_mappings_from_cyborg(inner)
        total = 0.0
        for _ in range(500):
            actions = {}
            for agent_idx, agent_name in enumerate(env.agents):
                mask = np.array(compute_blue_action_mask(const, agent_idx), dtype=bool)
                valid = np.where(mask)[0]
                action_idx = int(rng.choice(valid))
                actions[agent_name] = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
            _, rewards, _, _, _ = env.step(actions=actions)
            total += mean(rewards.values())
        totals.append(total)
    return mean(totals)


def run_verbose_trace(network, params, steps=20, seed=42):
    env = make_cyborg_env(seed=seed)
    observations, _ = env.reset()
    inner = env.env
    const = build_const_from_cyborg(inner)
    mappings = build_mappings_from_cyborg(inner)

    from jaxborg.actions.encoding import BLUE_ANALYSE_END
    from jaxborg.topology import BLUE_AGENT_SUBNETS, SUBNET_IDS

    print("\nMASK VALIDATION (step 0):")
    for agent_idx, agent_name in enumerate(env.agents):
        mask = np.array(compute_blue_action_mask(const, agent_idx), dtype=bool)
        valid_indices = np.where(mask)[0]
        agent_subnets = BLUE_AGENT_SUBNETS[agent_idx]
        agent_subnet_ids = [SUBNET_IDS[s] for s in agent_subnets]

        valid_analyse = valid_indices[(valid_indices >= BLUE_ANALYSE_START) & (valid_indices < BLUE_ANALYSE_END)]
        valid_host_indices = valid_analyse - BLUE_ANALYSE_START

        wrong_subnet_hosts = []
        for hidx in valid_host_indices:
            h_subnet = int(const.host_subnet[hidx])
            if h_subnet not in agent_subnet_ids:
                hostname = mappings.idx_to_hostname.get(int(hidx), f"host_{hidx}")
                wrong_subnet_hosts.append((int(hidx), hostname, h_subnet))

        print(
            f"  {agent_name}: subnets={agent_subnets}, "
            f"valid_analyse_hosts={len(valid_analyse)}, "
            f"wrong_subnet={len(wrong_subnet_hosts)}"
        )
        if wrong_subnet_hosts:
            for hidx, hname, hsub in wrong_subnet_hosts[:5]:
                print(f"    BUG: host_idx={hidx} {hname} in subnet {hsub} allowed!")

    total = 0.0
    for step in range(steps):
        actions = {}
        step_actions_desc = []
        for agent_idx, agent_name in enumerate(env.agents):
            obs_jax = jnp.array(observations[agent_name], dtype=jnp.float32)
            mask = compute_blue_action_mask(const, agent_idx)
            mask_np = np.array(mask, dtype=bool)

            pi, _value = network.apply(params, obs_jax, mask)
            action_idx = int(jnp.argmax(pi.logits))
            is_valid = bool(mask_np[action_idx])
            cyborg_action = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
            actions[agent_name] = cyborg_action

            desc = describe_blue_action(action_idx, mappings, const=const)
            cyborg_cls = type(cyborg_action).__name__
            valid_str = "OK" if is_valid else "MASKED!"
            step_actions_desc.append(
                f"  {agent_name}: idx={action_idx:4d} [{valid_str:7s}] -> {desc:45s} -> CybORG:{cyborg_cls}"
            )

        observations, rewards, _, _, _ = env.step(actions=actions)
        step_reward = mean(rewards.values())
        total += step_reward

        if step < 10 or step % 5 == 0:
            print(f"\nStep {step}: reward={step_reward:.2f}  cumulative={total:.2f}")
            for desc in step_actions_desc:
                print(desc)

    print(f"\nVerbose trace total reward ({steps} steps): {total:.2f}")


def print_mask_summary():
    print("\n--- Action Mask Summary ---")
    env = FsmRedCC4Env(num_steps=100)
    key = jax.random.PRNGKey(42)
    _, env_state = env.reset(key)

    for agent_idx in range(NUM_BLUE_AGENTS):
        mask = np.array(compute_blue_action_mask(env_state.const, agent_idx, env_state.state))
        total_valid = mask.sum()
        by_type = []
        for name, (start, end) in zip(ACTION_TYPE_NAMES, ACTION_TYPE_RANGES):
            count = mask[start:end].sum()
            if count > 0:
                by_type.append(f"{name}={count}")
        print(f"  blue_{agent_idx}: {total_valid} valid actions: {', '.join(by_type)}")


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="Evaluate JAXborg-trained policy: rollout, transfer, baselines")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_final.pkl")
    parser.add_argument("--episodes", type=int, default=3, help="Rollout episodes (default 3)")
    parser.add_argument("--stochastic", action="store_true", help="Sample from policy instead of argmax")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument("--baselines", action="store_true", help="Run sleep + random baselines")
    parser.add_argument("--verbose", type=int, default=0, help="Step-by-step CybORG trace for N steps")
    parser.add_argument("--plot", action="store_true", help="Save action dist + training curve PNGs")
    parser.add_argument("--mask-summary", action="store_true", help="Print per-agent mask breakdown")
    args = parser.parse_args()

    deterministic = not args.stochastic

    print(f"Loading checkpoint: {args.checkpoint}")
    network, params = load_checkpoint(args.checkpoint)

    if args.mask_summary:
        print_mask_summary()

    # JAXborg rollout
    print("\n" + "=" * 70)
    print("JAXBORG ROLLOUT")
    print("=" * 70)
    jax_actions, jax_rewards, jax_results = rollout_jaxborg(network, params, args.episodes, deterministic)

    # Per-agent action dist (all episodes pooled)
    jax_pooled_by_agent = [[a for ep in jax_results for a in ep.actions_by_agent[i]] for i in range(NUM_BLUE_AGENTS)]
    print_per_agent_action_dist(jax_pooled_by_agent, label="JAXborg")

    # Trajectory summary for last episode
    print_trajectory_summary(jax_results[-1].trajectory, label=f"JAXborg ep {len(jax_results)}")

    # CybORG rollout
    print("\n" + "=" * 70)
    print("CYBORG ROLLOUT")
    print("=" * 70)
    cyborg_actions, cyborg_rewards, cyborg_actions_by_agent = rollout_cyborg(
        network, params, args.episodes, deterministic, seed=args.seed
    )

    print_per_agent_action_dist(cyborg_actions_by_agent, label="CybORG")

    # Comparison report
    print_comparison_report(jax_actions, jax_rewards, cyborg_actions, cyborg_rewards)

    # Optional: baselines
    if args.baselines:
        print("\n" + "=" * 70)
        print("SLEEP BASELINE")
        print("=" * 70)
        sleep_score = run_sleep_baseline(args.episodes)
        print(f"Sleep baseline ({args.episodes} episodes): {sleep_score:.1f}")

        print("\n" + "=" * 70)
        print("RANDOM POLICY (with JAXborg action mask)")
        print("=" * 70)
        random_score = run_random_baseline(args.episodes, seed=args.seed)
        print(f"Random policy ({args.episodes} episodes): {random_score:.1f}")

    # Optional: verbose trace
    if args.verbose > 0:
        print("\n" + "=" * 70)
        print(f"VERBOSE CYBORG TRACE ({args.verbose} steps, deterministic)")
        print("=" * 70)
        run_verbose_trace(network, params, steps=args.verbose, seed=args.seed)

    # Optional: plots
    if args.plot:
        output_dir = EXP_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_action_distribution(jax_actions, "JAXborg Action Distribution", output_dir / "jax_action_dist.png")
        plot_action_distribution(cyborg_actions, "CybORG Action Distribution", output_dir / "cyborg_action_dist.png")

        try:
            save_reward_plot(jax_rewards, cyborg_rewards)
        except Exception as e:
            print(f"(Skipped reward plot: {e})")

        metrics_path = EXP_DIR / "ippo_cc4" / "metrics.jsonl"
        if metrics_path.exists():
            plot_training_curves(metrics_path, output_dir / "training_curves.png")
        else:
            print(f"No metrics file at {metrics_path}, skipping training curves")


if __name__ == "__main__":
    main()
