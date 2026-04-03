"""Cross-backend sleep baseline comparison: JAXborg vs CybORG.

Runs both backends with Sleep-only blue actions (no trained policy) and compares
mean episode rewards. If the means differ significantly, there is a simulation-level
reward gap independent of any policy. If they match, the L4 TOST gap comes from
policy-environment interaction (trained policy doesn't transfer across RNG streams).

Usage:
    export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
    uv run python scripts/diagnose_sleep_comparison.py --episodes 10
"""

import os

os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jaxborg/xla"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import argparse
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from jaxborg.constants import NUM_BLUE_AGENTS  # noqa: E402
from jaxborg.fsm_red_env import FsmRedCC4Env  # noqa: E402


def run_jaxborg_sleep(num_episodes: int, seed: int, bank_size: int) -> list[float]:
    """Run JAXborg episodes with Sleep-only blue actions."""
    env = FsmRedCC4Env(
        num_steps=500,
        topology_mode="cyborg_bank",
        topology_bank_size=bank_size,
    )
    totals = []
    for ep in range(num_episodes):
        ep_seed = seed + ep * 100
        key = jax.random.PRNGKey(ep_seed)
        t0 = time.perf_counter()

        obs, state = env.reset(key)
        total = 0.0
        sleep_actions = {f"blue_{i}": jnp.int32(0) for i in range(NUM_BLUE_AGENTS)}

        for step in range(500):
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, infos = env.step(step_key, state, sleep_actions)
            step_reward = float(np.mean([float(rewards[f"blue_{i}"]) for i in range(NUM_BLUE_AGENTS)]))
            total += step_reward

        elapsed = time.perf_counter() - t0
        totals.append(total)
        print(f"  JAXborg ep {ep + 1}: {total:.1f} ({elapsed:.1f}s)")
    return totals


def run_cyborg_sleep(num_episodes: int, seed: int, bank_size: int) -> list[float]:
    """Run CybORG episodes with Sleep-only blue actions."""
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.topology import cyborg_bank_seed_from_seed

    totals = []
    for ep in range(num_episodes):
        ep_seed = seed + ep * 100
        actual_seed = cyborg_bank_seed_from_seed(ep_seed, bank_size)
        t0 = time.perf_counter()

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=500,
        )
        cyborg = CybORG(sg, "sim", seed=actual_seed)
        wrapper = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        wrapper.reset()

        total = 0.0
        for step in range(500):
            from CybORG.Simulator.Actions import Sleep

            actions = {a: Sleep() for a in wrapper.agents}
            obs, rews, dones, info = wrapper.env.parallel_step(actions, skip_valid_action_check=True)
            # Extract only BlueRewardMachine (match eval_transfer.py extraction)
            step_rewards = []
            for agent in wrapper.possible_agents:
                if agent in rews:
                    step_rewards.append(rews[agent].get("BlueRewardMachine", sum(rews[agent].values())))
            total += float(np.mean(step_rewards))

        elapsed = time.perf_counter() - t0
        totals.append(total)
        print(f"  CybORG  ep {ep + 1}: {total:.1f} ({elapsed:.1f}s)")
    return totals


def main():
    parser = argparse.ArgumentParser(description="Cross-backend sleep baseline comparison")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bank-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 70)
    print("CROSS-BACKEND SLEEP BASELINE COMPARISON")
    print("=" * 70)
    print(f"Episodes: {args.episodes}, Seed: {args.seed}, Bank size: {args.bank_size}")
    print("Both backends use Sleep-only blue (no trained policy).")
    print("Red uses independent native FSM (no cross-backend sync).")
    print()

    print("--- JAXborg (FsmRedCC4Env, cyborg_bank) ---")
    jax_totals = run_jaxborg_sleep(args.episodes, args.seed, args.bank_size)

    print()
    print("--- CybORG (FiniteStateRedAgent, matched bank seeds) ---")
    cyborg_totals = run_cyborg_sleep(args.episodes, args.seed, args.bank_size)

    jax_mean = mean(jax_totals)
    cyborg_mean = mean(cyborg_totals)
    gap = jax_mean - cyborg_mean

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'':>15s} {'JAXborg':>10s} {'CybORG':>10s} {'Gap':>10s}")
    print("-" * 50)
    for i in range(args.episodes):
        ep_gap = jax_totals[i] - cyborg_totals[i]
        print(f"Episode {i + 1:>5d}  {jax_totals[i]:>10.1f} {cyborg_totals[i]:>10.1f} {ep_gap:>+10.1f}")
    print("-" * 50)
    print(f"{'Mean':>15s} {jax_mean:>10.1f} {cyborg_mean:>10.1f} {gap:>+10.1f}")
    if args.episodes > 1:
        jax_sd = stdev(jax_totals)
        cyborg_sd = stdev(cyborg_totals)
        print(f"{'Stdev':>15s} {jax_sd:>10.1f} {cyborg_sd:>10.1f}")

    # Simple two-sample TOST
    if args.episodes >= 2:
        from scipy import stats

        delta = 1000.0
        t_stat, p_two = stats.ttest_ind(jax_totals, cyborg_totals, equal_var=False)  # noqa: F841
        var_j = np.var(jax_totals, ddof=1) / len(jax_totals)
        var_c = np.var(cyborg_totals, ddof=1) / len(cyborg_totals)
        se = np.sqrt(var_j + var_c)
        df_num = (var_j + var_c) ** 2
        df_den = var_j**2 / (len(jax_totals) - 1) + var_c**2 / (len(cyborg_totals) - 1)
        df = df_num / df_den

        t_upper = (gap - delta) / se
        t_lower = (gap + delta) / se
        p_upper = stats.t.cdf(t_upper, df)
        p_lower = 1.0 - stats.t.cdf(t_lower, df)
        p_tost = max(p_upper, p_lower)
        equivalent = p_tost < 0.05

        ci_t = stats.t.ppf(0.975, df)
        ci_low = gap - ci_t * se
        ci_high = gap + ci_t * se

        print()
        print("=" * 70)
        print(f"SLEEP BASELINE TOST (Δ={delta:.0f})")
        print("=" * 70)
        print(f"  Mean gap:    {gap:+.1f}")
        print(f"  95% CI:      [{ci_low:+.1f}, {ci_high:+.1f}]")
        print(f"  p_TOST:      {p_tost:.4f}")
        print(f"  Verdict:     {'EQUIVALENT' if equivalent else 'NOT EQUIVALENT'}")
        if equivalent:
            print("  -> Sleep baselines match: simulation produces equivalent rewards.")
            print("     L4 trained-policy gap is from policy-env interaction, not sim bug.")
        else:
            print(f"  -> Sleep baselines DIFFER by {gap:+.1f}.")
            print("     Investigate green/red dynamics for simulation-level reward gap.")


if __name__ == "__main__":
    main()
