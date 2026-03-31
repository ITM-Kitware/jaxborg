"""Compare raw green+red dynamics without learned blue policy (blue=Sleep).

If JAX produces worse rewards than CybORG with blue=Sleep, the dynamics
themselves are systematically different. If similar, the gap is from
policy-dynamics interaction.

Usage:
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
    uv run python scripts/diagnose_baseline_dynamics.py --episodes 10
"""

# ruff: noqa: E402

import os

os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jaxborg/xla"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import argparse
import sys
from pathlib import Path
from statistics import mean, stdev

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from eval_transfer import (
    DEFAULT_BANK_SIZE,
    NUM_BLUE_AGENTS,
    _raw_cyborg_step_with_flat_obs,
)

from jaxborg.actions.encoding import BLUE_SLEEP
from jaxborg.fsm_red_env import FsmRedCC4Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology-bank-size", type=int, default=DEFAULT_BANK_SIZE)
    parser.add_argument("--no-green", action="store_true", help="Disable green agents too")
    args = parser.parse_args()

    jax_rewards = []
    cy_rewards = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep * 100

        # --- JAX env with blue=Sleep ---
        jax_env = FsmRedCC4Env(
            num_steps=500,
            topology_mode="cyborg_bank",
            topology_bank_size=args.topology_bank_size,
        )
        key = jax.random.PRNGKey(ep_seed)
        jax_obs, jax_state = jax_env.reset(key)
        if args.no_green:
            jax_state = jax_state.replace(
                const=jax_state.const.replace(
                    num_green_agents=jnp.int32(0),
                    green_agents_active=jnp.array(False),
                )
            )
        sleep_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

        jax_total = 0.0
        for step in range(500):
            key, step_key = jax.random.split(key)
            jax_obs, jax_state, jax_step_rewards, _, _ = jax_env.step(
                step_key, jax_state, sleep_actions
            )
            jax_reward = float(
                np.asarray(
                    jnp.stack([jax_step_rewards[f"blue_{i}"] for i in range(NUM_BLUE_AGENTS)]).mean()
                )
            )
            jax_total += jax_reward

        # --- CybORG env with blue=Sleep ---
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.topology import cyborg_bank_seed_from_seed

        actual_seed = cyborg_bank_seed_from_seed(ep_seed, args.topology_bank_size)
        green_cls = SleepAgent if args.no_green else EnterpriseGreenAgent
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=green_cls,
            red_agent_class=FiniteStateRedAgent,
            steps=500,
        )
        cyborg = CybORG(sg, "sim", seed=actual_seed)
        cyborg_env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        cyborg_obs, _ = cyborg_env.reset()
        cyborg_agent_names = [f"blue_agent_{i}" for i in range(NUM_BLUE_AGENTS)]

        cy_total = 0.0
        for step in range(500):
            from CybORG.Simulator.Actions import Sleep

            actions = {name: Sleep() for name in cyborg_agent_names}
            cyborg_obs, cy_step_rewards, _, _, _ = _raw_cyborg_step_with_flat_obs(
                cyborg_env, actions=actions
            )
            cy_total += float(mean(cy_step_rewards.values()))

        jax_rewards.append(jax_total)
        cy_rewards.append(cy_total)
        gap = jax_total - cy_total
        print(f"  Ep {ep + 1}: JAX={jax_total:.1f}  CybORG={cy_total:.1f}  gap={gap:+.1f}")

    jax_mean = mean(jax_rewards)
    cy_mean = mean(cy_rewards)
    gap = jax_mean - cy_mean
    print(f"\n  JAX mean:   {jax_mean:.1f} (std={stdev(jax_rewards):.1f})")
    print(f"  CybORG mean: {cy_mean:.1f} (std={stdev(cy_rewards):.1f})")
    print(f"  Gap: {gap:+.1f}")
    if abs(gap) < 200:
        print("  → Dynamics are EQUIVALENT (gap within ±200)")
    else:
        direction = "JAX worse" if gap < 0 else "CybORG worse"
        print(f"  → Dynamics DIFFER systematically ({direction})")


if __name__ == "__main__":
    main()
