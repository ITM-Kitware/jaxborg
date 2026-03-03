"""Diagnose JAXborg → CybORG policy transfer gap.

Runs side-by-side comparison of observations, action masks, and
action translations. Also runs a sleep baseline and a random-policy
baseline to quantify the transfer gap.
"""

import pickle
import sys
from pathlib import Path
from statistics import mean

import jax.numpy as jnp
import numpy as np
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.topology import build_const_from_cyborg
from jaxborg.translate import (
    build_mappings_from_cyborg,
    describe_blue_action,
    jax_blue_to_cyborg,
)

SEED = 42


def make_env(seed=SEED):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    return BlueFlatWrapper(env=cyborg, pad_spaces=True)


def run_sleep_baseline(episodes=5):
    from CybORG.Simulator.Actions import Sleep

    env = make_env()
    totals = []
    for _ in range(episodes):
        env.reset()
        total = 0.0
        for _ in range(500):
            actions = {a: Sleep() for a in env.agents}
            _, rewards, _, _, _ = env.step(actions=actions)
            total += mean(rewards.values())
        totals.append(total)
    return mean(totals)


def run_random_policy(episodes=5):
    env = make_env()
    rng = np.random.default_rng(SEED)
    totals = []
    for _ in range(episodes):
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
                actions[agent_name] = jax_blue_to_cyborg(action_idx, agent_idx, mappings)
            _, rewards, _, _, _ = env.step(actions=actions)
            total += mean(rewards.values())
        totals.append(total)
    return mean(totals)


def run_trained_verbose(checkpoint_path, steps=20):
    sys.path.insert(0, str(Path(__file__).parent))
    from train_ippo_cc4 import ActorCritic

    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    network = ActorCritic(
        action_dim=ckpt["action_dim"],
        hidden_dim=ckpt["hidden_dim"],
        activation=ckpt["activation"],
    )
    params = ckpt["params"]

    env = make_env()
    observations, _ = env.reset()
    inner = env.env
    const = build_const_from_cyborg(inner)
    mappings = build_mappings_from_cyborg(inner)

    print(f"\nNetwork action_dim={ckpt['action_dim']}, hidden_dim={ckpt['hidden_dim']}")

    # Check mask validity at step 0
    print("\nMASK VALIDATION (step 0):")
    from jaxborg.topology import BLUE_AGENT_SUBNETS, SUBNET_IDS

    for agent_idx, agent_name in enumerate(env.agents):
        mask = np.array(compute_blue_action_mask(const, agent_idx), dtype=bool)
        valid_indices = np.where(mask)[0]
        agent_subnets = BLUE_AGENT_SUBNETS[agent_idx]
        agent_subnet_ids = [SUBNET_IDS[s] for s in agent_subnets]

        # Check which hosts are marked valid for host-based actions
        from jaxborg.actions.encoding import BLUE_ANALYSE_END, BLUE_ANALYSE_START

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
            obs_vec = observations[agent_name]
            obs_jax = jnp.array(obs_vec, dtype=jnp.float32)
            mask = compute_blue_action_mask(const, agent_idx)
            mask_np = np.array(mask, dtype=bool)

            pi, value = network.apply(params, obs_jax, mask)
            action_idx = int(jnp.argmax(pi.logits))
            is_valid = bool(mask_np[action_idx])
            cyborg_action = jax_blue_to_cyborg(action_idx, agent_idx, mappings)
            actions[agent_name] = cyborg_action

            desc = describe_blue_action(action_idx, mappings)
            cyborg_cls = type(cyborg_action).__name__
            valid_str = "OK" if is_valid else "MASKED!"
            step_actions_desc.append(
                f"  {agent_name}: idx={action_idx:4d} [{valid_str:7s}] → {desc:45s} → CybORG:{cyborg_cls}"
            )

        observations, rewards, _, _, _ = env.step(actions=actions)
        step_reward = mean(rewards.values())
        total += step_reward

        if step < 10 or step % 5 == 0:
            print(f"\nStep {step}: reward={step_reward:.2f}  cumulative={total:.2f}")
            for desc in step_actions_desc:
                print(desc)

    return total


def main():
    print("=" * 70)
    print("SLEEP BASELINE")
    print("=" * 70)
    sleep_score = run_sleep_baseline(3)
    print(f"Sleep baseline (3 episodes): {sleep_score:.1f}")

    print("\n" + "=" * 70)
    print("RANDOM POLICY (with JAXborg action mask)")
    print("=" * 70)
    random_score = run_random_policy(3)
    print(f"Random policy (3 episodes): {random_score:.1f}")

    ckpt_path = Path(__file__).parent.parent / "jaxborg-exp" / "ippo_cc4" / "checkpoint_final.pkl"
    if ckpt_path.exists():
        print("\n" + "=" * 70)
        print("TRAINED POLICY (first 20 steps, deterministic)")
        print("=" * 70)
        run_trained_verbose(str(ckpt_path), steps=20)
    else:
        print(f"\nNo checkpoint at {ckpt_path}")
        print("Run: uv run python -u scripts/train_ippo_cc4.py TOTAL_TIMESTEPS=5000")


if __name__ == "__main__":
    main()
