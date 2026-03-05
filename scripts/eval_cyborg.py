"""Evaluate a JAXborg-trained policy in pure CybORG."""

import argparse
import pickle
from statistics import mean, stdev

import jax
import jax.numpy as jnp
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from train_ippo_cc4 import ActorCritic

from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.topology import build_const_from_cyborg
from jaxborg.translate import build_mappings_from_cyborg, jax_blue_to_cyborg

EPISODE_LENGTH = 500


def make_env(seed=None):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    return BlueFlatWrapper(env=cyborg, pad_spaces=True)


def load_checkpoint(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    network = ActorCritic(
        action_dim=ckpt["action_dim"],
        hidden_dim=ckpt["hidden_dim"],
        activation=ckpt["activation"],
    )
    return network, ckpt["params"]


def run_episode(env, network, params, deterministic, rng):
    observations, _ = env.reset()
    inner_cyborg = env.env
    const = build_const_from_cyborg(inner_cyborg)
    mappings = build_mappings_from_cyborg(inner_cyborg)

    total = 0.0
    for _ in range(EPISODE_LENGTH):
        actions = {}
        for agent_idx, agent_name in enumerate(env.agents):
            obs_vec = observations[agent_name]
            obs_jax = jnp.array(obs_vec, dtype=jnp.float32)

            mask = compute_blue_action_mask(const, agent_idx)

            pi, _ = network.apply(params, obs_jax, mask)

            if deterministic:
                action_idx = int(jnp.argmax(pi.logits))
            else:
                rng, _rng = jax.random.split(rng)
                action_idx = int(pi.sample(seed=_rng))

            cyborg_action = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
            actions[agent_name] = cyborg_action

        observations, rewards, _, _, _ = env.step(actions=actions)
        total += mean(rewards.values())

    return total


def evaluate(checkpoint_path, episodes, seed, deterministic):
    network, params = load_checkpoint(checkpoint_path)
    env = make_env(seed)
    rng = jax.random.PRNGKey(seed if seed is not None else 0)

    episode_rewards = []
    for ep in range(episodes):
        rng, _rng = jax.random.split(rng)
        reward = run_episode(env, network, params, deterministic, _rng)
        episode_rewards.append(reward)
        print(f"Episode {ep + 1}: {reward:.4f}")

    print(f"\nepisodes:  {episodes}")
    print(f"mean:      {mean(episode_rewards):.4f}")
    if len(episode_rewards) > 1:
        print(f"stdev:     {stdev(episode_rewards):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate JAXborg policy in CybORG")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_final.pkl")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes, args.seed, args.deterministic)
