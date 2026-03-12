"""Evaluate a JAXborg-trained policy in pure CybORG."""

import argparse
import pickle
from pathlib import Path
from statistics import mean, stdev

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyApache import ApacheDecoyFactory
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyHarakaSMPT import HarakaDecoyFactory
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyTomcat import TomcatDecoyFactory
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyVsftpd import VsftpdDecoyFactory
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from flax.linen.initializers import constant, orthogonal
from train_ippo_cc4 import ActorCritic

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP, encode_blue_action
from jaxborg.topology import build_const_from_cyborg
from jaxborg.translate import (
    build_mappings_from_cyborg,
    cyborg_blue_to_jax,
    jax_blue_to_cyborg,
)

EPISODE_LENGTH = 500
DECOY_FACTORY_ACTIONS = (
    (HarakaDecoyFactory(), "DeployDecoy_HarakaSMPT"),
    (ApacheDecoyFactory(), "DeployDecoy_Apache"),
    (TomcatDecoyFactory(), "DeployDecoy_Tomcat"),
    (VsftpdDecoyFactory(), "DeployDecoy_Vsftpd"),
)


class LegacyActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, avail_actions=None):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        actor_mean = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        if avail_actions is not None:
            action_logits = action_logits - ((1 - avail_actions) * 1e10)

        return distrax.Categorical(logits=action_logits)


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
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    nested_params = ckpt["params"].get("params", {})

    if "actor_head" in nested_params:
        policy = ActorCritic(
            action_dim=ckpt["action_dim"],
            hidden_dim=ckpt["hidden_dim"],
            activation=ckpt["activation"],
        )
        return policy, ckpt["params"], "current"

    if "Dense_0" in nested_params:
        if ckpt["action_dim"] != BLUE_ALLOW_TRAFFIC_END:
            raise ValueError(
                f"Legacy checkpoint action_dim={ckpt['action_dim']} is incompatible with current action space "
                f"{BLUE_ALLOW_TRAFFIC_END}"
            )
        policy = LegacyActor(
            action_dim=ckpt["action_dim"],
            hidden_dim=ckpt["hidden_dim"],
            activation=ckpt["activation"],
        )
        return policy, ckpt["params"], "legacy"

    raise ValueError(f"Unrecognized checkpoint format: nested params keys={sorted(nested_params.keys())}")


def policy_dist(policy, params, policy_kind, obs_jax, mask):
    if policy_kind == "current":
        return policy.apply(params, obs_jax, mask, method=ActorCritic.actor)
    return policy.apply(params, obs_jax, mask)


def _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state):
    cls_name = type(action).__name__
    agent_id = int(agent_name.split("_")[-1])

    if label.startswith("[Padding]"):
        return []

    if cls_name == "Sleep" and not label.startswith("[Invalid]"):
        return [BLUE_SLEEP]

    if cls_name == "Sleep" and label.startswith("[Invalid]"):
        return []

    if cls_name == "DeployDecoy":
        if action.hostname not in mappings.hostname_to_idx:
            return []
        host = cyborg_state.hosts[action.hostname]
        host_idx = mappings.hostname_to_idx[action.hostname]
        return [
            encode_blue_action(action_name, host_idx, agent_id, const=const)
            for factory, action_name in DECOY_FACTORY_ACTIONS
            if factory.is_host_compatible(host)
        ]

    try:
        return [cyborg_blue_to_jax(action, agent_name, mappings, const=const)]
    except (KeyError, ValueError):
        return []


def _build_action_lookup(env, agent_name, mappings, const):
    """Precompute cyborg_action_idx -> list[jax_idx] for one agent. Call once per episode."""
    controller = env.env.environment_controller
    cyborg_actions = env.actions(agent_name)
    cyborg_labels = env.action_labels(agent_name)
    cyborg_state = controller.state
    lookup = []
    for action, label in zip(cyborg_actions, cyborg_labels):
        jax_indices = _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state)
        lookup.append(jax_indices)
    return lookup


def _live_cyborg_mask_in_jax_space(env, agent_name, mappings, const, lookup=None):
    controller = env.env.environment_controller
    pending = controller.actions_in_progress.get(agent_name)
    if pending is not None and pending["remaining_ticks"] > 0:
        jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
        label = f"[Pending] {type(pending['action']).__name__}"
        for jax_idx in _cyborg_action_to_jax_indices(
            pending["action"], label, agent_name, mappings, const, controller.state
        ):
            jax_mask[jax_idx] = True
        return jnp.array(jax_mask)

    jax_mask = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
    cyborg_mask = env.get_action_space(agent_name)["mask"]

    if lookup is not None:
        for cyborg_idx, valid in enumerate(cyborg_mask):
            if valid:
                for jax_idx in lookup[cyborg_idx]:
                    jax_mask[jax_idx] = True
    else:
        cyborg_actions = env.actions(agent_name)
        cyborg_labels = env.action_labels(agent_name)
        cyborg_state = controller.state
        for action, valid, label in zip(cyborg_actions, cyborg_mask, cyborg_labels):
            if not valid:
                continue
            for jax_idx in _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const, cyborg_state):
                jax_mask[jax_idx] = True

    return jnp.array(jax_mask)


def _raw_cyborg_step_with_flat_obs(wrapper, actions, messages=None):
    """Step underlying CybORG with raw actions, then flatten blue observations via the wrapper."""
    obs, rews, dones, info = wrapper.env.parallel_step(
        actions,
        messages=messages,
        skip_valid_action_check=True,
    )

    observations = {
        agent: wrapper.observation_change(agent, obs[agent]) for agent in wrapper.possible_agents if agent in obs
    }
    rewards = {agent: sum(rews[agent].values()) for agent in wrapper.possible_agents if agent in rews}
    terminated = {agent: bool(dones[agent]) for agent in wrapper.possible_agents if agent in dones}
    truncated = terminated.copy()
    info = {agent: {"action_mask": wrapper.get_action_space(agent)["mask"]} for agent in wrapper.possible_agents}
    wrapper.agents = [agent for agent in wrapper.possible_agents if not terminated.get(agent, False)]
    return observations, rewards, terminated, truncated, info


def run_episode(env, policy, params, policy_kind, deterministic, rng):
    observations, _ = env.reset()
    inner_cyborg = env.env
    const = build_const_from_cyborg(inner_cyborg)
    mappings = build_mappings_from_cyborg(inner_cyborg)

    # Precompute action translation tables (once per episode, ~600x faster per step)
    action_lookups = {agent_name: _build_action_lookup(env, agent_name, mappings, const) for agent_name in env.agents}

    total = 0.0
    for _ in range(EPISODE_LENGTH):
        actions = {}
        for agent_idx, agent_name in enumerate(env.agents):
            obs_vec = observations[agent_name]
            obs_jax = jnp.array(obs_vec, dtype=jnp.float32)

            mask = _live_cyborg_mask_in_jax_space(env, agent_name, mappings, const, lookup=action_lookups[agent_name])

            pi = policy_dist(policy, params, policy_kind, obs_jax, mask)

            if deterministic:
                action_idx = int(jnp.argmax(pi.logits))
            else:
                rng, _rng = jax.random.split(rng)
                action_idx = int(pi.sample(seed=_rng))

            cyborg_action = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)
            actions[agent_name] = cyborg_action

        observations, rewards, terminations, truncations, _ = _raw_cyborg_step_with_flat_obs(env, actions=actions)
        total += mean(rewards.values())
        if terminations.get("__all__", False) or truncations.get("__all__", False):
            break

    return total


def evaluate(checkpoint_path, episodes, seed, deterministic):
    policy, params, policy_kind = load_checkpoint(checkpoint_path)
    env = make_env(seed)
    rng = jax.random.PRNGKey(seed if seed is not None else 0)

    episode_rewards = []
    for ep in range(episodes):
        rng, _rng = jax.random.split(rng)
        reward = run_episode(env, policy, params, policy_kind, deterministic, _rng)
        episode_rewards.append(reward)
        print(f"Episode {ep + 1}: {reward:.4f}")

    print(f"\nepisodes:  {episodes}")
    print(f"mean:      {mean(episode_rewards):.4f}")
    if len(episode_rewards) > 1:
        print(f"stdev:     {stdev(episode_rewards):.4f}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate JAXborg policy in CybORG")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_final.pkl")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.deterministic and args.stochastic:
        raise ValueError("Choose at most one of --deterministic or --stochastic")
    deterministic = not args.stochastic
    if args.deterministic:
        deterministic = True
    evaluate(args.checkpoint, args.episodes, args.seed, deterministic)


if __name__ == "__main__":
    main()
