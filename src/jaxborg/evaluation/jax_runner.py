"""JAX-checkpoint → CybORG rollout runner.

The cross-backend transfer eval: load a Flax `.pkl` checkpoint trained by
`scripts/train/algorithms/ippo_jax.py`, run it in pure CybORG, return per-
episode rewards. Action translation handles the JAX action space (~300
indices) vs CybORG's `BlueFlatWrapper` indices.

Used by `scripts/eval/eval_recipe.py` when the model file is a Flax `.pkl`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from statistics import mean
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP, encode_blue_action
from jaxborg.parity.translate import build_mappings_from_cyborg, cyborg_blue_to_jax, jax_blue_to_cyborg
from jaxborg.policies import make_jax_policy
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg, cyborg_bank_seed_from_seed

EPISODE_LENGTH = 500


def make_env(seed: int | None = None, bank_match_size: int | None = None):
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    actual_seed = cyborg_bank_seed_from_seed(seed, bank_match_size) if bank_match_size is not None else seed
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=actual_seed)
    return BlueFlatWrapper(env=cyborg, pad_spaces=True)


def load_jax_checkpoint(path: str | Path) -> tuple[Any, dict, dict]:
    """Load a Flax `.pkl` and return (policy_module, params, recipe).

    Requires a recipe sidecar next to the checkpoint. Pre-refactor
    checkpoints without sidecars are no longer supported — re-train
    under the recipe-driven layout.
    """
    from jaxborg.checkpoint import read_sidecar

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    with p.open("rb") as f:
        ckpt = pickle.load(f)

    params = ckpt["params"]
    action_dim = ckpt.get("action_dim", BLUE_ALLOW_TRAFFIC_END)
    hidden_dim = ckpt.get("hidden_dim", 256)
    activation = ckpt.get("activation", "tanh")

    recipe = read_sidecar(p)
    arch = recipe["arch"]
    policy = make_jax_policy(
        arch["name"],
        action_dim=action_dim,
        hidden_dim=int(arch.get("hidden_dim", hidden_dim)),
        hidden_layers=int(arch.get("hidden_layers", 2)),
        activation=arch.get("activation", activation),
    )
    return policy, params, recipe


def _policy_dist(policy: Any, params: dict, obs_jax, mask):
    """Run actor head on a recipe-driven Flax policy module."""
    pi, _ = policy.apply(params, obs_jax, mask)
    return pi


def _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const):
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
        host_idx = mappings.hostname_to_idx[action.hostname]
        jax_idx = encode_blue_action("DeployDecoy", host_idx, agent_id, const=const)
        if jax_idx == BLUE_SLEEP:
            return []
        return [jax_idx]
    try:
        jax_idx = cyborg_blue_to_jax(action, agent_name, mappings, const=const)
        if jax_idx == BLUE_SLEEP:
            return []
        return [jax_idx]
    except (KeyError, ValueError):
        return []


def _build_action_lookup(env, agent_name, mappings, const):
    cyborg_actions = env.actions(agent_name)
    cyborg_labels = env.action_labels(agent_name)
    return [
        _cyborg_action_to_jax_indices(action, label, agent_name, mappings, const)
        for action, label in zip(cyborg_actions, cyborg_labels)
    ]


def _live_cyborg_mask_in_jax_space(env, agent_name, lookup):
    controller = env.env.environment_controller
    pending = controller.actions_in_progress.get(agent_name)
    if pending is not None and pending["remaining_ticks"] > 0:
        # Force Sleep during pending ticks — CybORG silently drops resubmitted
        # actions and re-charges action_cost otherwise.
        m = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
        m[BLUE_SLEEP] = True
        return jnp.array(m)

    m = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
    cyborg_mask = env.get_action_space(agent_name)["mask"]
    for cyborg_idx, valid in enumerate(cyborg_mask):
        if valid:
            for jax_idx in lookup[cyborg_idx]:
                m[jax_idx] = True
    return jnp.array(m)


def _raw_step(wrapper, actions):
    obs, rews, dones, _ = wrapper.env.parallel_step(actions, messages=None, skip_valid_action_check=True)
    observations = {a: wrapper.observation_change(a, obs[a]) for a in wrapper.possible_agents if a in obs}
    rewards = {a: sum(rews[a].values()) for a in wrapper.possible_agents if a in rews}
    terminated = {a: bool(dones[a]) for a in wrapper.possible_agents if a in dones}
    truncated = terminated.copy()
    wrapper.agents = [a for a in wrapper.possible_agents if not terminated.get(a, False)]
    return observations, rewards, terminated, truncated


def run_episode(env, policy, params, deterministic: bool, rng) -> float:
    observations, _ = env.reset()
    inner = env.env
    const = build_const_from_cyborg(inner)
    mappings = build_mappings_from_cyborg(inner)

    lookups = {a: _build_action_lookup(env, a, mappings, const) for a in env.agents}

    total = 0.0
    for _ in range(EPISODE_LENGTH):
        actions = {}
        for agent_idx, agent_name in enumerate(env.agents):
            obs_jax = jnp.array(observations[agent_name], dtype=jnp.float32)
            mask = _live_cyborg_mask_in_jax_space(env, agent_name, lookups[agent_name])
            pi = _policy_dist(policy, params, obs_jax, mask)
            if deterministic:
                action_idx = int(jnp.argmax(pi.logits))
            else:
                rng, _rng = jax.random.split(rng)
                action_idx = int(pi.sample(seed=_rng))
            actions[agent_name] = jax_blue_to_cyborg(action_idx, agent_idx, mappings, const=const)

        observations, rewards, terms, truncs = _raw_step(env, actions)
        total += mean(rewards.values())
        if terms.get("__all__", False) or truncs.get("__all__", False):
            break
    return total


def evaluate_jax_on_cyborg(
    checkpoint_path: str | Path,
    *,
    seeds: list[int],
    episodes_per_seed: int,
    deterministic: bool = False,
    bank_match_size: int | None = None,
    progress: bool = True,
) -> tuple[list[float], list[int], dict]:
    """Load a JAX `.pkl`, evaluate against CybORG. Returns (rewards, seed_log, recipe)."""
    policy, params, recipe = load_jax_checkpoint(checkpoint_path)
    rng = jax.random.PRNGKey(seeds[0] if seeds else 0)

    rewards: list[float] = []
    seed_log: list[int] = []
    total_eps = len(seeds) * episodes_per_seed
    n = 0
    for s in seeds:
        for ep in range(episodes_per_seed):
            env = make_env(seed=s + ep, bank_match_size=bank_match_size)
            rng, _rng = jax.random.split(rng)
            r = run_episode(env, policy, params, deterministic, _rng)
            rewards.append(r)
            seed_log.append(s + ep)
            n += 1
            if progress:
                print(f"  ep {n}/{total_eps} (seed={s + ep}): {r:.1f}", flush=True)
    return rewards, seed_log, recipe
