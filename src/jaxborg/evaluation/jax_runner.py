"""JAX-checkpoint → CybORG rollout runner.

The cross-backend transfer eval: load a Flax `.safetensors` checkpoint
trained by `scripts/train/algorithms/ippo_jax.py`, run it in pure CybORG,
return per-episode rewards. Action translation handles the JAX action space
(~300 indices) vs CybORG's `BlueFlatWrapper` indices.

Used by `scripts/eval/eval_recipe.py` when the model file is a Flax
`.safetensors`.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
from pathlib import Path
from statistics import mean
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from CybORG.Agents.Wrappers import BlueFlatWrapper

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP, encode_blue_action
from jaxborg.checkpoint import load_jax_params
from jaxborg.evaluation.cyborg_env_factory import make_cyborg_env, reset_cyborg_env
from jaxborg.parity.translate import build_mappings_from_cyborg, cyborg_blue_to_jax, jax_blue_to_cyborg
from jaxborg.policies import make_jax_policy
from jaxborg.scenarios.cc4.game_variant import GameVariant
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

EPISODE_LENGTH = 500


def load_jax_checkpoint(path: str | Path) -> tuple[Any, dict, dict]:
    """Load a Flax `.safetensors` and return (policy_module, params, recipe).

    Requires a recipe sidecar next to the checkpoint.
    """
    from jaxborg.checkpoint import read_sidecar

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    params, action_dim = load_jax_params(p)
    if action_dim == 0:
        action_dim = BLUE_ALLOW_TRAFFIC_END

    recipe = read_sidecar(p)
    arch = recipe["arch"]
    policy = make_jax_policy(
        arch["name"],
        action_dim=action_dim,
        hidden_dim=int(arch.get("hidden_dim", 256)),
        hidden_layers=int(arch.get("hidden_layers", 2)),
        activation=arch.get("activation", "tanh"),
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


def run_episode(env, variant: GameVariant, ep_seed: int, policy, params, deterministic: bool, rng) -> float:
    r = reset_cyborg_env(env, variant, ep_seed=ep_seed)
    observations = r.obs
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


def _jax_worker(args):
    """Pool worker: load checkpoint once, run a chunk of (idx, seed, rng_seed) episodes."""
    checkpoint_path, deterministic, variant, items = args
    policy, params, _ = load_jax_checkpoint(checkpoint_path)
    out = []
    for idx, seed, rng_seed in items:
        env = make_cyborg_env(variant, seed, wrapper_class=BlueFlatWrapper, wrapper_kwargs={"pad_spaces": True})
        r = run_episode(
            env,
            variant,
            ep_seed=seed,
            policy=policy,
            params=params,
            deterministic=deterministic,
            rng=jax.random.PRNGKey(rng_seed),
        )
        out.append((idx, seed, r))
    return out


def evaluate_jax_on_cyborg(
    checkpoint_path: str | Path,
    *,
    variant: GameVariant,
    seeds: list[int],
    episodes_per_seed: int,
    deterministic: bool = False,
    workers: int = 1,
    progress: bool = True,
) -> tuple[list[float], list[int], dict]:
    """Load a JAX `.safetensors`, evaluate against CybORG. Returns (rewards, seed_log, recipe).

    Episodes are independent — set `workers > 1` to fan out across processes
    (each worker spawns a clean Python interpreter and loads the model).
    """
    flat = [s + ep for s in seeds for ep in range(episodes_per_seed)]
    total = len(flat)
    base_seed = seeds[0] if seeds else 0
    items = [(idx, env_seed, base_seed * 100003 + idx) for idx, env_seed in enumerate(flat)]
    rewards: list[float] = [0.0] * total
    seed_log: list[int] = [0] * total

    policy, params, recipe = load_jax_checkpoint(checkpoint_path)

    if workers <= 1:
        for idx, seed, rng_seed in items:
            env = make_cyborg_env(variant, seed, wrapper_class=BlueFlatWrapper, wrapper_kwargs={"pad_spaces": True})
            r = run_episode(
                env,
                variant,
                ep_seed=seed,
                policy=policy,
                params=params,
                deterministic=deterministic,
                rng=jax.random.PRNGKey(rng_seed),
            )
            rewards[idx] = r
            seed_log[idx] = seed
            if progress:
                print(f"  ep {idx + 1}/{total} (seed={seed}): {r:.1f}", flush=True)
        return rewards, seed_log, recipe

    n_workers = min(workers, total)
    chunks = [items[i::n_workers] for i in range(n_workers)]
    pargs = [(str(checkpoint_path), deterministic, variant, c) for c in chunks]
    completed = 0
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        for chunk_results in ex.map(_jax_worker, pargs):
            for idx, seed, r in chunk_results:
                rewards[idx] = r
                seed_log[idx] = seed
                completed += 1
                if progress:
                    print(f"  ep {completed}/{total} (seed={seed}): {r:.1f}", flush=True)
    return rewards, seed_log, recipe
