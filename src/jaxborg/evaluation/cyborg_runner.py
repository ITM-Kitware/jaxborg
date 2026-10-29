"""CybORG rollout runner — load a torch checkpoint, evaluate on real CybORG.

Used by `scripts/eval/eval_recipe.py` for the `.pt` (torch state_dict)
path. JAX `.pkl` checkpoints take the parallel path through
`jaxborg.evaluation.jax_runner` (same target environment, different model
loader and action translation).
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from CybORG.Agents.Wrappers import EnterpriseMAE

from jaxborg.blue_observation_contract import blue_obs_size, enhanced_obs_enabled
from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.evaluation.cyborg_env_factory import CyborgReset, make_cyborg_env, reset_cyborg_env
from jaxborg.evaluation.episode_seeds import expand_episode_seeds
from jaxborg.policies import make_torch_policy
from jaxborg.scenarios.cc4.game_variant import GameVariant

os.environ.setdefault("JAX_PLATFORMS", "cpu")

NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]
OBS_DIM = BLUE_OBS_SIZE
ACT_DIM = 242
EPISODE_LENGTH = 500


def _pad_obs_mask(obs_dict, info_dict):
    obs = np.zeros((NUM_AGENTS, max(OBS_DIM, max(len(o) for o in obs_dict.values()))), dtype=np.float32)
    mask = np.zeros((NUM_AGENTS, ACT_DIM), dtype=np.float32)
    for i, aid in enumerate(AGENT_IDS):
        raw_o = np.asarray(obs_dict[aid], dtype=np.float32)
        raw_m = np.asarray(info_dict[aid]["action_mask"], dtype=np.float32)
        obs[i, : len(raw_o)] = raw_o
        mask[i, : len(raw_m)] = raw_m
    return obs, mask


def rollout_episode(
    env,
    variant: GameVariant,
    ep_seed: int,
    agent,
    *,
    deterministic: bool,
    policy_seed: int | None = None,
    on_reset: Callable[[Any, CyborgReset], None] | None = None,
    on_complete: Callable[[float, int], None] | None = None,
) -> float:
    if policy_seed is None:
        torch.manual_seed(ep_seed)
    else:
        torch.random.default_generator.manual_seed(policy_seed)
    r = reset_cyborg_env(env, variant, ep_seed=ep_seed)
    if on_reset is not None:
        on_reset(env, r)
    obs_d, info_d = r.obs, r.info
    total = 0.0
    steps = 0
    for _ in range(variant.num_steps):
        obs, mask = _pad_obs_mask(obs_d, info_d)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs)
            mask_t = torch.from_numpy(mask)
            if deterministic:
                act = agent.deterministic_action(obs_t, mask_t).cpu().numpy()
            else:
                a, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
                act = a.cpu().numpy()
        action_dict = {AGENT_IDS[i]: int(act[i]) for i in range(NUM_AGENTS)}
        obs_d, rew_d, term_d, trunc_d, info_d = env.step(action_dict)
        steps += 1
        total += float(rew_d[AGENT_IDS[0]])
        if any(term_d.values()) or any(trunc_d.values()):
            break
    if on_complete is not None:
        on_complete(total, steps)
    return total


def torch_policy_configuration(recipe: dict[str, Any]) -> dict[str, Any]:
    """Resolve exactly the factory arguments consumed by the checkpoint loader."""
    arch = recipe["arch"]
    return {
        "name": arch["name"],
        "obs_dim": blue_obs_size(enhanced_obs_enabled(recipe)),
        "action_dim": ACT_DIM,
        "hidden_dim": int(arch.get("hidden_dim", 256)),
        "hidden_layers": int(arch.get("hidden_layers", 2)),
    }


@dataclass(frozen=True)
class TorchEvaluationSettings:
    variant: GameVariant
    policy: dict[str, Any]
    deterministic: bool


def resolve_torch_evaluation_settings(
    recipe: dict[str, Any], *, deterministic: bool = False, horizon: int | None = None
) -> TorchEvaluationSettings:
    """Resolve native evaluation settings using the existing recipe contract."""
    from jaxborg.recipe import eval_variant

    if not isinstance(deterministic, bool):
        raise ValueError("deterministic must be a boolean")
    variant = eval_variant(recipe)
    if horizon is not None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 500:
            raise ValueError("horizon must be an integer between 1 and 500")
        variant = replace(variant, num_steps=horizon)
    return TorchEvaluationSettings(variant, torch_policy_configuration(recipe), deterministic)


def load_torch_policy_from_recipe(recipe: dict[str, Any], state_dict: dict[str, torch.Tensor]):
    agent = make_torch_policy(**torch_policy_configuration(recipe))
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def load_torch_policy(model_path: str | Path):
    """Load a torch policy + recipe from a checkpoint.

    Tries the sidecar first; falls back to key-based architecture detection
    for legacy checkpoints (no recipe_<tag>.yaml). Returns (agent, recipe).
    """
    import warnings

    from jaxborg.checkpoint import load_torch_policy as load_bundle_policy
    from jaxborg.checkpoint import read_sidecar

    model_path = Path(model_path)
    entry = load_bundle_policy(
        model_path,
        "blue",
        expected_obs_dim=None,
        expected_action_dim=ACT_DIM,
    )
    state_dict = entry.weights
    try:
        recipe = read_sidecar(model_path)
    except FileNotFoundError:
        if not entry.arch.get("name"):
            warnings.warn(
                f"No recipe sidecar next to {model_path}; falling back to "
                "key-based arch detection. Re-train under the new layout to "
                "remove this fallback.",
                DeprecationWarning,
                stacklevel=2,
            )
        is_separate = any(k.startswith("actor_features.") for k in state_dict)
        arch = (
            dict(entry.arch)
            if entry.arch.get("name")
            else {
                "name": "separate" if is_separate else "shared",
                "hidden_dim": 256,
                "hidden_layers": 2,
                "activation": "tanh",
            }
        )
        recipe = {
            "meta": {"name": "legacy", "source": "fallback (no sidecar)"},
            "algorithm": "ippo",
            "cage4_enhanced_obs": entry.obs_dim == blue_obs_size(True),
            "arch": arch,
        }
    if entry.arch.get("name"):
        recipe = dict(recipe)
        recipe["arch"] = dict(entry.arch)
    expected_obs_dim = blue_obs_size(enhanced_obs_enabled(recipe))
    if entry.obs_dim not in (0, expected_obs_dim):
        raise ValueError(
            f"Blue checkpoint observation dimension {entry.obs_dim} disagrees with recipe {expected_obs_dim}"
        )
    agent = load_torch_policy_from_recipe(recipe, state_dict)
    return agent, recipe


@contextmanager
def _policy_rng(seed: int | None):
    if seed is None:
        yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            yield


def _run_torch_items(
    model_path,
    deterministic,
    variant,
    items,
    construction_seed,
    *,
    on_policy=None,
    on_reset=None,
    on_complete=None,
    progress=False,
    total=0,
):
    with _policy_rng(construction_seed):
        policy, recipe = load_torch_policy(model_path)
    if variant.cage4_enhanced_obs != enhanced_obs_enabled(recipe):
        raise ValueError("checkpoint and evaluation cage4_enhanced_obs must match")
    if on_policy is not None:
        on_policy(policy, recipe)
    out = []
    for index, seed, policy_seed in items:

        def reset_observer(env, reset):
            if on_reset is not None:
                on_reset(index, seed, env, reset)

        def completion_observer(reward, steps):
            if on_complete is not None:
                on_complete(index, seed, seed if policy_seed is None else policy_seed, reward, steps)

        with _policy_rng(policy_seed):
            env = make_cyborg_env(variant, seed, wrapper_class=EnterpriseMAE)
            reward = rollout_episode(
                env,
                variant,
                ep_seed=seed,
                agent=policy,
                deterministic=deterministic,
                policy_seed=policy_seed,
                on_reset=reset_observer if on_reset is not None else None,
                on_complete=completion_observer if on_complete is not None else None,
            )
        out.append((index, seed, reward))
        if progress:
            print(f"  ep {index + 1}/{total} (seed={seed}): {reward:.1f}", flush=True)
    return out


def _cyborg_worker(args):
    """Load one policy and execute a strided chunk of the global schedule."""
    return _run_torch_items(*args)


def _validate_policy_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise ValueError("policy seeds must be integers between 0 and 2**63 - 1")


def run_torch_episodes(
    model_path: str | Path,
    *,
    variant: GameVariant,
    episode_seeds: list[int],
    deterministic: bool = False,
    policy_seeds: list[int] | None = None,
    workers: int = 1,
    progress: bool = False,
    on_policy: Callable[[Any, dict], None] | None = None,
    on_reset: Callable[[int, int, Any, CyborgReset], None] | None = None,
    on_complete: Callable[[int, int, int | None, float, int], None] | None = None,
) -> tuple[list[float], list[int]]:
    """Execute an ordered explicit episode schedule with optional serial observers.

    Supplied policy seeds isolate CPU Torch randomness during construction and
    each episode. Omitting them seeds policy sampling from each environment seed.

    Observers run synchronously and require serial execution. Policy observation
    runs after loading, outside RNG isolation; reset and completion observers
    run inside their episode's RNG context. Observer exceptions stop execution.
    """
    if not episode_seeds:
        raise ValueError("At least one episode seed is required")
    if workers > 1 and any(callback is not None for callback in (on_policy, on_reset, on_complete)):
        raise ValueError("Evaluation observers require serial execution")
    if policy_seeds is not None:
        if len(policy_seeds) != len(episode_seeds):
            raise ValueError("policy_seeds must have the same length as episode_seeds")
        for seed in policy_seeds:
            _validate_policy_seed(seed)
    construction_seed = policy_seeds[0] if policy_seeds is not None else None
    items = [
        (index, seed, policy_seeds[index] if policy_seeds is not None else None)
        for index, seed in enumerate(episode_seeds)
    ]
    total = len(items)
    rewards: list[float] = [0.0] * total
    seed_log: list[int] = [0] * total
    if workers <= 1:
        results = _run_torch_items(
            model_path,
            deterministic,
            variant,
            items,
            construction_seed,
            on_policy=on_policy,
            on_reset=on_reset,
            on_complete=on_complete,
            progress=progress,
            total=total,
        )
        for index, seed, reward in results:
            rewards[index], seed_log[index] = reward, seed
        return rewards, seed_log

    n_workers = min(workers, total)
    chunks = [items[i::n_workers] for i in range(n_workers)]
    pargs = [(str(model_path), deterministic, variant, chunk, construction_seed) for chunk in chunks]
    completed = 0
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        for chunk_results in ex.map(_cyborg_worker, pargs):
            for index, seed, reward in chunk_results:
                rewards[index], seed_log[index] = reward, seed
                completed += 1
                if progress:
                    print(f"  ep {completed}/{total} (seed={seed}): {reward:.1f}", flush=True)
    return rewards, seed_log


def evaluate_on_cyborg(
    model_path: str | Path,
    *,
    variant: GameVariant,
    seeds: list[int],
    episodes_per_seed: int,
    deterministic: bool = False,
    workers: int = 1,
    progress: bool = True,
    policy_seed: int | None = None,
) -> tuple[list[float], list[int]]:
    """Expand roots with the shared unique episode-seed mapping.

    Returns rewards and environment seeds in expanded order. An optional policy
    seed assigns consecutive CPU Torch seeds by global episode index.
    """
    flat = expand_episode_seeds(seeds, episodes_per_seed)
    if policy_seed is not None:
        _validate_policy_seed(policy_seed)
    policy_seeds = None if policy_seed is None else [policy_seed + index for index in range(len(flat))]
    return run_torch_episodes(
        model_path,
        variant=variant,
        episode_seeds=flat,
        deterministic=deterministic,
        policy_seeds=policy_seeds,
        workers=workers,
        progress=progress,
    )
