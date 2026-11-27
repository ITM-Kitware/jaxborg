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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from CybORG.Agents.Wrappers import EnterpriseMAE

from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.evaluation.cyborg_env_factory import make_cyborg_env, reset_cyborg_env
from jaxborg.policies import make_torch_policy
from jaxborg.scenarios.cc4.game_variant import GameVariant

os.environ.setdefault("JAX_PLATFORMS", "cpu")

NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]
OBS_DIM = BLUE_OBS_SIZE
ACT_DIM = 242
EPISODE_LENGTH = 500


def _pad_obs_mask(obs_dict, info_dict):
    obs = np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)
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
    horizon=EPISODE_LENGTH,
    on_reset=None,
    on_complete=None,
) -> float:
    r = reset_cyborg_env(env, variant, ep_seed=ep_seed)
    if on_reset is not None:
        on_reset(env, r)
    obs_d, info_d = r.obs, r.info
    total = 0.0
    for step in range(1, horizon + 1):
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
        total += float(rew_d[AGENT_IDS[0]])
        if any(term_d.values()) or any(trunc_d.values()):
            break
    if on_complete is not None:
        on_complete(total, step)
    return total


def torch_policy_configuration(recipe: dict[str, Any]) -> dict[str, Any]:
    """Resolve exactly the factory arguments consumed by the checkpoint loader."""
    arch = recipe["arch"]
    return {
        "name": arch["name"],
        "obs_dim": OBS_DIM,
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

    from jaxborg.checkpoint import read_sidecar

    model_path = Path(model_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    try:
        recipe = read_sidecar(model_path)
    except FileNotFoundError:
        warnings.warn(
            f"No recipe sidecar next to {model_path}; falling back to "
            "key-based arch detection. Re-train under the new layout to "
            "remove this fallback.",
            DeprecationWarning,
            stacklevel=2,
        )
        is_separate = any(k.startswith("actor_features.") for k in state_dict)
        recipe = {
            "meta": {"name": "legacy", "source": "fallback (no sidecar)"},
            "algorithm": "ippo",
            "arch": {
                "name": "separate" if is_separate else "shared",
                "hidden_dim": 256,
                "hidden_layers": 2,
                "activation": "tanh",
            },
        }
    agent = load_torch_policy_from_recipe(recipe, state_dict)
    return agent, recipe


def run_torch_episodes(
    model_path, *, variant, episode_seeds, deterministic, policy_seeds, on_policy, on_reset, on_complete
):
    """Record serial episodes with an explicit environment seed and policy seed per case."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(policy_seeds[0])
        policy, recipe = load_torch_policy(model_path)
        on_policy(policy, recipe)
        for index, (seed, policy_seed) in enumerate(zip(episode_seeds, policy_seeds, strict=True)):
            torch.random.default_generator.manual_seed(policy_seed)
            env = make_cyborg_env(variant, seed, wrapper_class=EnterpriseMAE)
            rollout_episode(
                env,
                variant,
                seed,
                policy,
                deterministic=deterministic,
                horizon=variant.num_steps,
                on_reset=lambda env, reset: on_reset(index, seed, env, reset),
                on_complete=lambda reward, steps: on_complete(index, seed, policy_seed, reward, steps),
            )


def _cyborg_worker(args):
    """Pool worker: load model once, run a chunk of (idx, seed) episodes."""
    model_path, deterministic, variant, items = args
    agent, _ = load_torch_policy(model_path)
    out = []
    for idx, seed in items:
        env = make_cyborg_env(variant, seed, wrapper_class=EnterpriseMAE)
        r = rollout_episode(env, variant, ep_seed=seed, agent=agent, deterministic=deterministic)
        out.append((idx, seed, r))
    return out


def evaluate_on_cyborg(
    model_path: str | Path,
    *,
    variant: GameVariant,
    seeds: list[int],
    episodes_per_seed: int,
    deterministic: bool = False,
    workers: int = 1,
    progress: bool = True,
) -> tuple[list[float], list[int]]:
    """Run `episodes_per_seed` episodes per seed. Returns (rewards, seed_for_each_ep).

    Episodes are independent — set `workers > 1` to fan out across processes
    (each worker spawns a clean Python interpreter and loads the model).
    """
    flat = [s + ep for s in seeds for ep in range(episodes_per_seed)]
    total = len(flat)
    items = list(enumerate(flat))
    rewards: list[float] = [0.0] * total
    seed_log: list[int] = [0] * total

    if workers <= 1:
        agent, _ = load_torch_policy(model_path)
        for idx, seed in items:
            env = make_cyborg_env(variant, seed, wrapper_class=EnterpriseMAE)
            r = rollout_episode(env, variant, ep_seed=seed, agent=agent, deterministic=deterministic)
            rewards[idx] = r
            seed_log[idx] = seed
            if progress:
                print(f"  ep {idx + 1}/{total} (seed={seed}): {r:.1f}", flush=True)
        return rewards, seed_log

    n_workers = min(workers, total)
    chunks = [items[i::n_workers] for i in range(n_workers)]
    pargs = [(str(model_path), deterministic, variant, c) for c in chunks]
    completed = 0
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        for chunk_results in ex.map(_cyborg_worker, pargs):
            for idx, seed, r in chunk_results:
                rewards[idx] = r
                seed_log[idx] = seed
                completed += 1
                if progress:
                    print(f"  ep {completed}/{total} (seed={seed}): {r:.1f}", flush=True)
    return rewards, seed_log
