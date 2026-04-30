"""CybORG rollout runner — load a torch checkpoint, evaluate on real CybORG.

Used by `scripts/eval/eval_recipe.py` for the `.pt` (torch state_dict)
path. JAX `.pkl` checkpoints take the parallel path through
`jaxborg.eval.jax_runner` (same target environment, different model loader
and action translation).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.policies import make_torch_policy

os.environ.setdefault("JAX_PLATFORMS", "cpu")

NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]
OBS_DIM = BLUE_OBS_SIZE
ACT_DIM = 242
EPISODE_LENGTH = 500


def make_env(seed: int):
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import EnterpriseMAE
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    return EnterpriseMAE(CybORG(sg, "sim", seed=seed))


def _pad_obs_mask(obs_dict, info_dict):
    obs = np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)
    mask = np.zeros((NUM_AGENTS, ACT_DIM), dtype=np.float32)
    for i, aid in enumerate(AGENT_IDS):
        raw_o = np.asarray(obs_dict[aid], dtype=np.float32)
        raw_m = np.asarray(info_dict[aid]["action_mask"], dtype=np.float32)
        obs[i, : len(raw_o)] = raw_o
        mask[i, : len(raw_m)] = raw_m
    return obs, mask


def rollout_episode(env, agent, *, deterministic: bool) -> float:
    obs_d, info_d = env.reset()
    total = 0.0
    for _ in range(EPISODE_LENGTH):
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
    return total


def load_torch_policy_from_recipe(recipe: dict[str, Any], state_dict: dict[str, torch.Tensor]):
    arch = recipe["arch"]
    agent = make_torch_policy(
        arch["name"],
        obs_dim=OBS_DIM,
        action_dim=ACT_DIM,
        hidden_dim=int(arch.get("hidden_dim", 256)),
        hidden_layers=int(arch.get("hidden_layers", 2)),
    )
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


def evaluate_on_cyborg(
    agent,
    *,
    seeds: list[int],
    episodes_per_seed: int,
    deterministic: bool = False,
    progress: bool = True,
) -> tuple[list[float], list[int]]:
    """Run `episodes_per_seed` episodes per seed. Returns (rewards, seed_for_each_ep)."""
    rewards: list[float] = []
    seed_log: list[int] = []
    total_eps = len(seeds) * episodes_per_seed
    n = 0
    for s in seeds:
        for ep in range(episodes_per_seed):
            env = make_env(s + ep)
            r = rollout_episode(env, agent, deterministic=deterministic)
            rewards.append(r)
            seed_log.append(s + ep)
            n += 1
            if progress:
                print(f"  ep {n}/{total_eps} (seed={s + ep}): {r:.1f}", flush=True)
    return rewards, seed_log
