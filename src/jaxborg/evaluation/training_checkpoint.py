"""Evaluate immutable training checkpoints for experiment tracking.

The training loops deliberately call this module only after writing a model
bundle and its recipe sidecar.  Evaluation therefore measures the exact
checkpoint uploaded to the tracker rather than mutable in-memory parameters.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from jaxborg.mlflow_setup import CheckpointEvalSettings
from jaxborg.recipe import eval_variant, project_eval, training_teams


def _uses_learned_red(recipe: dict) -> bool:
    teams = training_teams(recipe)
    opponents = recipe.get("train", {}).get("opponents") or {}
    return "red" in teams or bool(opponents.get("red"))


@contextmanager
def _paired_eval_rng(*, seed: int, torch_rng: bool) -> Iterator[None]:
    """Use paired eval randomness without perturbing the trainer's RNG streams."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = None
    torch = None
    if torch_rng:
        import torch as torch_module

        torch = torch_module
        torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        if torch is not None:
            torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None and torch_state is not None:
            torch.random.set_rng_state(torch_state)


def evaluate_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    backend: str,
    recipe: dict,
    seed: int,
    episodes: int = 10,
) -> dict[str, float]:
    """Return mean checkpoint rewards for each policy team being evaluated.

    Learned Blue/Red matchups use the existing JAX-native matchup evaluator
    and load both policies from the same versioned bundle.  Legacy Blue-only
    runs retain their existing CybORG contract evaluator.  A fixed seed range
    is reused at every checkpoint so the sparse evaluation curve is paired.
    """

    if episodes < 1:
        raise ValueError(f"episodes must be positive, got {episodes}")
    backend_name = "cyborg" if backend in ("cleanrl", "torch") else backend
    if backend_name not in ("jax", "cyborg"):
        raise ValueError(f"backend must be 'jax' or 'cyborg', got {backend!r}")

    checkpoint_path = Path(checkpoint_path)
    settings = CheckpointEvalSettings.from_recipe(recipe)
    configured_seed = settings.seed
    eval_seed = int(seed) + 100_000 if configured_seed is None else int(configured_seed)
    deterministic = settings.deterministic
    variant = eval_variant(recipe)
    trainable_teams = training_teams(recipe)

    with _paired_eval_rng(seed=eval_seed, torch_rng=backend_name == "cyborg"):
        if _uses_learned_red(recipe):
            from jaxborg.evaluation.matchup_runner import evaluate_matchup

            eval_config = project_eval(recipe, materialize_topologies=True)
            topology_bank = eval_config["TOPOLOGY_BANK"] or None
            result = evaluate_matchup(
                checkpoint_path,
                checkpoint_path,
                backend=backend_name,
                variant=variant,
                seeds=[eval_seed],
                episodes_per_seed=episodes,
                deterministic=deterministic,
                progress=False,
                topology_path=topology_bank,
                topology_sampling=eval_config["TOPOLOGY_SAMPLING"],
            )
            means = {
                "blue": float(np.mean(result.blue_returns)),
                "red": float(np.mean(result.red_returns)),
            }
            return {team: means[team] for team in trainable_teams}

        if backend_name == "jax":
            from jaxborg.evaluation.jax_runner import evaluate_jax_on_cyborg

            rewards, _seed_log, _checkpoint_recipe = evaluate_jax_on_cyborg(
                checkpoint_path,
                variant=variant,
                seeds=[eval_seed],
                episodes_per_seed=episodes,
                deterministic=deterministic,
                workers=1,
                progress=False,
            )
        else:
            from jaxborg.evaluation.cyborg_runner import evaluate_on_cyborg

            rewards, _seed_log = evaluate_on_cyborg(
                checkpoint_path,
                variant=variant,
                seeds=[eval_seed],
                episodes_per_seed=episodes,
                deterministic=deterministic,
                workers=1,
                progress=False,
            )
        return {"blue": float(np.mean(rewards))}


__all__ = ["evaluate_training_checkpoint"]
