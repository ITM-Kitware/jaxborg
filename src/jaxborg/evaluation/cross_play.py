"""Cross-play matrix over durable checkpoints of a simultaneous Blue/Red run.

``play_priors`` compares each checkpoint with the one immediately before it.
That answers "is today better than yesterday?", and under a rock-paper-scissors
cycle the answer is yes on every single day while the policy walks in a circle.
Cycling is a statement about *non-transitivity over training time*, so it only
shows up away from the diagonal.

This module fills the rest of the matrix: for checkpoints ``c_0..c_{n-1}`` it
evaluates Blue from ``c_i`` against Red from ``c_j`` for every ordered pair,
including the ``i == j`` diagonal as the self-play reference.  The headline
readout is each policy's *worst* payoff against the whole of its own history.
Genuine progress makes that curve climb; a cycle leaves it flat or falling
while the adjacent-pair numbers still look like steady improvement.

The matrix costs ``n**2`` matchups, so ``max_checkpoints`` strides the run down
to an evenly spaced subsample that always keeps the first and last checkpoint.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from jaxborg.evaluation.play_priors import (
    PeriodicCheckpoint,
    _git_commit,
    _parse_seeds,
    find_periodic_checkpoints,
    select_checkpoints,
)

_ALLOWED_SETTINGS = {
    "enabled",
    "seeds",
    "episodes_per_seed",
    "deterministic",
    "required",
    "max_checkpoints",
}


@dataclass(frozen=True)
class CrossPlaySettings:
    """Validated ``eval.cross_play`` settings."""

    enabled: bool = False
    seeds: tuple[int, ...] = tuple(range(1000, 1003))
    episodes_per_seed: int = 1
    deterministic: bool = False
    required: bool = True
    max_checkpoints: int = 6

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> CrossPlaySettings:
        eval_config = recipe.get("eval", {})
        if eval_config is None:
            eval_config = {}
        if not isinstance(eval_config, Mapping):
            raise ValueError("eval must be a mapping")
        raw = eval_config.get("cross_play", False)
        if raw is None or raw is False:
            return cls()
        if raw is True:
            return cls(enabled=True)
        if not isinstance(raw, Mapping):
            raise ValueError("eval.cross_play must be a boolean or mapping")
        unknown = set(raw) - _ALLOWED_SETTINGS
        if unknown:
            raise ValueError(f"eval.cross_play has unknown settings: {sorted(unknown)}")

        enabled = raw.get("enabled", True)
        deterministic = raw.get("deterministic", False)
        required = raw.get("required", True)
        episodes_per_seed = raw.get("episodes_per_seed", 1)
        max_checkpoints = raw.get("max_checkpoints", 6)
        for name, value in (("enabled", enabled), ("deterministic", deterministic), ("required", required)):
            if not isinstance(value, bool):
                raise ValueError(f"eval.cross_play.{name} must be a boolean")
        for name, value in (("episodes_per_seed", episodes_per_seed), ("max_checkpoints", max_checkpoints)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"eval.cross_play.{name} must be an integer")
        if episodes_per_seed < 1:
            raise ValueError("eval.cross_play.episodes_per_seed must be positive")
        # Two checkpoints is the smallest matrix with an off-diagonal cell.
        if max_checkpoints < 2:
            raise ValueError("eval.cross_play.max_checkpoints must be at least 2")
        return cls(
            enabled=enabled,
            seeds=_parse_seeds(raw.get("seeds", "1000-1002")),
            episodes_per_seed=episodes_per_seed,
            deterministic=deterministic,
            required=required,
            max_checkpoints=max_checkpoints,
        )


def summarize_matrix(
    matrix: list[list[float]],
    checkpoints: list[PeriodicCheckpoint],
) -> dict[str, Any]:
    """Reduce a Blue-payoff matrix to progress and forgetting curves.

    ``matrix[i][j]`` is Blue's mean return with Blue from checkpoint ``i`` and
    Red from checkpoint ``j``; the game is zero-sum, so Red's payoff is its
    negation.  Blue prefers larger values, Red smaller.
    """
    n = len(checkpoints)
    blue_worst, blue_mean, red_worst, red_mean, diagonal = [], [], [], [], []
    for index in range(n):
        # Blue i has only met Red 0..i during training; later Reds are unseen.
        history = [matrix[index][j] for j in range(index + 1)]
        blue_worst.append(min(history))
        blue_mean.append(mean(history))
        # Red j against every Blue up to its own step, sign-flipped.
        red_history = [-matrix[i][index] for i in range(index + 1)]
        red_worst.append(min(red_history))
        red_mean.append(mean(red_history))
        diagonal.append(matrix[index][index])

    # Forgetting: a later policy doing worse than an earlier one against the
    # *same* older opponent. This is the cell-level signature of a cycle.
    blue_regressions = blue_total = 0
    red_regressions = red_total = 0
    for older in range(n):
        for early in range(older, n):
            for late in range(early + 1, n):
                blue_total += 1
                if matrix[late][older] < matrix[early][older]:
                    blue_regressions += 1
                red_total += 1
                if -matrix[older][late] < -matrix[older][early]:
                    red_regressions += 1

    return {
        "steps": [checkpoint.steps for checkpoint in checkpoints],
        "blue_payoff_matrix": matrix,
        "blue_self_play": diagonal,
        "blue_worst_vs_history": blue_worst,
        "blue_mean_vs_history": blue_mean,
        "red_worst_vs_history": red_worst,
        "red_mean_vs_history": red_mean,
        # Positive gain = the worst case against all of history improved.
        "blue_worst_vs_history_gain": blue_worst[-1] - blue_worst[0],
        "red_worst_vs_history_gain": red_worst[-1] - red_worst[0],
        "blue_forgetting_rate": (blue_regressions / blue_total) if blue_total else 0.0,
        "red_forgetting_rate": (red_regressions / red_total) if red_total else 0.0,
    }


def _cell_row(
    *,
    evaluation: Any,
    recipe: Mapping[str, Any],
    backend: str,
    blue: PeriodicCheckpoint,
    red: PeriodicCheckpoint,
    blue_index: int,
    red_index: int,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    deterministic: bool,
    wall_time_s: float,
    eval_id: str,
) -> dict[str, Any]:
    blue_mean = mean(evaluation.blue_returns)
    blue_std = stdev(evaluation.blue_returns) if len(evaluation.blue_returns) > 1 else 0.0
    row: dict[str, Any] = {
        "eval_id": f"{eval_id}_{blue.steps}_{red.steps}",
        "eval_name": "cross_play",
        "suite": "cross_play",
        "comparison": f"blue_{blue_index}_vs_red_{red_index}",
        "recipe_name": recipe.get("meta", {}).get("name", ""),
        "trained_backend": backend,
        "policy_backend": backend,
        "eval_env": "jax_joint",
        "variant": recipe.get("eval", {}).get("variant") or recipe.get("train", {}).get("variant"),
        "blue_checkpoint": str(blue.path),
        "red_checkpoint": str(red.path),
        "blue_checkpoint_index": blue_index,
        "red_checkpoint_index": red_index,
        "blue_step": blue.steps,
        "red_step": red.steps,
        "is_diagonal": blue_index == red_index,
        "policies": evaluation.policies,
        "seeds": list(seeds),
        "episodes_per_seed": episodes_per_seed,
        "total_episodes": len(evaluation.blue_returns),
        "stochastic": not deterministic,
        "mean_reward": blue_mean,
        "std_reward": blue_std,
        "n_episodes": len(evaluation.blue_returns),
        "blue_mean_return": blue_mean,
        "blue_std_return": blue_std,
        "red_mean_return": -blue_mean,
        "red_std_return": blue_std,
        "wall_time_s": wall_time_s,
        "git_commit": _git_commit(),
        "train_run_id": recipe.get("run", {}).get("train_run_id"),
        "train_seed": recipe.get("run", {}).get("seed"),
        "per_episode_blue_returns": evaluation.blue_returns,
        "per_episode_red_returns": evaluation.red_returns,
        "per_episode_seeds": evaluation.episode_seeds,
        "topology_paths": evaluation.topology_paths,
        "topology_sampling": evaluation.topology_sampling,
    }
    cia_summary = getattr(evaluation, "cia_summary", None)
    if cia_summary is not None:
        row["cia_metric"] = getattr(evaluation, "cia_metric", None)
        row["cia_summary"] = cia_summary
    return row


def run_cross_play(
    final_model: str | Path,
    recipe: Mapping[str, Any],
    *,
    output: str | Path | None = None,
    evaluate_fn: Callable[..., Any] | None = None,
    attach_metrics_fn: Callable[..., Any] | None = None,
) -> Path | None:
    """Evaluate every ordered checkpoint pair and summarize the matrix."""

    settings = CrossPlaySettings.from_recipe(recipe)
    if not settings.enabled:
        return None
    if recipe.get("train", {}).get("teams", "blue") != "both":
        raise ValueError("eval.cross_play is only supported when train.teams is 'both'")

    resolved_model = Path(final_model).expanduser().resolve()
    backend = "cyborg" if resolved_model.suffix == ".pt" else "jax"
    checkpoints = select_checkpoints(
        find_periodic_checkpoints(resolved_model, recipe),
        settings.max_checkpoints,
    )
    if len(checkpoints) < 2:
        raise ValueError(
            "eval.cross_play requires at least two saved periodic checkpoints; "
            f"found {len(checkpoints)} in {resolved_model.parent}"
        )

    from jaxborg.recipe import eval_variant, project_eval

    if evaluate_fn is None:
        from jaxborg.evaluation.matchup_runner import evaluate_matchup

        evaluate = evaluate_matchup
    else:
        evaluate = evaluate_fn
    if attach_metrics_fn is None:
        from jaxborg.mlflow_setup import attach_eval_metrics

        attach = attach_eval_metrics
    else:
        attach = attach_metrics_fn

    projected_eval = project_eval(dict(recipe), materialize_topologies=True)
    topology_paths = list(projected_eval["TOPOLOGY_BANK"]) or None
    topology_sampling = projected_eval.get(
        "TOPOLOGY_SAMPLING",
        (recipe.get("eval") or {}).get("topology_sampling", "exhaustive"),
    )
    from jaxborg.evaluation.cia.config import CIAEvalSettings

    cia_config = projected_eval.get("CIA", CIAEvalSettings.from_recipe(recipe).as_dict())
    variant = eval_variant(dict(recipe))
    eval_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"

    n = len(checkpoints)
    rows: list[dict[str, Any]] = []
    matrix = [[0.0] * n for _ in range(n)]
    print(f"Cross-play evaluation: {n} checkpoints, {n * n} matchups", flush=True)
    for blue_index, blue in enumerate(checkpoints):
        for red_index, red in enumerate(checkpoints):
            started = time.perf_counter()
            evaluation_kwargs: dict[str, Any] = {
                "backend": backend,
                "variant": variant,
                "seeds": list(settings.seeds),
                "episodes_per_seed": settings.episodes_per_seed,
                "deterministic": settings.deterministic,
                "topology_path": topology_paths,
                "topology_sampling": topology_sampling,
            }
            if cia_config["enabled"]:
                evaluation_kwargs["cia"] = cia_config
            evaluation = evaluate(blue.path, red.path, **evaluation_kwargs)
            row = _cell_row(
                evaluation=evaluation,
                recipe=recipe,
                backend=backend,
                blue=blue,
                red=red,
                blue_index=blue_index,
                red_index=red_index,
                seeds=settings.seeds,
                episodes_per_seed=settings.episodes_per_seed,
                deterministic=settings.deterministic,
                wall_time_s=time.perf_counter() - started,
                eval_id=eval_id,
            )
            rows.append(row)
            matrix[blue_index][red_index] = row["blue_mean_return"]
            print(
                f"  blue@{blue.steps:,} vs red@{red.steps:,}: {row['blue_mean_return']:.2f}",
                flush=True,
            )

    summary = summarize_matrix(matrix, checkpoints)
    rows.append(
        {
            "eval_id": f"{eval_id}_summary",
            "eval_name": "cross_play_summary",
            "suite": "cross_play",
            "recipe_name": recipe.get("meta", {}).get("name", ""),
            "git_commit": _git_commit(),
            "train_run_id": recipe.get("run", {}).get("train_run_id"),
            "train_seed": recipe.get("run", {}).get("seed"),
            "checkpoint_steps": summary["steps"],
            **summary,
        }
    )

    train_run_id = recipe.get("run", {}).get("train_run_id")
    if train_run_id:
        try:
            for index, checkpoint in enumerate(checkpoints):
                attach(
                    train_run_id,
                    {
                        "eval.cross_play.blue.self_play": summary["blue_self_play"][index],
                        "eval.cross_play.blue.worst_vs_history": summary["blue_worst_vs_history"][index],
                        "eval.cross_play.blue.mean_vs_history": summary["blue_mean_vs_history"][index],
                        "eval.cross_play.red.worst_vs_history": summary["red_worst_vs_history"][index],
                        "eval.cross_play.red.mean_vs_history": summary["red_mean_vs_history"][index],
                    },
                    step=checkpoint.steps,
                )
            attach(
                train_run_id,
                {
                    "eval.cross_play.blue.worst_vs_history_gain": summary["blue_worst_vs_history_gain"],
                    "eval.cross_play.red.worst_vs_history_gain": summary["red_worst_vs_history_gain"],
                    "eval.cross_play.blue.forgetting_rate": summary["blue_forgetting_rate"],
                    "eval.cross_play.red.forgetting_rate": summary["red_forgetting_rate"],
                    "eval.cross_play.checkpoints": float(n),
                },
                step=checkpoints[-1].steps,
            )
        except Exception as exc:
            print(f"MLflow attach warning for {train_run_id}: {exc}", flush=True)

    exp_dir = Path(os.environ.get("JAXBORG_EXP_DIR", resolved_model.parents[2])).expanduser().resolve()
    run_tag = resolved_model.stem.removeprefix("model_")
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else exp_dir / "eval" / f"{run_tag}_cross_play_{eval_id}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(
        f"Cross-play summary: blue worst-vs-history gain "
        f"{summary['blue_worst_vs_history_gain']:+.2f}, forgetting rate "
        f"{summary['blue_forgetting_rate']:.2%}",
        flush=True,
    )
    print(f"Wrote cross-play results: {output_path}", flush=True)
    return output_path


__all__ = [
    "CrossPlaySettings",
    "run_cross_play",
    "summarize_matrix",
]
