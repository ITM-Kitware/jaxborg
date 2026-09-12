"""Adjacent-checkpoint cross-play for simultaneous Blue/Red training.

For every pair of durable periodic checkpoints ``(i - 1, i)``, evaluate the
current Blue policy against the prior Red policy and the current Red policy
against the prior Blue policy.  The same evaluation seeds are used in both
directions so the two curves remain paired.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

_CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)$")
_ALLOWED_SETTINGS = {
    "enabled",
    "seeds",
    "episodes_per_seed",
    "deterministic",
    "required",
    "max_checkpoints",
}


def _parse_seeds(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        seeds: set[int] = set()
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError("eval.play_priors.seeds ranges must be ascending")
                seeds.update(range(start, end + 1))
            else:
                seeds.add(int(token))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
            raise ValueError("eval.play_priors.seeds must contain only integers")
        seeds = set(value)
    else:
        raise ValueError("eval.play_priors.seeds must be a range string or list of integers")
    if not seeds:
        raise ValueError("eval.play_priors.seeds must contain at least one seed")
    if min(seeds) < 0:
        raise ValueError("eval.play_priors.seeds must be non-negative")
    return tuple(sorted(seeds))


@dataclass(frozen=True)
class PlayPriorsSettings:
    """Validated ``eval.play_priors`` settings."""

    enabled: bool = False
    seeds: tuple[int, ...] = tuple(range(1000, 1010))
    episodes_per_seed: int = 1
    deterministic: bool = False
    required: bool = True
    max_checkpoints: int = 6

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> PlayPriorsSettings:
        eval_config = recipe.get("eval", {})
        if eval_config is None:
            eval_config = {}
        if not isinstance(eval_config, Mapping):
            raise ValueError("eval must be a mapping")
        raw = eval_config.get("play_priors", False)
        if raw is None or raw is False:
            return cls()
        if raw is True:
            return cls(enabled=True)
        if not isinstance(raw, Mapping):
            raise ValueError("eval.play_priors must be a boolean or mapping")
        unknown = set(raw) - _ALLOWED_SETTINGS
        if unknown:
            raise ValueError(f"eval.play_priors has unknown settings: {sorted(unknown)}")

        enabled = raw.get("enabled", True)
        deterministic = raw.get("deterministic", False)
        required = raw.get("required", True)
        episodes_per_seed = raw.get("episodes_per_seed", 1)
        max_checkpoints = raw.get("max_checkpoints", 6)
        if not isinstance(enabled, bool):
            raise ValueError("eval.play_priors.enabled must be a boolean")
        if not isinstance(deterministic, bool):
            raise ValueError("eval.play_priors.deterministic must be a boolean")
        if not isinstance(required, bool):
            raise ValueError("eval.play_priors.required must be a boolean")
        for name, value in (("episodes_per_seed", episodes_per_seed), ("max_checkpoints", max_checkpoints)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"eval.play_priors.{name} must be an integer")
        if episodes_per_seed < 1:
            raise ValueError("eval.play_priors.episodes_per_seed must be positive")
        # Two checkpoints is the smallest subsample with an adjacent pair.
        if max_checkpoints < 2:
            raise ValueError("eval.play_priors.max_checkpoints must be at least 2")
        seeds = _parse_seeds(raw.get("seeds", "1000-1009"))
        return cls(
            enabled=enabled,
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            deterministic=deterministic,
            required=required,
            max_checkpoints=max_checkpoints,
        )


@dataclass(frozen=True)
class PeriodicCheckpoint:
    path: Path
    steps: int


def _backend_from_model(model_path: Path) -> str:
    if model_path.suffix == ".pt":
        return "cyborg"
    if model_path.suffix == ".safetensors":
        return "jax"
    raise ValueError(f"cannot detect trained backend from model suffix: {model_path}")


def _periodic_step_stride(recipe: Mapping[str, Any], backend: str) -> int:
    train = recipe.get("train", {})
    if not isinstance(train, Mapping):
        raise ValueError("train must be a mapping")
    episode_length = int(train["episode_length"])
    if backend == "jax":
        section_name = "jax"
        backend_config = recipe.get("jax", {}) or {}
        checkpoint_every = int(backend_config.get("checkpoint_every_updates", 50))
        steps_per_update = int(backend_config.get("num_envs", 1024)) * episode_length
    else:
        section_name = "cleanrl"
        backend_config = recipe.get("cleanrl", {}) or {}
        checkpoint_every = int(backend_config.get("checkpoint_every_updates", 50))
        num_envs = int(backend_config.get("num_envs", 48))
        rollout_length = int(backend_config.get("rollout_length", episode_length))
        per_rollout = num_envs * rollout_length
        if "num_rollouts_per_update" in backend_config:
            rollouts_per_update = int(backend_config["num_rollouts_per_update"])
        else:
            rollouts_per_update = max(1, math.ceil(int(train["buffer_size"]) / per_rollout))
        steps_per_update = per_rollout * rollouts_per_update
    if checkpoint_every <= 0:
        raise ValueError(f"eval.play_priors requires {section_name}.checkpoint_every_updates to be positive")
    return checkpoint_every * steps_per_update


def find_periodic_checkpoints(
    final_model: str | Path,
    recipe: Mapping[str, Any],
) -> list[PeriodicCheckpoint]:
    """Return durable periodic checkpoints, excluding MLflow-only snapshots."""

    model_path = Path(final_model).expanduser().resolve()
    backend = _backend_from_model(model_path)
    stride = _periodic_step_stride(recipe, backend)
    checkpoints: list[PeriodicCheckpoint] = []
    for path in model_path.parent.glob(f"checkpoint_*{model_path.suffix}"):
        match = _CHECKPOINT_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        steps = int(match.group(1))
        if steps % stride == 0:
            checkpoints.append(PeriodicCheckpoint(path.resolve(), steps))
    checkpoints.sort(key=lambda checkpoint: checkpoint.steps)
    return checkpoints


def select_checkpoints(
    checkpoints: list[PeriodicCheckpoint],
    max_checkpoints: int,
) -> list[PeriodicCheckpoint]:
    """Stride to an evenly spaced subsample keeping the first and last.

    Every checkpoint evaluation scales in the number of checkpoints -- play
    priors linearly, cross-play quadratically -- so each of them caps the count
    under its own ``max_checkpoints`` key rather than sharing one global.
    """
    if max_checkpoints < 2:
        raise ValueError("max_checkpoints must be at least 2")
    total = len(checkpoints)
    if total <= max_checkpoints:
        return list(checkpoints)
    picked = sorted({round(i * (total - 1) / (max_checkpoints - 1)) for i in range(max_checkpoints)})
    return [checkpoints[index] for index in picked]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _result_row(
    *,
    evaluation: Any,
    recipe: Mapping[str, Any],
    backend: str,
    focal_team: str,
    current: PeriodicCheckpoint,
    prior: PeriodicCheckpoint,
    current_index: int,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    deterministic: bool,
    wall_time_s: float,
    eval_id: str,
) -> dict[str, Any]:
    focal_returns = evaluation.blue_returns if focal_team == "blue" else evaluation.red_returns
    focal_mean = mean(focal_returns)
    focal_std = stdev(focal_returns) if len(focal_returns) > 1 else 0.0
    blue_mean = mean(evaluation.blue_returns)
    blue_std = stdev(evaluation.blue_returns) if len(evaluation.blue_returns) > 1 else 0.0
    cia_fields: dict[str, Any] = {}
    cia_summary = getattr(evaluation, "cia_summary", None)
    if cia_summary is not None:
        from jaxborg.evaluation.cia.config import CIAEvalSettings

        settings = CIAEvalSettings.from_recipe(recipe)
        cia_fields = {
            "cia_metric": getattr(evaluation, "cia_metric", None) or settings.metric,
            "cia_config": getattr(evaluation, "cia_config", None) or settings.as_dict(),
            "cia_summary": cia_summary,
            "per_episode_cia": getattr(evaluation, "per_episode_cia", []),
            "episode_role_map_ids": getattr(evaluation, "episode_role_map_ids", []),
            "per_episode_topology_fingerprints": getattr(
                evaluation,
                "episode_topology_fingerprints",
                [],
            ),
            "topology_role_maps": getattr(evaluation, "topology_role_maps", []),
        }
    return {
        "eval_id": f"{eval_id}_{current.steps}_{focal_team}",
        "eval_name": "play_priors",
        "suite": "play_priors",
        "comparison": f"current_{focal_team}_vs_prior_{'red' if focal_team == 'blue' else 'blue'}",
        "focal_team": focal_team,
        "recipe_name": recipe.get("meta", {}).get("name", ""),
        "recipe_path": recipe.get("__source_path__", recipe.get("meta", {}).get("source_path", "")),
        "trained_backend": backend,
        "policy_backend": backend,
        "eval_env": "jax_joint",
        "variant": recipe.get("eval", {}).get("variant") or recipe.get("train", {}).get("variant"),
        "model": str(current.path),
        "current_checkpoint": str(current.path),
        "prior_checkpoint": str(prior.path),
        "current_checkpoint_index": current_index,
        "prior_checkpoint_index": current_index - 1,
        "current_step": current.steps,
        "prior_step": prior.steps,
        "policies": evaluation.policies,
        "seeds": list(seeds),
        "episodes_per_seed": episodes_per_seed,
        "total_episodes": len(focal_returns),
        "stochastic": not deterministic,
        "mean_reward": focal_mean,
        "std_reward": focal_std,
        "n_episodes": len(focal_returns),
        "blue_mean_return": blue_mean,
        "blue_std_return": blue_std,
        "red_mean_return": -blue_mean,
        "red_std_return": blue_std,
        "wall_time_s": wall_time_s,
        "git_commit": _git_commit(),
        "train_run_id": recipe.get("run", {}).get("train_run_id"),
        "train_seed": recipe.get("run", {}).get("seed"),
        "per_episode": focal_returns,
        "per_episode_blue_returns": evaluation.blue_returns,
        "per_episode_red_returns": evaluation.red_returns,
        "per_episode_seeds": evaluation.episode_seeds,
        "topology_paths": evaluation.topology_paths,
        "topology_sampling": evaluation.topology_sampling,
        "per_episode_topology_paths": evaluation.episode_topology_paths,
        **cia_fields,
    }


def run_play_priors(
    final_model: str | Path,
    recipe: Mapping[str, Any],
    *,
    output: str | Path | None = None,
    evaluate_fn: Callable[..., Any] | None = None,
    attach_metrics_fn: Callable[..., Any] | None = None,
) -> Path | None:
    """Evaluate every adjacent durable checkpoint pair in both directions."""

    settings = PlayPriorsSettings.from_recipe(recipe)
    if not settings.enabled:
        return None
    if recipe.get("train", {}).get("teams", "blue") != "both":
        raise ValueError("eval.play_priors is only supported when train.teams is 'both'")

    resolved_model = Path(final_model).expanduser().resolve()
    backend = _backend_from_model(resolved_model)
    checkpoints = find_periodic_checkpoints(resolved_model, recipe)
    if len(checkpoints) < 2:
        raise ValueError(
            "eval.play_priors requires at least two saved periodic checkpoints; "
            f"found {len(checkpoints)} in {resolved_model.parent}"
        )
    # Pairs are adjacent within the subsample, not on disk: a 960k-step stride
    # puts neighbours close enough that the two policies barely differ, and the
    # untrimmed pair count is what makes this the longest eval in the chain.
    checkpoints = select_checkpoints(checkpoints, settings.max_checkpoints)

    from jaxborg.recipe import eval_variant, project_eval

    if evaluate_fn is None:
        from functools import partial

        from jaxborg.evaluation.matchup_runner import MatchupEvaluationContext, evaluate_matchup

        evaluate = partial(evaluate_matchup, context=MatchupEvaluationContext())
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
    rows: list[dict[str, Any]] = []

    print(f"Play-priors evaluation: {len(checkpoints)} checkpoints, {len(checkpoints) - 1} adjacent pairs")
    for current_index, (prior, current) in enumerate(zip(checkpoints, checkpoints[1:]), start=2):
        pair_rows: dict[str, dict[str, Any]] = {}
        matchups = {
            "blue": (current.path, prior.path),
            "red": (prior.path, current.path),
        }
        for focal_team, (blue_path, red_path) in matchups.items():
            started = time.perf_counter()
            evaluation_kwargs = {
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
            evaluation = evaluate(blue_path, red_path, **evaluation_kwargs)
            row = _result_row(
                evaluation=evaluation,
                recipe=recipe,
                backend=backend,
                focal_team=focal_team,
                current=current,
                prior=prior,
                current_index=current_index,
                seeds=settings.seeds,
                episodes_per_seed=settings.episodes_per_seed,
                deterministic=settings.deterministic,
                wall_time_s=time.perf_counter() - started,
                eval_id=eval_id,
            )
            rows.append(row)
            pair_rows[focal_team] = row
            print(
                f"  checkpoint {current_index} ({current.steps:,}) {row['comparison']}: "
                f"{row['mean_reward']:.2f} ± {row['std_reward']:.2f}",
                flush=True,
            )

        train_run_id = recipe.get("run", {}).get("train_run_id")
        if train_run_id:
            try:
                metrics = {
                    "eval.play_priors.blue_vs_prior_red.mean_reward": pair_rows["blue"]["mean_reward"],
                    "eval.play_priors.red_vs_prior_blue.mean_reward": pair_rows["red"]["mean_reward"],
                    "eval.play_priors.prior_step": float(prior.steps),
                }
                if cia_config["enabled"]:
                    from jaxborg.evaluation.cia.reporting import cia_mlflow_metrics

                    metrics.update(
                        cia_mlflow_metrics(
                            "eval.play_priors.blue_vs_prior_red.cia",
                            pair_rows["blue"]["cia_summary"],
                        )
                    )
                    metrics.update(
                        cia_mlflow_metrics(
                            "eval.play_priors.red_vs_prior_blue.cia",
                            pair_rows["red"]["cia_summary"],
                        )
                    )
                attach(train_run_id, metrics, step=current.steps)
            except Exception as exc:
                print(f"MLflow attach warning for {train_run_id}: {exc}", flush=True)

    exp_dir = Path(os.environ.get("JAXBORG_EXP_DIR", resolved_model.parents[2])).expanduser().resolve()
    run_tag = resolved_model.stem.removeprefix("model_")
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else exp_dir / "eval" / f"{run_tag}_play_priors_{eval_id}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote play-priors results: {output_path}", flush=True)
    return output_path


__all__ = [
    "PeriodicCheckpoint",
    "PlayPriorsSettings",
    "find_periodic_checkpoints",
    "run_play_priors",
    "select_checkpoints",
]
