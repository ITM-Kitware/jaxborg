"""Blue checkpoint curves against fixed final Reds from other training seeds."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from jaxborg.evaluation.play_priors import (
    _backend_from_model,
    _git_commit,
    _parse_seeds,
    find_periodic_checkpoints,
    select_checkpoints,
    with_final_checkpoint,
)


@dataclass(frozen=True)
class CrossSeedPlaySettings:
    enabled: bool = False
    seeds: tuple[int, ...] = tuple(range(1000, 1010))
    episodes_per_seed: int = 1
    deterministic: bool = False
    required: bool = True
    max_checkpoints: int = 1  # One preserves final-only evaluation; larger values include history.

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> "CrossSeedPlaySettings":
        raw = (recipe.get("eval") or {}).get("cross_seed_play", False)
        if raw is None or raw is False:
            return cls()
        if raw is True:
            return cls(enabled=True)
        if not isinstance(raw, Mapping):
            raise ValueError("eval.cross_seed_play must be a boolean or mapping")
        unknown = set(raw) - {"enabled", "seeds", "episodes_per_seed", "deterministic", "required", "max_checkpoints"}
        if unknown:
            raise ValueError(f"eval.cross_seed_play has unknown settings: {sorted(unknown)}")
        flags = {
            name: raw.get(name, default)
            for name, default in (("enabled", True), ("deterministic", False), ("required", True))
        }
        for name, value in flags.items():
            if not isinstance(value, bool):
                raise ValueError(f"eval.cross_seed_play.{name} must be a boolean")
        episodes = raw.get("episodes_per_seed", 1)
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
            raise ValueError("eval.cross_seed_play.episodes_per_seed must be a positive integer")
        maximum = raw.get("max_checkpoints", 1)
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError("eval.cross_seed_play.max_checkpoints must be a positive integer")
        try:
            seeds = _parse_seeds(raw.get("seeds", "1000-1009"))
        except ValueError as exc:
            raise ValueError(str(exc).replace("eval.play_priors", "eval.cross_seed_play")) from exc
        return cls(**flags, seeds=seeds, episodes_per_seed=episodes, max_checkpoints=maximum)


def validate_cross_seed_models(blue_model: str | Path, red_model: str | Path) -> Path:
    """Reject self-play, missing models and ambiguous training provenance."""
    from jaxborg.checkpoint import read_sidecar

    blue, red = (Path(path).expanduser().resolve() for path in (blue_model, red_model))
    for path in (blue, red):
        if not path.is_file():
            raise FileNotFoundError(f"cross-seed-play final model is missing: {path}")
        if not path.name.startswith("model_"):
            raise ValueError(f"cross-seed-play requires a final model bundle, not a periodic checkpoint: {path}")
    if blue == red:
        raise ValueError("cross-seed-play requires different Blue and Red bundles")
    if blue.suffix != red.suffix:
        raise ValueError("cross-seed-play requires Blue and Red to use the same backend")
    seeds = [read_sidecar(path).get("run", {}).get("seed") for path in (blue, red)]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("cross-seed-play requires run.seed in both model recipe sidecars")
    if seeds[0] == seeds[1]:
        raise ValueError("cross-seed-play requires different training seeds")
    return red


def validate_cross_seed_pool(blue_model: str | Path, red_models: Sequence[str | Path]) -> list[Path]:
    """Keep one final Red per held-out training seed, without duplicate weighting."""
    from jaxborg.checkpoint import read_sidecar

    reds = [validate_cross_seed_models(blue_model, red) for red in red_models]
    if not reds:
        raise ValueError("cross-seed-play requires at least one other-seed final Red")
    seeds = [read_sidecar(red)["run"]["seed"] for red in reds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("cross-seed-play requires distinct Red training seeds")
    return [red for _, red in sorted(zip(seeds, reds))]


def run_cross_seed_play(final_model, red_models, recipe, *, output=None, evaluate_fn=None, attach_metrics_fn=None):
    """Replay Blue history against every supplied final Red and log at Blue's steps."""
    from jaxborg.checkpoint import read_sidecar
    from jaxborg.evaluation.cia.reporting import cia_mlflow_metrics
    from jaxborg.recipe import eval_variant, project_eval

    settings = CrossSeedPlaySettings.from_recipe(recipe)
    if not settings.enabled:
        return None
    final_model = Path(final_model).expanduser().resolve()
    reds = validate_cross_seed_pool(final_model, red_models)
    run = recipe.get("run", {})
    # Use the final bundle's own provenance, including when an eval recipe overrides settings.
    saved_run = read_sidecar(final_model).get("run", {})
    if run != saved_run:
        raise ValueError("cross-seed-play recipe must preserve the final model's run provenance")
    checkpoints = find_periodic_checkpoints(final_model, recipe) if settings.max_checkpoints > 1 else []
    checkpoints = with_final_checkpoint(checkpoints, final_model, recipe, setting="eval.cross_seed_play")
    if settings.max_checkpoints > 1:
        checkpoints = select_checkpoints(checkpoints, settings.max_checkpoints)

    if evaluate_fn is None:
        from functools import partial

        from jaxborg.evaluation.matchup_runner import MatchupEvaluationContext, evaluate_matchup

        evaluate_fn = partial(evaluate_matchup, context=MatchupEvaluationContext())
    if attach_metrics_fn is None:
        from jaxborg.mlflow_setup import attach_eval_metrics

        attach_metrics_fn = attach_eval_metrics
    projection = project_eval(dict(recipe), materialize_topologies=True)
    backend = _backend_from_model(final_model)
    kwargs = {
        "backend": backend,
        "variant": eval_variant(dict(recipe)),
        "seeds": list(settings.seeds),
        "episodes_per_seed": settings.episodes_per_seed,
        "deterministic": settings.deterministic,
        "topology_path": list(projection["TOPOLOGY_BANK"]) or None,
        "topology_sampling": projection["TOPOLOGY_SAMPLING"],
        "cia": projection["CIA"],
    }
    eval_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
    exp_dir = Path(os.environ.get("JAXBORG_EXP_DIR", final_model.parents[2]))
    output = Path(output) if output else exp_dir / "eval" / f"{final_model.stem}_cross_seed_{eval_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    red_runs = {red: read_sidecar(red)["run"] for red in reds}
    prefix = "eval.cross_seed_play"
    commit = _git_commit()
    print(f"Cross-seed play: {len(checkpoints)} Blue checkpoints x {len(reds)} fixed final Reds", flush=True)
    with output.open("w") as stream:
        for blue in checkpoints:
            rows, metrics = [], {}
            for red in reds:
                red_seed = red_runs[red]["seed"]
                started = time.perf_counter()
                result = evaluate_fn(blue.path, red, **kwargs)
                reward = mean(result.blue_returns)
                spread = stdev(result.blue_returns) if len(result.blue_returns) > 1 else 0.0
                row = {
                    "eval_id": f"{eval_id}_{blue.steps}_{red_seed}",
                    "eval_name": "cross_seed_play",
                    "suite": "cross_seed_play",
                    "recipe_name": recipe.get("meta", {}).get("name", ""),
                    "train_run_id": run.get("train_run_id"),
                    "train_seed": run["seed"],
                    "red_train_seed": red_seed,
                    "blue_checkpoint": str(blue.path),
                    "red_checkpoint": str(red),
                    "blue_step": blue.steps,
                    "checkpoint_step": blue.steps,
                    "red_step": red_runs[red]["total_steps"],
                    "mean_reward": reward,
                    "std_reward": spread,
                    "n_episodes": len(result.blue_returns),
                    "blue_mean_return": reward,
                    "blue_std_return": spread,
                    "red_mean_return": -reward,
                    "red_std_return": spread,
                    "total_episodes": len(result.blue_returns),
                    "seeds": list(settings.seeds),
                    "episodes_per_seed": settings.episodes_per_seed,
                    "stochastic": not settings.deterministic,
                    "policy_backend": backend,
                    "trained_backend": backend,
                    "eval_env": "jax_joint",
                    "variant": kwargs["variant"].name,
                    "wall_time_s": time.perf_counter() - started,
                    "git_commit": commit,
                    "per_episode_blue_returns": result.blue_returns,
                    "per_episode_red_returns": result.red_returns,
                    "per_episode_seeds": result.episode_seeds,
                    "per_episode_topology_paths": result.episode_topology_paths,
                    "per_episode_topology_fingerprints": result.episode_topology_fingerprints,
                    **{
                        key: getattr(result, key)
                        for key in (
                            "policies",
                            "topology_paths",
                            "topology_sampling",
                            "episode_topology_paths",
                            "cia_metric",
                            "cia_config",
                            "cia_summary",
                            "per_episode_cia",
                            "episode_role_map_ids",
                            "episode_topology_fingerprints",
                            "topology_role_maps",
                        )
                    },
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
                if blue.path == final_model:
                    # Preserve single-matchup artifacts used by existing final-score plots.
                    final_row = {**row, "eval_name": "cross-seed-play"}
                    final_path = output.with_name(
                        f"{row['recipe_name']}_matchup_cross-seed-play_{eval_id}_red{red_seed}.jsonl"
                    )
                    final_path.write_text(json.dumps(final_row, indent=2) + "\n")
                opponent_prefix = f"{prefix}.red_seed_{red_seed}"
                metrics[f"{opponent_prefix}.blue_mean"] = reward
                metrics[f"{opponent_prefix}.blue_std"] = spread
                if result.cia_summary is not None:
                    metrics.update(cia_mlflow_metrics(f"{opponent_prefix}.cia", result.cia_summary))
                print(
                    f"  Blue seed {run['seed']} @{blue.steps:,} vs final Red seed {red_seed}: {reward:.2f}", flush=True
                )
            metrics[f"{prefix}.blue.mean_reward"] = mean(row["mean_reward"] for row in rows)
            metrics[f"{prefix}.blue.worst_reward"] = min(row["mean_reward"] for row in rows)
            summary = {
                "eval_name": "cross_seed_play_summary",
                "eval_id": f"{eval_id}_{blue.steps}",
                "recipe_name": recipe.get("meta", {}).get("name", ""),
                "train_run_id": run.get("train_run_id"),
                "train_seed": run["seed"],
                "blue_step": blue.steps,
                "red_train_seeds": [red_runs[red]["seed"] for red in reds],
                "mean_reward": metrics[f"{prefix}.blue.mean_reward"],
                "worst_reward": metrics[f"{prefix}.blue.worst_reward"],
            }
            stream.write(json.dumps(summary) + "\n")
            stream.flush()
            if blue.path == final_model:
                # The legacy final metric now averages all held-out Reds instead of being
                # overwritten by whichever opponent happened to run last.
                returns = [reward for row in rows for reward in row["per_episode_blue_returns"]]
                legacy = "eval.after_training.cross-seed-play.jax_matchup"
                metrics.update(
                    {
                        f"{legacy}.blue_mean": mean(returns),
                        f"{legacy}.red_mean": -mean(returns),
                        f"{legacy}.blue_std": stdev(returns) if len(returns) > 1 else 0.0,
                        f"{legacy}.red_std": stdev(returns) if len(returns) > 1 else 0.0,
                        f"{legacy}.episodes": len(returns),
                    }
                )
                episode_cia = [episode for row in rows for episode in row["per_episode_cia"]]
                if episode_cia:
                    from jaxborg.evaluation.cia.jax_resilience import CIA_KEYS, summarize_resilience_episodes

                    pooled_cia = summarize_resilience_episodes([[ep[key] for key in CIA_KEYS] for ep in episode_cia])
                    metrics.update(cia_mlflow_metrics(f"{legacy}.cia", pooled_cia))
            if run.get("train_run_id"):
                try:
                    attach_metrics_fn(run["train_run_id"], metrics, step=blue.steps)
                except Exception as exc:
                    print(f"MLflow attach warning for {run['train_run_id']}: {exc}", flush=True)
    print(f"Cross-seed results: {output}", flush=True)
    return output
