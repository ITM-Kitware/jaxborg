"""MLflow run setup and periodic frozen-checkpoint evaluation.

Both algorithm scripts call `start_run(recipe, backend=...)` to:
- point MLflow at `$JAXBORG_EXP_DIR/mlflow.db`
- start a run named `<algorithm>-<backend>-<recipe_name>-seed<n>`
- tag the run with recipe.{name,source,path}, algorithm, backend,
  arch.name, git.{commit,branch}
- log the resolved recipe as flat dotted-key params
- log the source recipe yaml as an artifact

Returns the active mlflow.ActiveRun. Trainer is responsible for
mlflow.end_run() at finish.

``MlflowCheckpointEvaluator`` copies portable model bundles into the active
run's artifacts, evaluates those exact files, and logs step-indexed reward
metrics that MLflow renders as time-series curves.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import mlflow

from jaxborg.recipe import flatten_for_logging, training_teams


@dataclass(frozen=True)
class CheckpointEvalSettings:
    """Validated periodic checkpoint-evaluation settings from a recipe."""

    every_steps: int = 0
    episodes_per_seed: int = 10
    seed: int | None = None
    deterministic: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.every_steps, bool) or not isinstance(self.every_steps, int):
            raise ValueError("mlflow.checkpoint_eval.every_steps must be an integer")
        if self.every_steps < 0:
            raise ValueError("mlflow.checkpoint_eval.every_steps must be non-negative")
        if isinstance(self.episodes_per_seed, bool) or not isinstance(self.episodes_per_seed, int):
            raise ValueError("mlflow.checkpoint_eval.episodes_per_seed must be an integer")
        if self.episodes_per_seed < 1:
            raise ValueError("mlflow.checkpoint_eval.episodes_per_seed must be positive")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("mlflow.checkpoint_eval.seed must be an integer or null")
        if self.seed is not None and self.seed < 0:
            raise ValueError("mlflow.checkpoint_eval.seed must be non-negative")
        if not isinstance(self.deterministic, bool):
            raise ValueError("mlflow.checkpoint_eval.deterministic must be a boolean")

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> CheckpointEvalSettings:
        mlflow_config = recipe.get("mlflow", {})
        if mlflow_config is None:
            mlflow_config = {}
        if not isinstance(mlflow_config, Mapping):
            raise ValueError("mlflow must be a mapping")
        eval_config = mlflow_config.get("checkpoint_eval", {})
        if eval_config is None:
            eval_config = {}
        if not isinstance(eval_config, Mapping):
            raise ValueError("mlflow.checkpoint_eval must be a mapping")
        if "episodes_per_seed" in eval_config and "episodes" in eval_config:
            raise ValueError("mlflow.checkpoint_eval may set episodes_per_seed or legacy episodes, not both")
        return cls(
            every_steps=eval_config.get("every_steps", 0),
            episodes_per_seed=eval_config.get("episodes_per_seed", eval_config.get("episodes", 10)),
            seed=eval_config.get("seed"),
            deterministic=eval_config.get("deterministic", False),
        )

    @property
    def episodes(self) -> int:
        """Deprecated compatibility alias for callers using the old field name."""

        return self.episodes_per_seed


def checkpoint_eval_due(
    previous_steps: int,
    completed_steps: int,
    every_steps: int,
    *,
    final: bool = False,
) -> bool:
    """Return whether an update crossed an evaluation boundary or is final."""

    for name, value in (("previous_steps", previous_steps), ("completed_steps", completed_steps)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if completed_steps < previous_steps:
        raise ValueError("completed_steps must not be less than previous_steps")
    if isinstance(every_steps, bool) or not isinstance(every_steps, int):
        raise ValueError("every_steps must be an integer")
    if every_steps < 0:
        raise ValueError("every_steps must be non-negative")
    if not isinstance(final, bool):
        raise ValueError("final must be a boolean")
    if every_steps == 0:
        return False
    crossed_boundary = previous_steps // every_steps < completed_steps // every_steps
    return crossed_boundary or final


def _numeric(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


class MlflowCheckpointEvaluator:
    """Freeze, evaluate, and log portable training checkpoints to MLflow."""

    def __init__(self, recipe: Mapping[str, Any], *, mlflow_module: Any | None = None) -> None:
        self.settings = CheckpointEvalSettings.from_recipe(recipe)
        self.trainable_teams = training_teams(dict(recipe))
        self._mlflow = mlflow if mlflow_module is None else mlflow_module

    @property
    def enabled(self) -> bool:
        return self.settings.every_steps > 0

    def due(self, previous_steps: int, completed_steps: int, *, final: bool = False) -> bool:
        return checkpoint_eval_due(
            previous_steps,
            completed_steps,
            self.settings.every_steps,
            final=final,
        )

    def on_checkpoint(
        self,
        checkpoint_path: str | Path,
        sidecar_path: str | Path,
        *,
        env_steps: int,
        evaluate_fn: Callable[[int], Mapping[str, float]],
    ) -> dict[str, float]:
        """Copy one exact bundle into MLflow, evaluate it, and log its curve point."""

        if not self.enabled:
            return {}
        if isinstance(env_steps, bool) or not isinstance(env_steps, int) or env_steps < 0:
            raise ValueError("env_steps must be a non-negative integer")

        artifact_path = f"checkpoints/step-{env_steps}"
        self._mlflow.log_artifact(str(checkpoint_path), artifact_path=artifact_path)
        self._mlflow.log_artifact(str(sidecar_path), artifact_path=artifact_path)

        raw_means = evaluate_fn(self.settings.episodes_per_seed)
        if not isinstance(raw_means, Mapping):
            raise ValueError("evaluate_fn must return a mapping of team names to mean rewards")

        means: dict[str, float] = {}
        for team, value in raw_means.items():
            if team not in ("blue", "red"):
                raise ValueError(f"unknown evaluation team {team!r}")
            if not _numeric(value):
                raise ValueError(f"evaluation reward for {team} must be numeric")
            means[team] = float(value)

        missing = set(self.trainable_teams) - set(means)
        if missing:
            raise ValueError(f"evaluation did not return trained teams: {sorted(missing)}")
        trained_means = {team: means[team] for team in self.trainable_teams}
        self._mlflow.log_metrics(
            {f"eval.checkpoint.{team}.mean_reward": value for team, value in trained_means.items()},
            step=env_steps,
        )
        return trained_means


def _exp_dir() -> Path:
    return Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()


def _git(arg: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", arg], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def configure(experiment: str | None = None) -> Path:
    """Point MLflow at $JAXBORG_EXP_DIR/mlflow.db. Return the db path."""
    exp_dir = _exp_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)
    db_path = exp_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    if experiment:
        mlflow.set_experiment(experiment)
    return db_path


def start_run(
    recipe: dict[str, Any],
    *,
    backend: str,
    seed: int,
    extra_tags: dict[str, str] | None = None,
    extra_params: dict[str, Any] | None = None,
):
    """Start an MLflow run and stamp it with recipe + git tags. Returns ActiveRun."""
    name = recipe.get("meta", {}).get("name", "unnamed")
    algorithm = recipe.get("algorithm", "ippo")
    arch_name = recipe.get("arch", {}).get("name", "shared")

    configure(experiment=f"{algorithm}-cc4")
    run_name = f"{algorithm}-{backend}-{name}-seed{seed}"
    run = mlflow.start_run(run_name=run_name)

    tags = {
        "recipe.name": name,
        "recipe.source": recipe.get("meta", {}).get("source", ""),
        "recipe.path": str(recipe.get("__source_path__", "")),
        "algorithm": algorithm,
        "backend": backend,
        "arch.name": arch_name,
        "seed": str(seed),
        "git.commit": _git("HEAD"),
        "git.branch": _git("--abbrev-ref HEAD"),
    }
    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags(tags)

    flat = flatten_for_logging(recipe)
    params = {f"recipe.{k}": v for k, v in flat.items()}
    if extra_params:
        params.update(extra_params)
    # MLflow caps param values at 500 chars and rejects unknown types.
    safe = {}
    for k, v in params.items():
        if v is None:
            continue
        s = str(v)
        if len(s) > 500:
            s = s[:497] + "..."
        safe[k] = s
    mlflow.log_params(safe)

    src = recipe.get("__source_path__")
    if src and Path(src).exists():
        try:
            mlflow.log_artifact(src)
        except Exception:
            pass

    return run


def attach_eval_metrics(
    train_run_id: str,
    metrics: dict[str, float],
) -> None:
    """Append eval metrics to the train run (used by eval_recipe.py)."""
    configure()
    with mlflow.start_run(run_id=train_run_id):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
