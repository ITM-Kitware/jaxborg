"""Optional Weights & Biases tracking for training and checkpoint evaluation.

The module deliberately imports :mod:`wandb` only after a recipe opts in. This
keeps local training and test environments independent of W&B when tracking is
disabled.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from jaxborg.recipe import flatten_for_logging, training_teams


@dataclass(frozen=True)
class WandbSettings:
    """Validated W&B settings projected from a training recipe."""

    enabled: bool = False
    project: str = "jaxborg"
    run_name: str = "jaxborg"
    eval_episodes: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("wandb must be a boolean")
        if not isinstance(self.project, str) or not self.project.strip():
            raise ValueError("wandb_project must be a non-empty string")
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise ValueError("wandb_run_name must be a non-empty string")
        if isinstance(self.eval_episodes, bool) or not isinstance(self.eval_episodes, int):
            raise ValueError("wandb_eval_episodes must be an integer")
        if self.eval_episodes < 1:
            raise ValueError("wandb_eval_episodes must be positive")

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> WandbSettings:
        """Resolve top-level W&B settings, including the source filename default."""

        configured_name = recipe.get("wandb_run_name")
        if configured_name is None:
            source_path = recipe.get("__source_path__")
            source_name = Path(str(source_path)).stem if source_path else ""
            configured_name = source_name or recipe.get("meta", {}).get("name") or "jaxborg"
        return cls(
            enabled=recipe.get("wandb", False),
            project=recipe.get("wandb_project", "jaxborg"),
            run_name=configured_name,
            eval_episodes=recipe.get("wandb_eval_episodes", 10),
        )


def milestones_for_update(
    completed_updates: int,
    total_updates: int,
    interval_percent: int = 5,
) -> list[int]:
    """Return percentage milestones crossed by one completed training update.

    Integer comparisons avoid floating-point boundary errors. When a short run
    crosses several thresholds in one update, all thresholds are returned so a
    caller can attach several aliases to one checkpoint instead of saving it
    repeatedly. The terminal update always includes 100 percent.
    """

    for name, value in (("completed_updates", completed_updates), ("total_updates", total_updates)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if total_updates < 1:
        raise ValueError("total_updates must be positive")
    if completed_updates < 0 or completed_updates > total_updates:
        raise ValueError("completed_updates must be between zero and total_updates")
    if isinstance(interval_percent, bool) or not isinstance(interval_percent, int):
        raise ValueError("interval_percent must be an integer")
    if not 1 <= interval_percent <= 100:
        raise ValueError("interval_percent must be between 1 and 100")
    if completed_updates == 0:
        return []

    thresholds = list(range(interval_percent, 100, interval_percent)) + [100]
    previous_progress = (completed_updates - 1) * 100
    current_progress = completed_updates * 100
    return [percent for percent in thresholds if previous_progress < percent * total_updates <= current_progress]


def _numeric(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


class WandbCallback:
    """Small framework-neutral W&B hook for the repository's training loops."""

    def __init__(
        self,
        recipe: Mapping[str, Any],
        *,
        backend: str,
        seed: int,
        wandb_module: Any | None = None,
    ) -> None:
        self.settings = WandbSettings.from_recipe(recipe)
        self.trainable_teams = training_teams(dict(recipe))
        self.run: Any | None = None
        self._wandb: Any | None = None
        self._finished = False

        if not self.settings.enabled:
            return

        self._wandb = wandb_module if wandb_module is not None else importlib.import_module("wandb")
        config = flatten_for_logging(dict(recipe))
        config.update({"backend": backend, "seed": seed})
        self.run = self._wandb.init(
            project=self.settings.project,
            name=self.settings.run_name,
            config=config,
        )
        self.run.define_metric("global_step")
        self.run.define_metric("train/*", step_metric="global_step")
        self.run.define_metric("eval/*", step_metric="global_step")

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def milestones(self, completed_updates: int, total_updates: int) -> list[int]:
        """Return due 5-percent milestones only when W&B tracking is enabled."""

        if not self.enabled:
            return []
        return milestones_for_update(completed_updates, total_updates)

    def log_training(self, row: Mapping[str, Any]) -> None:
        """Log a standardized metrics row and stable team reward aliases."""

        if self.run is None:
            return
        if not _numeric(row.get("env_steps")):
            raise ValueError("training metrics require a numeric env_steps value")

        payload = {f"train/{key}": float(value) for key, value in row.items() if _numeric(value)}
        payload["global_step"] = int(row["env_steps"])

        for team in self.trainable_teams:
            reward = self._team_training_reward(row, team)
            if reward is not None:
                payload[f"train/{team}_reward"] = reward
        self.run.log(payload)

    def _team_training_reward(self, row: Mapping[str, Any], team: str) -> float | None:
        for key in (
            f"team.{team}.return",
            f"team.{team}.train_episode_reward_mean",
            f"team.{team}.reward",
        ):
            value = row.get(key)
            if _numeric(value):
                return float(value)

        if len(self.trainable_teams) == 1:
            value = row.get("train_episode_reward_mean")
            if _numeric(value):
                return float(value)
        return None

    def on_checkpoint(
        self,
        checkpoint_path: str | Path,
        sidecar_path: str | Path,
        env_steps: int,
        milestones: Sequence[int],
        evaluate_fn: Callable[[int], Mapping[str, float]],
    ) -> dict[str, float]:
        """Evaluate and version one milestone checkpoint, returning team means."""

        if self.run is None:
            return {}
        milestone_values = self._validate_milestones(milestones)
        if not milestone_values:
            return {}
        if isinstance(env_steps, bool) or not isinstance(env_steps, int) or env_steps < 0:
            raise ValueError("env_steps must be a non-negative integer")

        raw_means = evaluate_fn(self.settings.eval_episodes)
        if not isinstance(raw_means, Mapping):
            raise ValueError("evaluate_fn must return a mapping of team names to mean rewards")
        eval_means: dict[str, float] = {}
        for team, value in raw_means.items():
            if team not in ("blue", "red"):
                raise ValueError(f"unknown evaluation team {team!r}")
            if not _numeric(value):
                raise ValueError(f"evaluation reward for {team} must be numeric")
            eval_means[team] = float(value)

        eval_metrics = {f"eval/{team}_reward": value for team, value in eval_means.items()}
        self.run.log({"global_step": env_steps, **eval_metrics})

        artifact_id = getattr(self.run, "id", self.settings.run_name)
        metadata = {
            "global_step": env_steps,
            "training_percent": max(milestone_values),
            "training_milestones": milestone_values,
            **eval_metrics,
        }
        artifact = self._wandb.Artifact(
            f"checkpoint-{artifact_id}",
            type="model",
            metadata=metadata,
        )
        artifact.add_file(str(checkpoint_path))
        artifact.add_file(str(sidecar_path))
        aliases = [f"percent-{percent:02d}" for percent in milestone_values]
        aliases.extend((f"step-{env_steps}", "latest"))
        self.run.log_artifact(artifact, aliases=aliases)
        return eval_means

    @staticmethod
    def _validate_milestones(milestones: Sequence[int]) -> list[int]:
        values = []
        for milestone in milestones:
            if isinstance(milestone, bool) or not isinstance(milestone, int) or not 1 <= milestone <= 100:
                raise ValueError("milestones must be integer percentages between 1 and 100")
            values.append(milestone)
        return sorted(set(values))

    def finish(self) -> None:
        """Flush the active W&B run once; disabled callbacks are no-ops."""

        if self.run is not None and not self._finished:
            self.run.finish()
            self._finished = True


__all__ = ["WandbCallback", "WandbSettings", "milestones_for_update"]
