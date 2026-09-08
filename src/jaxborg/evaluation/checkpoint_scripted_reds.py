"""Scripted-Red and CIA curves across a run's durable checkpoints.

``eval.after_training`` runs the scripted-Red suite once, on the final model.
That gives a single point, so it cannot show *when* Blue acquired (or lost) its
defence against a fixed adversary — and in co-training, "lost" is a real
outcome: Blue chases the learned Red and can quietly regress against the
scripted FSMs it is ultimately measured on.

This module replays the same evaluator at every strided durable checkpoint and
logs each result at ``step=checkpoint.steps``, turning the final-model number
into a curve.  Scripted opponents are fixed, so unlike the self-play return
these values are an *absolute* scale: they move only when Blue changes.

The suite is Blue-only, so it works for single-team runs as well as co-training
pairs.  Cost is ``max_checkpoints`` times the final-model evaluation, so the
checkpoint list is strided the same way as ``cross_play``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxborg.evaluation.play_priors import _parse_seeds, find_periodic_checkpoints, select_checkpoints

_ALLOWED_SETTINGS = {
    "enabled",
    "reds",
    "seeds",
    "episodes_per_seed",
    "deterministic",
    "required",
    "max_checkpoints",
}


@dataclass(frozen=True)
class CheckpointScriptedRedsSettings:
    """Validated ``eval.checkpoint_scripted_reds`` settings."""

    enabled: bool = False
    reds: tuple[str, ...] = ("fsm", "cia_c", "cia_i", "cia_a")
    seeds: tuple[int, ...] = tuple(range(1000, 1010))
    episodes_per_seed: int = 1
    deterministic: bool = False
    required: bool = True
    max_checkpoints: int = 6

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> CheckpointScriptedRedsSettings:
        eval_config = recipe.get("eval", {})
        if eval_config is None:
            eval_config = {}
        if not isinstance(eval_config, Mapping):
            raise ValueError("eval must be a mapping")
        raw = eval_config.get("checkpoint_scripted_reds", False)
        if raw is None or raw is False:
            return cls()
        if raw is True:
            return cls(enabled=True)
        if not isinstance(raw, Mapping):
            raise ValueError("eval.checkpoint_scripted_reds must be a boolean or mapping")
        unknown = set(raw) - _ALLOWED_SETTINGS
        if unknown:
            raise ValueError(f"eval.checkpoint_scripted_reds has unknown settings: {sorted(unknown)}")

        enabled = raw.get("enabled", True)
        deterministic = raw.get("deterministic", False)
        required = raw.get("required", True)
        episodes_per_seed = raw.get("episodes_per_seed", 1)
        max_checkpoints = raw.get("max_checkpoints", 6)
        for name, value in (("enabled", enabled), ("deterministic", deterministic), ("required", required)):
            if not isinstance(value, bool):
                raise ValueError(f"eval.checkpoint_scripted_reds.{name} must be a boolean")
        for name, value in (("episodes_per_seed", episodes_per_seed), ("max_checkpoints", max_checkpoints)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"eval.checkpoint_scripted_reds.{name} must be an integer")
        if episodes_per_seed < 1:
            raise ValueError("eval.checkpoint_scripted_reds.episodes_per_seed must be positive")
        if max_checkpoints < 2:
            raise ValueError("eval.checkpoint_scripted_reds.max_checkpoints must be at least 2")

        raw_reds = raw.get("reds", cls.reds)
        if isinstance(raw_reds, str) or not isinstance(raw_reds, Sequence):
            raise ValueError("eval.checkpoint_scripted_reds.reds must be a list of scripted Red names")
        reds = tuple(str(red) for red in raw_reds)
        if not reds:
            raise ValueError("eval.checkpoint_scripted_reds.reds must contain at least one Red")
        return cls(
            enabled=enabled,
            reds=reds,
            seeds=_parse_seeds(raw.get("seeds", "1000-1009")),
            episodes_per_seed=episodes_per_seed,
            deterministic=deterministic,
            required=required,
            max_checkpoints=max_checkpoints,
        )


def _checkpoint_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Per-Red rewards and CIA, plus the worst case over the scripted suite."""
    from jaxborg.evaluation.cia.reporting import cia_mlflow_metrics

    metrics: dict[str, float] = {}
    rewards: list[float] = []
    for row in rows:
        prefix = f"eval.checkpoint_scripted_reds.{row['eval_red']}.blue"
        reward = float(row["mean_reward"])
        rewards.append(reward)
        metrics[f"{prefix}.mean_reward"] = reward
        metrics[f"{prefix}.std_reward"] = float(row["std_reward"])
        metrics[f"{prefix}.episodes"] = float(row["n_episodes"])
        cia_summary = row.get("cia_summary")
        if cia_summary:
            # Rewards are the primary signal, so a malformed CIA summary must
            # not take the whole checkpoint's metrics down with it.
            try:
                metrics.update(cia_mlflow_metrics(f"{prefix}.cia", cia_summary))
            except (KeyError, TypeError, ValueError) as exc:
                print(f"Skipping CIA metrics for {row['eval_red']}: {exc}", flush=True)
    if rewards:
        metrics["eval.checkpoint_scripted_reds.blue.mean_reward"] = sum(rewards) / len(rewards)
        # Blue's worst scripted matchup: the robustness number that a mean
        # across Reds can hide.
        metrics["eval.checkpoint_scripted_reds.blue.worst_reward"] = min(rewards)
    return metrics


def run_checkpoint_scripted_reds(
    final_model: str | Path,
    recipe: Mapping[str, Any],
    *,
    output: str | Path | None = None,
    evaluate_fn: Callable[..., Any] | None = None,
    attach_metrics_fn: Callable[..., Any] | None = None,
) -> Path | None:
    """Evaluate every strided checkpoint against the scripted Reds."""

    settings = CheckpointScriptedRedsSettings.from_recipe(recipe)
    if not settings.enabled:
        return None

    resolved_model = Path(final_model).expanduser().resolve()
    checkpoints = select_checkpoints(
        find_periodic_checkpoints(resolved_model, recipe),
        settings.max_checkpoints,
    )
    if not checkpoints:
        raise ValueError(
            "eval.checkpoint_scripted_reds requires at least one saved periodic checkpoint; "
            f"found none in {resolved_model.parent}"
        )

    from jaxborg.recipe import eval_variant, project_eval

    if evaluate_fn is None:
        from jaxborg.evaluation.jax_scripted_red import evaluate_jax_scripted_reds

        evaluate = evaluate_jax_scripted_reds
    else:
        evaluate = evaluate_fn
    if attach_metrics_fn is None:
        from jaxborg.mlflow_setup import attach_eval_metrics

        attach = attach_eval_metrics
    else:
        attach = attach_metrics_fn

    projected_eval = project_eval(dict(recipe), materialize_topologies=True)
    topology_paths = list(projected_eval["TOPOLOGY_BANK"]) or None
    base_variant = eval_variant(dict(recipe))
    train_run_id = recipe.get("run", {}).get("train_run_id")

    rows: list[dict[str, Any]] = []
    print(
        f"Checkpoint scripted-Red evaluation: {len(checkpoints)} checkpoints x {len(settings.reds)} Reds",
        flush=True,
    )
    for checkpoint in checkpoints:
        started = time.perf_counter()
        checkpoint_rows = evaluate(
            checkpoint.path,
            base_variant=base_variant,
            topology_paths=topology_paths,
            reds=list(settings.reds),
            seeds=list(settings.seeds),
            episodes_per_seed=settings.episodes_per_seed,
            deterministic=settings.deterministic,
            progress=True,
            eval_name="checkpoint_scripted_reds",
            recipe=recipe,
        )
        elapsed = time.perf_counter() - started
        for row in checkpoint_rows:
            enriched = {
                **row,
                "suite": "checkpoint_scripted_reds",
                "checkpoint": str(checkpoint.path),
                "checkpoint_step": checkpoint.steps,
                "wall_time_s": elapsed,
            }
            rows.append(enriched)
            print(
                f"  checkpoint {checkpoint.steps:,} vs {row['eval_red']}: {float(row['mean_reward']):.2f}",
                flush=True,
            )

        if train_run_id:
            try:
                attach(str(train_run_id), _checkpoint_metrics(checkpoint_rows), step=checkpoint.steps)
            except Exception as exc:
                print(f"MLflow attach warning for {train_run_id}: {exc}", flush=True)

    exp_dir = Path(os.environ.get("JAXBORG_EXP_DIR", resolved_model.parents[2])).expanduser().resolve()
    run_tag = resolved_model.stem.removeprefix("model_")
    eval_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else exp_dir / "eval" / f"{run_tag}_checkpoint_scripted_reds_{eval_id}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote checkpoint scripted-Red results: {output_path}", flush=True)
    return output_path


__all__ = [
    "CheckpointScriptedRedsSettings",
    "run_checkpoint_scripted_reds",
]
