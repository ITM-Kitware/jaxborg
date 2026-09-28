"""Paired default/shaping scores on the same evaluation trajectories.

CIA records are episode means. CC4 evaluation always runs the configured
fixed horizon, so multiplying by that horizon recovers the undiscounted sum.
Legacy evaluations without role maps explicitly report shaping as unavailable.
"""

import math
from collections.abc import Mapping, Sequence
from statistics import mean, stdev

from jaxborg.reward_config import RewardConfig


def reward_fields(default_returns: Sequence[float], cia_records, *, steps: int, recipe: Mapping) -> dict:
    config = RewardConfig.from_recipe(recipe)
    defaults = [float(v) for v in default_returns]
    shaped = None
    if cia_records:
        if len(cia_records) != len(defaults):
            raise ValueError("CIA records and default returns must describe the same episodes")
        shaped = [
            None
            if cia is None
            else score + config.scale * steps * sum(w * float(cia[k]) for w, k in zip(config.weights, ("c", "i", "a")))
            for score, cia in zip(defaults, cia_records, strict=True)
        ]
    return reward_return_fields(defaults, shaped, config=config)


def reward_return_fields(defaults, shaped, *, config: RewardConfig) -> dict:
    defaults = [float(value) for value in defaults]
    if shaped is None:
        shaped = [None] * len(defaults)
    if len(shaped) != len(defaults):
        raise ValueError("default and shaped returns must describe the same episodes")
    shaped = [float(value) if value is not None and math.isfinite(value) else None for value in shaped]
    available = bool(shaped) and all(value is not None for value in shaped)
    fields = {"reward_config": config.as_dict(), "reward_team": "blue", "shaping_available": available}
    for name, values in (("default", defaults), ("shaping", shaped)):
        fields[f"per_episode_reward_{name}"] = values
        # Do not average a subset: that would destroy the paired comparison.
        complete = bool(values) and all(value is not None for value in values)
        fields[f"mean_reward_{name}"] = mean(values) if complete else None
        fields[f"std_reward_{name}"] = (stdev(values) if len(values) > 1 else 0.0) if complete else None
    return fields


def reward_mlflow_metrics(prefix: str, row: Mapping) -> dict[str, float]:
    return {
        f"{prefix}.{key}": float(row[key])
        for name in ("default", "shaping")
        for key in (f"mean_reward_{name}", f"std_reward_{name}")
        if row.get(key) is not None and math.isfinite(row[key])
    }


def matchup_reward_fields(evaluation, recipe):
    from jaxborg.recipe import eval_variant

    return reward_fields(
        evaluation.blue_returns,
        getattr(evaluation, "per_episode_cia", []),
        steps=eval_variant(recipe).num_steps,
        recipe=recipe,
    )
