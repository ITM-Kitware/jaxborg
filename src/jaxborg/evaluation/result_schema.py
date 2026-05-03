"""Standardized result-row schema for eval output.

One JSON object per eval invocation, written to
`$JAXBORG_EXP_DIR/eval/<recipe-name>_<tag>_<eval-id>.jsonl`.
"""

from __future__ import annotations

from typing import TypedDict


class EvalRow(TypedDict, total=False):
    eval_id: str
    model: str
    recipe_name: str
    recipe_path: str
    trained_backend: str  # "cyborg" or "jax" — which trainer produced the model
    eval_env: str  # always "cyborg" today (CC4 contract eval)
    seeds: list[int]
    episodes_per_seed: int
    stochastic: bool
    mean_reward: float
    std_reward: float
    n_episodes: int
    wall_time_s: float
    git_commit: str
    train_run_id: str | None
    per_episode: list[float]
    per_episode_seeds: list[int]
