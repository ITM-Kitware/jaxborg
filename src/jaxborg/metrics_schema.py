"""Standardized metrics.jsonl schema.

Both algorithm scripts emit one JSONL line per training-side update with
the fields below. Backend-specific extras live under `backend.<name>.*`
(optional). Keep this module the single source of truth — algorithm
scripts import the keys list when constructing their per-update record so
field names cannot drift.
"""

from __future__ import annotations

from typing import TypedDict


class MetricsRow(TypedDict, total=False):
    update_idx: int
    env_steps: int
    wall_time_s: float
    throughput_sps: float

    # Episode-level (mean over the rollout / recent window)
    train_episode_reward_mean: float
    train_episode_reward_std: float
    train_episode_length_mean: float

    # Loss components
    loss_policy: float
    loss_value: float
    loss_entropy: float
    loss_total: float

    # PPO diagnostics
    ppo_kl_divergence: float
    ppo_clip_fraction: float
    ppo_explained_variance: float
    ppo_grad_norm: float
    ppo_pre_clip_grad_norm: float

    lr: float

    # Dual-team runs use dotted, team-qualified keys in the serialized row,
    # e.g. ``team.blue.loss_policy`` and ``team.red.return``.  TypedDict
    # cannot spell every dotted extension cleanly, so these remain optional
    # runtime fields added by ``add_team_metrics`` below.


REQUIRED_KEYS: tuple[str, ...] = (
    "update_idx",
    "env_steps",
    "wall_time_s",
    "throughput_sps",
    "loss_policy",
    "loss_value",
    "loss_entropy",
    "loss_total",
    "ppo_kl_divergence",
    "ppo_clip_fraction",
    "ppo_explained_variance",
    "lr",
)


def make_row(
    *,
    update_idx: int,
    env_steps: int,
    wall_time_s: float,
    throughput_sps: float,
    loss_policy: float,
    loss_value: float,
    loss_entropy: float,
    loss_total: float,
    ppo_kl_divergence: float,
    ppo_clip_fraction: float,
    ppo_explained_variance: float,
    lr: float,
    train_episode_reward_mean: float | None = None,
    train_episode_reward_std: float | None = None,
    train_episode_length_mean: float | None = None,
    ppo_grad_norm: float | None = None,
    ppo_pre_clip_grad_norm: float | None = None,
    backend_extras: dict | None = None,
) -> dict:
    row: dict = {
        "update_idx": int(update_idx),
        "env_steps": int(env_steps),
        "wall_time_s": float(wall_time_s),
        "throughput_sps": float(throughput_sps),
        "loss_policy": float(loss_policy),
        "loss_value": float(loss_value),
        "loss_entropy": float(loss_entropy),
        "loss_total": float(loss_total),
        "ppo_kl_divergence": float(ppo_kl_divergence),
        "ppo_clip_fraction": float(ppo_clip_fraction),
        "ppo_explained_variance": float(ppo_explained_variance),
        "lr": float(lr),
    }
    if train_episode_reward_mean is not None:
        row["train_episode_reward_mean"] = float(train_episode_reward_mean)
    if train_episode_reward_std is not None:
        row["train_episode_reward_std"] = float(train_episode_reward_std)
    if train_episode_length_mean is not None:
        row["train_episode_length_mean"] = float(train_episode_length_mean)
    if ppo_grad_norm is not None:
        row["ppo_grad_norm"] = float(ppo_grad_norm)
    if ppo_pre_clip_grad_norm is not None:
        row["ppo_pre_clip_grad_norm"] = float(ppo_pre_clip_grad_norm)
    if backend_extras:
        for k, v in backend_extras.items():
            row[f"backend.{k}"] = v
    return row


def add_team_metrics(row: dict, team: str, metrics: dict) -> dict:
    """Add scalar metrics under a stable ``team.<colour>.`` namespace.

    The input row is updated and returned so callers can use the helper in a
    fluent style.  Legacy top-level fields remain untouched; joint trainers
    populate those from Blue for dashboard compatibility.
    """
    if team not in ("blue", "red"):
        raise ValueError(f"unknown team {team!r}")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            row[f"team.{team}.{key}"] = float(value)
    return row
