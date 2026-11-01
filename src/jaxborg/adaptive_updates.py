"""Adapt the GAPT two-phase schedule to Red PPO and raw Blue returns."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class AdaptiveUpdateSettings:
    phase_switch_blue_updates: int = 200
    reward_threshold: float = -2000.0
    window_blue_updates: int = 5
    max_frozen_blue_rollouts: int = 50

    @classmethod
    def from_config(cls, raw) -> AdaptiveUpdateSettings | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("train.adaptive_updates must be a mapping")
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"train.adaptive_updates has unknown settings: {sorted(unknown)}")
        settings = cls(**raw)
        for name, minimum in (
            ("phase_switch_blue_updates", 0),
            ("window_blue_updates", 1),
            ("max_frozen_blue_rollouts", 1),
        ):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"train.adaptive_updates.{name} must be an integer >= {minimum}")
        threshold = settings.reward_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("train.adaptive_updates.reward_threshold must be a finite number")
        return settings


class AdaptiveUpdateState(NamedTuple):
    blue_returns: jax.Array
    window_count: jax.Array
    frozen_rollouts: jax.Array


def initial_adaptive_update_state(settings: AdaptiveUpdateSettings) -> AdaptiveUpdateState:
    return AdaptiveUpdateState(
        blue_returns=jnp.zeros(settings.window_blue_updates, dtype=jnp.float32),
        window_count=jnp.array(0, dtype=jnp.int32),
        frozen_rollouts=jnp.array(0, dtype=jnp.int32),
    )


def adaptive_red_update(
    settings: AdaptiveUpdateSettings,
    state: AdaptiveUpdateState,
    *,
    blue_return: jax.Array,
    rollout_number: jax.Array,
    warmup_update_every: int,
) -> tuple[jax.Array, AdaptiveUpdateState, dict[str, jax.Array]]:
    """Decide after collection, before PPO, using the current on-policy batch.

    ``rollout_number`` is one-based; Blue updates on every rollout. History
    includes phase one and the current rollout. A closed gate forces one PPO
    update at the end of every ``max_frozen_blue_rollouts`` consecutive closed
    rollouts. Any Red update resets that counter; phase-one skips do not count.
    """
    returns = jnp.roll(state.blue_returns, -1).at[-1].set(blue_return)
    count = jnp.minimum(state.window_count + 1, settings.window_blue_updates)
    mean_return = returns.sum() / count
    adaptive = rollout_number > settings.phase_switch_blue_updates
    gate_open = adaptive & (count == settings.window_blue_updates) & (mean_return > settings.reward_threshold)
    closed_rollouts = state.frozen_rollouts + 1
    forced = adaptive & ~gate_open & (closed_rollouts >= settings.max_frozen_blue_rollouts)
    did_update = jnp.where(adaptive, gate_open | forced, rollout_number % warmup_update_every == 0)
    frozen_rollouts = jnp.where(adaptive & ~did_update, closed_rollouts, 0)
    next_state = AdaptiveUpdateState(returns, count, frozen_rollouts)
    metrics = {
        "adaptive_phase": jnp.where(adaptive, 2, 1),
        "reward_window_mean": mean_return,
        "reward_window_count": count,
        "gate_open": gate_open.astype(jnp.float32),
        "forced_update": forced.astype(jnp.float32),
        "frozen_rollouts": frozen_rollouts,
    }
    return did_update, next_state, metrics
