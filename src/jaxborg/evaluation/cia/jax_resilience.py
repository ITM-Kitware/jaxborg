"""JAX-native CIA resilience scoring helpers.

The rollout-facing functions in this module operate only on JAX arrays and
are safe to use from ``jit``, ``vmap``, and ``lax.scan``.  Conversion to the
JSON/MLflow representation is deliberately kept in the eager summary helpers
at the bottom of the module.

Scores use the existing signed convention: zero means healthy and every
impacted resilience role contributes ``-10`` to each CIA component that
depends on that role.  The returned array order is always C, I, A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from jaxborg.scenarios.cc4.topology_roles import ROLE_AUTH, ROLE_DB, ROLE_WEB

CIA_KEYS = ("c", "i", "a")
CIA_DROP = 10.0


def resilience_host_impacted(state: Any) -> jnp.ndarray:
    """Return the per-host resilience impact predicate for a simulator state.

    Leading batch dimensions are supported.  A host is impacted when its OT
    service is stopped or any of its service/decoy reliability values is below
    100.  This mirrors the persistent simulator effects of successful Impact
    and DegradeServices actions; Restore clears all three conditions.
    """

    service_degraded = jnp.any(jnp.asarray(state.host_service_reliability) < 100, axis=-1)
    decoy_degraded = jnp.any(jnp.asarray(state.host_decoy_reliability) < 100, axis=-1)
    return jnp.asarray(state.ot_service_stopped, dtype=jnp.bool_) | service_degraded | decoy_degraded


def score_resilience_impacts(
    impacted: jnp.ndarray,
    host_resilience_role: jnp.ndarray,
    *,
    drop: float = CIA_DROP,
) -> jnp.ndarray:
    """Score an impact mask and role map, returning ``[..., (C, I, A)]``.

    ``impacted`` and ``host_resilience_role`` must have host as their final
    axis.  Normal JAX broadcasting permits one fixed ``(hosts,)`` role map to
    score a batched ``(..., hosts)`` impact mask.
    """

    impacted = jnp.asarray(impacted, dtype=jnp.bool_)
    roles = jnp.asarray(host_resilience_role)
    auth = roles == ROLE_AUTH
    database = roles == ROLE_DB
    web = roles == ROLE_WEB

    confidentiality = jnp.sum(impacted & (auth | database), axis=-1)
    integrity = jnp.sum(impacted & (auth | web), axis=-1)
    availability = jnp.sum(impacted & (auth | database | web), axis=-1)
    counts = jnp.stack((confidentiality, integrity, availability), axis=-1)
    return -jnp.asarray(drop, dtype=jnp.float32) * counts.astype(jnp.float32)


def score_resilience_state(
    state: Any,
    host_resilience_role: jnp.ndarray,
    *,
    drop: float = CIA_DROP,
) -> jnp.ndarray:
    """Return the instantaneous ``[..., (C, I, A)]`` score for ``state``."""

    return score_resilience_impacts(
        resilience_host_impacted(state),
        host_resilience_role,
        drop=drop,
    )


def mean_resilience_episode(
    step_scores: jnp.ndarray,
    valid_steps: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Reduce per-step scores to temporal episode means.

    ``step_scores`` has shape ``(..., steps, 3)``.  ``valid_steps``, when
    supplied, has shape ``(..., steps)`` and lets padded post-terminal steps be
    excluded without leaving JAX tracing.  An all-false mask returns zeros.
    """

    scores = jnp.asarray(step_scores, dtype=jnp.float32)
    if scores.ndim < 2 or scores.shape[-1] != len(CIA_KEYS):
        raise ValueError(f"step_scores must have shape (..., steps, 3), got {scores.shape}")
    if valid_steps is None:
        return jnp.mean(scores, axis=-2)

    valid = jnp.asarray(valid_steps, dtype=jnp.bool_)
    weighted = jnp.where(valid[..., None], scores, jnp.float32(0.0))
    count = jnp.sum(valid, axis=-1, keepdims=True)
    return jnp.where(count > 0, jnp.sum(weighted, axis=-2) / jnp.maximum(count, 1), jnp.float32(0.0))


@dataclass(frozen=True)
class ResilienceSummary:
    """Across-episode CIA means and sample standard deviations."""

    n: int
    mean: dict[str, float]
    std: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON/MLflow-friendly aggregate representation."""

        return {
            "n": self.n,
            **{key: {"mean": self.mean[key], "std": self.std[key]} for key in CIA_KEYS},
        }


def _episode_score_array(episode_scores: Sequence[Sequence[float]] | np.ndarray | jnp.ndarray) -> np.ndarray:
    values = np.asarray(episode_scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CIA_KEYS):
        raise ValueError(f"episode_scores must have shape (episodes, 3), got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("episode_scores must contain at least one episode")
    return values


def summarize_resilience_episodes(
    episode_scores: Sequence[Sequence[float]] | np.ndarray | jnp.ndarray,
) -> ResilienceSummary:
    """Build an eager aggregate using sample std (zero for one episode)."""

    values = _episode_score_array(episode_scores)
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0, ddof=1) if len(values) > 1 else np.zeros(len(CIA_KEYS))
    return ResilienceSummary(
        n=len(values),
        mean={key: float(means[i]) for i, key in enumerate(CIA_KEYS)},
        std={key: float(stds[i]) for i, key in enumerate(CIA_KEYS)},
    )


def resilience_episode_records(
    episode_scores: Sequence[Sequence[float]] | np.ndarray | jnp.ndarray,
) -> list[dict[str, float]]:
    """Convert ``(episodes, 3)`` scores to JSON-friendly C/I/A records."""

    values = _episode_score_array(episode_scores)
    return [{key: float(row[i]) for i, key in enumerate(CIA_KEYS)} for row in values]
