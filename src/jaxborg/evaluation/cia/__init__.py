"""Resilience CIA scorer — post-hoc metric over trajectory JSONL.

Single metric (``ResilienceMetric``) keyed by recipe ``cia_metric`` value.
"""

from __future__ import annotations

from typing import Any

from jaxborg.evaluation.cia.resilience_metric import (
    ResilienceEpisodeScore,
    ResilienceMetric,
)
from jaxborg.scenarios.cc4.topology_roles import ROLE_AUTH, ROLE_DB, ROLE_NONE, ROLE_WEB


def get_cia_scorer(eval_cfg: dict[str, Any]):
    """Return a ``(path: Path) -> score`` callable for the given eval config.

    Args:
        eval_cfg: the dict returned by ``recipe.project_eval()``.

    The only supported metric today is ``resilience``; ``cia_metric`` may be
    omitted (it defaults to ``resilience``). Keeping the registry indirection
    in place so future metrics register here without script churn.
    """
    metric = eval_cfg.get("cia_metric", "resilience")
    if metric == "resilience":
        return ResilienceMetric().score_trajectory_file
    raise ValueError(f"Unknown CIA metric: {metric!r}")


__all__ = [
    "ROLE_AUTH",
    "ROLE_DB",
    "ROLE_NONE",
    "ROLE_WEB",
    "ResilienceEpisodeScore",
    "ResilienceMetric",
    "get_cia_scorer",
]
