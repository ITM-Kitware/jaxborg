"""CC4-aware CIA alignment metric — trajectory recording + post-hoc scoring."""

from __future__ import annotations

from typing import Any

from jaxborg.evaluation.cia.cc4_cia_metric import (
    CIA_COMPOSITE_WEIGHT,
    CIAEpisodeScore,
    cc4_host_weight,
    score_episode,
    score_trajectory_file,
)
from jaxborg.evaluation.cia.resilience_metric import (
    ROLE_AUTH,
    ROLE_DB,
    ROLE_NONE,
    ROLE_WEB,
    ResilienceEpisodeScore,
    ResilienceMetric,
)


def get_cia_scorer(eval_cfg: dict[str, Any]):
    """Return a ``score_trajectory_file``-compatible callable for the given eval config.

    Args:
        eval_cfg: the dict returned by ``recipe.project_eval()``.

    Returns:
        A callable with signature ``(path: Path) -> score`` that scores one
        ``.jsonl`` trajectory file.
    """
    metric = eval_cfg.get("cia_metric", "cc4")
    if metric == "cc4":
        return score_trajectory_file
    if metric == "resilience":
        return ResilienceMetric().score_trajectory_file
    raise ValueError(f"Unknown metric: {metric!r}")


__all__ = [
    "CIA_COMPOSITE_WEIGHT",
    "CIAEpisodeScore",
    "ROLE_AUTH",
    "ROLE_DB",
    "ROLE_NONE",
    "ROLE_WEB",
    "ResilienceEpisodeScore",
    "ResilienceMetric",
    "cc4_host_weight",
    "get_cia_scorer",
    "score_episode",
    "score_trajectory_file",
]
