"""Offline and JAX-native resilience CIA evaluation helpers."""

from __future__ import annotations

from typing import Any

from jaxborg.evaluation.cia.config import (
    CIAEvalSettings,
    coerce_cia_settings,
    validate_cia_evaluation,
)
from jaxborg.evaluation.cia.fixed_topology import (
    FIXED_ROLE_ASSIGNMENT_VERSION,
    TOPOLOGY_FINGERPRINT_VERSION,
    EvaluationCase,
    FixedRoleAssignment,
    build_evaluation_cases,
    canonical_topology_fingerprint,
    eligible_operational_server_indices,
    fixed_role_assignment,
    validate_fixed_role_eligibility,
)
from jaxborg.evaluation.cia.jax_resilience import (
    CIA_DROP,
    CIA_KEYS,
    ResilienceSummary,
    mean_resilience_episode,
    resilience_episode_records,
    resilience_host_impacted,
    score_resilience_impacts,
    score_resilience_state,
    summarize_resilience_episodes,
)
from jaxborg.evaluation.cia.reporting import cia_mlflow_metrics, cia_summary_dict
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
    "CIA_DROP",
    "CIAEvalSettings",
    "CIA_KEYS",
    "FIXED_ROLE_ASSIGNMENT_VERSION",
    "ROLE_AUTH",
    "ROLE_DB",
    "ROLE_NONE",
    "ROLE_WEB",
    "TOPOLOGY_FINGERPRINT_VERSION",
    "EvaluationCase",
    "FixedRoleAssignment",
    "ResilienceEpisodeScore",
    "ResilienceMetric",
    "ResilienceSummary",
    "build_evaluation_cases",
    "canonical_topology_fingerprint",
    "cia_mlflow_metrics",
    "cia_summary_dict",
    "coerce_cia_settings",
    "eligible_operational_server_indices",
    "fixed_role_assignment",
    "get_cia_scorer",
    "mean_resilience_episode",
    "resilience_episode_records",
    "resilience_host_impacted",
    "score_resilience_impacts",
    "score_resilience_state",
    "summarize_resilience_episodes",
    "validate_cia_evaluation",
    "validate_fixed_role_eligibility",
]
