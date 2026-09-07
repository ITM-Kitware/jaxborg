"""Recipe settings for opt-in CIA evaluation.

The metric is deliberately independent from resilience role *enablement*.
Selecting ``eval.cia_metric`` (the legacy spelling) chooses a metric but does
not turn collection on; only ``eval.cia.enabled`` does that.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from jaxborg.scenarios.cc4.game_variant import GameVariant


@dataclass(frozen=True)
class CIAEvalSettings:
    """Validated ``eval.cia`` settings."""

    enabled: bool = False
    metric: str = "resilience"
    role_assignment: str = "fixed_per_topology"

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> CIAEvalSettings:
        eval_config = recipe.get("eval", {})
        if eval_config is None:
            eval_config = {}
        if not isinstance(eval_config, Mapping):
            raise ValueError("eval must be a mapping")

        legacy_metric = eval_config.get("cia_metric")
        raw = eval_config.get("cia", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("eval.cia must be a mapping")
        unknown = set(raw) - {"enabled", "metric", "role_assignment"}
        if unknown:
            raise ValueError(f"eval.cia has unknown settings: {sorted(unknown)}")

        enabled = raw.get("enabled", False)
        metric = raw["metric"] if "metric" in raw else legacy_metric if legacy_metric is not None else "resilience"
        role_assignment = raw.get("role_assignment", "fixed_per_topology")
        if not isinstance(enabled, bool):
            raise ValueError("eval.cia.enabled must be a boolean")
        if legacy_metric is not None and "metric" in raw and metric != legacy_metric:
            raise ValueError("eval.cia.metric and eval.cia_metric must agree when both are set")
        if metric != "resilience":
            raise ValueError("eval.cia.metric must be 'resilience'")
        if role_assignment != "fixed_per_topology":
            raise ValueError("eval.cia.role_assignment must be 'fixed_per_topology'")
        return cls(enabled=enabled, metric=metric, role_assignment=role_assignment)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for results and projections."""

        return asdict(self)


def coerce_cia_settings(value: CIAEvalSettings | Mapping[str, Any] | None) -> CIAEvalSettings:
    """Normalize a projected CIA mapping accepted by evaluator APIs."""

    if value is None:
        return CIAEvalSettings()
    if isinstance(value, CIAEvalSettings):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("CIA settings must be a mapping")
    return CIAEvalSettings.from_recipe({"eval": {"cia": dict(value)}})


def validate_cia_evaluation(
    settings: CIAEvalSettings,
    *,
    variant: GameVariant,
    topology_sampling: str,
    topology_paths: Sequence[str | Path],
    inspect_snapshots: bool = False,
) -> None:
    """Fail an invalid CIA contract before constructing a rollout environment."""

    if not settings.enabled:
        return
    if not variant.resilience_roles:
        raise ValueError(
            "eval.cia.enabled requires an evaluation variant with resilience_roles=True; "
            "select eval.variant: cia_resilience (or another resilience variant)"
        )
    if topology_sampling != "exhaustive":
        raise ValueError("eval.cia.enabled requires eval.topology_sampling: exhaustive")
    if not topology_paths:
        raise ValueError("eval.cia.enabled requires a non-empty eval topology bank")
    if not inspect_snapshots:
        return

    from jaxborg.scenarios.cc4.topology import load_topology
    from jaxborg.scenarios.cc4.topology_roles import count_resilience_candidates

    for path in topology_paths:
        count = count_resilience_candidates(load_topology(path))
        if count < 3:
            raise ValueError(
                f"topology snapshot {path} has only {count} eligible operational servers; "
                "CIA evaluation requires at least 3 for AUTH/DB/WEB"
            )


__all__ = ["CIAEvalSettings", "coerce_cia_settings", "validate_cia_evaluation"]
