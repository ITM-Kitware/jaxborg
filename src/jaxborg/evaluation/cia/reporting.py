"""Shared serialization and MLflow naming for CIA evaluation results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jaxborg.evaluation.cia.jax_resilience import CIA_KEYS, ResilienceSummary


def cia_summary_dict(summary: ResilienceSummary | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a resilience aggregate to the public JSON representation."""

    raw = summary.to_dict() if isinstance(summary, ResilienceSummary) else dict(summary)
    return {
        "n": int(raw["n"]),
        **{
            key: {
                "mean": float(raw[key]["mean"]),
                "std": float(raw[key]["std"]),
            }
            for key in CIA_KEYS
        },
    }


def cia_mlflow_metrics(
    prefix: str,
    summary: ResilienceSummary | Mapping[str, Any],
) -> dict[str, float]:
    """Flatten a CIA summary under ``<prefix>.{c,i,a}.{mean,std}``."""

    raw = cia_summary_dict(summary)
    return {
        f"{prefix}.{component}.{statistic}": float(raw[component][statistic])
        for component in CIA_KEYS
        for statistic in ("mean", "std")
    }


__all__ = ["cia_mlflow_metrics", "cia_summary_dict"]
