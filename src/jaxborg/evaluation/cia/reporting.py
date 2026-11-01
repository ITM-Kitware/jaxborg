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
    """Flatten a CIA summary under ``<prefix>.{c,i,a}.{mean,std,...}``.

    Besides the raw ``mean`` and ``std``, each component also gets the two
    envelope series ``mean_minus_std`` and ``mean_plus_std``.  The MLflow UI
    cannot draw error bars, so reading the spread off the ``std`` chart means
    eyeballing it against a second, separately scaled chart.  Charting the
    three ``mean*`` keys together instead puts the +/-1 sigma envelope around
    the mean on one axis.  ``std`` stays logged because it is the value the
    exported figures in ``plots/plot_mlflow.py`` pool across seeds.
    """

    raw = cia_summary_dict(summary)
    metrics: dict[str, float] = {}
    for component in CIA_KEYS:
        mean = float(raw[component]["mean"])
        std = float(raw[component]["std"])
        metrics[f"{prefix}.{component}.mean"] = mean
        metrics[f"{prefix}.{component}.std"] = std
        metrics[f"{prefix}.{component}.mean_minus_std"] = mean - std
        metrics[f"{prefix}.{component}.mean_plus_std"] = mean + std
    return metrics


__all__ = ["cia_mlflow_metrics", "cia_summary_dict"]
