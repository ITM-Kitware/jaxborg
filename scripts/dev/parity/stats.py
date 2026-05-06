"""Statistics and action-distribution helpers for parity checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from jaxborg.actions.encoding import (
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_ANALYSE_START,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_DECOY_START,
    BLUE_MONITOR,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
)
from jaxborg.constants import BLUE_ACTION_HOST_SLOTS

ACTION_TYPE_NAMES = [
    "Sleep",
    "Monitor",
    "Analyse",
    "Remove",
    "Restore",
    "Decoy",
    "BlockTraffic",
    "AllowTraffic",
]

ACTION_TYPE_RANGES = [
    (BLUE_SLEEP, BLUE_SLEEP + 1),
    (BLUE_MONITOR, BLUE_MONITOR + 1),
    (BLUE_ANALYSE_START, BLUE_ANALYSE_START + BLUE_ACTION_HOST_SLOTS),
    (BLUE_REMOVE_START, BLUE_REMOVE_START + BLUE_ACTION_HOST_SLOTS),
    (BLUE_RESTORE_START, BLUE_RESTORE_START + BLUE_ACTION_HOST_SLOTS),
    (BLUE_DECOY_START, BLUE_BLOCK_TRAFFIC_START),
    (BLUE_BLOCK_TRAFFIC_START, BLUE_ALLOW_TRAFFIC_START),
    (BLUE_ALLOW_TRAFFIC_START, BLUE_ALLOW_TRAFFIC_END),
]


def classify_action(action_idx: int) -> int:
    for i, (start, end) in enumerate(ACTION_TYPE_RANGES):
        if start <= action_idx < end:
            return i
    return 0


def action_distribution(actions):
    counts = np.zeros(len(ACTION_TYPE_NAMES))
    for a in actions:
        counts[classify_action(int(a))] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def l1_distribution_distance(p, q) -> float:
    # Sum of absolute differences; 0 = identical, 2 = disjoint. Halve for "total variation".
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.abs(p - q).sum())


def tost_equivalence(
    perf_rewards: np.ndarray | list[float],
    ref_rewards: np.ndarray | list[float],
    margin: float,
    alpha: float = 0.05,
    paired: bool = False,
) -> dict[str, Any]:
    """Run two one-sided tests for equivalence of mean episode rewards.

    ``perf_rewards`` is the performance backend (JAXborg here), ``ref_rewards``
    is the reference backend (CybORG). If ``paired`` is true, the inputs must be
    episode-aligned and the test is run on per-episode differences.
    """
    from scipy import stats

    perf = np.asarray(perf_rewards, dtype=float)
    ref = np.asarray(ref_rewards, dtype=float)
    if len(perf) < 2 or len(ref) < 2:
        raise ValueError("TOST requires at least two rewards from each backend")

    if paired:
        if len(perf) != len(ref):
            raise ValueError(f"Paired TOST requires equal-length arrays, got {len(perf)} vs {len(ref)}")
        diffs = perf - ref
        n = len(diffs)
        mean_diff = float(np.mean(diffs))
        se = float(np.std(diffs, ddof=1) / np.sqrt(n))
        df = n - 1
    else:
        n_perf, n_ref = len(perf), len(ref)
        mean_diff = float(np.mean(perf) - np.mean(ref))
        s1 = float(np.var(perf, ddof=1))
        s2 = float(np.var(ref, ddof=1))
        n_perf_f = float(n_perf)
        n_ref_f = float(n_ref)
        se = float(np.sqrt(s1 / n_perf_f + s2 / n_ref_f))
        nu_num = (s1 / n_perf_f + s2 / n_ref_f) ** 2
        nu_den = (s1 / n_perf_f) ** 2 / (n_perf_f - 1) + (s2 / n_ref_f) ** 2 / (n_ref_f - 1)
        df = nu_num / nu_den if nu_den > 0 else min(n_perf, n_ref) - 1

    if se < 1e-12:
        return {
            "equivalent": abs(mean_diff) < margin,
            "p_upper": 0.0 if mean_diff < margin else 1.0,
            "p_lower": 0.0 if mean_diff > -margin else 1.0,
            "mean_diff": mean_diff,
            "margin": margin,
            "ci_lower": mean_diff,
            "ci_upper": mean_diff,
            "paired": paired,
        }

    t_upper = (mean_diff - margin) / se
    p_upper = float(stats.t.cdf(t_upper, df))
    t_lower = (mean_diff + margin) / se
    p_lower = float(1.0 - stats.t.cdf(t_lower, df))

    t_crit = float(stats.t.ppf(1 - alpha, df))
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        "equivalent": p_upper < alpha and p_lower < alpha,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "mean_diff": mean_diff,
        "margin": margin,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "paired": paired,
    }
