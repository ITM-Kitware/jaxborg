"""Fit mask-conditioned fixed preferences and summarize matched return panels."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import t

LOGIT_PENALTY = 1e-8


def _fit_agent(probabilities, masks):
    count = len(probabilities)
    if not count:
        raise ValueError("Each agent needs genuine calibration decisions")
    if np.any(masks.sum(axis=-1) <= 1):
        raise ValueError("Genuine calibration rows must have multiple legal actions")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Teacher probabilities must be finite and nonnegative")
    if np.any(probabilities[~masks] > 1e-7):
        raise ValueError("Teacher probabilities assign mass to illegal actions")
    totals = probabilities.sum(axis=-1)
    if not np.allclose(totals, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("Teacher probabilities must sum to one")
    probabilities = np.where(masks, probabilities, 0.0)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    available = masks.any(axis=0)
    patterns, frequency = np.unique(masks[:, available], axis=0, return_counts=True)
    weights = frequency / count
    target = probabilities[:, available].mean(axis=0)

    def objective(logits):
        masked = np.where(patterns, logits[None, :], -np.inf)
        normalizers = logsumexp(masked, axis=-1)
        predicted = np.exp(masked - normalizers[:, None])
        value = weights @ normalizers - target @ logits + 0.5 * LOGIT_PENALTY * np.mean(logits**2)
        gradient = weights @ predicted - target + LOGIT_PENALTY * logits / len(logits)
        return float(value), gradient

    result = minimize(
        objective,
        np.zeros(int(available.sum()), dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 2000, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"Fixed preference fit did not converge: {result.message}")
    logits = np.zeros(masks.shape[-1], dtype=np.float64)
    logits[available] = result.x - result.x.mean()
    active = logits[available]
    regularized_loss, gradient = objective(active)
    cross_entropy = regularized_loss - 0.5 * LOGIT_PENALTY * np.mean(active**2)
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])) / count)
    return logits.tolist(), {
        "genuine_decisions": count,
        "unique_legal_masks": len(patterns),
        "never_legal_actions": np.flatnonzero(~available).tolist(),
        "converged": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "maximum_gradient": float(np.max(np.abs(gradient))),
        "regularized_objective": regularized_loss,
        "teacher_entropy_nats": entropy,
        "cross_entropy_nats": float(cross_entropy),
        "teacher_to_baseline_kl_nats": float(max(0.0, cross_entropy - entropy)),
    }


def fit_baseline(probabilities: np.ndarray, masks: np.ndarray, genuine: np.ndarray) -> dict:
    """Distill one fixed categorical vector per agent from calibration inputs.

    Arrays have shapes ``(rows, agents, actions)`` and ``(rows, agents)``.
    Minimize teacher cross entropy after legal-mask renormalization. Identical
    masks are aggregated exactly; every genuine decision receives equal weight.
    A fixed L2 penalty on mean squared available-action logits identifies their
    common offset. Never-legal actions keep logit zero and are reported.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    genuine = np.asarray(genuine, dtype=bool)
    if probabilities.ndim != 3 or masks.shape != probabilities.shape or genuine.shape != probabilities.shape[:2]:
        raise ValueError("Expected matching row/agent/action probability and mask arrays, plus row/agent flags")
    if probabilities.shape[-1] < 2 or not probabilities.shape[1]:
        raise ValueError("Calibration requires agents and at least two canonical actions")
    fitted = [
        _fit_agent(probabilities[genuine[:, agent], agent], masks[genuine[:, agent], agent])
        for agent in range(probabilities.shape[1])
    ]
    return {
        "kind": "mask_renormalized_fixed_preferences",
        "method": "teacher_cross_entropy",
        "conditioning": "agent and legal mask only; one vector across all calibration opponents",
        "optimizer": "scipy.optimize.minimize L-BFGS-B; deterministic zero initialization",
        "logit_penalty": LOGIT_PENALTY,
        "penalty_definition": "0.5 * logit_penalty * mean(available_action_logits ** 2)",
        "calibration_rows": probabilities.shape[0],
        "agent_count": probabilities.shape[1],
        "action_count": probabilities.shape[2],
        "logits": [value[0] for value in fitted],
        "agents": [value[1] for value in fitted],
        "converged": all(value[1]["converged"] for value in fitted),
    }


def summarize_comparison(differences: np.ndarray, margin: float = 100) -> dict:
    """Return a root-cluster t interval conditional on frozen opponents/layouts."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 3 or any(size == 0 for size in differences.shape) or differences.shape[0] < 2:
        raise ValueError("Expected at least two complete roots, each with the same opponent/layout panel")
    if not np.isfinite(differences).all() or not np.isfinite(margin) or margin <= 0:
        raise ValueError("Differences must be finite and the practical margin must be positive")
    root_means = differences.mean(axis=(1, 2))
    mean = float(root_means.mean())
    standard_error = float(root_means.std(ddof=1) / np.sqrt(len(root_means)))
    half_width = float(t.ppf(0.975, df=len(root_means) - 1) * standard_error)
    lower, upper = mean - half_width, mean + half_width
    if lower > margin:
        classification = "useful_full_policy_advantage"
    elif upper < -margin:
        classification = "meaningful_baseline_advantage"
    elif lower >= -margin and upper <= margin:
        classification = "comparable_within_margin"
    else:
        classification = "inconclusive"
    return {
        "difference": "full_policy_return_minus_frozen_baseline_return",
        "mean_difference": mean,
        "confidence_level": 0.95,
        "confidence_interval": [lower, upper],
        "standard_error": standard_error,
        "practical_margin": float(margin),
        "classification": classification,
        "roots": differences.shape[0],
        "opponents": differences.shape[1],
        "layouts": differences.shape[2],
        "matched_episode_pairs": int(differences.size),
        "root_mean_differences": root_means.tolist(),
        "uncertainty": "Student t interval across independent episode-root means; fixed opponent/layout panel",
        "scope": (
            "Conditional on selected checkpoints and the fitted baseline; evaluation roots are not training replicates"
        ),
    }
