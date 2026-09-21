"""Training curves, reward components, and action statistics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import CONDITIONS, FAMILY_ORDER, MAX_STEPS, TEAMS, _seed_band
from .style import (
    _TICK_LABELS,
    COMPONENT_COLORS,
    COND_COLORS,
    COND_SHORT,
    FAMILY_LABELS,
    PANEL_W,
    TEAM_LABELS,
    _band_legend,
    _bar_grid,
    _cond_legend,
    _grid,
    _lighten,
)


def _training_panel(ax: plt.Axes, data: pd.DataFrame, *, linewidth: float) -> None:
    grid = np.linspace(0, MAX_STEPS, 141)
    for cond in CONDITIONS:
        runs = [run for _, run in data[data["condition"] == cond].groupby("seed")]
        if not runs:
            continue
        mean, low, high = _seed_band(runs, grid, "step", "value")
        ax.fill_between(grid / 1e6, low, high, color=_lighten(COND_COLORS[cond]), alpha=0.55, lw=0)
        ax.plot(grid / 1e6, mean, color=COND_COLORS[cond], lw=linewidth)
    ax.set_xlim(0, MAX_STEPS / 1e6)
    _grid(ax)


def fig_training_curves(training: pd.DataFrame) -> plt.Figure:
    data = training[(training["team"] == "blue") & (training["metric"] == "return")]
    families = [f for f in FAMILY_ORDER if f in set(data["family"])]
    fig, axes = plt.subplots(1, len(families), figsize=(PANEL_W * len(families), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, families):
        _training_panel(ax, data[data["family"] == fam], linewidth=1.8)
        ax.set_title(FAMILY_LABELS[fam])
        ax.set_xlabel("Environment steps (M)")
    axes[0].set_ylabel("Blue training return")
    _band_legend(axes[0])
    fig.tight_layout()
    return fig


def fig_training_by_team(training: pd.DataFrame) -> plt.Figure:
    families = [f for f in FAMILY_ORDER if f in set(training["family"])]
    metrics = [
        ("return", "training return"),
        ("entropy", "policy entropy"),
        ("explained_variance", "value expl. var."),
    ]
    rows = [(metric, team, f"{TEAM_LABELS[team]}\n{label}") for metric, label in metrics for team in TEAMS]
    fig, axes = plt.subplots(
        len(rows),
        len(families),
        figsize=(PANEL_W * len(families), 2.7 * len(rows)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for r, (metric, team, ylabel) in enumerate(rows):
        for c, fam in enumerate(families):
            ax = axes[r, c]
            sub = training[(training["family"] == fam) & (training["metric"] == metric) & (training["team"] == team)]
            _training_panel(ax, sub, linewidth=1.6)
            if r == 0:
                ax.set_title(FAMILY_LABELS[fam])
            if r == len(rows) - 1:
                ax.set_xlabel("Environment steps (M)")
        axes[r, 0].set_ylabel(ylabel)
    _band_legend(axes[0, 0])
    fig.tight_layout()
    return fig


def fig_red_reward_sources(components: pd.DataFrame) -> plt.Figure | None:
    """Stacked Red-reward sources, grouped by learner family with one bar per condition."""
    if components.empty:
        return None
    parts = [
        ("team.red.reward_ria", "RIA: Red impacts", COMPONENT_COLORS["ria"]),
        ("team.red.reward_lwf", "LWF: green work failures", COMPONENT_COLORS["lwf"]),
        ("team.red.reward_asf", "ASF: Blue's own blocking", COMPONENT_COLORS["asf"]),
        ("team.red.action_cost", "Blue Restore cost", COMPONENT_COLORS["cost"]),
    ]
    families = [
        f
        for f in FAMILY_ORDER
        if any(not components[(components["family"] == f) & (components["condition"] == c)].empty for c in CONDITIONS)
    ]
    if not families:
        return None

    # One group per family; the two conditions sit side by side inside it.
    offset, bar_h = 0.21, 0.38
    bars = []  # (y, family, condition, seed-mean row)
    for fam_idx, fam in enumerate(families):
        for cond_idx, cond in enumerate(CONDITIONS):
            sub = components[(components["family"] == fam) & (components["condition"] == cond)]
            if sub.empty:
                continue
            bars.append((fam_idx + (cond_idx - 0.5) * 2 * offset, fam, cond, sub.mean(numeric_only=True)))

    fig, ax = plt.subplots(figsize=(10, 1.45 * len(families) + 2.4))
    ys = np.array([b[0] for b in bars])
    left = np.zeros(len(bars))
    for key, label, color in parts:
        widths = np.array([b[3][key] for b in bars])
        ax.barh(ys, widths, height=bar_h, left=left, color=color, edgecolor="none", label=label)
        for y, start_x, w in zip(ys, left, widths):
            if w > 450:
                ax.text(start_x + w / 2, y, f"{w:,.0f}", ha="center", va="center", fontsize=11, color="black")
        left += widths
    for y, total in zip(ys, left):
        ax.text(total + 70, y, f"{total:,.0f}", va="center", fontsize=12, color="black")

    # Two-level y axis: condition per bar (major), family per group (minor, padded out).
    ax.set_yticks(ys)
    # Axis text stays ink: the segment fills are the only thing carrying colour identity here.
    ax.set_yticklabels([COND_SHORT[b[2]] for b in bars], fontsize=12)
    ax.set_yticks(range(len(families)), minor=True)
    ax.set_yticklabels([FAMILY_LABELS[f] for f in families], minor=True, fontsize=14)
    ax.tick_params(axis="y", which="major", length=0)
    ax.tick_params(axis="y", which="minor", length=0, pad=58)
    ax.set_ylim(len(families) - 0.5, -0.5)
    ax.set_xlim(0, left.max() * 1.12)
    ax.set_xlabel("Red reward per training episode (last 30 updates, seed mean)")
    ax.set_title("Where Red's reward comes from", pad=34)
    # A single row above the axes: a legend inside the axes overlaps the longest bars.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=len(parts), frameon=False)
    _bar_grid(ax, "x")
    fig.tight_layout()
    return fig


def fig_action_stats(components: pd.DataFrame) -> plt.Figure | None:
    if components.empty:
        return None
    metrics = [
        ("Blue Restores / episode", lambda d: -d["team.blue.action_cost"]),
        ("Red impact events / episode", lambda d: d["backend.jax.game.impact_count"]),
        ("Green access blocked by Blue", lambda d: d["backend.jax.game.green_asf_count"]),
        ("Green work failures / episode", lambda d: d["backend.jax.game.green_lwf_count"]),
        ("Blue decision share (%)", lambda d: 100 * d["team.blue.actor_fraction"]),
        ("Red active decision share (%)", lambda d: 100 * d["team.red.actor_fraction"]),
    ]
    families = [f for f in FAMILY_ORDER if f in set(components["family"])]
    fig, axes = plt.subplots(2, 3, figsize=(3 * PANEL_W, 8.4))
    width = 0.38
    for ax, (title, get) in zip(axes.flat, metrics):
        for cond_idx, cond in enumerate(CONDITIONS):
            for fam_idx, fam in enumerate(families):
                sub = components[(components["family"] == fam) & (components["condition"] == cond)]
                if sub.empty:
                    continue
                values = get(sub)
                x = fam_idx + (cond_idx - 0.5) * width
                ax.bar(x, values.mean(), width=width, color=COND_COLORS[cond], edgecolor="none")
                ax.scatter(np.full(len(values), x), values, s=26, color="black", alpha=0.6, zorder=3)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels([_TICK_LABELS[f] for f in families])
        ax.set_title(title)
        _bar_grid(ax, "y")
    _cond_legend(axes.flat[2], kind="patch", loc="upper right")
    fig.tight_layout()
    return fig
