"""Held-out scripted-Red, learned matchup, and checkpoint cross-play figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .data import CONDITIONS, MAX_STEPS, _seed_band, paired_families
from .style import (
    COND_COLORS,
    COND_LABELS,
    COND_MARKERS,
    FAMILY_LABELS,
    PANEL_W,
    RED_LABELS,
    RED_ORDER,
    TEAM_LABELS,
    _bar_grid,
    _cond_legend,
    _grid,
)


def fig_scripted_reds(scripted: pd.DataFrame) -> plt.Figure | None:
    families = paired_families(scripted)
    if not families:
        return None
    fig, axes = plt.subplots(1, len(families), figsize=(PANEL_W * len(families), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    offsets = {"single": -0.17, "diverse": 0.17}
    for ax, fam in zip(axes, families):
        sub = scripted[scripted["family"] == fam]
        for red_idx, red in enumerate(RED_ORDER):
            means = {}
            for cond in CONDITIONS:
                pts = sub[(sub["condition"] == cond) & (sub["red"] == red)]["reward"]
                if not pts.empty:
                    means[cond] = pts.mean()
                    y = red_idx + offsets[cond]
                    ax.scatter(
                        pts, np.full(len(pts), y), s=40, marker=COND_MARKERS[cond], color=COND_COLORS[cond], alpha=0.4
                    )
            if len(means) == 2:
                ax.plot(
                    [means["single"], means["diverse"]],
                    [red_idx + offsets["single"], red_idx + offsets["diverse"]],
                    color="gray",
                    lw=1.34,
                    alpha=0.7,
                )
            for cond, value in means.items():
                y = red_idx + offsets[cond]
                ax.scatter([value], [y], s=100, marker=COND_MARKERS[cond], color=COND_COLORS[cond], zorder=3)
                ax.annotate(
                    f"{value:,.0f}",
                    (value, y),
                    xytext=(0, 9 if cond == "single" else -9),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if cond == "single" else "top",
                    fontsize=11,
                    color="black",
                )
        ax.set_yticks(range(len(RED_ORDER)))
        ax.set_yticklabels([RED_LABELS[r] for r in RED_ORDER])
        ax.set_ylim(len(RED_ORDER) + 0.35, -0.8)  # room for the legend below and value labels above
        ax.set_title(FAMILY_LABELS[fam])
        ax.set_xlabel("Blue return vs scripted Red")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.margins(x=0.08)
        _grid(ax)
    axes[0].set_ylabel("Scripted Red")
    axes[0].legend(
        handles=[
            Line2D([], [], color=COND_COLORS[c], lw=0, marker=COND_MARKERS[c], markersize=10, label=COND_LABELS[c])
            for c in CONDITIONS
        ],
        loc="lower center",
        ncol=2,
    )
    fig.tight_layout()
    return fig


def fig_cia_drops(scripted: pd.DataFrame) -> plt.Figure | None:
    families = paired_families(scripted)
    if not families:
        return None
    dims = ("c", "i", "a")
    fig, axes = plt.subplots(
        len(families),
        len(RED_ORDER),
        figsize=(0.75 * PANEL_W * len(RED_ORDER), 3.3 * len(families)),
        sharey="row",
        squeeze=False,
    )
    width = 0.38
    for row, fam in enumerate(families):
        sub = scripted[scripted["family"] == fam]
        for col, red in enumerate(RED_ORDER):
            ax = axes[row, col]
            for cond_idx, cond in enumerate(CONDITIONS):
                part = sub[(sub["condition"] == cond) & (sub["red"] == red)]
                if part.empty:
                    continue
                means = [part[d].mean() for d in dims]
                xs = np.arange(len(dims)) + (cond_idx - 0.5) * width
                bars = ax.bar(xs, means, width=width, color=COND_COLORS[cond], edgecolor="none")
                ax.bar_label(bars, labels=[f"{m:.1f}" for m in means], fontsize=10, padding=2, color="black")
            ax.axhline(0, color="black", lw=1.0)
            ax.set_xticks(range(len(dims)))
            ax.set_xticklabels([d.upper() for d in dims])
            if row == 0:
                ax.set_title(f"{RED_LABELS[red]} Red")
            ax.margins(y=0.14)
            _bar_grid(ax, "y")
        axes[row, 0].set_ylabel(f"{FAMILY_LABELS[fam]}\nmean per-step drop")
    _cond_legend(axes[0, 0], kind="patch", loc="lower right")
    fig.tight_layout()
    return fig


def fig_learned_reds(matchups: pd.DataFrame) -> plt.Figure | None:
    families = paired_families(matchups)
    if not families:
        return None
    teams = [
        ("blue", "blue_return", [("self", "vs own Red"), ("cross_seed", "vs cross-seed Red")]),
        ("red", "red_return", [("self", "vs own Blue"), ("cross_seed", "vs cross-seed Blue")]),
    ]
    width = 0.38
    fig, axes = plt.subplots(2, len(families), figsize=(PANEL_W * len(families), 8.0), sharey="row", squeeze=False)
    for r, (team, column, kinds) in enumerate(teams):
        for c, fam in enumerate(families):
            ax = axes[r, c]
            sub = matchups[matchups["family"] == fam]
            for cond_idx, cond in enumerate(CONDITIONS):
                for kind_idx, (kind, _) in enumerate(kinds):
                    part = sub[(sub["condition"] == cond) & (sub["kind"] == kind)][column]
                    if part.empty:
                        continue
                    x = kind_idx + (cond_idx - 0.5) * width
                    ax.bar(x, part.mean(), width=width, color=COND_COLORS[cond], edgecolor="none")
                    ax.scatter(np.full(len(part), x), part, s=30, color="black", alpha=0.6, zorder=3)
            ax.axhline(0, color="black", lw=1.0)
            ax.set_xticks(range(len(kinds)))
            ax.set_xticklabels([label for _, label in kinds])
            if r == 0:
                ax.set_title(FAMILY_LABELS[fam])
            _bar_grid(ax, "y")
        axes[r, 0].set_ylabel(f"{TEAM_LABELS[team]} return (held-out bank)")
    _cond_legend(axes[0, -1], kind="patch", loc="lower right")
    fig.tight_layout()
    return fig


def _step_panel(ax: plt.Axes, sub: pd.DataFrame, value: str) -> None:
    """Thin per-seed lines plus the seed mean on the union of checkpoint steps."""
    for cond in CONDITIONS:
        color = COND_COLORS[cond]
        part = sub[sub["condition"] == cond]
        runs = [run for _, run in part.groupby("seed")]
        if not runs:
            continue
        for run in runs:
            run = run.sort_values("step")
            ax.plot(run["step"] / 1e6, run[value], color=color, lw=0.9, alpha=0.35)
        grid = np.unique(part["step"])
        mean, _, _ = _seed_band(runs, grid, "step", value)
        ax.plot(grid / 1e6, mean, color=color, lw=1.34, alpha=0.7)
        ax.scatter(grid / 1e6, mean, s=100, marker=COND_MARKERS[cond], color=color, zorder=3)


def fig_eval_by_step(by_step: pd.DataFrame) -> plt.Figure | None:
    families = paired_families(by_step)
    if not families:
        return None
    rows = [
        ("blue", "Blue return vs the run's\nRed checkpoints"),
        ("red", "Red return vs the run's\nBlue checkpoints"),
    ]
    fig, axes = plt.subplots(
        2, len(families), figsize=(PANEL_W * len(families), 7.6), sharex=True, sharey="row", squeeze=False
    )
    for r, (team, ylabel) in enumerate(rows):
        for c, fam in enumerate(families):
            ax = axes[r, c]
            _step_panel(ax, by_step[(by_step["family"] == fam) & (by_step["team"] == team)], "return")
            if r == 0:
                ax.set_title(FAMILY_LABELS[fam])
            if r == 1:
                ax.set_xlabel("Checkpoint environment steps (M)")
            ax.set_xlim(-2, MAX_STEPS / 1e6 + 2)
            _grid(ax)
        axes[r, 0].set_ylabel(ylabel)
    _cond_legend(axes[0, -1], loc="lower right")
    fig.tight_layout()
    return fig


def fig_scripted_reds_by_step(checkpoint_scripted: pd.DataFrame) -> plt.Figure | None:
    families = paired_families(checkpoint_scripted)
    if not families:
        return None
    fig, axes = plt.subplots(
        len(families),
        len(RED_ORDER),
        figsize=(0.75 * PANEL_W * len(RED_ORDER), 3.6 * len(families)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for r, fam in enumerate(families):
        for c, red in enumerate(RED_ORDER):
            ax = axes[r, c]
            sub = checkpoint_scripted[(checkpoint_scripted["family"] == fam) & (checkpoint_scripted["red"] == red)]
            _step_panel(ax, sub, "reward")
            if r == 0:
                ax.set_title(f"{RED_LABELS[red]} Red")
            if r == len(families) - 1:
                ax.set_xlabel("Checkpoint environment steps (M)")
            ax.set_xlim(-2, MAX_STEPS / 1e6 + 2)
            _grid(ax)
        axes[r, 0].set_ylabel(f"{FAMILY_LABELS[fam]}\nBlue return")
    _cond_legend(axes[0, 0])
    fig.tight_layout()
    return fig


def fig_zero_sum_decomposition(decomposition: pd.DataFrame) -> plt.Figure | None:
    families = paired_families(decomposition)
    if not families:
        return None
    parts = [
        ("selfplay_change", "Blue return,\nself-play"),
        ("blue_vs_frozen_red", "Blue return vs\nfrozen first Red"),
        ("red_vs_frozen_blue", "Red return vs\nfrozen first Blue"),
    ]
    width = 0.38
    fig, axes = plt.subplots(1, len(families), figsize=(PANEL_W * len(families), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, families):
        for cond_idx, cond in enumerate(CONDITIONS):
            sub = decomposition[(decomposition["family"] == fam) & (decomposition["condition"] == cond)]
            for part_idx, (column, _) in enumerate(parts):
                values = sub[column]
                x = part_idx + (cond_idx - 0.5) * width
                ax.bar(x, values.mean(), width=width, color=COND_COLORS[cond], edgecolor="none")
                ax.scatter(np.full(len(values), x), values, s=26, color="black", alpha=0.6, zorder=3)
        ax.axhline(0, color="black", lw=1.0)
        ax.set_xticks(range(len(parts)))
        ax.set_xticklabels([label for _, label in parts], fontsize=11)
        ax.set_title(FAMILY_LABELS[fam])
        _bar_grid(ax, "y")
    axes[0].set_ylabel("Change, first to last checkpoint")
    _cond_legend(axes[0], kind="patch", loc="upper left")
    fig.tight_layout()
    return fig


def fig_cross_play(cross_play: dict, family_name: str, seed: int) -> plt.Figure | None:
    keys = [(family_name, cond, seed) for cond in CONDITIONS]
    if not all(k in cross_play for k in keys):
        return None
    mats = [cross_play[k]["matrix"] for k in keys]
    vmin = min(m.min() for m in mats)
    vmax = max(m.max() for m in mats)
    fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_W, 4.3), gridspec_kw={"width_ratios": (1, 1.22)})
    for ax, key, mat in zip(axes, keys, mats):
        steps = [f"{s / 1e6:.0f}M" for s in cross_play[key]["steps"]]
        sns.heatmap(
            mat,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap="Spectral",
            annot=mat / 1000,
            fmt=".1f",
            annot_kws={"size": 10},
            cbar=ax is axes[-1],
            cbar_kws={"label": "Blue return (Red = −Blue)"},
            xticklabels=steps,
            yticklabels=steps,
            linewidths=0,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        ax.set_title(COND_LABELS[key[1]])
        ax.set_xlabel("Red checkpoint")
        ax.set_ylabel("Blue checkpoint")
    fig.suptitle(f"{FAMILY_LABELS[family_name]} seed {seed}: checkpoint cross-play (labels in thousands)")
    fig.tight_layout()
    return fig
