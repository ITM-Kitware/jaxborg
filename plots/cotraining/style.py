"""Shared colors, labels, legends, and axis styling."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .data import CONDITIONS

DEFAULT_COLORS = [to_rgb(c) for c in plt.rcParamsDefault["axes.prop_cycle"].by_key()["color"]]


COND_COLORS = {"single": DEFAULT_COLORS[0], "diverse": DEFAULT_COLORS[1]}


COND_MARKERS = {"single": "o", "diverse": "X"}


COND_LABELS = {"single": "Single topology", "diverse": "Diverse bank (100)"}


_DEEP = [to_rgb(c) for c in sns.color_palette("deep", 10)]


COMPONENT_COLORS = {
    "ria": _DEEP[3],  # red
    "lwf": _DEEP[6],  # pink
    "asf": _DEEP[2],  # green
    "cost": _DEEP[4],  # purple
}


TEAM_LABELS = {"blue": "Blue", "red": "Red"}


FAMILY_LABELS = {
    "ippo": "IPPO feed-forward",
    "lstm": "IPPO-LSTM",
    "gru": "IPPO-GRU",
    "mappo": "MAPPO",
    "mappo_joint_obs": "MAPPO joint-obs",
}


COND_SHORT = {"single": "Single", "diverse": "Diverse"}


_TICK_LABELS = {
    "ippo": "IPPO",
    "lstm": "IPPO-LSTM",
    "gru": "IPPO-GRU",
    "mappo": "MAPPO",
    "mappo_joint_obs": "MAPPO\njoint-obs",
}


RED_LABELS = {"fsm": "FSM (stock)", "cia_c": "CIA-C", "cia_i": "CIA-I", "cia_a": "CIA-A"}


RED_ORDER = ("fsm", "cia_c", "cia_i", "cia_a")


PANEL_W = 4.7  # inches per panel; three panels span a 14 in slide or page width


def _style() -> None:
    sns.set_style("white")
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "normal",
            "figure.titlesize": 16,
            "figure.titleweight": "normal",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 14,
            "axes.linewidth": 1.5,
            "lines.linewidth": 1.34,
            "svg.fonttype": "path",  # text as outlines, so SVG, PNG and PDF render identically
        }
    )


def _grid(ax: plt.Axes) -> None:
    """Dashed light grid of the policy-shaping pareto figure (line and scatter panels)."""
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.3)
    ax.set_axisbelow(True)


def _bar_grid(ax: plt.Axes, axis: str) -> None:
    """Value-axis grid of the attribute-distribution bar figure."""
    ax.grid(True, axis=axis, linestyle="--", linewidth=0.65, alpha=0.99)
    ax.set_axisbelow(True)


def _cond_legend(ax: plt.Axes, kind: str = "marker", loc: str = "best") -> None:
    if kind == "patch":
        handles = [Patch(facecolor=COND_COLORS[c], edgecolor="none", label=COND_LABELS[c]) for c in CONDITIONS]
    else:
        handles = [
            Line2D(
                [],
                [],
                color=COND_COLORS[c],
                lw=1.34 if kind == "marker" else 2.2,
                marker=COND_MARKERS[c] if kind == "marker" else None,
                markersize=9,
                label=COND_LABELS[c],
            )
            for c in CONDITIONS
        ]
    ax.legend(handles=handles, loc=loc)


def _band_legend(ax: plt.Axes, loc: str = "best") -> None:
    """Condition legend for mean-and-band panels: line = seed mean, shaded patch = ±1 SD."""
    handles = [
        (
            Patch(facecolor=_lighten(COND_COLORS[c]), alpha=0.55, edgecolor="none"),
            Line2D([], [], color=COND_COLORS[c], lw=1.8),
        )
        for c in CONDITIONS
    ]
    ax.legend(handles, [f"{COND_LABELS[c]} (mean ± SD)" for c in CONDITIONS], loc=loc)


def _lighten(color, amount: float = 0.55) -> tuple[float, float, float]:
    """Blend ``color`` toward white; used for error bands so they read as a lighter shade of the line."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)
