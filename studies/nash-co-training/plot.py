"""Plot the matched response and CPU cross-play comparisons from a replay."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxborg-matplotlib")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from plots.cotraining.style import _style  # noqa: E402
from plots.plot_cross_play_fsm import heatmap  # noqa: E402

STEPS = ("99840000", "124800000", "150000000")
AGES = ("99.84M", "124.8M", "150M")
RED = "#a7384c"
BLUE = "#28598e"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_effects(ax, labels, rows, color):
    means = np.array([float(row["own_return_first_minus_second"]) for row in rows])
    low = np.array([float(row["simultaneous_95_low"]) for row in rows])
    high = np.array([float(row["simultaneous_95_high"]) for row in rows])
    positions = np.arange(len(rows))
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1.1)
    ax.grid(axis="x", color="#d9dfe7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.errorbar(means, positions, xerr=[means - low, high - means], color=color, fmt="o", capsize=4, markersize=7)
    ax.set_yticks(positions, labels)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    for position, mean, lower, upper in zip(positions, means, low, high, strict=True):
        ax.text(
            1.025,
            position,
            f"{mean:+.2f} [{lower:.2f}, {upper:.2f}]".replace("-", "−"),
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=10,
        )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def fresh_response(contrasts, selected):
    rows = {row["contrast"]: row for row in contrasts if row["family"] == "fresh_response_4"}
    targets = STEPS[1:]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.7), sharex=True, layout="constrained")
    fig.suptitle("Fresh Red responses improve on saved Red 150M", fontweight="bold")
    plot_effects(
        axes[0],
        [f"Against Blue {age}\nRed seed {selected[step]['seed']}" for age, step in zip(AGES[1:], targets, strict=True)],
        [rows[f"selected_red_gain_at_blue{step}"] for step in targets],
        RED,
    )
    axes[0].set_title("Red return · selected attacker minus saved Red 150M", loc="left", fontsize=13)
    plot_effects(
        axes[1],
        [
            f"Attacker selected for Blue {age}\nRed seed {selected[step]['seed']}"
            for age, step in zip(AGES[1:], targets, strict=True)
        ],
        [rows[f"blue150_minus_blue125_vs_selected_for{step}"] for step in targets],
        BLUE,
    )
    axes[1].set_title("Blue return · Blue 150M minus Blue 124.8M against the same attacker", loc="left", fontsize=13)
    axes[1].set(xlim=(-140, 590), xlabel="Paired own-return difference")
    fig.supxlabel(
        "One defender trajectory and topology · 4,096 held-out episode roots per contrast\n"
        "Whiskers show approximate simultaneous 95% intervals across four registered contrasts",
        fontsize=10,
    )
    return fig


def cpu_crossplay(matchups, contrasts):
    cells = {
        (row["blue_steps"], row["red_steps"]): float(row["blue_mean_return"])
        for row in matchups
        if row["cohort"] == "cpu-matrix" and row["status"] == "complete"
    }
    matrix = np.array([[cells[blue, red] for red in STEPS] for blue in STEPS])
    effects = {
        row["opponent_steps_or_panel"]: row
        for row in contrasts
        if row["family"] == "cpu_crossplay_24"
        and row["team"] == "red"
        and row["scope"] == "fixed_opponent"
        and (row["first_policy_steps"], row["second_policy_steps"]) == (STEPS[0], STEPS[2])
    }

    fig, (payoff, reversal) = plt.subplots(1, 2, figsize=(12.2, 5.2), width_ratios=[1, 1.25], layout="constrained")
    fig.suptitle("Saved Red rankings depend on the Blue opponent", fontweight="bold")
    limits = (matrix.min(), matrix.max())
    heatmap(payoff, matrix, AGES, AGES, limits)
    payoff.set(title="CPU payoff matrix", xlabel="Red checkpoint", ylabel="Blue checkpoint")
    fig.colorbar(
        ScalarMappable(norm=Normalize(*limits), cmap="Spectral"),
        ax=payoff,
        location="bottom",
        shrink=0.86,
        label="Mean Blue return (higher favors Blue)",
    )
    plot_effects(reversal, AGES, [effects[step] for step in STEPS], RED)
    reversal.set(
        title="Red 99.84M minus Red 150M",
        xlabel="Paired Red return difference",
        ylabel="Fixed Blue checkpoint",
        xlim=(-54, 60),
    )
    fig.supxlabel(
        "One MAPPO trajectory and topology · 4,096 aligned episode roots per cell\n"
        "Whiskers show approximate simultaneous 95% intervals across 24 registered contrasts",
        fontsize=10,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analysis = args.analysis.resolve()
    output = args.output.resolve()
    if output.is_relative_to(analysis) or output.is_relative_to(REPO):
        parser.error("Choose a new figure directory outside the analysis and source checkout.")
    contrasts = read_rows(analysis / "contrasts.csv")
    matchups = read_rows(analysis / "matchups.csv")
    selected = json.loads((analysis / "response-selection.json").read_text(encoding="utf-8"))["selected"]
    output.mkdir(parents=True)
    _style()
    plt.rcParams["svg.hashsalt"] = "jaxborg-nash"
    for name, fig in (
        ("fresh-red-response", fresh_response(contrasts, selected)),
        ("cpu-saved-policy-crossplay", cpu_crossplay(matchups, contrasts)),
    ):
        fig.savefig(output / f"{name}.png", dpi=180)
        fig.savefig(output / f"{name}.svg", metadata={"Date": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
