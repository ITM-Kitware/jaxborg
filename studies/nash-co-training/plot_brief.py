"""Render the study summary charts from replayed contrasts."""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxborg-matplotlib")

import matplotlib.pyplot as plt  # noqa: E402

from plots.cotraining.style import _style  # noqa: E402

INK = "#202735"
EARLIER = "#276f86"
LATER = "#854a91"


def interval(row):
    return tuple(
        float(row[key]) for key in ("own_return_first_minus_second", "simultaneous_95_low", "simultaneous_95_high")
    )


def return_axis(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dce2e9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK, linewidth=1.2)


def fresh_red(row):
    mean, low, high = interval(row)
    fig, ax = plt.subplots(figsize=(7.9, 2.65), layout="constrained")
    fig.suptitle(f"A new attacker scored {mean:.0f} points more", fontweight="bold")
    ax.set_title("New attacker compared with the original attacker against the same defender", fontsize=11)
    ax.barh(0, mean, height=0.52, color=LATER)
    ax.errorbar(mean, 0.38, xerr=[[mean - low], [high - mean]], fmt="none", color=INK, capsize=4, linewidth=1.4)
    ax.text(mean - 16, 0, f"{mean:+.0f} points", ha="right", va="center", color="white", fontweight="bold")
    return_axis(ax)
    ax.set(xlim=(0, 575), ylim=(-0.43, 0.62), yticks=[], xlabel="Gain in attacker reward (points)")
    fig.supxlabel(
        f"Approximate adjusted 95% interval {low:.2f} to {high:.2f} points.",
        fontsize=9.5,
    )
    return fig


def saved_red(rows):
    fig, ax = plt.subplots(figsize=(7.9, 3.6), layout="constrained")
    fig.suptitle("The higher-scoring attacker depends on the defender", fontweight="bold")
    ax.set_title("Earlier attacker’s reward minus later attacker’s reward", fontsize=11)
    labels = ("Blue 99.84M", "Blue 124.8M", "Blue 150M")
    for position, row in enumerate(rows):
        mean, low, high = interval(row)
        ax.barh(position, mean, height=0.55, color=EARLIER if mean > 0 else LATER)
        ax.errorbar(
            mean, position - 0.29, xerr=[[mean - low], [high - mean]], fmt="none", color=INK, capsize=3, linewidth=1.2
        )
        ax.text(
            mean / 2,
            position,
            f"{mean:+.0f}".replace("-", "−"),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11,
        )
    return_axis(ax)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlim=(-54, 54), ylim=(2.48, -0.55), xlabel="← Favors later attacker          Favors earlier attacker →")
    fig.supxlabel("Whiskers show approximate 95% intervals adjusted for multiple comparisons.", fontsize=9.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with (args.analysis / "contrasts.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    response = next(
        row
        for row in rows
        if row["family"] == "fresh_response_4" and row["contrast"] == "selected_red_gain_at_blue150000000"
    )
    saved = {
        row["opponent_steps_or_panel"]: row
        for row in rows
        if row["family"] == "cpu_crossplay_24"
        and row["team"] == "red"
        and row["scope"] == "fixed_opponent"
        and row["first_policy_steps"] == "99840000"
        and row["second_policy_steps"] == "150000000"
    }
    _style()
    args.output.mkdir(parents=True)
    for name, figure in (
        ("fresh-red-gain", fresh_red(response)),
        ("saved-red-ranking", saved_red([saved[steps] for steps in ("99840000", "124800000", "150000000")])),
    ):
        figure.savefig(args.output / f"{name}.png", dpi=145)
        plt.close(figure)


if __name__ == "__main__":
    main()
