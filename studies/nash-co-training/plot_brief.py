"""Render the challenger Red comparison from replayed outcomes."""

import argparse
import csv
import os
import sys
from math import isclose
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxborg-matplotlib")

import matplotlib.pyplot as plt  # noqa: E402

from plots.cotraining.style import _style  # noqa: E402

CHALLENGER = "#854a91"


def interval(row):
    return tuple(
        float(row[key]) for key in ("own_return_first_minus_second", "simultaneous_95_low", "simultaneous_95_high")
    )


def fresh_red(row, original_matchup, challenger_matchup):
    original = float(original_matchup["red_mean_return"])
    adapted = float(challenger_matchup["red_mean_return"])
    challenger_steps = int(challenger_matchup["red_steps"])
    gain, low, high = interval(row)
    if not isclose(adapted - original, gain):
        raise ValueError("Confirmation means disagree with the paired gain")

    fig, ax = plt.subplots(figsize=(7.9, 4.0), layout="constrained")
    fig.suptitle(f"Challenger Red scored {gain:.0f} points more", fontweight="bold")
    ax.set_title("Both Reds face frozen Blue from the 150M checkpoint\n4,096 matched test episodes", fontsize=11)
    bars = ax.bar(
        (
            "Co-trained Red\n150M checkpoint",
            f"Challenger Red\n{challenger_steps / 1e6:g}M steps from scratch\nagainst frozen Blue",
        ),
        (original, adapted),
        width=0.58,
        color=("#687587", CHALLENGER),
    )
    ax.bar_label(bars, labels=[f"{score:,.0f}" for score in (original, adapted)], padding=5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#dce2e9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set(ylim=(0, max(original, adapted) * 1.17), ylabel="Average Red reward per episode")
    fig.supxlabel(
        f"Gain: {gain:+.2f} points per episode. Approximate adjusted 95% interval: {low:.2f} to {high:.2f}.",
        fontsize=9.5,
    )
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
    with (args.analysis / "matchups.csv").open(encoding="utf-8", newline="") as stream:
        confirmation = {
            row["source_run_id"]: row
            for row in csv.DictReader(stream)
            if row["study"] == "matched-red-response"
            and row["cohort"] == "confirmation"
            and row["status"] == "complete"
        }
    original = confirmation[response["second_eval_reference"]]
    adapted = confirmation[response["first_eval_reference"]]
    _style()
    args.output.mkdir(parents=True)
    figure = fresh_red(response, original, adapted)
    figure.savefig(args.output / "fresh-red-gain.png", dpi=145)
    plt.close(figure)


if __name__ == "__main__":
    main()
