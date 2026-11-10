"""Render the completed matched-return report with its retained plotting inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

POLICIES = {
    "lstm42-final": "LSTM 42\n124.992M training steps",
    "lstm100-final": "LSTM 100\n124.992M training steps",
    "mappo43-final": "MAPPO 43\n150M training steps",
}
TITLE = "Original policies outperform the constant mix"
STEM = "original-versus-constant-mix"


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def selected_rows(report):
    rows = report["summary"]
    selected = {row["blue"]: row for row in rows}
    if len(rows) != len(POLICIES) or set(selected) != set(POLICIES):
        raise ValueError("Require one completed comparison for each of the three final Blue policies")
    ordered = [selected[blue] for blue in POLICIES]
    for row in ordered:
        estimate = row["mean_difference"]
        lower, upper = row["confidence_interval"]
        if not all(math.isfinite(value) for value in (estimate, lower, upper, row["full_mean"], row["blind_mean"])):
            raise ValueError("The report contains nonfinite return estimates")
        if not lower <= estimate <= upper:
            raise ValueError("The confidence interval must contain its return estimate")
        if row["confidence_level"] != 0.95 or row["practical_margin"] != 100:
            raise ValueError("This figure requires 95% intervals and the declared 100-point practical margin")
        if row["difference"] != "full_policy_return_minus_frozen_baseline_return":
            raise ValueError("Unexpected return-difference direction")
        if not math.isclose(estimate, row["full_mean"] - row["blind_mean"], rel_tol=0, abs_tol=1e-8):
            raise ValueError("The report's return means disagree with its paired difference")
        dimensions = [row[name] for name in ("roots", "opponents", "layouts")]
        if any(not isinstance(value, int) or value < 1 for value in dimensions):
            raise ValueError("The report must identify a complete root/opponent/layout panel")
        if row["matched_episode_pairs"] != math.prod(dimensions):
            raise ValueError("The report's matched episode count disagrees with its panel")
    panel = [(row["roots"], row["opponents"], row["layouts"]) for row in ordered]
    if len(set(panel)) != 1:
        raise ValueError("The three policies must use the same declared comparison panel")
    if report["scored_test_episodes"] != 2 * sum(row["matched_episode_pairs"] for row in ordered):
        raise ValueError("The completed test-episode count disagrees with the plotted comparisons")
    return ordered


def plot(report_path, output_dir):
    report_path, output_dir = report_path.expanduser().resolve(), output_dir.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if output_dir.is_relative_to(checkout):
        raise ValueError("Generated figures must be saved outside the source checkout")
    report_identity = identity(report_path)
    report = json.loads(report_path.read_text())
    rows = selected_rows(report)
    roots, opponents, layouts = (rows[0][name] for name in ("roots", "opponents", "layouts"))
    scope = (
        f"{roots} fresh episode roots × {opponents} selected Reds × {layouts} familiar layouts per policy; 500 steps.\n"
        "Intervals describe episode-root variation for this fixed panel, conditional on the fitted baselines."
    )
    source_identity = identity(Path(__file__).resolve())
    estimates = [row["mean_difference"] for row in rows]
    lower = [row["confidence_interval"][0] for row in rows]
    upper = [row["confidence_interval"][1] for row in rows]
    minimum, maximum = min(-100, *lower), max(100, *upper)
    padding = (maximum - minimum) * 0.08
    output_dir.mkdir(parents=True, exist_ok=False)
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "observation-value",
        }
    ):
        figure, axis = plt.subplots(figsize=(10.5, 4.7))
        figure.subplots_adjust(left=0.23, right=0.72, top=0.76, bottom=0.32)
        figure.suptitle(TITLE, fontsize=15, x=0.03, ha="left", y=0.97)
        axis.set_title("Mean score advantage and 95% confidence interval", fontsize=10, loc="left", pad=20)
        axis.axvspan(-100, 100, color="#edf0f3", zorder=0)
        axis.axvline(0, color="#6b7280", linestyle="--", linewidth=1.0, zorder=1)
        axis.errorbar(
            estimates,
            range(len(rows)),
            xerr=[
                [estimate - bound for estimate, bound in zip(estimates, lower, strict=True)],
                [bound - estimate for estimate, bound in zip(estimates, upper, strict=True)],
            ],
            fmt="o",
            color="#245b82",
            ecolor="#245b82",
            markersize=7,
            elinewidth=2,
            capsize=5,
            zorder=3,
        )
        axis.set_yticks(range(len(rows)), [POLICIES[row["blue"]] for row in rows])
        axis.set_ylim(len(rows) - 0.5, -0.5)
        axis.set_xlim(minimum - padding, maximum + padding)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.grid(axis="x", color="#d8dee5", linewidth=0.6, zorder=1)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#a3aab4")
        axis.tick_params(axis="y", length=0, pad=12)
        axis.set_xlabel("Score advantage per episode (original sampling minus constant-mix sampling)", labelpad=10)
        axis.text(1.04, 1.02, "Difference [95% CI]", transform=axis.transAxes, fontsize=10)
        for position, row in enumerate(rows):
            estimate = row["mean_difference"]
            left, right = row["confidence_interval"]
            axis.text(
                1.04,
                position,
                f"{estimate:+.1f}  [{left:+.1f}, {right:+.1f}]",
                transform=axis.get_yaxis_transform(),
                va="center",
                fontsize=10,
            )
        figure.text(
            0.23,
            0.17,
            "Shaded band: ±100-point practical margin. Dashed line: no return difference.",
            fontsize=9,
            color="#4b5563",
        )
        figure.text(0.03, 0.04, scope, fontsize=9, linespacing=1.5, color="#4b5563")
        png_path, svg_path = output_dir / f"{STEM}.png", output_dir / f"{STEM}.svg"
        figure.savefig(png_path, dpi=200, bbox_inches="tight")
        figure.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
        plt.close(figure)
    if identity(report_path) != report_identity or identity(Path(__file__).resolve()) != source_identity:
        raise ValueError("The report or plotting source changed while exporting the figure")
    data = {
        "study_id": report["study_id"],
        "input_report": report_identity,
        "plotting_source": source_identity,
        "title": TITLE,
        "scope": scope,
        "rows": [
            {
                key: row[key]
                for key in (
                    "blue",
                    "full_mean",
                    "blind_mean",
                    "mean_difference",
                    "confidence_interval",
                    "confidence_level",
                    "practical_margin",
                    "roots",
                    "opponents",
                    "layouts",
                    "matched_episode_pairs",
                    "baseline_sha256",
                )
            }
            for row in rows
        ],
        "x_limits": [minimum - padding, maximum + padding],
        "artifacts": [identity(png_path), identity(svg_path)],
    }
    with (output_dir / f"{STEM}-data.json").open("x") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "artifacts": [item["path"] for item in data["artifacts"]]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    plot(arguments.report, arguments.output_dir)
