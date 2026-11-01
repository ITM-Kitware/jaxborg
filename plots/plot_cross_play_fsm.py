#!/usr/bin/env python3
"""Plot saved IPPO-LSTM cross-play and matched checkpoint-vs-FSM results.

Example:
    uv run python plots/plot_cross_play_fsm.py --fsm-results exp/diagnostics/lstm_seed42_checkpoint_fsm.jsonl

Outputs default to ``<exp-dir>/plots/cotraining_diversity`` so figures stay
with the experiment snapshot that supplied their evaluation data.

FSM has no learned checkpoint axis: its comparison has one column per training
condition. Both figures and the combined figure use the same reward color scale.
Grouped bar plots use two fixed Spectral colors for the training conditions;
their error bars show one standard deviation across evaluation episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxborg-matplotlib")
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from plots.cotraining.data import CONDITIONS
from plots.cotraining.style import COND_LABELS, _style


def read_cross_play(eval_dir: Path, seed: int):
    result = {}
    for condition in CONDITIONS:
        recipe = "cotraining_lstm" + ("_env_diversity" if condition == "diverse" else "")
        candidates = sorted(eval_dir.glob(f"{recipe}_seed{seed}_cross_play_*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No {recipe} seed {seed} cross-play result")
        source = candidates[-1]
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        summary = rows[-1]
        assert summary["eval_name"] == "cross_play_summary"
        steps = summary["steps"]
        cells = rows[:-1]
        if len(cells) != len(steps) ** 2:
            raise ValueError("Incomplete cross-play matrix")
        for cell in cells:
            if cell["seeds"] != cells[0]["seeds"] or cell["topology_paths"] != cells[0]["topology_paths"]:
                raise ValueError("Cross-play cells use different evaluation cases")
        result[condition] = dict(
            source=source,
            summary=summary,
            cells=cells,
            steps=steps,
            matrix=np.array(summary["blue_payoff_matrix"], dtype=float),
        )
    return result


def read_fsm(path: Path, cross):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {}
    for condition in CONDITIONS:
        data = cross[condition]
        values = []
        for step in data["steps"]:
            matching = [
                row
                for row in rows
                if row["source_cross_play_eval_id"] == data["summary"]["eval_id"]
                and row["checkpoint_step"] == step
                and row["eval_red"] == "fsm"
            ]
            if not matching:
                raise ValueError(f"Missing FSM result: {condition}, {step}")
            row = matching[-1]
            cell = next(c for c in data["cells"] if c["blue_step"] == step)
            assert row["seeds"] == cell["seeds"]
            assert row["n_episodes"] == cell["n_episodes"]
            assert row["stochastic"] == cell["stochastic"]
            assert [Path(p).name for p in row["topology_paths"]] == [Path(p).name for p in cell["topology_paths"]]
            assert np.isclose(np.mean(row["per_episode"]), row["mean_reward"])
            values.append(row)
        result[condition] = values
    return result


def ticks(steps):
    return [f"{step / 1e6:g}M" for step in steps]


def heatmap(ax, matrix, xlabels, ylabels, limits):
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="Spectral",
        vmin=limits[0],
        vmax=limits[1],
        annot=True,
        fmt=".0f",
        annot_kws={"size": 12},
        cbar=False,
        xticklabels=xlabels,
        yticklabels=ylabels,
    )
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)


def finish(fig, axes, limits, title, episodes):
    fig.colorbar(
        ScalarMappable(norm=Normalize(*limits), cmap="Spectral"),
        ax=axes,
        label="Mean Blue episode return (higher is better)",
        shrink=0.85,
    )
    fig.suptitle(title, fontsize=17)
    fig.supxlabel(f"{episodes} episodes per cell · 500 steps · held-out ops3 bank · original observations", fontsize=11)


def grouped_return_bars(means, deviations, groups, title, group_label, episodes):
    """Plot raw negative returns from zero, with descriptive episode variability."""
    palette = plt.get_cmap("Spectral")
    colors = {"single": palette(0.96), "diverse": palette(0.78)}
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    centers = np.arange(len(groups), dtype=float)
    height = 0.32
    lower = min(np.min(means[c] - deviations[c]) for c in CONDITIONS)
    upper = max(0.0, max(np.max(means[c] + deviations[c]) for c in CONDITIONS))
    span = upper - lower
    for index, condition in enumerate(CONDITIONS):
        positions = centers + (index - 0.5) * (height + 0.04)
        ax.barh(
            positions,
            means[condition],
            height=height,
            color=colors[condition],
            edgecolor="none",
            label=COND_LABELS[condition],
            xerr=deviations[condition],
            error_kw={"ecolor": "#333333", "elinewidth": 1.1, "capsize": 4, "capthick": 1.1},
        )
        for position, value in zip(positions, means[condition], strict=True):
            ax.text(
                -0.025 * span,
                position,
                f"{value:,.0f}",
                ha="right",
                va="center",
                color="white" if condition == "single" else "#17332f",
                fontsize=12,
            )
    ax.set_yticks(centers, groups)
    ax.invert_yaxis()
    ax.set_xlim(lower - 0.06 * span, upper)
    ax.set_xlabel("Mean Blue episode return\nHigher (closer to zero) is better", fontsize=12)
    ax.set_ylabel(group_label)
    ax.axvline(0, color="#333333", lw=1.2)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=11)
    fig.suptitle(title.replace(": ", "\n", 1), fontsize=16)
    fig.supxlabel(
        f"Whiskers: ±1 episode standard deviation · {episodes} episodes per bar\n"
        "500 steps · held-out ops3 bank · original observations",
        fontsize=10,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", type=Path, default=Path("remote/jaxborg-exp"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory; defaults to <exp-dir>/plots/cotraining_diversity",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fsm-results", type=Path)
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf"], default=["png", "pdf"])
    args = parser.parse_args()
    out_dir = args.out_dir or args.exp_dir / "plots" / "cotraining_diversity"
    _style()
    cross = read_cross_play(args.exp_dir / "eval", args.seed)
    fsm = read_fsm(args.fsm_results, cross) if args.fsm_results else None
    all_values = [data["matrix"].ravel() for data in cross.values()]
    if fsm:
        all_values += [np.array([row["mean_reward"] for row in fsm[condition]]) for condition in CONDITIONS]
    limits = (min(values.min() for values in all_values), max(values.max() for values in all_values))
    episodes = cross["single"]["cells"][0]["n_episodes"]
    figures = {}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), layout="constrained")
    for ax, condition in zip(axes, CONDITIONS):
        data = cross[condition]
        heatmap(ax, data["matrix"], ticks(data["steps"]), ticks(data["steps"]), limits)
        ax.set(title=COND_LABELS[condition], xlabel="Learned Red checkpoint", ylabel="Blue checkpoint")
    finish(fig, axes, limits, f"IPPO-LSTM seed {args.seed}: checkpoint cross-play", episodes)
    figures[f"cross_play_lstm_seed{args.seed}"] = fig

    if fsm:
        if cross["single"]["steps"] != cross["diverse"]["steps"]:
            raise ValueError("The FSM comparison requires matching Blue checkpoint steps")
        matrix = np.column_stack([[r["mean_reward"] for r in fsm[c]] for c in CONDITIONS])
        fig, ax = plt.subplots(figsize=(8.0, 5.1), layout="constrained")
        heatmap(ax, matrix, [COND_LABELS[c] for c in CONDITIONS], ticks(cross["single"]["steps"]), limits)
        ax.set(xlabel="Blue training condition", ylabel="Blue checkpoint")
        finish(fig, ax, limits, f"IPPO-LSTM seed {args.seed}: against scripted FSM Red", episodes)
        figures[f"fsm_lstm_seed{args.seed}"] = fig
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.1), layout="constrained")
        for ax, condition in zip(axes, CONDITIONS):
            data = cross[condition]
            matrix = np.column_stack([data["matrix"], [row["mean_reward"] for row in fsm[condition]]])
            heatmap(ax, matrix, ticks(data["steps"]) + ["FSM"], ticks(data["steps"]), limits)
            ax.axvline(len(data["steps"]), color="black", lw=2)
            ax.set(title=COND_LABELS[condition], xlabel="Learned Red checkpoint / fixed FSM", ylabel="Blue checkpoint")
        finish(fig, axes, limits, f"IPPO-LSTM seed {args.seed}: learned Red and scripted FSM", episodes)
        figures[f"cross_play_lstm_seed{args.seed}_with_fsm"] = fig
        means = {c: np.array([row["mean_reward"] for row in fsm[c]]) for c in CONDITIONS}
        deviations = {c: np.array([row["std_reward"] for row in fsm[c]]) for c in CONDITIONS}
        figures[f"fsm_lstm_seed{args.seed}_bars"] = grouped_return_bars(
            means,
            deviations,
            ticks(cross["single"]["steps"]),
            f"IPPO-LSTM seed {args.seed}: against scripted FSM Red",
            "Blue checkpoint",
            episodes,
        )
        latest_cells = {
            c: next(
                row
                for row in cross[c]["cells"]
                if row["blue_step"] == cross[c]["steps"][-1] and row["red_step"] == cross[c]["steps"][-1]
            )
            for c in CONDITIONS
        }
        figures[f"latest_opponents_lstm_seed{args.seed}_bars"] = grouped_return_bars(
            {c: np.array([latest_cells[c]["mean_reward"], fsm[c][-1]["mean_reward"]]) for c in CONDITIONS},
            {c: np.array([latest_cells[c]["std_reward"], fsm[c][-1]["std_reward"]]) for c in CONDITIONS},
            ["Own latest\nlearned Red", "Scripted FSM"],
            f"IPPO-LSTM seed {args.seed}: latest Blue checkpoint ({ticks(cross['single']['steps'])[-1]})",
            "Red opponent",
            episodes,
        )
        table = pd.DataFrame(
            [
                dict(
                    condition=c,
                    step=r["checkpoint_step"],
                    mean_return=r["mean_reward"],
                    std_return=r["std_reward"],
                    episodes=r["n_episodes"],
                )
                for c in CONDITIONS
                for r in fsm[c]
            ]
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / f"fsm_lstm_seed{args.seed}.csv", index=False)
        print(table.to_string(index=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    cross_rows = [
        {
            "condition": condition,
            "blue_step": row["blue_step"],
            "red_step": row["red_step"],
            "mean_return": row["mean_reward"],
            "std_return": row["std_reward"],
            "episodes": row["n_episodes"],
            "source": str(cross[condition]["source"]),
        }
        for condition in CONDITIONS
        for row in cross[condition]["cells"]
    ]
    pd.DataFrame(cross_rows).to_csv(out_dir / f"cross_play_lstm_seed{args.seed}.csv", index=False)
    for name, fig in figures.items():
        for ext in args.formats:
            path = out_dir / f"{name}.{ext}"
            # Preserve the square canvas for bars; constrained layout fits the labels.
            fig.savefig(path, dpi=220, bbox_inches=None if name.endswith("_bars") else "tight", pad_inches=0.05)
            print(f"Wrote {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
