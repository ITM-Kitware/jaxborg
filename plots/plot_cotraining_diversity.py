#!/usr/bin/env python3
"""Export cotraining diversity figures (PNG/PDF/SVG) and summary CSV tables.

Read <exp-dir>/mlflow.db and <exp-dir>/eval; write to
<exp-dir>/plots/cotraining_diversity unless --out-dir is supplied.
Only completed runs and paired evaluation conditions are compared.

Example: uv run python plots/plot_cotraining_diversity.py --formats png pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Support both direct execution and python -m plots.plot_cotraining_diversity.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from plots.cotraining import data, evaluation, training
from plots.cotraining.style import _style


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", default=os.environ.get("JAXBORG_EXP_DIR", "remote/jaxborg-exp"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory; defaults to <exp-dir>/plots/cotraining_diversity",
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--cross-play-family", default="lstm", choices=list(data.FAMILY_ORDER))
    parser.add_argument("--cross-play-seed", type=int, default=42)
    args = parser.parse_args(argv)

    exp_dir = Path(args.exp_dir)
    eval_dir = exp_dir / "eval"
    db_path = exp_dir / "mlflow.db"
    out_dir = args.out_dir or exp_dir / "plots" / "cotraining_diversity"
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()

    runs = data.completed_runs(db_path)
    scripted = data.load_scripted(eval_dir, runs.ids)
    checkpoint_scripted = data.load_checkpoint_scripted(eval_dir, runs.ids)
    matchups = data.load_matchups(eval_dir, runs.ids)
    cross_play = data.load_cross_play(eval_dir, runs.ids)
    by_step = data.cross_play_by_step(cross_play)
    decomposition = data.zero_sum_decomposition(cross_play)
    training_data = data.load_training(db_path, runs)
    components = data.load_team_components(db_path, runs)

    figures = {
        "training_curves": training.fig_training_curves(training_data),
        "training_by_team": training.fig_training_by_team(training_data),
        "scripted_reds": evaluation.fig_scripted_reds(scripted),
        "cia_drops": evaluation.fig_cia_drops(scripted),
        "learned_reds": evaluation.fig_learned_reds(matchups),
        "eval_by_step": evaluation.fig_eval_by_step(by_step),
        "scripted_reds_by_step": evaluation.fig_scripted_reds_by_step(checkpoint_scripted),
        "red_reward_sources": training.fig_red_reward_sources(components),
        "action_stats": training.fig_action_stats(components),
        "zero_sum_decomposition": evaluation.fig_zero_sum_decomposition(decomposition),
        f"cross_play_{args.cross_play_family}_seed{args.cross_play_seed}": evaluation.fig_cross_play(
            cross_play, args.cross_play_family, args.cross_play_seed
        ),
    }
    for name, fig in figures.items():
        if fig is None:
            print(f"skip {name}: no paired results yet")
            continue
        for ext in args.formats:
            path = out_dir / f"{name}.{ext}"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.05, transparent=False)
            print(f"wrote {path}")
        plt.close(fig)

    tables: dict[str, pd.DataFrame] = {
        "team_components.csv": components.set_index(["family", "condition", "seed"]).sort_index(),
    }
    if not scripted.empty:
        tables["scripted_reds_summary.csv"] = scripted.groupby(["family", "condition", "red"]).agg(
            reward=("reward", "mean"), seeds=("seed", "count"), c=("c", "mean"), i=("i", "mean"), a=("a", "mean")
        )
    if not matchups.empty:
        tables["learned_reds_summary.csv"] = matchups.groupby(["family", "condition", "kind"]).agg(
            blue_return=("blue_return", "mean"), red_return=("red_return", "mean"), n=("blue_return", "count")
        )
    if not decomposition.empty:
        tables["zero_sum_decomposition.csv"] = decomposition.set_index(["family", "condition", "seed"]).sort_index()
    if not by_step.empty:
        tables["eval_by_step.csv"] = by_step.set_index(["family", "condition", "seed", "team", "step"]).sort_index()
    if not checkpoint_scripted.empty:
        index = ["family", "condition", "seed", "red", "step"]
        tables["scripted_reds_by_step.csv"] = checkpoint_scripted.set_index(index).sort_index()
    for filename, table in tables.items():
        table.reset_index().to_csv(out_dir / filename, index=False)
        print(f"wrote {out_dir / filename}")


if __name__ == "__main__":
    main()
