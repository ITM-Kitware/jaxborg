#!/usr/bin/env python3
"""Compare two cotraining recipes against the same cross-seed learned Reds."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxborg.evaluation.env_diversity import build_plan, run_comparison
from jaxborg.evaluation.play_priors import _parse_seeds
from jaxborg.recipe import load


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_recipe", help="Non-diverse training recipe name or YAML path")
    parser.add_argument("diverse_recipe", help="Diverse training recipe name or YAML path")
    parser.add_argument("--exp-dir", default=os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp"))
    parser.add_argument("--train-seeds", help="Training seeds to compare; default: intersection of available seeds")
    parser.add_argument("--seeds", default="1000-1009", help="Evaluation episode seeds, separate from training seeds")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument(
        "--checkpoint-step", type=int, help="Exact periodic checkpoint step; default: completed final models"
    )
    parser.add_argument("--baseline-tag", default="*", help="Run-directory glob to disambiguate baseline reruns")
    parser.add_argument("--diverse-tag", default="*", help="Run-directory glob to disambiguate diverse reruns")
    parser.add_argument("--eval-recipe", help="Optional recipe supplying a common eval section for both conditions")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", help="New result directory; default: timestamped directory under EXP_DIR/eval")
    parser.add_argument("--resume", action="store_true", help="Resume the identical plan in --output-dir")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show models and matchups without rollouts or file writes"
    )
    args = parser.parse_args(argv)
    if args.resume and not args.output_dir:
        parser.error("--resume requires --output-dir")
    try:
        plan = build_plan(
            load(args.baseline_recipe),
            load(args.diverse_recipe),
            args.exp_dir,
            train_seeds=None if args.train_seeds is None else _parse_seeds(args.train_seeds),
            seeds=_parse_seeds(args.seeds),
            episodes_per_seed=args.episodes_per_seed,
            checkpoint_step=args.checkpoint_step,
            baseline_tag=args.baseline_tag,
            diverse_tag=args.diverse_tag,
            eval_recipe=None if args.eval_recipe is None else load(args.eval_recipe),
            deterministic=args.deterministic,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Training seeds: {plan['train_seeds']}; evaluation seeds: {plan['seeds']}", flush=True)
    for model in plan["models"]:
        print(f"  {model['condition']} seed={model['seed']} step={model['steps']}: {model['path']}", flush=True)
    for note in plan["notes"]:
        print(f"NOTE: {note}", flush=True)
    print(f"{len(plan['matchups'])} matchups x {plan['episodes_per_matchup']} episodes; same-seed opponents excluded.")
    if args.dry_run:
        for matchup in plan["matchups"]:
            print(f"  {matchup['id']}")
        return
    output = args.output_dir or str(
        Path(args.exp_dir)
        / "eval"
        / f"env_diversity_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
    )
    summary = run_comparison(plan, output, resume=args.resume)
    for metric, values in summary["overall"]["metrics"].items():
        print(
            f"{metric}: baseline={values['baseline']:.3f} diverse={values['diverse']:.3f} delta={values['delta']:+.3f}"
        )
    print(f"Results: {Path(output).resolve() / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
