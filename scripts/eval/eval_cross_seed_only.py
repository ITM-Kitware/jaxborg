#!/usr/bin/env python3
"""Replay only a recipe's cyclic final-model cross-seed evaluation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxborg.evaluation.cross_seed_play import CrossSeedPlaySettings, validate_cross_seed_models
from jaxborg.evaluation.env_diversity import discover_models
from jaxborg.evaluation.play_priors import _parse_seeds
from jaxborg.recipe import load, project_eval


def main(argv=None, *, run_subprocess=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="cotraining_mappo", help="Recipe name or YAML path")
    parser.add_argument("--exp-dir", default=os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp"))
    parser.add_argument(
        "--train-seeds", help="Training seeds in the cycle; default: all available completed seeds, sorted"
    )
    parser.add_argument("--model-tag", default="*", help="Run-directory glob to disambiguate reruns")
    parser.add_argument(
        "--output-dir", help="New result directory; defaults to a timestamped directory under EXP_DIR/eval"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the exact evaluation commands without running them"
    )
    args = parser.parse_args(argv)
    exp_dir = Path(args.exp_dir).expanduser().resolve()
    try:
        recipe = load(args.recipe)
        settings = CrossSeedPlaySettings.from_recipe(recipe)
        if not settings.enabled:
            raise ValueError("The supplied recipe must enable eval.cross_seed_play; use the current training recipe")
        evaluation = project_eval(recipe)
        models, notes = discover_models(recipe, "model", exp_dir, tag=args.model_tag)
        seeds = sorted(models) if args.train_seeds is None else list(_parse_seeds(args.train_seeds))
        missing = sorted(set(seeds) - set(models))
        if missing:
            raise ValueError(f"Completed final models are missing for training seeds {missing}")
        if len(seeds) < 2:
            raise ValueError(f"Need at least two completed training seeds; found {seeds}. {'; '.join(notes)}")
        if len({models[seed].steps for seed in seeds}) != 1:
            raise ValueError("Selected final models have different completed training steps")
        for seed in seeds:
            candidate = copy.deepcopy(models[seed].recipe)
            candidate["eval"] = copy.deepcopy(recipe["eval"])
            # project_eval also checks this bank against each actual train pool.
            if project_eval(candidate)["EVAL_VARIANT"] != evaluation["EVAL_VARIANT"]:
                raise ValueError(f"Training rules for seed {seed} conflict with the evaluation recipe")
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else exp_dir / "eval" / f"{recipe['meta']['name']}_cross_seed_{time.time_ns()}"
    )
    recipe_path = output / "recipe_eval.yaml"
    records = []
    for index, seed in enumerate(seeds):
        opponent = seeds[(index + 1) % len(seeds)]
        blue, red = models[seed].path, models[opponent].path
        validate_cross_seed_models(blue, red)
        command = [
            sys.executable,
            str(ROOT / "scripts/eval/eval_matchup.py"),
            "--recipe",
            str(recipe_path),
            "--policy-backend",
            "jax",
            "--blue-path",
            blue,
            "--red-path",
            red,
            "--name",
            "cross-seed-play",
            "--mlflow-source-team",
            "blue",
            "--seeds",
            ",".join(map(str, settings.seeds)),
            "--episodes-per-seed",
            str(settings.episodes_per_seed),
            "--output",
            str(output / f"blue_{seed}_vs_red_{opponent}.json"),
        ]
        if settings.deterministic:
            command.append("--deterministic")
        records.append({"blue_seed": seed, "red_seed": opponent, "command": command, "status": "pending"})
    print(f"Only cross-seed-play: recipe={recipe['meta']['name']}, training seed cycle={seeds}", flush=True)
    for note in notes:
        print(f"NOTE: {note}", flush=True)
    if args.dry_run:
        for record in records:
            print(shlex.join(record["command"]))
        print("Dry run: recipe snapshot and output files will be created only during execution.")
        return

    output.mkdir(parents=True, exist_ok=False)
    snapshot = {k: v for k, v in recipe.items() if not k.startswith("__")}
    recipe_path.write_text(yaml.safe_dump(snapshot, sort_keys=False))
    manifest = {"recipe": args.recipe, "suite": "cross-seed-play", "notes": notes, "evaluations": records}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = {**os.environ, "JAXBORG_EXP_DIR": str(exp_dir), "PYTHONUNBUFFERED": "1"}
    for record in records:
        print(f"Blue {record['blue_seed']} vs Red {record['red_seed']}", flush=True)
        result = run_subprocess(record["command"], cwd=ROOT, env=env, check=False)
        record["returncode"] = result.returncode
        record["status"] = "succeeded" if result.returncode == 0 else "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Results: {output}", flush=True)
    if any(record["status"] == "failed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
