"""Evaluate H-MARL or a trained Blue bundle against the four H-MARL red adversaries.

Uses the same exhaustive topology cases, recurrent/stateful Blue rollout, CIA
metrics and JSONL/MLflow reporting as the existing JAX scripted-Red evaluator.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from jaxborg.scenarios.cc4.hmarl_reds import HMARL_REDS


def main(argv: Sequence[str] | None = None) -> None:
    from dataclasses import replace

    import jax

    from jaxborg.evaluation.jax_scripted_red import (
        attach_results_to_mlflow,
        evaluate_jax_scripted_reds,
        run_jax_scripted_red_episode,
        write_results,
    )
    from jaxborg.recipe import REPO_ROOT, load, project_eval
    from jaxborg.topology_banks import validate_eval_topology_override

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Trained Blue bundle, or override H-MARL's actor cache")
    parser.add_argument("--recipe", default=os.environ.get("JAXBORG_RECIPE_PATH"), help="Recipe name or YAML path")
    parser.add_argument("--reds", nargs="+", choices=HMARL_REDS, default=list(HMARL_REDS))
    parser.add_argument("--seeds", help="Seed list/range; defaults to eval.seeds or 1000-1009")
    parser.add_argument("--episodes-per-seed", type=int, help="Defaults to eval.episodes_per_seed or 6")
    parser.add_argument("--episode-length", type=int, help="Override duration for a short smoke run")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--topology-path", type=Path, action="append", help="Override held-out snapshot; repeat for a bank"
    )
    parser.add_argument("--upstream-dir", type=Path, help="Local upstream checkout for importing missing H-MARL actors")
    parser.add_argument("--name", default=os.environ.get("JAXBORG_EVAL_NAME", "hmarl-reds"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    if not args.recipe and args.model is None:
        parser.error("provide --recipe for H-MARL, or --model for a trained Blue bundle")
    if args.episode_length is not None and args.episode_length < 1:
        parser.error("--episode-length must be positive")
    if args.episodes_per_seed is not None and args.episodes_per_seed < 1:
        parser.error("--episodes-per-seed must be positive")

    if args.recipe:
        recipe = load(args.recipe)
    else:
        from jaxborg.checkpoint import read_sidecar

        recipe = read_sidecar(args.model)
    ev = recipe.get("eval", {})
    pretrained = recipe.get("kind") == "pretrained_eval"
    policy_loader = None
    model = args.model
    if pretrained:
        from jaxborg.pretrained.hmarl import load_policy
        from jaxborg.pretrained.hmarl_import import import_checkpoint

        config = recipe["pretrained"]
        model = (model or Path(config["checkpoint"])).expanduser()
        if not model.is_absolute():
            model = REPO_ROOT / model
        if not model.is_file() and args.model is None:
            import_checkpoint(model, upstream_dir=args.upstream_dir)
        policy = load_policy(model, variant=config["variant"])

        def policy_loader(*args, **kwargs):
            return policy
    elif model is None:
        parser.error("--model is required for a trained Blue recipe")
    elif args.upstream_dir is not None:
        parser.error("--upstream-dir is only supported for H-MARL recipes")

    if args.topology_path:
        topologies = tuple(path.expanduser().resolve() for path in args.topology_path)
        validate_eval_topology_override(recipe, topologies, repo_root=REPO_ROOT)
        projection = project_eval(recipe)
    else:
        projection = project_eval(recipe, materialize_topologies=True)
        topologies = projection["TOPOLOGY_BANK"]
    variant = projection["EVAL_VARIANT"]
    if args.episode_length is not None:
        variant = replace(variant, num_steps=args.episode_length)
    # Preserve H-MARL's CPU smoke-run convention: a singleton vmap expands
    # simulator branches without providing parallelism.
    scalar_cpu = jax.default_backend() == "cpu" and "JAXBORG_EVAL_BATCH_SIZE" not in os.environ
    rows = evaluate_jax_scripted_reds(
        model,
        base_variant=variant,
        topology_paths=topologies,
        reds=args.reds,
        seeds=args.seeds if args.seeds is not None else ev.get("seeds", "1000-1009"),
        episodes_per_seed=(
            args.episodes_per_seed if args.episodes_per_seed is not None else ev.get("episodes_per_seed", 6)
        ),
        deterministic=args.deterministic if args.deterministic is not None else ev.get("deterministic", False),
        progress=args.progress,
        eval_name=args.name,
        recipe=recipe,
        policy_loader=policy_loader,
        episode_runner=run_jax_scripted_red_episode if scalar_cpu else None,
    )
    for row in rows:
        row["suite"] = "hmarl_reds"
        row["episode_length"] = variant.num_steps
        if pretrained:
            row["trained_backend"] = "rllib_torch"
            row["pretrained"] = recipe["pretrained"]
    output = write_results(rows, args.output)
    print(f"Wrote H-MARL Red sweep: {output}", flush=True)
    if not args.no_mlflow:
        try:
            attach_results_to_mlflow(rows)
        except Exception as exc:
            print(f"MLflow attach warning: {exc}", flush=True)


if __name__ == "__main__":
    main()
