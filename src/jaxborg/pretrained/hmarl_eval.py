"""Recipe-driven evaluation of the released H-MARL Expert and Meta actors."""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from pathlib import Path
from statistics import mean, stdev

import jax

from jaxborg.evaluation.jax_scripted_red import (
    DEFAULT_SCRIPTED_REDS,
    _git_commit,
    _normalise_reds,
    _parse_seeds,
    evaluate_jax_scripted_reds,
    run_jax_scripted_red_episode,
    write_results,
)
from jaxborg.evaluation.matchup_runner import MatchupEvaluationContext, _eval_batch_size, evaluate_matchup
from jaxborg.pretrained.hmarl import load_policy
from jaxborg.pretrained.hmarl_import import import_checkpoint

ARCHITECTURE = {"hidden_layers": 2, "hidden_dim": 256, "activation": "tanh", "observation_filter": "NoFilter"}


def validate_recipe(recipe, *, source):
    if any(key in recipe for key in ("train", "core", "algorithm")):
        raise ValueError(f"{source}: pretrained_eval recipes must not contain training configuration")
    if not isinstance(recipe.get("meta"), dict) or not recipe["meta"].get("name"):
        raise ValueError(f"{source}: meta.name is required")
    config = recipe.get("pretrained", {})
    if config.get("family") != "hmarl" or config.get("variant") not in ("expert", "meta"):
        raise ValueError(f"{source}: pretrained family must be hmarl with variant expert or meta")
    if config.get("architecture") != ARCHITECTURE:
        raise ValueError(f"{source}: pretrained.architecture must match the released 256x256 tanh NoFilter actors")
    if not isinstance(config.get("checkpoint"), str) or not config["checkpoint"]:
        raise ValueError(f"{source}: pretrained.checkpoint is required")
    if recipe.get("cage4_enhanced_obs") is not True:
        raise ValueError(f"{source}: H-MARL needs cage4_enhanced_obs: true for observed file evidence")
    ev = recipe.get("eval", {})
    _parse_seeds(ev.get("seeds", ""))
    _normalise_reds(ev.get("reds", []))
    for key in ("episodes_per_seed", "episode_length"):
        value = ev.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{source}: eval.{key} must be a positive integer")
    if not isinstance(ev.get("deterministic"), bool):
        raise ValueError(f"{source}: eval.deterministic must be a boolean")
    if ev.get("topology_sampling") != "exhaustive":
        raise ValueError(f"{source}: H-MARL CIA evaluation requires exhaustive topology sampling")
    if not ev.get("cia", {}).get("enabled"):
        raise ValueError(f"{source}: eval.cia.enabled must be true")
    from jaxborg.recipe import project_eval

    project_eval(recipe)


def final_red_metadata(path):
    """Require the final cotraining bundle, never a historical checkpoint."""
    from jaxborg.checkpoint import read_sidecar

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix != ".safetensors" or not path.name.startswith("model_"):
        raise ValueError(f"Use the final model_<tag>.safetensors bundle: {path}")
    recipe = read_sidecar(path)
    run = recipe.get("run", {})
    if run.get("model") != path.name or set(run.get("trainable_teams", [])) != {"blue", "red"}:
        raise ValueError(f"Expected a final cotrained Blue/Red bundle with matching sidecar: {path}")
    return {"recipe_name": recipe["meta"]["name"], "seed": run["seed"], "total_steps": run["total_steps"]}


def evaluate_learned_red(path, red_path, *, recipe, projection, topology_paths, context, progress, metadata):
    ev = recipe["eval"]
    started = time.perf_counter()
    result = evaluate_matchup(
        path,
        red_path,
        backend="jax",
        variant=projection["EVAL_VARIANT"],
        seeds=_parse_seeds(ev["seeds"]),
        episodes_per_seed=ev["episodes_per_seed"],
        deterministic=ev["deterministic"],
        progress=progress,
        topology_path=topology_paths,
        topology_sampling=ev["topology_sampling"],
        cia=projection["CIA"],
        context=context,
    )
    average = mean(result.blue_returns)
    deviation = stdev(result.blue_returns) if len(result.blue_returns) > 1 else 0.0
    return {
        "eval_id": f"{time.time_ns()}_learned_red",
        "eval_name": recipe["meta"]["name"],
        "suite": "learned_red",
        "recipe_name": recipe["meta"]["name"],
        "recipe_path": recipe.get("__source_path__", ""),
        "model": str(path),
        "red_model": str(red_path),
        "red_checkpoint": "final",
        "red_training": metadata,
        "eval_red": "learned_red",
        "policy_team": "blue",
        "policy_backend": "jax",
        "eval_env": "jax_joint",
        "policies": result.policies,
        "variant": projection["EVAL_VARIANT"].name,
        "seeds": _parse_seeds(ev["seeds"]),
        "episodes_per_seed": ev["episodes_per_seed"],
        "stochastic": not ev["deterministic"],
        "mean_reward": average,
        "std_reward": deviation,
        "n_episodes": len(result.blue_returns),
        "blue_mean_return": average,
        "blue_std_return": deviation,
        "red_mean_return": -average,
        "red_std_return": deviation,
        "per_episode": result.blue_returns,
        "per_episode_blue_returns": result.blue_returns,
        "per_episode_red_returns": result.red_returns,
        "per_episode_seeds": result.episode_seeds,
        "topology_paths": result.topology_paths,
        "topology_sampling": result.topology_sampling,
        "per_episode_topology_paths": result.episode_topology_paths,
        "cia_metric": result.cia_metric,
        "cia_config": result.cia_config,
        "cia_summary": result.cia_summary,
        "per_episode_cia": result.per_episode_cia,
        "episode_role_map_ids": result.episode_role_map_ids,
        "per_episode_topology_fingerprints": result.episode_topology_fingerprints,
        "topology_role_maps": result.topology_role_maps,
        "cage4_enhanced_obs": True,
        "blue_observation_version": projection["EVAL_VARIANT"].blue_observation_version,
        "wall_time_s": time.perf_counter() - started,
        "git_commit": _git_commit(),
    }


def main(argv=None):
    from jaxborg.recipe import REPO_ROOT, load, project_eval
    from jaxborg.topology_banks import validate_eval_topology_override

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="hmarl_expert or hmarl_meta (or a YAML path)")
    parser.add_argument("--model", type=Path, help="Override the imported actor bundle")
    parser.add_argument(
        "--upstream-dir", type=Path, help="Import from a local upstream checkout if weights are missing"
    )
    parser.add_argument("--prepare-only", action="store_true", help="Fetch/convert/validate weights without evaluation")
    parser.add_argument("--reds", nargs="+", choices=DEFAULT_SCRIPTED_REDS)
    parser.add_argument(
        "--red-model",
        action="append",
        type=Path,
        default=[],
        help="Final cotrained Red bundle; repeat to evaluate multiple runs after FSM/CIA",
    )
    parser.add_argument("--learned-only", action="store_true", help="Skip scripted Reds; requires --red-model")
    parser.add_argument("--seeds")
    parser.add_argument("--episodes-per-seed", type=int)
    parser.add_argument("--episode-length", type=int, help="Override duration for a short smoke run")
    parser.add_argument("--topology-path", action="append", type=Path)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.learned_only and not args.red_model:
        parser.error("--learned-only requires --red-model")
    if args.prepare_only and (args.red_model or args.learned_only):
        parser.error("--prepare-only cannot be combined with learned-Red evaluation")
    recipe = load(args.recipe)
    if recipe.get("kind") != "pretrained_eval":
        parser.error("Use an evaluation-only pretrained recipe")
    config, ev = recipe["pretrained"], recipe["eval"]
    for name in ("reds", "seeds", "episodes_per_seed", "episode_length", "deterministic"):
        if getattr(args, name) is not None:
            ev[name] = getattr(args, name)
    validate_recipe(recipe, source=args.recipe)
    path = args.model or Path(config["checkpoint"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        if args.model:
            raise FileNotFoundError(f"Explicit model not found: {path}")
        import_checkpoint(path, upstream_dir=args.upstream_dir)
    loader = partial(load_policy, variant=config["variant"])
    policy = loader(path)
    if args.prepare_only:
        print(f"Ready: H-MARL {config['variant']} at {path}", flush=True)
        return
    red_paths = [p.expanduser().resolve() for p in args.red_model]
    if len(red_paths) != len(set(red_paths)):
        parser.error("--red-model paths must be distinct")
    context = MatchupEvaluationContext()
    context.policies[(path.resolve(), "blue", "jax")] = policy
    red_metadata = {}
    for red_path in red_paths:
        red_metadata[red_path] = final_red_metadata(red_path)
        context.load_policy(red_path, team="red", backend="jax")
    if args.topology_path:
        topology_paths = tuple(path.expanduser().resolve() for path in args.topology_path)
        validate_eval_topology_override(recipe, topology_paths, repo_root=REPO_ROOT)
        projection = project_eval(recipe)
    else:
        projection = project_eval(recipe, materialize_topologies=True)
        topology_paths = projection["TOPOLOGY_BANK"]
    print(
        f"H-MARL {config['variant']}: {len(topology_paths)} topologies, seeds={ev['seeds']}, "
        f"{ev['episodes_per_seed']} episodes/seed, {ev['episode_length']} steps, reds={ev['reds']}",
        flush=True,
    )
    batch_size = (
        1 if jax.default_backend() == "cpu" and "JAXBORG_EVAL_BATCH_SIZE" not in os.environ else _eval_batch_size()
    )
    rows = (
        []
        if args.learned_only
        else evaluate_jax_scripted_reds(
            path,
            base_variant=projection["EVAL_VARIANT"],
            topology_paths=topology_paths,
            reds=ev["reds"],
            seeds=ev["seeds"],
            episodes_per_seed=ev["episodes_per_seed"],
            deterministic=ev["deterministic"],
            progress=args.progress,
            recipe=recipe,
            eval_name=recipe["meta"]["name"],
            policy_loader=lambda *a, **kw: policy,
            # A singleton vmap expands simulator conditionals into selects for no
            # parallelism gain. Use the same scalar scan for CPU smoke runs.
            episode_runner=run_jax_scripted_red_episode if batch_size == 1 else None,
        )
    )

    def save_rows():
        for row in rows:
            row["trained_backend"] = "rllib_torch"
            row["episode_length"] = ev["episode_length"]
            row["evaluation_batch_size"] = batch_size
            row["pretrained"] = config
        return write_results(rows, output)

    # Save each completed opponent so a later failure leaves useful results.
    output = args.output
    if rows:
        output = save_rows()
    previous_batch_size = os.environ.get("JAXBORG_EVAL_BATCH_SIZE")
    os.environ["JAXBORG_EVAL_BATCH_SIZE"] = str(batch_size)
    try:
        for red_path in red_paths:
            print(f"H-MARL {config['variant']} vs final Red: {red_path}", flush=True)
            rows.append(
                evaluate_learned_red(
                    path,
                    red_path,
                    recipe=recipe,
                    projection=projection,
                    topology_paths=topology_paths,
                    context=context,
                    progress=args.progress,
                    metadata=red_metadata[red_path],
                )
            )
            output = save_rows()
    finally:
        if previous_batch_size is None:
            os.environ.pop("JAXBORG_EVAL_BATCH_SIZE", None)
        else:
            os.environ["JAXBORG_EVAL_BATCH_SIZE"] = previous_batch_size
    print(f"Wrote: {output}", flush=True)


if __name__ == "__main__":
    main()
