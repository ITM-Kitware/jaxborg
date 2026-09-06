"""Evaluate learned Blue and Red policies together in the JAX CC4 simulator.

The existing ``eval_recipe.py --model`` command remains the contract
evaluation for one learned Blue policy against scripted Red inside CybORG.
This entry point is specifically for learned-vs-learned matchups.

``--episodes-per-seed`` is the canonical rollout-count option. The older
``--episodes`` spelling remains as a command-line compatibility alias.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from jaxborg.evaluation.matchup_runner import evaluate_matchup
from jaxborg.mlflow_setup import attach_eval_metrics
from jaxborg.recipe import eval_variant, load, project_eval, resolve_eval_policies
from jaxborg.topology_banks import validate_eval_topology_override

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()
_EVAL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _normalise_eval_name(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    if not _EVAL_NAME_PATTERN.fullmatch(value):
        raise ValueError("evaluation name may contain only letters, numbers, '.', '_' and '-'")
    return value


def _parse_seeds(spec: str) -> list[int]:
    values: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(token))
    if not values:
        raise ValueError("at least one evaluation seed is required")
    return sorted(values)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _policy_override(path: str | None, experiment: str | None):
    if path:
        return {"path": path}
    if experiment:
        return {"experiment": experiment}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned Blue vs learned Red in JAX CC4")
    parser.add_argument("--recipe", required=True, help="Recipe name or YAML path; its eval.variant is authoritative")
    parser.add_argument("--policy-backend", choices=("jax", "cyborg"), default=None)
    parser.add_argument("--blue-path", "--blue-model", dest="blue_path", default=None)
    parser.add_argument("--blue-experiment", default=None)
    parser.add_argument("--red-path", "--red-model", dest="red_path", default=None)
    parser.add_argument("--red-experiment", default=None)
    parser.add_argument(
        "--episodes-per-seed",
        "--episodes",
        dest="episodes_per_seed",
        type=int,
        default=10,
        help="Episodes generated from each seed (--episodes is a deprecated alias)",
    )
    parser.add_argument("--seeds", default="42-51")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--topology-path",
        action="append",
        default=None,
        help="Topology snapshot to sample during evaluation; repeat for a held-out bank",
    )
    parser.add_argument(
        "--topology-sampling",
        choices=("exhaustive", "random"),
        default=None,
        help="Bank assignment (default: recipe value or exhaustive)",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--name",
        default=os.environ.get("JAXBORG_EVAL_NAME"),
        help="Optional evaluation name used in the result filename and MLflow keys",
    )
    args = parser.parse_args()

    recipe = copy.deepcopy(load(args.recipe))
    eval_cfg = recipe.setdefault("eval", {})
    if args.policy_backend:
        eval_cfg["policy_backend"] = args.policy_backend
    policies_cfg = eval_cfg.setdefault("policies", {})
    blue_override = _policy_override(args.blue_path, args.blue_experiment)
    red_override = _policy_override(args.red_path, args.red_experiment)
    if blue_override:
        policies_cfg["blue"] = blue_override
    if red_override:
        policies_cfg["red"] = red_override

    backend = eval_cfg.get("policy_backend")
    if backend not in ("jax", "cyborg"):
        raise ValueError("eval.policy_backend (or --policy-backend) must be 'jax' or 'cyborg'")
    missing = {"blue", "red"} - set(policies_cfg)
    if missing:
        raise ValueError(f"learned matchup requires both eval policies; missing {sorted(missing)}")
    model_paths = resolve_eval_policies(recipe, exp_dir=EXP_DIR)
    variant = eval_variant(recipe)
    seeds = _parse_seeds(args.seeds)
    eval_name = _normalise_eval_name(args.name)
    if args.topology_path:
        topology_paths = [Path(path).expanduser().resolve() for path in args.topology_path]
        validate_eval_topology_override(recipe, topology_paths, repo_root=_REPO_ROOT)
    else:
        topology_paths = list(project_eval(recipe, materialize_topologies=True)["TOPOLOGY_BANK"]) or None
    topology_sampling = args.topology_sampling or eval_cfg.get("topology_sampling", "exhaustive")

    print(
        f"JAX matchup: backend={backend} variant={variant.name} seeds={seeds} episodes/seed={args.episodes_per_seed}",
        flush=True,
    )
    print(f"  Blue: {model_paths['blue']}", flush=True)
    print(f"  Red:  {model_paths['red']}", flush=True)
    if topology_paths:
        print(
            f"  Topology bank: {len(topology_paths)} snapshots ({topology_sampling})",
            flush=True,
        )
    t0 = time.perf_counter()
    result = evaluate_matchup(
        model_paths["blue"],
        model_paths["red"],
        backend=backend,
        variant=variant,
        seeds=seeds,
        episodes_per_seed=args.episodes_per_seed,
        deterministic=args.deterministic,
        topology_path=topology_paths,
        topology_sampling=topology_sampling,
    )
    wall = time.perf_counter() - t0
    blue_mean = mean(result.blue_returns)
    blue_std = stdev(result.blue_returns) if len(result.blue_returns) > 1 else 0.0
    red_mean = -blue_mean
    red_std = blue_std
    eval_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{seeds[0]}"
    row = {
        "eval_id": eval_id,
        "eval_name": eval_name,
        "recipe_name": recipe.get("meta", {}).get("name", ""),
        "recipe_path": recipe.get("__source_path__", ""),
        "eval_env": "jax_joint",
        "variant": variant.name,
        "policy_backend": backend,
        "policies": result.policies,
        "seeds": seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "episodes_per_topology": (
            len(seeds) * args.episodes_per_seed if result.topology_sampling == "exhaustive" else None
        ),
        "total_episodes": len(result.blue_returns),
        "stochastic": not args.deterministic,
        "blue_mean_return": blue_mean,
        "blue_std_return": blue_std,
        "red_mean_return": red_mean,
        "red_std_return": red_std,
        # Stable aliases keep score-oriented consumers simple.
        "mean_reward": blue_mean,
        "std_reward": blue_std,
        "n_episodes": len(result.blue_returns),
        "wall_time_s": wall,
        "git_commit": _git_commit(),
        "per_episode_blue_returns": result.blue_returns,
        "per_episode_red_returns": result.red_returns,
        "per_episode_seeds": result.episode_seeds,
        "topology_paths": result.topology_paths,
        "topology_sampling": result.topology_sampling,
        "per_episode_topology_paths": result.episode_topology_paths,
    }

    name = f"_{eval_name}" if eval_name else ""
    output = (
        Path(args.output).expanduser()
        if args.output
        else EXP_DIR / "eval" / f"{row['recipe_name']}_matchup{name}_{eval_id}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2) + "\n")
    print(f"\nBlue: {blue_mean:.2f} ± {blue_std:.2f}", flush=True)
    print(f"Red:  {red_mean:.2f} ± {red_std:.2f}", flush=True)
    print(f"wrote: {output}", flush=True)

    # Attach team-qualified results to every distinct source run when IDs are
    # available. Evaluation still succeeds if the tracking server is absent.
    run_ids = {source.get("train_run_id") for source in result.policies.values() if source.get("train_run_id")}
    for run_id in run_ids:
        try:
            prefix = f"eval.after_training.{eval_name}.jax_matchup" if eval_name else "eval.jax_matchup"
            attach_eval_metrics(
                run_id,
                {
                    f"{prefix}.blue_mean": blue_mean,
                    f"{prefix}.red_mean": red_mean,
                    f"{prefix}.episodes": len(result.blue_returns),
                },
            )
        except Exception as exc:
            print(f"MLflow attach warning for {run_id}: {exc}", flush=True)


if __name__ == "__main__":
    main()
