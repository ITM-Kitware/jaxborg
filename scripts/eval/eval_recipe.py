"""Recipe-driven evaluation against CybORG (the CC4 contract eval).

Loads a model + sibling `recipe_<tag>.yaml`, instantiates the right policy
from `recipe.arch`, and rolls out N episodes per seed against CybORG.
Each independent evaluation owns an MLflow run and a verified result artifact.
Use --reuse explicitly to reuse a matching complete evaluation; --output exports
that artifact. Corrected evaluations link back via --supersedes-eval-run-id.

Single entrypoint for both trained backends:
- `.pt`  → torch state_dict from `algorithms/ippo_cyborg.py` (loaded via
  `jaxborg.evaluation.cyborg_runner`)
- `.safetensors` → Flax params from `algorithms/ippo_jax.py` (loaded via
  `jaxborg.evaluation.jax_runner`, which translates JAX action space to CybORG
  per step — the cross-backend transfer eval)

Usage:
    uv run python scripts/eval/eval_recipe.py \
        --model jaxborg-exp/ippo_cyborg/<tag>/model_<tag>.pt \
        --episodes 10 --seeds 42-51

    uv run python scripts/eval/eval_recipe.py \
        --model jaxborg-exp/ippo_jax/<tag>/model_<tag>.safetensors \
        --episodes 10 --seeds 42-51
"""

# ruff: noqa: E402

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlflow

from jaxborg.checkpoint import read_sidecar
from jaxborg.tracking import (
    Run,
    assigned_devices,
    evaluation_fingerprint,
    export_artifact,
    find_reusable,
    input_artifact,
    resolve_artifact,
    tracked_entrypoint,
)


def _parse_seeds(spec: str) -> list[int]:
    """'42,43,44' or '42-51' or '42-44,50,52' -> sorted unique list."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for s in range(int(a), int(b) + 1):
                out.add(s)
        else:
            out.add(int(part))
    return sorted(out)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _detect_trained_backend(model_path: Path) -> str:
    """Determine which trainer produced this model from the file suffix."""
    if model_path.suffix == ".pt":
        return "cyborg"
    if model_path.suffix in (".safetensors", ".flax", ".orbax"):
        return "jax"
    raise ValueError(f"Cannot detect trained backend from suffix: {model_path}")


@tracked_entrypoint
def main():
    parser = argparse.ArgumentParser(description="Evaluate a recipe-trained policy on CybORG")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model_<tag>.pt (CybORG-trained) or .safetensors (JAX-trained)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per seed")
    parser.add_argument("--seeds", type=str, default="42-51", help="e.g. '42-51' or '42,43,44'")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Parallel rollout workers (1 = single process). Default: cpu_count() - 2.",
    )
    parser.add_argument("--output", type=str, default=None, help="Override result jsonl path")
    parser.add_argument("--reuse", action="store_true", help="Reuse a fully validated completed evaluation")
    parser.add_argument("--supersedes-eval-run-id", default=None, help="Original evaluation corrected by this new run")
    parser.add_argument("--bug-reference", default=None)
    args = parser.parse_args()

    model_path = resolve_artifact(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    trained_backend = _detect_trained_backend(model_path)
    if trained_backend == "jax":
        assigned_devices()
    seeds = _parse_seeds(args.seeds)
    if not seeds or args.episodes < 1:
        parser.error("At least one seed and one episode per seed are required")
    from jaxborg.recipe import eval_variant

    recipe = read_sidecar(model_path)
    variant = eval_variant(recipe)
    inputs = [input_artifact(args.model, role="Blue policy")]
    effective = {
        "variant": variant,
        "seeds": seeds,
        "episodes_per_seed": args.episodes,
        "deterministic": args.deterministic,
        "workers": args.workers,
        "eval_env": "cyborg",
        "episode_seed_spec": "base_seed + replica for each seed, replica in range(episodes_per_seed)",
        "episode_seeds": [seed + ep for seed in seeds for ep in range(args.episodes)],
        "policy_rng_contract": (
            "CybORG torch sampling uses process RNG; no per-episode torch seed is assigned by this runner"
            if trained_backend == "cyborg"
            else "JAX sampling key: PRNGKey(seeds[0] * 100003 + flattened episode index)"
        ),
    }
    fingerprint = evaluation_fingerprint(inputs, recipe, effective)
    if args.reuse and args.supersedes_eval_run_id is None:
        reused = find_reusable(fingerprint, ["evaluations/result.json"])
        if reused:
            reference = f"runs:/{reused}/evaluations/result.json"
            if args.output:
                export_artifact(reference, args.output)
            print(f"Reused evaluation run: {reused}\nCanonical: {reference}")
            return reused
    run = Run(
        recipe,
        backend="cyborg",
        kind="evaluation",
        config=effective,
        inputs=inputs,
        fingerprint=fingerprint,
        supersedes=args.supersedes_eval_run_id,
        bug_reference=args.bug_reference,
    )

    model_path = run.input_path(0)

    if trained_backend == "cyborg":
        from jaxborg.evaluation.cyborg_runner import evaluate_on_cyborg
        from jaxborg.recipe import eval_variant

        recipe = read_sidecar(model_path)
        variant = eval_variant(recipe)
        print(f"Loaded recipe sidecar: {recipe.get('meta', {}).get('name', '?')}", flush=True)
        print(
            f"  trained=cyborg arch={recipe['arch']['name']} seeds={seeds} "
            f"eps/seed={args.episodes} variant={variant.name} workers={args.workers}",
            flush=True,
        )

        if args.workers == 1:
            import torch

            run.write_json(
                "environment/torch_cpu_rng.json",
                {
                    "state_before_model_load": torch.get_rng_state().tolist(),
                    "contract": "restore before evaluate_on_cyborg for CPU replay",
                },
            )
        else:
            run.update(policy_rng_replay="unknown per-worker torch RNG; existing sampling behavior preserved")
        t0 = time.perf_counter()
        rewards, seed_log = evaluate_on_cyborg(
            model_path,
            variant=variant,
            seeds=seeds,
            episodes_per_seed=args.episodes,
            deterministic=args.deterministic,
            workers=args.workers,
        )
        wall = time.perf_counter() - t0
    else:
        from jaxborg.evaluation.jax_runner import evaluate_jax_on_cyborg
        from jaxborg.recipe import eval_variant

        recipe = read_sidecar(model_path)
        variant = eval_variant(recipe)

        t0 = time.perf_counter()
        rewards, seed_log, recipe = evaluate_jax_on_cyborg(
            model_path,
            variant=variant,
            seeds=seeds,
            episodes_per_seed=args.episodes,
            deterministic=args.deterministic,
            workers=args.workers,
        )
        wall = time.perf_counter() - t0
        print(f"Loaded recipe (sidecar or fallback): {recipe.get('meta', {}).get('name', '?')}", flush=True)
        print(
            f"  trained=jax arch={recipe['arch']['name']} seeds={seeds} "
            f"eps/seed={args.episodes} variant={variant.name} workers={args.workers}",
            flush=True,
        )

    m = mean(rewards)
    s = stdev(rewards) if len(rewards) > 1 else 0.0

    eval_id = run.run_id
    train_run_id = recipe.get("run", {}).get("train_run_id")
    row = {
        "eval_id": eval_id,
        "evaluator_source_sha": run.manifest["source"]["git_commit"],
        "model": str(model_path),
        "recipe_name": recipe.get("meta", {}).get("name", ""),
        "recipe_path": recipe.get("meta", {}).get("source_path") or recipe.get("__source_path__", ""),
        "trained_backend": trained_backend,
        "eval_env": "cyborg",
        "variant": variant.name,
        "red_agent": variant.red_agent,
        "seeds": seeds,
        "episodes_per_seed": args.episodes,
        "stochastic": not args.deterministic,
        "mean_reward": m,
        "std_reward": s,
        "n_episodes": len(rewards),
        "wall_time_s": wall,
        "git_commit": _git_commit(),
        "train_run_id": train_run_id,
        "per_episode": rewards,
        "per_episode_seeds": seed_log,
    }

    reference = run.write_json("evaluations/result.json", row)
    mlflow.log_metrics({"eval.cyborg.mean": m, "eval.cyborg.std": s, "eval.cyborg.episodes": len(rewards)})
    if args.output:
        run.export("evaluations/result.json", args.output)
    print(f"\nmean: {m:.2f} ± {s:.2f} (n={len(rewards)})", flush=True)
    print(f"Executed evaluation run: {run.run_id}\nCanonical: {reference}", flush=True)
    return run.run_id


if __name__ == "__main__":
    main()
