"""Recipe-driven evaluation against CybORG (the CC4 contract eval).

Loads a model + sibling `recipe_<tag>.yaml`, instantiates the right policy
from `recipe.arch`, and rolls out N episodes per seed against CybORG.
Writes a standardized result row under `$JAXBORG_EXP_DIR/eval/` and
attaches eval metrics to the train MLflow run when one is named in the
sidecar.

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
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from jaxborg.checkpoint import read_sidecar
from jaxborg.mlflow_setup import attach_eval_metrics

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()


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
    parser.add_argument("--output", type=str, default=None, help="Override result jsonl path")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    trained_backend = _detect_trained_backend(model_path)
    seeds = _parse_seeds(args.seeds)

    if trained_backend == "cyborg":
        import torch

        from jaxborg.evaluation.cyborg_runner import evaluate_on_cyborg, load_torch_policy_from_recipe
        from jaxborg.recipe import project_eval

        recipe = read_sidecar(model_path)
        eval_cfg = project_eval(recipe)
        red_agent = eval_cfg["red_agent"]
        target_weight = eval_cfg["resilience_target_weight"]
        print(f"Loaded recipe sidecar: {recipe.get('meta', {}).get('name', '?')}", flush=True)
        print(f"  trained=cyborg arch={recipe['arch']['name']} seeds={seeds} eps/seed={args.episodes} red_agent={red_agent}", flush=True)

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        agent = load_torch_policy_from_recipe(recipe, state_dict)

        t0 = time.perf_counter()
        rewards, seed_log = evaluate_on_cyborg(
            agent,
            seeds=seeds,
            episodes_per_seed=args.episodes,
            deterministic=args.deterministic,
            red_agent=red_agent,
            target_weight=target_weight,
        )
        wall = time.perf_counter() - t0
    else:
        from jaxborg.evaluation.jax_runner import evaluate_jax_on_cyborg

        t0 = time.perf_counter()
        rewards, seed_log, recipe = evaluate_jax_on_cyborg(
            model_path,
            seeds=seeds,
            episodes_per_seed=args.episodes,
            deterministic=args.deterministic,
        )
        wall = time.perf_counter() - t0
        print(f"Loaded recipe (sidecar or fallback): {recipe.get('meta', {}).get('name', '?')}", flush=True)
        print(f"  trained=jax arch={recipe['arch']['name']} seeds={seeds} eps/seed={args.episodes}", flush=True)

    m = mean(rewards)
    s = stdev(rewards) if len(rewards) > 1 else 0.0

    eval_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{seeds[0]}"
    train_run_id = recipe.get("run", {}).get("train_run_id")
    row = {
        "eval_id": eval_id,
        "model": str(model_path),
        "recipe_name": recipe.get("meta", {}).get("name", ""),
        "recipe_path": recipe.get("meta", {}).get("source_path") or recipe.get("__source_path__", ""),
        "trained_backend": trained_backend,
        "eval_env": "cyborg",
        "red_agent": red_agent if trained_backend == "cyborg" else "finite_state",
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

    out_dir = EXP_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = out_dir / f"{row['recipe_name']}_{model_path.stem}_{eval_id}.jsonl"
    out_path.write_text(json.dumps(row, indent=2) + "\n")
    print(f"\nmean: {m:.2f} ± {s:.2f} (n={len(rewards)})", flush=True)
    print(f"wrote: {out_path}", flush=True)

    if train_run_id:
        try:
            attach_eval_metrics(
                train_run_id,
                {
                    "eval.cyborg.mean": m,
                    "eval.cyborg.std": s,
                    "eval.cyborg.episodes": len(rewards),
                },
            )
            print(f"attached eval metrics to MLflow run {train_run_id}", flush=True)
        except Exception as e:
            print(f"MLflow attach warning: {e}", flush=True)


if __name__ == "__main__":
    main()
