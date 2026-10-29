"""Evaluate recipe-trained policies against CybORG.

Research evaluation retains native Torch results with explicit protocol and trial
membership. ``--development-no-capture`` selects unrecorded evaluation, including
the JAX-to-CybORG transfer path. See docs/evaluation-capture.md for research use.
"""

# ruff: noqa: E402

import argparse
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

from jaxborg.checkpoint import read_sidecar
from jaxborg.mlflow_setup import attach_eval_metrics

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()
_EVAL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _normalise_eval_name(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    if not _EVAL_NAME_PATTERN.fullmatch(value):
        raise ValueError("evaluation name may contain only letters, numbers, '.', '_' and '-'")
    return value


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
        return (
            subprocess.check_output(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
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
    parser.add_argument(
        "--episodes-per-seed",
        "--episodes",
        dest="episodes_per_seed",
        type=int,
        default=None,
        help="Episodes generated from each seed (--episodes is a deprecated alias)",
    )
    seed_mode = parser.add_mutually_exclusive_group()
    seed_mode.add_argument("--seeds", help="Root seeds, e.g. '42-51' or '42,43,44' (default: 42-51)")
    seed_mode.add_argument("--episode-seeds", nargs="+", type=int, help="Exact ordered episode seeds for research")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Research capture requires 1; development defaults to cpu_count() - 2.",
    )
    parser.add_argument("--output", type=str, default=None, help="Override result jsonl path")
    parser.add_argument(
        "--name",
        default=None,
        help="Optional evaluation name used in the result filename and MLflow keys",
    )
    parser.add_argument(
        "--eval-red",
        type=str,
        default=None,
        help=(
            "Force the eval env's red selector regardless of recipe. "
            "Precedence: CLI --eval-red > recipe eval.red > recipe eval.variant > "
            "train.variant. One of: fsm | cia_c | cia_i | cia_a | random | "
            "resilience | sleep."
        ),
    )
    parser.add_argument("--evidence-root", type=Path, help="Retained evidence store outside the checkout")
    parser.add_argument("--protocol", type=Path, help="Declared version 1 protocol JSON")
    parser.add_argument("--trial", help="Trial ID already declared in the protocol")
    parser.add_argument("--retry-of", help="Previous terminal attempt ID for the same trial")
    parser.add_argument(
        "--policy-seed", type=int, default=None, help="First explicit policy seed (research default: 0)"
    )
    parser.add_argument(
        "--source-revision", help="Expected full commit SHA pinned at submission (default: current HEAD)"
    )
    parser.add_argument("--horizon", type=int, default=None, help="Episode step limit, from 1 to 500")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="Retain a development checkout's changes in evidence"
    )
    parser.add_argument(
        "--development-no-capture", action="store_true", help="Run a development evaluation without common records"
    )
    args = parser.parse_args()
    if args.episode_seeds is not None and (args.development_no_capture or args.episodes_per_seed is not None):
        parser.error("--episode-seeds requires research capture and cannot be combined with --episodes-per-seed")
    args.episodes_per_seed = 10 if args.episodes_per_seed is None else args.episodes_per_seed
    membership = (args.evidence_root, args.protocol, args.trial)
    if args.development_no_capture:
        args.name = args.name or os.environ.get("JAXBORG_EVAL_NAME")
        if any(value is not None for value in membership) or args.retry_of or args.allow_dirty or args.source_revision:
            parser.error("development opt-out cannot be combined with evidence membership or capture options")
        args.workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 4) - 2)
    else:
        if not all(value is not None for value in membership):
            parser.error(
                "research evaluation requires --evidence-root, --protocol and --trial; "
                "use --development-no-capture only for unrecorded development"
            )
        if args.workers not in (None, 1):
            parser.error("research capture requires serial execution (--workers 1)")
        if args.output or args.name:
            parser.error("research evidence uses its immutable store; --output and --name require development opt-out")
        args.workers = 1
    if args.horizon is not None and not 1 <= args.horizon <= 500:
        parser.error("--horizon must be between 1 and 500")

    model_path = Path(args.model).resolve()
    if args.development_no_capture and not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    trained_backend = _detect_trained_backend(model_path)
    seeds = _parse_seeds("42-51" if args.seeds is None else args.seeds)
    eval_name = _normalise_eval_name(args.name)

    if not args.development_no_capture:
        if trained_backend != "cyborg":
            parser.error(
                "research capture currently supports native Torch .pt checkpoints; "
                "JAX transfer evaluation requires --development-no-capture"
            )
        from jaxborg.evaluation.capture import run_evaluation
        from jaxborg.evaluation.episode_seeds import expand_episode_seeds

        try:
            captured = run_evaluation(
                root=args.evidence_root,
                protocol=json.loads(args.protocol.read_text()),
                trial_id=args.trial,
                model=model_path,
                seeds=(
                    args.episode_seeds
                    if args.episode_seeds is not None
                    else expand_episode_seeds(seeds, args.episodes_per_seed)
                ),
                horizon=args.horizon if args.horizon is not None else 500,
                retry_of=args.retry_of,
                deterministic=args.deterministic,
                policy_seed=args.policy_seed if args.policy_seed is not None else 0,
                allow_dirty=args.allow_dirty,
                eval_red=args.eval_red,
                entrypoint=Path(__file__),
                expected_revision=args.source_revision or _git_commit(),
            )
        except (Exception, KeyboardInterrupt) as error:
            captured = getattr(error, "_evaluation_capture", None)
            receipt = {
                "status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            if captured is not None:
                receipt.update(
                    root=str(captured.root), attempt_id=captured.attempt_id, evaluation_id=captured.evaluation_id
                )
            print(json.dumps(receipt))
            raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 1) from None
        print(
            json.dumps(
                {
                    "root": str(captured.root),
                    "attempt_id": captured.attempt_id,
                    "evaluation_id": captured.evaluation_id,
                    "status": "succeeded",
                }
            )
        )
        return

    if trained_backend == "cyborg":
        from jaxborg.evaluation.cyborg_runner import evaluate_on_cyborg, resolve_torch_evaluation_settings

        recipe = read_sidecar(model_path)
        if args.eval_red is not None:
            recipe.setdefault("eval", {})["red"] = args.eval_red
        settings = resolve_torch_evaluation_settings(recipe, deterministic=args.deterministic, horizon=args.horizon)
        variant = settings.variant
        print(f"Loaded recipe sidecar: {recipe.get('meta', {}).get('name', '?')}", flush=True)
        print(
            f"  trained=cyborg arch={recipe['arch']['name']} seeds={seeds} "
            f"eps/seed={args.episodes_per_seed} variant={variant.name} workers={args.workers}",
            flush=True,
        )

        t0 = time.perf_counter()
        rewards, seed_log = evaluate_on_cyborg(
            model_path,
            variant=variant,
            seeds=seeds,
            episodes_per_seed=args.episodes_per_seed,
            deterministic=settings.deterministic,
            workers=args.workers,
            policy_seed=args.policy_seed,
        )
        wall = time.perf_counter() - t0
    else:
        from jaxborg.evaluation.jax_runner import evaluate_jax_on_cyborg
        from jaxborg.recipe import eval_variant

        recipe = read_sidecar(model_path)
        if args.eval_red is not None:
            recipe.setdefault("eval", {})["red"] = args.eval_red
        variant = eval_variant(recipe)
        if args.policy_seed is not None:
            parser.error("--policy-seed is supported only by the native Torch evaluator")
        if args.horizon is not None:
            from dataclasses import replace

            variant = replace(variant, num_steps=args.horizon)

        t0 = time.perf_counter()
        rewards, seed_log, recipe = evaluate_jax_on_cyborg(
            model_path,
            variant=variant,
            seeds=seeds,
            episodes_per_seed=args.episodes_per_seed,
            deterministic=args.deterministic,
            workers=args.workers,
        )
        wall = time.perf_counter() - t0
        print(f"Loaded recipe (sidecar or fallback): {recipe.get('meta', {}).get('name', '?')}", flush=True)
        print(
            f"  trained=jax arch={recipe['arch']['name']} seeds={seeds} "
            f"eps/seed={args.episodes_per_seed} variant={variant.name} workers={args.workers}",
            flush=True,
        )

    m = mean(rewards)
    s = stdev(rewards) if len(rewards) > 1 else 0.0

    eval_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{seeds[0]}"
    train_run_id = recipe.get("run", {}).get("train_run_id")
    row = {
        "eval_id": eval_id,
        "eval_name": eval_name,
        "model": str(model_path),
        "recipe_name": recipe.get("meta", {}).get("name", ""),
        "recipe_path": recipe.get("meta", {}).get("source_path") or recipe.get("__source_path__", ""),
        "trained_backend": trained_backend,
        "eval_env": "cyborg",
        "variant": variant.name,
        "red_agent": variant.red_agent,
        "seeds": seeds,
        "episodes_per_seed": args.episodes_per_seed,
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
        out_path = Path(args.output).expanduser()
    else:
        name = f"_{eval_name}" if eval_name else ""
        out_path = out_dir / f"{row['recipe_name']}_{model_path.stem}{name}_{eval_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, indent=2) + "\n")
    print(f"\nmean: {m:.2f} ± {s:.2f} (n={len(rewards)})", flush=True)
    print(f"wrote: {out_path}", flush=True)

    if train_run_id:
        try:
            prefix = f"eval.after_training.{eval_name}.cyborg" if eval_name else "eval.cyborg"
            attach_eval_metrics(
                train_run_id,
                {
                    f"{prefix}.mean": m,
                    f"{prefix}.std": s,
                    f"{prefix}.episodes": len(rewards),
                },
            )
            print(f"attached eval metrics to MLflow run {train_run_id}", flush=True)
        except Exception as e:
            print(f"MLflow attach warning: {e}", flush=True)


if __name__ == "__main__":
    main()
