"""Evaluate a trained Blue policy against the scripted CC4 Red suite.

The portable model bundles written by co-training contain both Blue and Red
policies.  This evaluator deliberately selects only the Blue entry and plays
it against CybORG's scripted FSM/CIA agents.  It is therefore separate from
the learned Blue-vs-learned Red checkpoint evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

DEFAULT_SCRIPTED_REDS = ("fsm", "cia_c", "cia_i", "cia_a")
_SUPPORTED_SCRIPTED_REDS = frozenset(DEFAULT_SCRIPTED_REDS)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVAL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _normalise_eval_name(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or not _EVAL_NAME_PATTERN.fullmatch(value):
        raise ValueError("evaluation name may contain only letters, numbers, '.', '_' and '-'")
    return value


def parse_seeds(value: str | int | Sequence[int]) -> tuple[int, ...]:
    """Parse ``1000-1099``, ``1000,1002``, an int, or an integer sequence."""

    if isinstance(value, bool):
        raise ValueError("eval.scripted_red.seeds must contain non-negative integers")
    if isinstance(value, int):
        seeds = {value}
    elif isinstance(value, str):
        seeds: set[int] = set()
        for raw_part in value.split(","):
            part = raw_part.strip()
            if not part:
                continue
            if "-" in part:
                bounds = part.split("-", 1)
                try:
                    start, end = (int(bound) for bound in bounds)
                except ValueError as exc:
                    raise ValueError(f"invalid evaluation seed range: {part!r}") from exc
                if end < start:
                    raise ValueError(f"evaluation seed range must be ascending: {part!r}")
                seeds.update(range(start, end + 1))
            else:
                try:
                    seeds.add(int(part))
                except ValueError as exc:
                    raise ValueError(f"invalid evaluation seed: {part!r}") from exc
    elif isinstance(value, Sequence):
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
            raise ValueError("eval.scripted_red.seeds must contain non-negative integers")
        seeds = set(value)
    else:
        raise ValueError("eval.scripted_red.seeds must be a seed, range string, or integer list")

    if not seeds:
        raise ValueError("eval.scripted_red.seeds must contain at least one seed")
    if min(seeds) < 0:
        raise ValueError("eval.scripted_red.seeds must contain non-negative integers")
    return tuple(sorted(seeds))


def _normalise_reds(value: str | Sequence[str]) -> tuple[str, ...]:
    reds = (value,) if isinstance(value, str) else tuple(value)
    if not reds:
        raise ValueError("eval.scripted_red.reds must contain at least one Red agent")
    if any(not isinstance(red, str) for red in reds):
        raise ValueError("eval.scripted_red.reds must be a string list")
    unknown = set(reds) - _SUPPORTED_SCRIPTED_REDS
    if unknown:
        raise ValueError(
            "eval.scripted_red.reds contains unsupported agents "
            f"{sorted(unknown)}; expected a subset of {list(DEFAULT_SCRIPTED_REDS)}"
        )
    if len(set(reds)) != len(reds):
        raise ValueError("eval.scripted_red.reds must not contain duplicates")
    return reds


@dataclass(frozen=True)
class ScriptedRedEvalSettings:
    """Validated ``eval.scripted_red`` recipe settings."""

    after_training: bool = False
    reds: tuple[str, ...] = DEFAULT_SCRIPTED_REDS
    seeds: tuple[int, ...] = tuple(range(1000, 1010))
    episodes_per_seed: int = 1
    deterministic: bool = False
    workers: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.after_training, bool):
            raise ValueError("eval.scripted_red.after_training must be a boolean")
        _normalise_reds(self.reds)
        parse_seeds(self.seeds)
        if isinstance(self.episodes_per_seed, bool) or not isinstance(self.episodes_per_seed, int):
            raise ValueError("eval.scripted_red.episodes_per_seed must be an integer")
        if self.episodes_per_seed < 1:
            raise ValueError("eval.scripted_red.episodes_per_seed must be positive")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int):
            raise ValueError("eval.scripted_red.workers must be an integer")
        if self.workers < 1:
            raise ValueError("eval.scripted_red.workers must be positive")
        if not isinstance(self.deterministic, bool):
            raise ValueError("eval.scripted_red.deterministic must be a boolean")

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> ScriptedRedEvalSettings:
        eval_config = recipe.get("eval", {})
        if eval_config is None:
            eval_config = {}
        if not isinstance(eval_config, Mapping):
            raise ValueError("eval must be a mapping")
        config = eval_config.get("scripted_red", {})
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise ValueError("eval.scripted_red must be a mapping")
        allowed = {
            "after_training",
            "reds",
            "seeds",
            "episodes_per_seed",
            "deterministic",
            "workers",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"eval.scripted_red has unknown settings: {sorted(unknown)}")
        return cls(
            after_training=config.get("after_training", False),
            reds=_normalise_reds(config.get("reds", DEFAULT_SCRIPTED_REDS)),
            seeds=parse_seeds(config.get("seeds", "1000-1009")),
            episodes_per_seed=config.get("episodes_per_seed", 1),
            deterministic=config.get("deterministic", False),
            workers=config.get("workers", 1),
        )


def _detect_backend(model_path: Path) -> str:
    if model_path.suffix == ".pt":
        return "cyborg"
    if model_path.suffix in (".safetensors", ".flax", ".orbax"):
        return "jax"
    raise ValueError(f"cannot detect trained backend from model suffix: {model_path}")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _load_trained_blue_contract(model_path: Path, backend: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate that the exact bundle contains a Blue policy trained in this run."""

    from jaxborg.checkpoint import load_jax_bundle, load_torch_bundle, read_sidecar
    from jaxborg.recipe import training_teams

    recipe = read_sidecar(model_path)
    if "blue" not in training_teams(recipe):
        raise ValueError(
            f"checkpoint sidecar says Blue was not trained in this run: {model_path}; "
            "refusing to evaluate a frozen Blue opponent"
        )
    bundle = load_jax_bundle(model_path) if backend == "jax" else load_torch_bundle(model_path)
    entry = bundle.policies.get("blue")
    if entry is None:
        raise ValueError(f"model bundle has no Blue policy: {model_path}")
    if not entry.trainable:
        raise ValueError(f"model bundle marks Blue as frozen, not trained: {model_path}")
    metadata = {
        "bundle_schema_version": bundle.schema_version,
        "bundle_legacy": bundle.legacy,
        "bundle_provenance": bundle.provenance,
        "blue_policy_trainable": entry.trainable,
        "blue_policy_source": entry.source,
    }
    return recipe, metadata


def _evaluate_cell(
    backend: str,
    model_path: Path,
    *,
    variant,
    seeds: list[int],
    episodes_per_seed: int,
    deterministic: bool,
    workers: int,
    progress: bool,
) -> tuple[list[float], list[int]]:
    if backend == "jax":
        from jaxborg.evaluation.jax_runner import evaluate_jax_on_cyborg

        rewards, seed_log, _recipe = evaluate_jax_on_cyborg(
            model_path,
            variant=variant,
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            deterministic=deterministic,
            workers=workers,
            progress=progress,
        )
        return rewards, seed_log

    from jaxborg.evaluation.cyborg_runner import evaluate_on_cyborg

    return evaluate_on_cyborg(
        model_path,
        variant=variant,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        deterministic=deterministic,
        workers=workers,
        progress=progress,
    )


def evaluate_scripted_reds(
    model_path: str | Path,
    *,
    reds: Sequence[str] = DEFAULT_SCRIPTED_REDS,
    seeds: str | int | Sequence[int] = "1000-1009",
    episodes_per_seed: int = 1,
    deterministic: bool = False,
    workers: int = 1,
    progress: bool = False,
    eval_name: str | None = None,
    cell_evaluator: Callable[..., tuple[list[float], list[int]]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate trained Blue against each requested scripted Red in CybORG."""

    from jaxborg.scenarios.cc4.game_variants import variant_for_red

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    normalised_reds = _normalise_reds(reds)
    parsed_seeds = parse_seeds(seeds)
    eval_name = _normalise_eval_name(eval_name)
    settings = ScriptedRedEvalSettings(
        reds=normalised_reds,
        seeds=parsed_seeds,
        episodes_per_seed=episodes_per_seed,
        deterministic=deterministic,
        workers=workers,
    )
    backend = _detect_backend(model_path)
    recipe, bundle_metadata = _load_trained_blue_contract(model_path, backend)
    evaluate_cell = cell_evaluator or _evaluate_cell
    git_commit = _git_commit()
    run = recipe.get("run", {})
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nonce = f"{time.time_ns() % 1_000_000_000:09d}"
    rows: list[dict[str, Any]] = []

    for red in settings.reds:
        variant = variant_for_red(red)
        print(
            f"Blue from {model_path.name} vs {red}: {len(settings.seeds) * settings.episodes_per_seed} episodes",
            flush=True,
        )
        t0 = time.perf_counter()
        rewards, seed_log = evaluate_cell(
            backend,
            model_path,
            variant=variant,
            seeds=list(settings.seeds),
            episodes_per_seed=settings.episodes_per_seed,
            deterministic=settings.deterministic,
            workers=settings.workers,
            progress=progress,
        )
        wall_time = time.perf_counter() - t0
        expected = len(settings.seeds) * settings.episodes_per_seed
        if len(rewards) != expected or len(seed_log) != expected:
            raise RuntimeError(
                f"{red} evaluator returned {len(rewards)} rewards and {len(seed_log)} seeds; expected {expected}"
            )
        reward_mean = mean(rewards)
        reward_std = stdev(rewards) if len(rewards) > 1 else 0.0
        row = {
            "eval_id": f"{timestamp}_{nonce}_{red}",
            "eval_name": eval_name,
            "suite": "scripted_red",
            "model": str(model_path),
            "policy_team": "blue",
            "recipe_name": recipe.get("meta", {}).get("name", ""),
            "recipe_path": recipe.get("meta", {}).get("source_path", ""),
            "trained_backend": backend,
            "eval_env": "cyborg",
            "eval_red": red,
            "variant": variant.name,
            "red_agent": variant.red_agent,
            "resilience_roles": variant.resilience_roles,
            "seeds": list(settings.seeds),
            "episodes_per_seed": settings.episodes_per_seed,
            "stochastic": not settings.deterministic,
            "mean_reward": reward_mean,
            "std_reward": reward_std,
            "n_episodes": len(rewards),
            "wall_time_s": wall_time,
            "git_commit": git_commit,
            "train_run_id": run.get("train_run_id"),
            "train_total_steps": run.get("total_steps"),
            "per_episode": rewards,
            "per_episode_seeds": seed_log,
            **bundle_metadata,
        }
        rows.append(row)
        print(f"  mean {reward_mean:.2f} +/- {reward_std:.2f} (n={len(rewards)})", flush=True)
    return rows


def _default_output_path(rows: Sequence[Mapping[str, Any]]) -> Path:
    exp_dir = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).expanduser().resolve()
    first = rows[0]
    eval_prefix = str(first["eval_id"]).rsplit("_", 1)[0]
    name = f"_{first['eval_name']}" if first.get("eval_name") else ""
    return (
        exp_dir
        / "eval"
        / (f"{first['recipe_name']}_{Path(str(first['model'])).stem}_scripted_red{name}_{eval_prefix}.jsonl")
    )


def write_results(rows: Sequence[Mapping[str, Any]], output_path: str | Path | None = None) -> Path:
    if not rows:
        raise ValueError("cannot write an empty scripted-Red evaluation")
    path = Path(output_path).expanduser().resolve() if output_path else _default_output_path(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, default=str) + "\n")
    return path


def attach_results_to_mlflow(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    run_id = rows[0].get("train_run_id")
    if not run_id:
        return
    metrics: dict[str, float] = {}
    for row in rows:
        eval_name = row.get("eval_name")
        prefix = (
            f"eval.after_training.{eval_name}.scripted_red.{row['eval_red']}.blue"
            if eval_name
            else f"eval.scripted_red.{row['eval_red']}.blue"
        )
        metrics[f"{prefix}.mean_reward"] = float(row["mean_reward"])
        metrics[f"{prefix}.std_reward"] = float(row["std_reward"])
        metrics[f"{prefix}.episodes"] = float(row["n_episodes"])
    from jaxborg.mlflow_setup import attach_eval_metrics

    attach_eval_metrics(str(run_id), metrics)


def run_configured_after_training(
    model_path: str | Path,
    recipe: Mapping[str, Any],
    *,
    run_subprocess: Callable[..., Any] = subprocess.run,
) -> bool:
    """Run the configured sweep in a fresh CPU process after final model save.

    The child process receives the exact durable model path from the trainer.
    ``check=True`` deliberately makes the overall training command non-zero if
    a required post-training evaluation fails; the already-saved model remains
    available for diagnosis or a manual rerun.
    """

    settings = ScriptedRedEvalSettings.from_recipe(recipe)
    if not settings.after_training:
        return False
    if os.environ.get("JAXBORG_SKIP_SCRIPTED_RED_EVAL") == "1":
        print("Skipping configured scripted-Red evaluation (JAXBORG_SKIP_SCRIPTED_RED_EVAL=1).", flush=True)
        return False

    from jaxborg.checkpoint import read_sidecar

    resolved_model = Path(model_path).expanduser().resolve()
    if not resolved_model.is_file():
        raise FileNotFoundError(f"final model is missing before scripted-Red evaluation: {resolved_model}")
    read_sidecar(resolved_model)  # Fail before launch if final provenance was not written.

    command = [
        sys.executable,
        "-m",
        "jaxborg.evaluation.scripted_red",
        "--model",
        str(resolved_model),
        "--reds",
        *settings.reds,
        "--seeds",
        ",".join(str(seed) for seed in settings.seeds),
        "--episodes-per-seed",
        str(settings.episodes_per_seed),
        "--workers",
        str(settings.workers),
    ]
    if settings.deterministic:
        command.append("--deterministic")
    child_env = os.environ.copy()
    child_env["JAX_PLATFORMS"] = "cpu"
    # Derive the experiment root from the canonical final-model layout so a
    # relative/unset parent environment cannot send results to a different DB.
    child_env["JAXBORG_EXP_DIR"] = str(resolved_model.parents[2])
    child_env["PYTHONUNBUFFERED"] = "1"
    print(f"Running post-training scripted-Red evaluation:\n  {shlex.join(command)}", flush=True)
    run_subprocess(command, check=True, cwd=_REPO_ROOT, env=child_env)
    return True


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Blue against CybORG FSM and CIA scripted Red agents")
    parser.add_argument("--model", required=True, help="Final .safetensors or .pt model bundle")
    parser.add_argument(
        "--reds",
        nargs="+",
        choices=DEFAULT_SCRIPTED_REDS,
        default=list(DEFAULT_SCRIPTED_REDS),
        help="Scripted Red opponents to evaluate",
    )
    parser.add_argument("--seeds", default="1000-1009", help="Seed list/range, e.g. 1000-1099")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", help="Use argmax Blue actions (debugging)")
    parser.add_argument("--progress", action="store_true", help="Print every episode")
    parser.add_argument(
        "--name",
        default=os.environ.get("JAXBORG_EVAL_NAME"),
        help="Optional evaluation name used in result files and MLflow metric keys",
    )
    parser.add_argument("--output", help="Override aggregate JSONL path")
    parser.add_argument("--no-mlflow", action="store_true", help="Do not attach opponent-qualified metrics")
    args = parser.parse_args(argv)

    rows = evaluate_scripted_reds(
        args.model,
        reds=args.reds,
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        deterministic=args.deterministic,
        workers=args.workers,
        progress=args.progress,
        eval_name=args.name,
    )
    output_path = write_results(rows, args.output)
    print(f"Wrote scripted-Red sweep: {output_path}", flush=True)
    if not args.no_mlflow:
        try:
            attach_results_to_mlflow(rows)
            if rows[0].get("train_run_id"):
                print(f"Attached scripted-Red metrics to MLflow run {rows[0]['train_run_id']}", flush=True)
        except Exception as exc:
            print(f"MLflow attach warning: {exc}", flush=True)


if __name__ == "__main__":
    main()
