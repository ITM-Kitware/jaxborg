"""Ordered, recipe-configured evaluation scripts for final checkpoints.

Each configured script runs in its own process after the trainer has saved the
final model bundle and recipe sidecar.  The runner passes the exact final model
through a configurable command-line flag and exposes useful paths through
environment variables.  A manifest records the commands and their outcomes.

The older ``eval.scripted_red.after_training`` setting remains supported when
``eval.after_training`` is not configured.
"""

from __future__ import annotations

import argparse
import copy
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
from typing import Any

from jaxborg.evaluation.checkpoint_scripted_reds import CheckpointScriptedRedsSettings
from jaxborg.evaluation.cross_play import CrossPlaySettings
from jaxborg.evaluation.cross_seed_play import CrossSeedPlaySettings, validate_cross_seed_models
from jaxborg.evaluation.play_priors import PlayPriorsSettings

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PLACEHOLDERS = frozenset({"model", "recipe", "backend", "exp_dir", "eval_dir", "name"})


def _normalise_arg(value: Any, *, location: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{location} must be a string or number")
    return str(value)


def _validate_placeholders(value: str, *, location: str) -> None:
    fields = re.findall(r"(?<!\{)\{([^{}]+)\}(?!\})", value)
    unknown = set(fields) - _PLACEHOLDERS
    if unknown:
        raise ValueError(f"{location} contains unknown placeholders: {sorted(unknown)}")


@dataclass(frozen=True)
class PostTrainingEval:
    """One Python evaluation script in an ordered post-training pipeline."""

    name: str
    script: str
    args: tuple[str, ...] = ()
    model_arg: str | None = "--model"
    required: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                "eval.after_training[].name must start with an alphanumeric character "
                "and contain only letters, numbers, '.', '_' or '-'"
            )
        if not isinstance(self.script, str) or not self.script.strip():
            raise ValueError("eval.after_training[].script must be a non-empty path")
        if Path(self.script).suffix != ".py":
            raise ValueError("eval.after_training[].script must point to a Python (.py) script")
        if self.model_arg is not None and (not isinstance(self.model_arg, str) or not self.model_arg.startswith("-")):
            raise ValueError("eval.after_training[].model_arg must be a command-line flag or null")
        if not isinstance(self.required, bool):
            raise ValueError("eval.after_training[].required must be a boolean")
        for index, arg in enumerate(self.args):
            if not isinstance(arg, str):
                raise ValueError(f"eval.after_training[].args[{index}] must be a string")
            _validate_placeholders(arg, location=f"eval.after_training[].args[{index}]")

    def resolve_script(self) -> Path:
        path = Path(self.script).expanduser()
        if not path.is_absolute():
            path = _REPO_ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"post-training evaluation script not found: {path}")
        return path


@dataclass(frozen=True)
class PostTrainingEvalSettings:
    """Validated ordered ``eval.after_training`` recipe entries."""

    evaluations: tuple[PostTrainingEval, ...] = ()

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> PostTrainingEvalSettings:
        eval_config = recipe.get("eval", {})
        if eval_config is None:
            eval_config = {}
        if not isinstance(eval_config, Mapping):
            raise ValueError("eval must be a mapping")
        configured = eval_config.get("after_training", [])
        if configured is None:
            configured = []
        if isinstance(configured, (str, bytes)) or not isinstance(configured, Sequence):
            raise ValueError("eval.after_training must be a list")

        evaluations: list[PostTrainingEval] = []
        for index, raw in enumerate(configured):
            location = f"eval.after_training[{index}]"
            if not isinstance(raw, Mapping):
                raise ValueError(f"{location} must be a mapping")
            allowed = {"name", "script", "args", "model_arg", "required"}
            unknown = set(raw) - allowed
            if unknown:
                raise ValueError(f"{location} has unknown settings: {sorted(unknown)}")
            missing = {key for key in ("name", "script") if key not in raw}
            if missing:
                raise ValueError(f"{location} is missing required settings: {sorted(missing)}")

            raw_args = raw.get("args", [])
            if isinstance(raw_args, (str, bytes)) or not isinstance(raw_args, Sequence):
                raise ValueError(f"{location}.args must be a list")
            args = tuple(
                _normalise_arg(value, location=f"{location}.args[{arg_index}]")
                for arg_index, value in enumerate(raw_args)
            )
            model_arg = raw.get("model_arg", "--model")
            evaluations.append(
                PostTrainingEval(
                    name=raw["name"],
                    script=raw["script"],
                    args=args,
                    model_arg=model_arg,
                    required=raw.get("required", True),
                )
            )

        names = [evaluation.name for evaluation in evaluations]
        if len(names) != len(set(names)):
            raise ValueError("eval.after_training evaluation names must be unique")
        return cls(tuple(evaluations))


def _play_priors_evaluation(recipe: Mapping[str, Any]) -> PostTrainingEval | None:
    settings = PlayPriorsSettings.from_recipe(recipe)
    if not settings.enabled:
        return None
    return PostTrainingEval(
        name="play_priors",
        script="scripts/eval/eval_play_priors.py",
        args=("--recipe", "{recipe}"),
        required=settings.required,
    )


def _cross_play_evaluation(recipe: Mapping[str, Any]) -> PostTrainingEval | None:
    settings = CrossPlaySettings.from_recipe(recipe)
    if not settings.enabled:
        return None
    return PostTrainingEval(
        name="cross_play",
        script="scripts/eval/eval_cross_play.py",
        args=("--recipe", "{recipe}"),
        required=settings.required,
    )


def _checkpoint_scripted_reds_evaluation(recipe: Mapping[str, Any]) -> PostTrainingEval | None:
    settings = CheckpointScriptedRedsSettings.from_recipe(recipe)
    if not settings.enabled:
        return None
    return PostTrainingEval(
        name="checkpoint_scripted_reds",
        script="scripts/eval/eval_checkpoint_scripted_reds.py",
        args=("--recipe", "{recipe}"),
        required=settings.required,
    )


def _cross_seed_play_evaluation(
    model_path: str | Path, recipe: Mapping[str, Any], red_model: str | Path | None
) -> PostTrainingEval | None:
    settings = CrossSeedPlaySettings.from_recipe(recipe)
    if not settings.enabled:
        return None
    if red_model is None:
        print("Skipping cross-seed-play: supply --cross-seed-red after the other seed finishes training.", flush=True)
        return None
    red = validate_cross_seed_models(model_path, red_model)
    return PostTrainingEval(
        name="cross-seed-play",
        script="scripts/eval/eval_matchup.py",
        model_arg="--blue-path",
        args=(
            "--recipe",
            "{recipe}",
            "--policy-backend",
            "{backend}",
            "--red-path",
            str(red),
            "--name",
            "{name}",
            "--seeds",
            ",".join(map(str, settings.seeds)),
            "--episodes-per-seed",
            str(settings.episodes_per_seed),
            "--mlflow-source-team",
            "blue",
        )
        + (("--deterministic",) if settings.deterministic else ()),
        required=settings.required,
    )


def _sidecar_path(model_path: Path) -> Path:
    name = model_path.name
    stem = name[len("model_") :] if name.startswith("model_") else model_path.stem
    stem = stem.rsplit(".", 1)[0]
    for suffix in (".yaml", ".yml"):
        candidate = model_path.with_name(f"recipe_{stem}{suffix}")
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"no recipe sidecar found next to final model: {model_path}")


def _format_args(evaluation: PostTrainingEval, replacements: Mapping[str, str]) -> list[str]:
    try:
        return [arg.format_map(replacements) for arg in evaluation.args]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"could not expand arguments for evaluation {evaluation.name!r}: {exc}") from exc


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _evaluation_jax_platforms(backend: str) -> str:
    """Select the child JAX platform while preserving an explicit override."""

    configured = os.environ.get("JAX_PLATFORMS", "").strip()
    if configured:
        return configured
    return "cuda" if backend == "jax" else "cpu"


def run_configured_evaluations_after_training(
    model_path: str | Path,
    recipe: Mapping[str, Any],
    *,
    cross_seed_red: str | Path | None = None,
    save_evaluation_recipe: bool = False,
    run_subprocess: Callable[..., Any] = subprocess.run,
) -> Path | None:
    """Run configured evaluation scripts sequentially and return the manifest.

    If the new list is absent, this delegates to the legacy scripted-Red hook.
    Required evaluations fail the training command after their failure has been
    recorded; optional evaluations are recorded and the next script still runs.
    """

    settings = PostTrainingEvalSettings.from_recipe(recipe)
    if os.environ.get("JAXBORG_SKIP_POST_TRAINING_EVAL") == "1":
        print("Skipping configured post-training evaluations (JAXBORG_SKIP_POST_TRAINING_EVAL=1).", flush=True)
        return None
    # Named built-in suites are also post-training evaluations. Their order is
    # fixed here (independent of YAML key order), before the explicit script list.
    built_in = tuple(
        evaluation
        for evaluation in (
            _cross_seed_play_evaluation(model_path, recipe, cross_seed_red),
            _play_priors_evaluation(recipe),
            _cross_play_evaluation(recipe),
            _checkpoint_scripted_reds_evaluation(recipe),
        )
        if evaluation is not None
    )
    evaluations = built_in + settings.evaluations
    if not evaluations:
        from jaxborg.evaluation.scripted_red import run_configured_after_training

        run_configured_after_training(model_path, recipe, run_subprocess=run_subprocess)
        return None
    resolved_model = Path(model_path).expanduser().resolve()
    if not resolved_model.is_file():
        raise FileNotFoundError(f"final model is missing before post-training evaluation: {resolved_model}")
    sidecar = _sidecar_path(resolved_model)
    if resolved_model.suffix == ".pt":
        backend = "cyborg"
    elif resolved_model.suffix in (".safetensors", ".flax", ".orbax"):
        backend = "jax"
    else:
        raise ValueError(f"cannot detect trained backend from model suffix: {resolved_model}")
    jax_platforms = _evaluation_jax_platforms(backend)
    # Match the canonical $EXP_DIR/<algorithm>_<backend>/<tag>/model layout.
    exp_dir = resolved_model.parents[2]
    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nonce = f"{time.time_ns() % 1_000_000_000:09d}"
    manifest_path = eval_dir / "manifests" / f"{resolved_model.stem}_{timestamp}_{nonce}.json"
    if save_evaluation_recipe:
        # A CLI evaluation override must reach child processes, too. Preserve
        # the original training sidecar; archive the effective recipe beside
        # this evaluation's manifest so the smaller/larger protocol is auditable.
        import yaml

        sidecar = manifest_path.with_suffix(".yaml")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in recipe.items() if not str(key).startswith("__")}
        sidecar.write_text(yaml.safe_dump(payload, sort_keys=False))
    manifest: dict[str, Any] = {
        "model": str(resolved_model),
        "recipe": str(sidecar),
        "backend": backend,
        "jax_platforms": jax_platforms,
        "eval_dir": str(eval_dir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evaluations": [],
    }
    replacements = {
        "model": str(resolved_model),
        "recipe": str(sidecar),
        "backend": backend,
        "exp_dir": str(exp_dir),
        "eval_dir": str(eval_dir),
    }

    for index, evaluation in enumerate(evaluations, 1):
        script = evaluation.resolve_script()
        job_replacements = {**replacements, "name": evaluation.name}
        command = [sys.executable, str(script)]
        if evaluation.model_arg is not None:
            command.extend((evaluation.model_arg, str(resolved_model)))
        command.extend(_format_args(evaluation, job_replacements))
        child_env = os.environ.copy()
        child_env.update(
            {
                "JAX_PLATFORMS": jax_platforms,
                "JAXBORG_EXP_DIR": str(exp_dir),
                "JAXBORG_EVAL_DIR": str(eval_dir),
                "JAXBORG_EVAL_NAME": evaluation.name,
                "JAXBORG_MODEL_PATH": str(resolved_model),
                "JAXBORG_RECIPE_PATH": str(sidecar),
                "JAXBORG_TRAINED_BACKEND": backend,
                "PYTHONUNBUFFERED": "1",
            }
        )
        record: dict[str, Any] = {
            "name": evaluation.name,
            "script": str(script),
            "command": command,
            "required": evaluation.required,
            "status": "running",
        }
        manifest["evaluations"].append(record)
        _write_manifest(manifest_path, manifest)
        print(
            f"Running post-training evaluation {index}/{len(evaluations)} ({evaluation.name}):\n"
            f"  JAX_PLATFORMS={jax_platforms}\n"
            f"  {shlex.join(command)}",
            flush=True,
        )
        try:
            completed = run_subprocess(command, check=True, cwd=_REPO_ROOT, env=child_env)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                record["returncode"] = exc.returncode
            if evaluation.required:
                manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            _write_manifest(manifest_path, manifest)
            if evaluation.required:
                print(f"Post-training evaluation manifest: {manifest_path}", flush=True)
                raise
            print(f"Optional evaluation {evaluation.name!r} failed: {exc}", flush=True)
        else:
            record["status"] = "succeeded"
            record["returncode"] = int(getattr(completed, "returncode", 0))
            _write_manifest(manifest_path, manifest)

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_manifest(manifest_path, manifest)
    print(f"Post-training evaluation manifest: {manifest_path}", flush=True)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a recipe's ordered post-training evaluation scripts")
    parser.add_argument("--model", required=True, help="Final model bundle to pass to every evaluation")
    parser.add_argument("--cross-seed-red", help="Final Red bundle from another training seed for eval.cross_seed_play")
    parser.add_argument(
        "--recipe",
        help="Override the eval section using this recipe; keep the model's training settings and provenance",
    )
    args = parser.parse_args(argv)

    from jaxborg.checkpoint import read_sidecar

    recipe = read_sidecar(args.model)
    if args.recipe:
        from jaxborg.recipe import load

        recipe = copy.deepcopy(recipe)
        recipe["eval"] = copy.deepcopy(load(args.recipe).get("eval", {}))
    manifest = run_configured_evaluations_after_training(
        args.model, recipe, cross_seed_red=args.cross_seed_red, save_evaluation_recipe=bool(args.recipe)
    )
    if manifest is None:
        from jaxborg.evaluation.scripted_red import ScriptedRedEvalSettings

        if not ScriptedRedEvalSettings.from_recipe(recipe).after_training:
            print("No post-training evaluations were configured.", flush=True)


__all__ = [
    "PostTrainingEval",
    "PostTrainingEvalSettings",
    "main",
    "run_configured_evaluations_after_training",
]


if __name__ == "__main__":
    main()
