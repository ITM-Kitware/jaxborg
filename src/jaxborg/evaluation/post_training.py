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


def run_configured_evaluations_after_training(
    model_path: str | Path,
    recipe: Mapping[str, Any],
    *,
    run_subprocess: Callable[..., Any] = subprocess.run,
) -> Path | None:
    """Run configured evaluation scripts sequentially and return the manifest.

    If the new list is absent, this delegates to the legacy scripted-Red hook.
    Required evaluations fail the training command after their failure has been
    recorded; optional evaluations are recorded and the next script still runs.
    """

    settings = PostTrainingEvalSettings.from_recipe(recipe)
    if not settings.evaluations:
        from jaxborg.evaluation.scripted_red import run_configured_after_training

        run_configured_after_training(model_path, recipe, run_subprocess=run_subprocess)
        return None
    if os.environ.get("JAXBORG_SKIP_POST_TRAINING_EVAL") == "1":
        print("Skipping configured post-training evaluations (JAXBORG_SKIP_POST_TRAINING_EVAL=1).", flush=True)
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
    # Match the canonical $EXP_DIR/<algorithm>_<backend>/<tag>/model layout.
    exp_dir = resolved_model.parents[2]
    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nonce = f"{time.time_ns() % 1_000_000_000:09d}"
    manifest_path = eval_dir / "manifests" / f"{resolved_model.stem}_{timestamp}_{nonce}.json"
    manifest: dict[str, Any] = {
        "model": str(resolved_model),
        "recipe": str(sidecar),
        "backend": backend,
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

    for index, evaluation in enumerate(settings.evaluations, 1):
        script = evaluation.resolve_script()
        job_replacements = {**replacements, "name": evaluation.name}
        command = [sys.executable, str(script)]
        if evaluation.model_arg is not None:
            command.extend((evaluation.model_arg, str(resolved_model)))
        command.extend(_format_args(evaluation, job_replacements))
        child_env = os.environ.copy()
        child_env.update(
            {
                "JAX_PLATFORMS": "cpu",
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
            f"Running post-training evaluation {index}/{len(settings.evaluations)} ({evaluation.name}):\n"
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
    parser.add_argument(
        "--recipe",
        help="Recipe name/path containing eval.after_training (default: model's recipe sidecar)",
    )
    args = parser.parse_args(argv)

    if args.recipe:
        from jaxborg.recipe import load

        recipe = load(args.recipe)
    else:
        from jaxborg.checkpoint import read_sidecar

        recipe = read_sidecar(args.model)
    manifest = run_configured_evaluations_after_training(args.model, recipe)
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
