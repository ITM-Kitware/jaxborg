"""Checkpoint sidecar — recipe travels with model weights.

Each training run writes:

    $JAXBORG_EXP_DIR/<algo>_<backend>/
        model_<tag>.pt              (or .pkl for jax)
        recipe_<tag>.yaml           ← this module writes it
        checkpoint_<tag>.pt         (full optimizer state, optional)

`recipe_<tag>.yaml` is the **resolved** recipe: the recipe dict that the
trainer actually consumed (post CLI overrides), plus a `run` block with
seed, commit, timestamp, total_steps, and (when known) the MLflow run id.

The eval script reads it back to instantiate the right architecture and to
attach eval metrics to the same MLflow run.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return ""


def _git_branch() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return ""


def write_sidecar(
    path: Path,
    recipe: dict[str, Any],
    *,
    seed: int,
    total_steps: int,
    backend: str,
    train_run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the resolved recipe + run metadata to `path`. Returns `path`.

    `recipe` must be the recipe dict as the trainer consumed it. Internal
    keys (`__source_path__`) are preserved under `meta.source_path` and the
    underscore key is dropped from the on-disk YAML.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {k: v for k, v in recipe.items() if not str(k).startswith("__")}
    src = recipe.get("__source_path__")
    if src:
        payload.setdefault("meta", {})["source_path"] = src

    payload["run"] = {
        "seed": int(seed),
        "total_steps": int(total_steps),
        "backend": backend,
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_run_id": train_run_id,
    }
    if extra:
        payload["run"].update(extra)

    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def read_sidecar(model_path: str | Path) -> dict[str, Any]:
    """Load `recipe_<tag>.{yaml|yml}` adjacent to `model_path`."""
    model_path = Path(model_path)
    name = model_path.name
    if name.startswith("model_"):
        stem = name[len("model_") :]
        stem = stem.rsplit(".", 1)[0]
    else:
        stem = model_path.stem
    candidates = [
        model_path.with_name(f"recipe_{stem}.yaml"),
        model_path.with_name(f"recipe_{stem}.yml"),
    ]
    for c in candidates:
        if c.exists():
            return yaml.safe_load(c.read_text())
    raise FileNotFoundError(f"No recipe sidecar found next to {model_path} (looked for {[str(c) for c in candidates]})")
