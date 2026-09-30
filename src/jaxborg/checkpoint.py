"""Checkpoint sidecar — recipe travels with model weights.

Every completed policy checkpoint has an adjacent recipe sidecar with the
resolved recipe and its producing run ID/step. MLflow storage is authoritative;
legacy filesystem checkpoint/sidecar pairs remain loadable. Policy weights are
portable inference artifacts, not complete resumable training state.
"""

from __future__ import annotations

import copy
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import yaml
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors import safe_open
from safetensors.flax import load_file, save_file

_FLAX_KEY_SEP = "/"


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

    payload = copy.deepcopy({k: v for k, v in recipe.items() if not str(k).startswith("__")})
    src = recipe.get("__source_path__")
    if src:
        payload.setdefault("meta", {})["source_path"] = src

    payload["run"] = {
        "seed": int(seed),
        "total_steps": int(total_steps),
        "backend": backend,
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_run_id": train_run_id,
    }
    if extra:
        payload["run"].update(extra)

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    temporary.replace(path)
    return path


def save_jax_params(path: str | Path, params: Any, *, action_dim: int) -> Path:
    """Write Flax params + minimal metadata to a safetensors file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_dict(jax.device_get(params), sep=_FLAX_KEY_SEP)
    temporary = path.with_name(path.name + ".tmp")
    save_file(flat, str(temporary), metadata={"action_dim": str(int(action_dim))})
    temporary.replace(path)
    return path


def load_jax_params(path: str | Path) -> tuple[dict, int]:
    """Inverse of `save_jax_params`. Returns `(params, action_dim)`."""
    path = Path(path)
    flat = load_file(str(path))
    params = unflatten_dict(flat, sep=_FLAX_KEY_SEP)
    with safe_open(str(path), framework="flax") as f:
        meta = f.metadata() or {}
    action_dim = int(meta.get("action_dim", 0))
    return params, action_dim


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
