"""Common MLflow run-start hook.

Both algorithm scripts call `start_run(recipe, backend=...)` to:
- point MLflow at `$JAXBORG_EXP_DIR/mlflow.db`
- start a run named `<algorithm>-<backend>-<recipe_name>-seed<n>`
- tag the run with recipe.{name,source,path}, algorithm, backend,
  arch.name, git.{commit,branch}
- log the resolved recipe as flat dotted-key params
- log the source recipe yaml as an artifact

Returns the active mlflow.ActiveRun. Trainer is responsible for
mlflow.end_run() at finish.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow

from jaxborg.recipe import flatten_for_logging


def _exp_dir() -> Path:
    return Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()


def _git(arg: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", arg], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def configure(experiment: str | None = None) -> Path:
    """Point MLflow at $JAXBORG_EXP_DIR/mlflow.db. Return the db path."""
    exp_dir = _exp_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)
    db_path = exp_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    if experiment:
        mlflow.set_experiment(experiment)
    return db_path


def start_run(
    recipe: dict[str, Any],
    *,
    backend: str,
    seed: int,
    extra_tags: dict[str, str] | None = None,
    extra_params: dict[str, Any] | None = None,
):
    """Start an MLflow run and stamp it with recipe + git tags. Returns ActiveRun."""
    name = recipe.get("meta", {}).get("name", "unnamed")
    algorithm = recipe.get("algorithm", "ippo")
    arch_name = recipe.get("arch", {}).get("name", "shared")

    configure(experiment=f"{algorithm}-cc4")
    run_name = f"{algorithm}-{backend}-{name}-seed{seed}"
    run = mlflow.start_run(run_name=run_name)

    tags = {
        "recipe.name": name,
        "recipe.source": recipe.get("meta", {}).get("source", ""),
        "recipe.path": str(recipe.get("__source_path__", "")),
        "algorithm": algorithm,
        "backend": backend,
        "arch.name": arch_name,
        "seed": str(seed),
        "git.commit": _git("HEAD"),
        "git.branch": _git("--abbrev-ref HEAD"),
    }
    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags(tags)

    flat = flatten_for_logging(recipe)
    params = {f"recipe.{k}": v for k, v in flat.items()}
    if extra_params:
        params.update(extra_params)
    # MLflow caps param values at 500 chars and rejects unknown types.
    safe = {}
    for k, v in params.items():
        if v is None:
            continue
        s = str(v)
        if len(s) > 500:
            s = s[:497] + "..."
        safe[k] = s
    mlflow.log_params(safe)

    src = recipe.get("__source_path__")
    if src and Path(src).exists():
        try:
            mlflow.log_artifact(src)
        except Exception:
            pass

    return run


def attach_eval_metrics(
    train_run_id: str,
    metrics: dict[str, float],
) -> None:
    """Append eval metrics to the train run (used by eval_recipe.py)."""
    configure()
    with mlflow.start_run(run_id=train_run_id):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
