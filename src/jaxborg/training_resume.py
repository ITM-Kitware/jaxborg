"""Atomic, full-state JAX training checkpoints, separate from policy exports."""

import copy
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import yaml
from flax import serialization

LATEST_CHECKPOINT = "training_latest.msgpack"
SCHEMA_VERSION = 1


def resume_request(recipe, override=None):
    """CLI takes precedence; YAML accepts false/null, true/latest, or a path."""
    value = recipe.get("train", {}).get("resume") if override is None else override
    if value is None or value is False:
        return None
    if value is True:
        return "latest"
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("train.resume must be false, null, true, 'latest', or a training checkpoint path")


def _training_contract(recipe):
    contract = copy.deepcopy(recipe)
    for key in ("meta", "eval", "mlflow", "plots", "cleanrl", "run", "__source_path__"):
        contract.pop(key, None)
    train = contract.setdefault("train", {})
    train.pop("resume", None)
    train.pop("total_timesteps", None)
    contract.setdefault("jax", {}).pop("checkpoint_every_updates", None)
    return contract


def _validate_recipe(saved, recipe):
    if saved["meta"]["name"] != recipe["meta"]["name"]:
        raise ValueError("Resume recipe name differs from the saved run")
    old, new = _training_contract(saved), _training_contract(recipe)
    for key in old.keys() | new.keys():
        if old.get(key) != new.get(key):
            raise ValueError(f"Resume recipe differs in {key}")
    old_budget = saved["train"].get("total_timesteps")
    new_budget = recipe["train"].get("total_timesteps")
    if old_budget != new_budget:
        if new_budget is None or old_budget is None or new_budget < old_budget:
            raise ValueError("Resume total_timesteps may only increase")
        overrides = saved["train"].get("team_overrides", {})
        cores = [{**saved.get("core", {}), **overrides.get(team, {}).get("core", {})} for team in ("blue", "red")]
        if any(core.get("anneal_lr", False) for core in cores):
            raise ValueError(
                "Cannot change total_timesteps when resuming anneal_lr: retain the original schedule horizon"
            )


@dataclass
class TrainingCheckpoint:
    path: Path
    completed_updates: int
    state: dict
    policies: dict
    train_run_id: str | None

    def restore(self, template):
        """Restore registered Flax types and optimizer tuples using a fresh runtime."""
        expected = serialization.to_state_dict(template)
        expected_leaves, expected_tree = jax.tree.flatten_with_path(expected)
        actual_leaves, actual_tree = jax.tree.flatten_with_path(self.state)
        if expected_tree != actual_tree:
            raise ValueError(f"Resume training state structure differs: {self.path}")
        for (key, target), (_, value) in zip(expected_leaves, actual_leaves):
            if np.shape(target) != np.shape(value) or (
                hasattr(target, "dtype") and np.dtype(target.dtype) != np.asarray(value).dtype
            ):
                raise ValueError(
                    f"Resume training state shape/dtype differs at {jax.tree_util.keystr(key)}: {self.path}"
                )
        return jax.device_put(serialization.from_state_dict(template, self.state))


def save_training_checkpoint(
    path, *, recipe, seed, completed_updates, rollout_steps, state, policies=None, train_run_id=None
):
    """Replace the latest snapshot only after its complete contents reach disk.

    The state includes TrainState (params, Adam moments and schedule counters),
    RNG, observations, environment, reward normalizers, and adaptive state.
    Runtime functions are rebuilt from the recipe, never pickled.
    """
    path = Path(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "recipe_yaml": yaml.safe_dump({key: value for key, value in recipe.items() if not key.startswith("__")}),
        "seed": int(seed),
        "completed_updates": int(completed_updates),
        "rollout_steps": int(rollout_steps),
        "policies_yaml": yaml.safe_dump(policies or {}),
        "train_run_id": train_run_id,
        "state": serialization.to_state_dict(jax.device_get(state)),
    }
    data = serialization.msgpack_serialize(payload, in_place=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def select_training_resume(save_dir, recipe, seed, rollout_steps, total_updates, requested):
    """Load the latest full snapshot, refusing to treat weights as training state."""
    if requested is None:
        return None
    save_dir = Path(save_dir)
    path = save_dir / LATEST_CHECKPOINT if requested == "latest" else Path(requested).expanduser()
    if path.suffix != ".msgpack":
        raise ValueError(
            "Training resume requires a full .msgpack checkpoint; policy-only exports lack optimizer state"
        )
    if not path.exists() and requested == "latest":
        metrics = save_dir / "metrics.jsonl"
        if any(save_dir.glob("*.safetensors")) or (metrics.exists() and metrics.stat().st_size):
            raise ValueError(
                f"No full training checkpoint in {save_dir}; existing weights/metrics cannot resume training"
            )
        return None
    try:
        payload = serialization.msgpack_restore(path.read_bytes())
    except FileNotFoundError:
        raise
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unreadable training checkpoint: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported training checkpoint schema: {path}")
    required = {"recipe_yaml", "seed", "completed_updates", "rollout_steps", "state", "policies_yaml", "train_run_id"}
    if not required <= payload.keys():
        raise ValueError(f"Incomplete training checkpoint: {path}")
    _validate_recipe(yaml.safe_load(payload["recipe_yaml"]), recipe)
    if payload["seed"] != seed:
        raise ValueError(f"Resume seed mismatch: {path}")
    updates = payload["completed_updates"]
    if isinstance(updates, bool) or not isinstance(updates, int) or not 0 < updates <= total_updates:
        raise ValueError(f"Invalid resume update count {updates}: {path}")
    if payload["rollout_steps"] != rollout_steps:
        raise ValueError(f"Resume rollout size differs: {path}")
    return TrainingCheckpoint(
        path=path,
        completed_updates=updates,
        state=payload["state"],
        policies=yaml.safe_load(payload["policies_yaml"]),
        train_run_id=payload["train_run_id"],
    )


def preserve_resume_metrics(path, steps):
    """Archive the original log and retain only metrics covered by the checkpoint."""
    path = Path(path)
    if not path.exists():
        return
    backup = path.with_name(f"metrics_before_resume_{time.time_ns()}.jsonl")
    shutil.copy2(path, backup)
    retained = []
    for line in backup.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # Disk exhaustion can leave a partial final line.
        if row["env_steps"] <= steps:
            retained.append(line + "\n")
    path.write_text("".join(retained))
