"""Recovery from joint policy exports (weights and progress, not optimizer state)."""

import json
import shutil
import time
from pathlib import Path

import yaml
from safetensors import SafetensorError

from jaxborg.checkpoint import load_jax_bundle, read_sidecar


def select_joint_resume(save_dir, tag, recipe, seed, rollout_steps, total_updates, requested="latest"):
    """Find the newest readable export; never silently restart a damaged run."""
    save_dir = Path(save_dir)
    final = save_dir / f"model_{tag}.safetensors"
    if requested == "latest":
        checkpoints = sorted(
            (p for p in save_dir.glob("checkpoint_*.safetensors") if p.stem.removeprefix("checkpoint_").isdigit()),
            key=lambda p: int(p.stem.removeprefix("checkpoint_")),
            reverse=True,
        )
        candidates = ([final] if final.exists() else []) + checkpoints
    else:
        candidates = [Path(requested)]
    for path in candidates:
        try:
            bundle = load_jax_bundle(path)
            saved_recipe = read_sidecar(path)
            if not isinstance(saved_recipe, dict) or not saved_recipe.get("run"):
                raise yaml.YAMLError("Incomplete recipe sidecar")
        except (SafetensorError, FileNotFoundError, yaml.YAMLError) as exc:
            if requested != "latest":
                raise
            print(f"Skipping incomplete resume checkpoint {path}: {exc}", flush=True)
            continue
        # Evaluation settings may evolve, but training must retain its provenance.
        for key in ("algorithm", "core", "arch", "train", "jax", "cage4_enhanced_obs"):
            if saved_recipe.get(key) != recipe.get(key):
                raise ValueError(f"Resume recipe differs in {key}: {path}")
        provenance = bundle.provenance
        if provenance.get("seed") != seed or provenance.get("recipe") != recipe["meta"]["name"]:
            raise ValueError(f"Resume seed/recipe mismatch: {path}")
        if set(bundle.policies) != {"blue", "red"}:
            raise ValueError(f"Resume requires both Blue and Red policies: {path}")
        steps = int(provenance.get("total_steps", -1))
        if steps <= 0 or steps % rollout_steps or steps > total_updates * rollout_steps:
            raise ValueError(f"Invalid resume step count {steps}: {path}")
        if saved_recipe.get("run", {}).get("total_steps") != steps:
            raise ValueError(f"Resume sidecar step count mismatch: {path}")
        if path == final and steps != total_updates * rollout_steps:
            raise ValueError(f"Final model has incomplete training progress: {path}")
        return path, bundle, steps // rollout_steps
    if candidates:
        raise ValueError(f"No intact checkpoint and recipe sidecar in {save_dir}; refusing to restart from scratch")
    metrics = save_dir / "metrics.jsonl"
    if metrics.exists() and metrics.stat().st_size:
        raise ValueError(f"Training metrics exist but no checkpoints in {save_dir}; refusing to overwrite them")
    return None


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
