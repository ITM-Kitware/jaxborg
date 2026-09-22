#!/usr/bin/env python3
"""Evaluate the Blue checkpoints from saved cross-play cells against fixed FSM Red.

Uses the original checkpoint sidecars, topology order, episode seeds and horizon.
Writes each checkpoint result immediately; completed matching records are reused.
No training or MLflow logging is performed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jaxborg-xla-cache")
os.environ.setdefault("JAXBORG_EVAL_BATCH_SIZE", "6")

from jaxborg.checkpoint import read_sidecar
from jaxborg.evaluation.episode_seeds import expand_episode_seeds
from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.evaluation.jax_scripted_red import evaluate_jax_scripted_reds
from jaxborg.recipe import eval_variant


def matched_checkpoints(cross_play: Path, exp_dir: Path, topology_dir: Path):
    records = [json.loads(line) for line in cross_play.read_text().splitlines() if line.strip()]
    summary = records[-1]
    if summary.get("eval_name") != "cross_play_summary":
        raise ValueError(f"{cross_play}: missing cross-play summary")
    cells = records[:-1]
    for step in summary["steps"]:
        cell = next(row for row in cells if row["blue_step"] == step)
        original = Path(cell["blue_checkpoint"])
        checkpoint = exp_dir / original.parent.parent.name / original.parent.name / original.name
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        recipe = read_sidecar(checkpoint)
        variant = eval_variant(recipe)
        if variant.name != cell["variant"]:
            raise ValueError("saved cross-play and checkpoint evaluation variants differ")
        topologies = [topology_dir / Path(path).name for path in cell["topology_paths"]]
        for path in topologies:
            if not path.is_file():
                raise FileNotFoundError(path)
        expected_seeds = expand_episode_seeds(cell["seeds"], cell["episodes_per_seed"]) * len(topologies)
        if cell.get("topology_sampling") != "exhaustive" or cell.get("per_episode_seeds") != expected_seeds:
            raise ValueError(
                "Saved cross-play uses a different episode seed or topology protocol; "
                "rerun cross-play before comparing it with the current scripted-Red evaluator"
            )
        signature = {
            "source_cross_play": str(cross_play.resolve()),
            "source_cross_play_eval_id": summary["eval_id"],
            "checkpoint_step": step,
            "checkpoint": str(checkpoint.resolve()),
            "seeds": cell["seeds"],
            "episodes_per_seed": cell["episodes_per_seed"],
            "stochastic": cell["stochastic"],
            "per_episode_seeds": expected_seeds,
            "topology_paths": [str(path.resolve()) for path in topologies],
        }
        yield checkpoint, recipe, variant, topologies, signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-play", nargs="+", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("remote/jaxborg-exp"))
    parser.add_argument("--topology-dir", type=Path, default=Path(".bank_cache/topologies/cotraining/eval_ops3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existing = [json.loads(line) for line in args.output.read_text().splitlines()] if args.output.exists() else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    envs = {}

    def cached_env(variant, **kwargs):
        key = (variant, tuple(str(p) for p in kwargs["topology_path"]))
        if key not in envs:
            envs[key] = make_jax_env(variant, **kwargs)
        return envs[key]

    for source in args.cross_play:
        for checkpoint, recipe, variant, topologies, signature in matched_checkpoints(
            source, args.exp_dir, args.topology_dir
        ):
            if any(all(row.get(k) == v for k, v in signature.items()) for row in existing):
                print(f"Reuse {checkpoint.parent.name} {signature['checkpoint_step']:,}", flush=True)
                continue
            print(f"Evaluate {checkpoint.parent.name} {signature['checkpoint_step']:,} vs FSM", flush=True)
            rows = evaluate_jax_scripted_reds(
                checkpoint,
                base_variant=variant,
                topology_paths=topologies,
                reds=["fsm"],
                seeds=signature["seeds"],
                episodes_per_seed=signature["episodes_per_seed"],
                deterministic=not signature["stochastic"],
                progress=True,
                eval_name="cross-play-fsm-comparison",
                recipe=recipe,
                env_factory=cached_env,
            )
            for row in rows:
                row.update(signature)
                row["observation_mode"] = "enhanced" if variant.cage4_enhanced_obs else "original"
                with args.output.open("a") as output:
                    output.write(json.dumps(row) + "\n")
                existing.append(row)
                print(
                    f"Saved mean {row['mean_reward']:.2f}, SD {row['std_reward']:.2f}, n={row['n_episodes']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
