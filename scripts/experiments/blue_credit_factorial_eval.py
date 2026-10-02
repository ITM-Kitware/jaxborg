"""Evaluate all registered factorial endpoints and original Blue on a fresh cohort."""

import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np
import yaml

from jaxborg.blue_learning_probe import make_evaluator
from jaxborg.checkpoint import load_jax_bundle
from jaxborg.evaluation.jax_env_factory import make_joint_jax_env
from jaxborg.policies import policy_from_arch
from jaxborg.recipe import load, project_jax
from jaxborg.research_tracking import parameter_hash
from jaxborg.tracking import Run, assigned_devices, file_hash
from scripts.experiments.analyze_blue_credit_factorial import ARMS, audit_arms
from scripts.experiments.blue_learning_mechanism import archived_baseline, inputs, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    devices = assigned_devices()
    if not devices or any(d.platform != "gpu" for d in jax.devices()):
        raise RuntimeError("factorial confirmation requires allocated GPU")
    protocol = yaml.safe_load(args.config.read_text())
    if protocol["resources"]["partition"] != "community":
        raise ValueError("community partition required")
    audit = audit_arms(args.study_root)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("retain previous attempts and use a fresh output")
    args.output.mkdir(parents=True, exist_ok=True)
    data = inputs(protocol, args.data_root)
    recipe = load(str(data["warm"]["recipe"]))
    topology = args.data_root / "recipes" / f"blue_source{protocol['source_steps']}" / "topology-seed0.npz"
    recipe["train"]["topology_bank"] = recipe["eval"]["topology_bank"] = [str(topology)]
    recipe["train"]["opponents"]["red"]["path"] = str(data["original"]["model"])
    config = project_jax(recipe, team="blue")
    models = {
        label: args.study_root / directory / "checkpoints/warm-20.safetensors" for label, directory in ARMS.items()
    }
    models.update(initial=data["warm"]["model"], original=data["original"]["model"])
    bundles = {label: load_jax_bundle(path) for label, path in models.items()}
    red = bundles["original"].policies["red"]
    if any(
        parameter_hash(bundle.policies["red"].weights) != parameter_hash(red.weights) for bundle in bundles.values()
    ):
        raise ValueError("frozen-opponent mismatch")
    entries = {"blue": bundles["initial"].policies["blue"], "red": red}
    if any(bundle.policies["blue"].arch != entries["blue"].arch for bundle in bundles.values()):
        raise ValueError("policy architecture mismatch")
    networks = {team: policy_from_arch(entry.arch, action_dim=entry.action_dim) for team, entry in entries.items()}
    env = make_joint_jax_env(config["EVAL_VARIANT"], topology_path=[topology], training_mode=False)
    evaluate = make_evaluator(env, networks)
    source_inputs = [{"role": label, "path": str(path), "sha256": file_hash(path)} for label, path in models.items()]
    with Run(
        recipe, backend="jax", kind="evaluation", name=protocol["name"], config=config, inputs=source_inputs
    ) as owner:
        owner.update(diagnostic_protocol=protocol, factorial_training_audit=audit)
        (args.output / "config.yaml").write_text(yaml.safe_dump(protocol))
        (args.output / "resolved-recipe.yaml").write_text(yaml.safe_dump(recipe))
        write_json(args.output / "checkpoint-index.json", source_inputs)
        write_json(
            args.output / "runtime.json",
            {
                "owner": owner.run_id,
                "devices": devices,
                "slurm_job": os.environ["SLURM_JOB_ID"],
                "jax": jax.__version__,
                "source_revision": owner.manifest["source"]["git_commit"],
            },
        )
        archived = archived_baseline(protocol, args.data_root)
        reproduced = evaluate(entries["blue"].weights, red.weights, archived["per_episode_seeds"][:8])
        if [r["blue_return"] for r in reproduced] != archived["per_episode_blue_returns"][:8]:
            raise ValueError("archived canonical score check failed")
        write_json(args.output / "archived-reproduction.json", {"all_returns_exact": True, "episodes": reproduced})
        seeds = list(
            range(
                protocol["seeds"]["confirmation_start"],
                protocol["seeds"]["confirmation_start"] + protocol["confirmation_episodes"],
            )
        )
        rows = {}
        for label, bundle in bundles.items():
            for restricted in (False, True):
                key = label + ("-no-block" if restricted else "")
                rows[key] = evaluate(bundle.policies["blue"].weights, red.weights, seeds, no_block=restricted)
                write_json(args.output / "per-episode.json", rows)
                print(key, np.mean([r["blue_return"] for r in rows[key]]), flush=True)
        # Independently reproduce same-cohort confirmations from each mixed arm.
        for label in ("a095-c100", "a100-c095"):
            old = json.loads((args.study_root / label / "confirmation-episodes.json").read_text())
            old_protocol = yaml.safe_load((args.study_root / label / "config.yaml").read_text())
            if old_protocol["seeds"]["confirmation_start"] != protocol["seeds"]["confirmation_start"]:
                raise ValueError("mixed arms and factorial confirmation cohorts differ")
            for source, target in (
                ("warm-final", label),
                ("warm-0", "initial"),
                ("original", "original"),
                ("control-final", "a095-c095"),
            ):
                for suffix in ("", "-no-block"):
                    if old[source + suffix] != rows[target + suffix]:
                        raise ValueError(f"factorial confirmation differs on replay: {label}/{source}{suffix}")
        write_json(
            args.output / "verification.json",
            {
                "all_arm_calibrated_states_exact": True,
                "frozen_red_parameters_exact": True,
                "archived_returns_exact": True,
                "all_mixed_arm_confirmation_replays_exact": True,
            },
        )
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                owner.publish(path, "factorial-confirmation/" + path.relative_to(args.output).as_posix())
    write_json(args.output / "manifest.json", owner.manifest)


if __name__ == "__main__":
    main()
