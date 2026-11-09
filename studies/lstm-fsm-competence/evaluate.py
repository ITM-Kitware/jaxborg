"""Evaluate owned LSTM checkpoints on the fixed native stock-game panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ARCH = {
    "name": "recurrent",
    "hidden_dim": 256,
    "hidden_layers": 1,
    "activation": "tanh",
    "cell": "lstm",
    "trunk": "separate",
}
RED_IDS = {f"red-{condition}-{seed}-final" for condition in ("single", "diverse") for seed in (42, 100)}
MILESTONES = {"smoke": 144000, "interim": 9600000, "primary": 49968000, "final": 124992000}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked_file(path, sha256):
    path = Path(path).resolve()
    if digest(path) != sha256:
        raise ValueError(f"Input hash differs from its declaration: {path}")
    return path


def resolve_training_attempt(records, trial, expected_revision, *, explicit=None, receipt_path=None):
    """A live attempt needs no terminal receipt; ambiguous histories require selection."""
    receipt = (
        read_json(receipt_path) if explicit is None and receipt_path is not None and Path(receipt_path).exists() else {}
    )
    identity = explicit or receipt.get("attempt_id")
    candidates = [
        record
        for record in records
        if record["kind"] == "attempt"
        and record.get("execution_kind") == "training"
        and record.get("trial_id") == trial
    ]
    if identity is None:
        if len(candidates) != 1:
            raise ValueError("Choose --training-attempt explicitly when a trial has zero or multiple training attempts")
        owner = candidates[0]
    else:
        matches = [record for record in candidates if record["id"] == identity]
        if len(matches) != 1:
            raise ValueError("Selected training attempt does not belong to the requested trial")
        owner = matches[0]
    if owner.get("facts", {}).get("source_revision", {}).get("value") != expected_revision:
        raise ValueError("Selected training attempt was produced by a different source revision")
    return owner["id"]


def checkpoint_selection(records, root, trial, attempt_id, *, target_steps=None, allow_running=False):
    """Select a published immutable milestone; live training is not declared complete."""
    indexed = {record["id"]: record for record in records}
    owner = indexed.get(attempt_id, {})
    if (
        owner.get("kind") != "attempt"
        or owner.get("execution_kind") != "training"
        or owner.get("status") not in ({"running", "succeeded"} if allow_running else {"succeeded"})
        or owner.get("trial_id") != trial
    ):
        raise ValueError("Select an eligible training attempt of the declared trial")
    budget = next(budget for budget in owner["budgets"] if budget["unit"] == "environment_steps")
    final_steps = budget["completed"] if target_steps is None else target_steps
    if final_steps <= 0 or final_steps > budget["completed"]:
        raise ValueError("Requested milestone is not yet completed; no later/earlier checkpoint is substituted")
    initial = owner["facts"].get("initial_checkpoint", {}).get("value", {})
    initial_owner = attempt_id
    if initial.get("environment_steps") != 0:
        origin = owner["facts"].get("training_origin", {}).get("value", {})
        origin_owner = indexed.get(origin.get("attempt_id"), {})
        initial = origin.get("checkpoint", {})
        if (
            origin_owner.get("kind") != "attempt"
            or origin_owner.get("execution_kind") != "training"
            or origin_owner.get("facts", {}).get("initial_checkpoint", {}).get("value") != initial
            or initial.get("environment_steps") != 0
        ):
            raise ValueError("Resumed training requires its verified original zero-step checkpoint")
        initial_owner = origin_owner["id"]
    selected = []
    for key, steps in (("initial_checkpoint", 0), (f"checkpoint_{final_steps}", final_steps)):
        value = initial if steps == 0 else owner["facts"].get(key, {}).get("value", {})
        if value.get("environment_steps") != steps:
            raise ValueError(f"Missing declared training checkpoint: {key}")
        checkpoint_owner = initial_owner if steps == 0 else attempt_id
        item = {"steps": steps, "training_attempt": checkpoint_owner}
        for role in ("model", "sidecar"):
            artifact = indexed.get(value.get(role), {})
            if artifact.get("kind") != "artifact" or artifact.get("producer_id") != checkpoint_owner:
                raise ValueError("Checkpoint and sidecar must belong to their declared training attempt")
            path = (Path(root) / artifact["uri"]).resolve()
            if not path.is_relative_to(Path(root).resolve()):
                raise ValueError("Training artifact escaped its evidence store")
            item[role] = str(checked_file(path, artifact["sha256"]))
            item[f"{role}_artifact"] = artifact["id"]
            item[f"{role}_sha256"] = artifact["sha256"]
        selected.append(item)
    return selected


def reference_case(checkpoint, variant, snapshots, seeds, revision, red=None):
    """Identity excludes evaluation stage but includes every frozen scientific input."""
    return {
        "checkpoint": checkpoint["model_artifact"],
        "checkpoint_sha256": checkpoint["model_sha256"],
        "sidecar_sha256": checkpoint["sidecar_sha256"],
        "variant": asdict(variant),
        "topologies": [item["sha256"] for item in snapshots],
        "seeds": seeds,
        "source_revision": revision,
        "red_id": "fsm" if red is None else red["id"],
        "red_sha256": None if red is None else red["sha256"],
        "red_sidecar_sha256": None if red is None else red["sidecar_sha256"],
        "deterministic": False,
        "episodes_per_seed": 1,
        "backend": "cpu",
    }


def validate_reference(receipt, case, records, evidence):
    """Validate a reused reference against its retained inputs and episode bytes."""
    if receipt.get("case") != case or receipt.get("checkpoint_steps") != 0:
        raise ValueError("Initial reference differs from the requested frozen case")
    indexed = {record["id"]: record for record in records}
    evaluation = indexed.get(receipt.get("evaluation_id"), {})
    attempt = indexed.get(evaluation.get("attempt_id"), {})
    if (
        evaluation.get("kind") != "evaluation"
        or evaluation.get("status") != "succeeded"
        or attempt.get("status") != "succeeded"
    ):
        raise ValueError("Initial reference must name a successful recorded evaluation")
    if case["checkpoint"] not in evaluation["inputs"]:
        raise ValueError("Initial reference does not consume the selected owned checkpoint")
    if indexed[case["checkpoint"]]["sha256"] != case["checkpoint_sha256"]:
        raise ValueError("Initial reference checkpoint bytes differ")
    directory = Path(evidence) / "attempts" / attempt["id"].rsplit(":", 1)[-1]
    configuration = read_json(directory / "configuration.json")
    scripted = case["red_id"] == "fsm"
    observed_seeds = configuration["seeds"] if scripted else configuration["seed_roles"]["episode_roots"]
    if (
        configuration["base_variant" if scripted else "variant"] != case["variant"]
        or configuration["topology"]["sha256"] != case["topologies"]
        or list(observed_seeds) != case["seeds"]
        or configuration["execution"]["source_revision"] != case["source_revision"]
        or configuration["deterministic"] is not False
        or configuration["episodes_per_seed"] != 1
    ):
        raise ValueError("Initial reference recorded another evaluation procedure")
    inputs = read_json(directory / "input-artifacts.json")
    sidecar_role = "source recipe" if scripted else "blue sidecar"
    if indexed[inputs[sidecar_role]]["sha256"] != case["sidecar_sha256"]:
        raise ValueError("Initial reference used another Blue sidecar")
    if scripted and configuration["reds"] != ["fsm"]:
        raise ValueError("Initial reference used another scripted Red")
    if not scripted:
        if (
            indexed[inputs["red checkpoint"]]["sha256"] != case["red_sha256"]
            or indexed[inputs["red sidecar"]]["sha256"] != case["red_sidecar_sha256"]
        ):
            raise ValueError("Initial reference used another Red checkpoint or sidecar")
    observations = [
        read_json(checked_file(Path(evidence) / record["uri"], record["sha256"]))
        for record in records
        if record["kind"] == "artifact"
        and record.get("producer_id") == evaluation["id"]
        and record["role"] == "evaluation episode observation"
    ]
    expected = len(case["topologies"]) * len(case["seeds"])
    if len(observations) != expected or receipt["episodes"] != expected:
        raise ValueError("Initial reference has a different episode count")
    mean = statistics.mean(row["reward"] for row in observations)
    if mean != receipt["mean_return"]:
        raise ValueError("Initial reference summary differs from retained episode observations")
    return {**receipt, "reused": True}


def validate_bundle(path, sidecar, steps):
    """Actually reload all tensors and reject an incompatible or nonfinite policy."""
    import jax
    import numpy as np
    import yaml

    from jaxborg.checkpoint import load_jax_bundle, sidecar_path

    if sidecar_path(path).resolve() != Path(sidecar).resolve():
        raise ValueError("Checkpoint does not use its owned adjacent sidecar")
    bundle = load_jax_bundle(path)
    if bundle.legacy or bundle.backend != "jax" or set(bundle.policies) != {"blue"}:
        raise ValueError("Expected a versioned Blue-only JAX policy bundle")
    blue = bundle.policies["blue"]
    if blue.arch != ARCH or blue.obs_dim != 450 or blue.action_dim != 242 or not blue.trainable:
        raise ValueError("Policy differs from the declared enhanced-observation separate LSTM")
    recipe = yaml.safe_load(Path(sidecar).read_text())
    if recipe.get("cage4_enhanced_obs") is not True or recipe["run"]["total_steps"] != steps:
        raise ValueError("Checkpoint sidecar differs from its declared observation/step contract")
    weights = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(blue.weights):
        array = np.asarray(jax.device_get(leaf))
        if not np.isfinite(array).all():
            raise ValueError("Checkpoint contains nonfinite weights")
        weights.update(str((array.dtype.str, array.shape)).encode())
        weights.update(array.tobytes(order="C"))
    return {
        "arch": blue.arch,
        "obs_dim": blue.obs_dim,
        "action_dim": blue.action_dim,
        "weights_sha256": weights.hexdigest(),
    }


def panel_from_manifest(manifest, stage):
    bank = manifest["topology_banks"]["unseen"]
    if [item["source_seed"] for item in bank["snapshots"]] != [200, 201, 202, 203]:
        raise ValueError("Expected the preserved stock-generator 200..203 panel")
    snapshots = bank["snapshots"][:1] if stage == "smoke" else bank["snapshots"]
    paths = [checked_file(item["path"], item["sha256"]) for item in snapshots]
    reds = manifest["policies"]["red"]
    if {red["id"] for red in reds} != RED_IDS or len(reds) != 4:
        raise ValueError("Expected exactly the four prespecified archived Red policies")
    for red in reds:
        checked_file(red["model"], red["sha256"])
        checked_file(red["sidecar"], red["sidecar_sha256"])
    return paths, snapshots, [] if stage in {"smoke", "interim"} else sorted(reds, key=lambda red: red["id"])


def run(args):
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[: int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))])
    import jax
    import yaml
    from prepare import verify

    from jaxborg.evaluation.capture_store import load_records
    from jaxborg.evaluation.joint_capture import run_matchup
    from jaxborg.evaluation.scripted_capture import run_scripted
    from jaxborg.scenarios.cc4.game_variant import GameVariant

    if os.environ.get("JAX_PLATFORMS") != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("This evaluation must run on CPU")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=PROJECT, text=True)
    if actual != args.expected_revision or len(actual) != 40 or dirty:
        raise ValueError("Evaluation requires clean source at the requested full revision")
    study = args.study_root.resolve()
    verify(study, args.expected_revision)
    evidence = study / "evidence"
    records = load_records(evidence)
    receipt_path = study / "runs" / args.training_trial / "receipt.json"
    attempt_id = resolve_training_attempt(
        records, args.training_trial, actual, explicit=args.training_attempt, receipt_path=receipt_path
    )
    checkpoints = checkpoint_selection(
        records,
        evidence,
        args.training_trial,
        attempt_id,
        target_steps=MILESTONES[args.stage],
        allow_running=args.stage in {"interim", "primary"},
    )
    if args.stage == "smoke":
        if args.training_trial != "smoke-seed42" or checkpoints[-1]["steps"] != 144000:
            raise ValueError("Smoke requires its separate completed three-update trial")
    elif args.training_trial not in {"fsm-seed42", "fsm-seed100"}:
        raise ValueError("Scientific evaluation requires a declared production trial")
    protocol = read_json(evidence / "protocol.json")
    transfer = read_json(args.transfer_manifest)
    transfer_hash = digest(args.transfer_manifest)
    paths, snapshots, reds = panel_from_manifest(transfer, args.stage)
    from jaxborg.evaluation.cia.fixed_topology import canonical_topology_fingerprint
    from jaxborg.scenarios.cc4.topology import load_topology

    prepared = yaml.safe_load((study / "inputs/recipe-full.yaml").read_text())
    prepared_fingerprints = [
        canonical_topology_fingerprint(load_topology(path)) for path in prepared["eval"]["topology_bank"]
    ]
    transfer_fingerprints = [item["fingerprint"] for item in transfer["topology_banks"]["unseen"]["snapshots"]]
    if prepared_fingerprints != transfer_fingerprints:
        raise ValueError("Prepared study test bank differs from the preserved transfer panel")
    seeds = [990001] if args.stage == "smoke" else list(range(21000, 21016))
    evaluation_dir = study / "evaluations" / args.training_trial / args.stage
    if evaluation_dir.exists():
        raise FileExistsError("Evaluation output already exists; retain it and make an explicit retry selection")
    evaluation_dir.mkdir(parents=True)
    reference_dir = study / "evaluations" / args.training_trial / "initial-references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints:
        checkpoint["loaded_contract"] = validate_bundle(checkpoint["model"], checkpoint["sidecar"], checkpoint["steps"])
    if checkpoints[0]["loaded_contract"]["weights_sha256"] == checkpoints[1]["loaded_contract"]["weights_sha256"]:
        raise ValueError("Training left all policy weights unchanged")
    plan = {
        "source_revision": actual,
        "training_attempt": attempt_id,
        "training_trial": args.training_trial,
        "training_status_at_selection": next(record["status"] for record in records if record["id"] == attempt_id),
        "stage": args.stage,
        "checkpoints": checkpoints,
        "topologies": snapshots,
        "seeds": seeds,
        "red_ids": ["fsm", *[red["id"] for red in reds]],
        "transfer_manifest_sha256": transfer_hash,
        "horizon": 500,
        "deterministic": False,
        "backend": "cpu",
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "scripted_role_labels": "CIA instrumentation only; normal stock FSM, topology, Blue actions and reward",
        "selection": "original initialization and exact prespecified milestone; no outcome-based checkpoint choice",
        "interim_only": args.stage == "interim",
    }
    write_json(evaluation_dir / "plan.json", plan)
    write_json(evaluation_dir / "transfer-manifest.json", transfer)
    effective = {
        "algorithm": "ippo",
        "cage4_enhanced_obs": True,
        "train": {"variant": "cc4_stock"},
        "eval": {
            "variant": "cc4_stock",
            "policy_backend": "jax",
            "topology_bank": [str(path) for path in paths],
            "topology_sampling": "exhaustive",
            "cia": {"enabled": True, "metric": "resilience", "role_assignment": "fixed_per_topology"},
        },
    }
    effective_path = evaluation_dir / "effective-recipe.yaml"
    effective_path.write_text(yaml.safe_dump(effective, sort_keys=False))
    selection, summaries = {}, []
    for checkpoint in checkpoints:
        for red in [None, *reds]:
            if digest(args.transfer_manifest) != transfer_hash:
                raise ValueError("Transfer manifest changed during evaluation")
            scripted = red is None
            opponent = "fsm" if scripted else red["id"]
            key = f"{checkpoint['steps']}:{opponent}"
            print(f"START {args.training_trial} {key}", flush=True)
            variant = GameVariant(name="cc4_stock", cage4_enhanced_obs=True, resilience_roles=scripted)
            case = reference_case(checkpoint, variant, snapshots, seeds, actual, red)
            reference_id = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
            reference_path = reference_dir / f"{reference_id}.json"
            if checkpoint["steps"] == 0 and reference_path.exists():
                reused = validate_reference(read_json(reference_path), case, load_records(evidence), evidence)
                selection[key] = reused["evaluation_id"]
                summaries.append(reused)
                write_json(evaluation_dir / f"receipt-0-{opponent}.json", reused)
                print(f"REUSE {args.training_trial} {key} {reused['evaluation_id']}", flush=True)
                continue

            def condition_check(configuration):
                actual_variant = configuration["base_variant" if scripted else "variant"]
                if actual_variant != asdict(variant):
                    raise ValueError("Evaluation resolved a different game/observation contract")
                if configuration["topology"]["sha256"] != [item["sha256"] for item in snapshots]:
                    raise ValueError("Evaluation resolved a different topology panel")
                if configuration["deterministic"] or configuration["episodes_per_seed"] != 1:
                    raise ValueError("Evaluation must use the declared stochastic episode roots")

            if scripted:
                capture, result = run_scripted(
                    checkpoint["model"],
                    context={
                        "root": evidence,
                        "checkpoint": checkpoint["model_artifact"],
                        "training_attempt_id": checkpoint["training_attempt"],
                        "revision": actual,
                        "trial_id": args.training_trial,
                        "source_recipe": checkpoint["sidecar"],
                        "effective_recipe": effective_path,
                        "entrypoint": Path(__file__),
                        "source_roots": (HERE,),
                        "condition_check": condition_check,
                    },
                    return_capture=True,
                    base_variant=variant,
                    topology_paths=paths,
                    reds=["fsm"],
                    seeds=seeds,
                    episodes_per_seed=1,
                    deterministic=False,
                    progress=False,
                    eval_name=None,
                    recipe=effective,
                )
                returns = result[0]["per_episode"]
            else:
                capture, result = run_matchup(
                    checkpoint["model"],
                    checked_file(red["model"], red["sha256"]),
                    root=evidence,
                    protocol=protocol,
                    trial_id=args.training_trial,
                    expected_revision=actual,
                    variant=variant,
                    seeds=seeds,
                    topology_path=paths,
                    condition_check=condition_check,
                    input_artifacts={
                        "blue checkpoint": checkpoint["model_artifact"],
                        "blue sidecar": checkpoint["sidecar_artifact"],
                    },
                    entrypoint=Path(__file__),
                    source_roots=(HERE,),
                    recipe=effective,
                    overrides={"plan_sha256": digest(evaluation_dir / "plan.json"), "opponent": red["id"]},
                )
                returns = result.blue_returns
            if len(returns) != len(paths) * len(seeds) or not all(math.isfinite(value) for value in returns):
                raise ValueError("Evaluator returned missing or nonfinite episodes")
            selection[key] = capture.evaluation_id
            summaries.append(
                {
                    "checkpoint_steps": checkpoint["steps"],
                    "opponent": opponent,
                    "episodes": len(returns),
                    "mean_return": statistics.mean(returns),
                    "evaluation_id": capture.evaluation_id,
                    "case": case,
                    "reused": False,
                }
            )
            write_json(evaluation_dir / f"receipt-{checkpoint['steps']}-{opponent}.json", summaries[-1])
            if checkpoint["steps"] == 0:
                write_json(reference_path, summaries[-1])
            print(json.dumps(summaries[-1]), flush=True)
    write_json(evaluation_dir / "selection.json", selection)
    write_json(evaluation_dir / "summary.json", summaries)
    print(json.dumps({"evaluation_directory": str(evaluation_dir), "selection": selection}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--training-trial", required=True)
    parser.add_argument("--training-attempt")
    parser.add_argument("--transfer-manifest", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--stage", choices=tuple(MILESTONES), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
