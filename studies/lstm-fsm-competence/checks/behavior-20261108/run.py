"""Freeze and execute a small behavior check within the competence study."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY_SOURCE = HERE.parents[1]
PROJECT = STUDY_SOURCE.parents[1]
SPEC = importlib.util.spec_from_file_location("competence_evaluate", STUDY_SOURCE / "evaluate.py")
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)
read_json = EVALUATE.read_json
write_json = EVALUATE.write_json
digest = EVALUATE.digest
checked_file = EVALUATE.checked_file


def source_revision(expected):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=PROJECT, text=True)
    if len(expected) != 40 or actual != expected or dirty:
        raise ValueError("Behavior check requires clean source at the requested full revision")
    return actual


def canonical_cases(study, receipt_path, checkpoint, snapshots, roots, opponent, records, training_revision):
    from jaxborg.scenarios.cc4.game_variant import GameVariant

    receipt = read_json(receipt_path)
    indexed = {record["id"]: record for record in records}
    evaluation = indexed[receipt["evaluation_id"]]
    attempt = indexed[evaluation["attempt_id"]]
    case = receipt["case"]
    if evaluation["status"] != "succeeded" or attempt["status"] != "succeeded":
        raise ValueError("Canonical evaluation must have succeeded")
    if (
        case["checkpoint"] != checkpoint["model_artifact"]
        or case["checkpoint_sha256"] != checkpoint["model_sha256"]
        or case["sidecar_sha256"] != checkpoint["sidecar_sha256"]
        or case["red_id"] != opponent
        or case["topologies"] != [item["sha256"] for item in snapshots]
    ):
        raise ValueError("Canonical receipt differs from the selected policy or topology")
    directory = study / "evidence/attempts" / attempt["id"].rsplit(":", 1)[-1]
    configuration = read_json(directory / "configuration.json")
    recorded_roots = configuration["seeds"] if opponent == "fsm" else configuration["seed_roles"]["episode_roots"]
    if recorded_roots != case["seeds"] or configuration["topology"]["sha256"] != case["topologies"]:
        raise ValueError("Canonical configuration differs from its receipt")
    scripted = opponent == "fsm"
    expected_variant = asdict(GameVariant(name="cc4_stock", cage4_enhanced_obs=True, resilience_roles=scripted))
    expected_procedure = (
        "jaxborg.evaluation.jax_scripted_red.evaluate_jax_scripted_reds"
        if scripted
        else "jaxborg.evaluation.matchup_runner.evaluate_matchup"
    )
    if (
        configuration["backend"] != "jax"
        or configuration["procedure"] != expected_procedure
        or configuration["deterministic"] is not False
        or configuration["episodes_per_seed"] != 1
        or configuration["base_variant" if scripted else "variant"] != expected_variant
        or case["variant"] != expected_variant
        or configuration["execution"]["source_revision"] != training_revision
        or case["source_revision"] != training_revision
    ):
        raise ValueError("Canonical evaluation has another scientific or source contract")
    inputs = read_json(directory / "input-artifacts.json")
    expected_inputs = {
        "blue checkpoint": checkpoint["model_sha256"],
        "source recipe" if scripted else "blue sidecar": checkpoint["sidecar_sha256"],
    }
    expected_inputs.update({f"topology {i}": item["sha256"] for i, item in enumerate(snapshots)})
    if not scripted:
        expected_inputs.update({"red checkpoint": case["red_sha256"], "red sidecar": case["red_sidecar_sha256"]})
    if inputs["blue checkpoint"] != checkpoint["model_artifact"]:
        raise ValueError("Canonical evaluation does not consume the owned checkpoint")
    for role, expected_hash in expected_inputs.items():
        artifact = indexed[inputs[role]]
        if artifact["sha256"] != expected_hash or artifact["id"] not in evaluation["inputs"]:
            raise ValueError("Canonical retained input differs from its receipt")
    artifacts = {record.get("uri"): record for record in records if record["kind"] == "artifact"}
    result = []
    for topology_index, snapshot in enumerate(snapshots):
        for root in roots:
            index = topology_index * len(recorded_roots) + recorded_roots.index(root)
            path = directory / "episodes" / f"{index}.json"
            record = artifacts[str(path.relative_to(study / "evidence"))]
            episode = read_json(checked_file(path, record["sha256"]))
            if episode["seed"] != root or episode["episode_index"] != index:
                raise ValueError("Canonical episode index/root mismatch")
            result.append(
                {
                    "topology_index": topology_index,
                    "topology_sha256": snapshot["sha256"],
                    "episode_root": root,
                    "canonical_episode_index": index,
                    "canonical_episode_artifact": record["id"],
                    "canonical_episode_sha256": record["sha256"],
                    "canonical_return": episode["reward"],
                }
            )
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": digest(receipt_path),
        "evaluation_id": receipt["evaluation_id"],
        "case": case,
        "episodes": result,
    }


def prepare(args):
    from jaxborg.evaluation.capture_store import load_records

    source_revision(args.expected_revision)
    protocol = read_json(HERE / "protocol.json")
    study = args.study_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(PROJECT) or not output.is_relative_to(study):
        raise ValueError("Output must be inside the existing external study and outside source")
    if output.exists():
        raise FileExistsError("Retain existing checks and choose a fresh output directory")
    evidence = study / "evidence"
    records = load_records(evidence)
    trial = protocol["training_trial"]
    owner = EVALUATE.resolve_training_attempt(
        records, trial, protocol["training_source_revision"], receipt_path=study / "runs" / trial / "receipt.json"
    )
    transfer = read_json(args.transfer_manifest)
    _, snapshots, reds = EVALUATE.panel_from_manifest(transfer, "final")
    red = next(item for item in reds if item["id"] == "red-diverse-100-final")
    cells = []
    for steps in protocol["checkpoint_steps"]:
        checkpoint = EVALUATE.checkpoint_selection(records, evidence, trial, owner, target_steps=steps)[-1]
        stage = "primary" if steps == 49968000 else "final"
        for opponent in protocol["opponents"]:
            reference = canonical_cases(
                study,
                study / f"evaluations/{trial}/{stage}/receipt-{steps}-{opponent}.json",
                checkpoint,
                snapshots,
                protocol["episode_roots"],
                opponent,
                records,
                protocol["training_source_revision"],
            )
            if opponent != "fsm" and (
                reference["case"]["red_sha256"] != red["sha256"]
                or reference["case"]["red_sidecar_sha256"] != red["sidecar_sha256"]
            ):
                raise ValueError("Canonical learned opponent differs from the frozen Red")
            cells.append(
                {"id": f"{steps}-{opponent}", "checkpoint": checkpoint, "opponent": opponent, "reference": reference}
            )
    plan = {
        "schema_version": 1,
        "source_revision": args.expected_revision,
        "study_root": str(study),
        "protocol": protocol,
        "protocol_sha256": digest(HERE / "protocol.json"),
        "helper_sha256": digest(__file__),
        "training_attempt": owner,
        "topologies": snapshots,
        "red": red,
        "cells": cells,
        "transfer_manifest_sha256": digest(args.transfer_manifest),
    }
    write_json(output / "plan.json", plan)
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"plan": str(output / "plan.json"), "cells": [c["id"] for c in cells], "episodes": 16}))


def run_cell(args):
    source_revision(args.expected_revision)
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("Set JAX_PLATFORMS=cpu before starting Python")
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[: int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))])
    import jax
    import yaml

    from jaxborg.evaluation.joint_capture import run_matchup
    from jaxborg.evaluation.scripted_capture import run_scripted
    from jaxborg.scenarios.cc4.game_variant import GameVariant

    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("Behavior check requires CPU")
    plan = read_json(args.plan)
    if plan["source_revision"] != args.expected_revision or plan["helper_sha256"] != digest(__file__):
        raise ValueError("Frozen check selected another implementation")
    cell = next(cell for cell in plan["cells"] if cell["id"] == args.cell)
    output = args.plan.resolve().parent / "cells" / cell["id"]
    output.mkdir(parents=True, exist_ok=False)
    study = Path(plan["study_root"])
    evidence = study / "evidence"
    checkpoint = cell["checkpoint"]
    for role in ("model", "sidecar"):
        checked_file(checkpoint[role], checkpoint[f"{role}_sha256"])
    EVALUATE.validate_bundle(checkpoint["model"], checkpoint["sidecar"], checkpoint["steps"])
    checked_file(cell["reference"]["receipt_path"], cell["reference"]["receipt_sha256"])
    paths = [checked_file(item["path"], item["sha256"]) for item in plan["topologies"]]
    scripted = cell["opponent"] == "fsm"
    variant = GameVariant(name="cc4_stock", cage4_enhanced_obs=True, resilience_roles=scripted)
    effective = {
        "algorithm": "ippo",
        "cage4_enhanced_obs": True,
        "train": {"variant": "cc4_stock"},
        "eval": {
            "variant": "cc4_stock",
            "policy_backend": "jax",
            "topology_bank": [str(p) for p in paths],
            "topology_sampling": "exhaustive",
            "cia": {"enabled": True, "metric": "resilience", "role_assignment": "fixed_per_topology"},
        },
    }
    effective_path = output / "effective-recipe.yaml"
    effective_path.write_text(yaml.safe_dump(effective, sort_keys=False))
    common = {
        "seeds": plan["protocol"]["episode_roots"],
        "episodes_per_seed": 1,
        "deterministic": False,
        "trajectories": True,
        "recipe": effective,
    }
    if scripted:
        capture, result = run_scripted(
            checkpoint["model"],
            context={
                "root": evidence,
                "checkpoint": checkpoint["model_artifact"],
                "training_attempt_id": checkpoint["training_attempt"],
                "revision": args.expected_revision,
                "trial_id": plan["protocol"]["training_trial"],
                "source_recipe": checkpoint["sidecar"],
                "effective_recipe": effective_path,
                "entrypoint": Path(__file__),
                "source_roots": (STUDY_SOURCE,),
            },
            return_capture=True,
            base_variant=variant,
            topology_paths=paths,
            reds=["fsm"],
            progress=False,
            eval_name=None,
            **common,
        )
        returns = result[0]["per_episode"]
    else:
        red = plan["red"]
        checked_file(red["sidecar"], red["sidecar_sha256"])
        capture, result = run_matchup(
            checkpoint["model"],
            checked_file(red["model"], red["sha256"]),
            root=evidence,
            protocol=read_json(evidence / "protocol.json"),
            trial_id=plan["protocol"]["training_trial"],
            expected_revision=args.expected_revision,
            variant=variant,
            topology_path=paths,
            input_artifacts={
                "blue checkpoint": checkpoint["model_artifact"],
                "blue sidecar": checkpoint["sidecar_artifact"],
            },
            entrypoint=Path(__file__),
            source_roots=(STUDY_SOURCE,),
            overrides={"behavior_plan_sha256": digest(args.plan), "cell": cell["id"]},
            **common,
        )
        returns = result.blue_returns
    comparisons = [
        {**case, "traced_return": float(value), "matches_canonical": float(value) == case["canonical_return"]}
        for case, value in zip(cell["reference"]["episodes"], returns, strict=True)
    ]
    receipt = {
        "cell": cell["id"],
        "evaluation_id": capture.evaluation_id,
        "plan_sha256": digest(args.plan),
        "episodes": comparisons,
        "canonical_parity": all(case["matches_canonical"] for case in comparisons),
    }
    write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt))
    if not receipt["canonical_parity"]:
        raise ValueError("Trace differs from canonical returns; discrepancy retained, behavior interpretation withheld")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--study-root", type=Path, required=True)
    prepare_parser.add_argument("--transfer-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run-cell")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--cell", required=True)
    for command in (prepare_parser, run_parser):
        command.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run_cell)(args)


if __name__ == "__main__":
    main()
