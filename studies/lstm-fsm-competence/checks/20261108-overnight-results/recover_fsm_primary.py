"""Recover only missing 49.968M cells using the unchanged pinned matchup API."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def setup(args):
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    if os.environ.get("JAX_PLATFORMS", "cpu") != "cpu" or os.environ.get("JAXBORG_EVAL_BATCH_SIZE", "1") != "1":
        raise ValueError("Recovery preserves CPU evaluation with batch size one")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAXBORG_EVAL_BATCH_SIZE"] = "1"
    os.chdir(args.source_root)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    if actual != args.expected_revision or dirty:
        raise ValueError("Recovery requires the clean pinned checkout")
    sys.path.insert(0, str(args.source_root / "src"))
    spec = importlib.util.spec_from_file_location(
        "pinned_evaluator", args.source_root / "studies/lstm-fsm-competence/evaluate.py"
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


def validate_receipt(receipt, case, records, evidence):
    """Verify successful cells by immutable records, inputs, configuration and raw episodes."""
    if receipt["case"] != case:
        raise ValueError("Completed receipt differs from the exact recovery case")
    indexed = {record["id"]: record for record in records}
    evaluation = indexed[receipt["evaluation_id"]]
    attempt = indexed[evaluation["attempt_id"]]
    if evaluation["kind"] != "evaluation" or evaluation["status"] != "succeeded" or attempt["status"] != "succeeded":
        raise ValueError("Only successful immutable evaluations can be reused")
    directory = evidence / "attempts" / attempt["id"].rsplit(":", 1)[-1]
    config = read(directory / "configuration.json")
    inputs = read(directory / "input-artifacts.json")
    scripted = case["red_id"] == "fsm"
    if (
        config["base_variant" if scripted else "variant"] != case["variant"]
        or config["topology"]["sha256"] != case["topologies"]
        or list(config["seeds"] if scripted else config["seed_roles"]["episode_roots"]) != case["seeds"]
        or config["execution"]["source_revision"] != case["source_revision"]
        or config["deterministic"] is not False
        or config["episodes_per_seed"] != 1
        or inputs["blue checkpoint"] != case["checkpoint"]
        or indexed[inputs["blue checkpoint"]]["sha256"] != case["checkpoint_sha256"]
        or indexed[inputs["source recipe" if scripted else "blue sidecar"]]["sha256"] != case["sidecar_sha256"]
    ):
        raise ValueError("Recorded evaluator inputs or procedure differ")
    if scripted:
        if config["reds"] != ["fsm"]:
            raise ValueError("Recorded scripted opponent differs")
    elif (indexed[inputs["red checkpoint"]]["sha256"], indexed[inputs["red sidecar"]]["sha256"]) != (
        case["red_sha256"],
        case["red_sidecar_sha256"],
    ):
        raise ValueError("Recorded learned opponent differs")
    observations = []
    for record in records:
        if (
            record["kind"] == "artifact"
            and record.get("producer_id") == evaluation["id"]
            and record["role"] == "evaluation episode observation"
        ):
            path = evidence / record["uri"]
            if sha(path) != record["sha256"]:
                raise ValueError("Episode artifact changed")
            observations.append(read(path))
    expected = len(case["topologies"]) * len(case["seeds"])
    if (
        len(observations) != expected
        or receipt["episodes"] != expected
        or statistics.mean(row["reward"] for row in observations) != receipt["mean_return"]
    ):
        raise ValueError("Receipt summary differs from retained complete episode observations")
    return receipt


def prepare(args):
    evaluator = setup(args)
    from jaxborg.evaluation.capture_store import load_records
    from jaxborg.scenarios.cc4.game_variant import GameVariant

    if args.recovery_root.exists():
        raise FileExistsError("Recovery output already exists; inspect its explicit plan")
    evidence = args.study_root / "evidence"
    records = load_records(evidence)
    indexed = {record["id"]: record for record in records}
    transfer = read(args.transfer_manifest)
    _, snapshots, reds = evaluator.panel_from_manifest(transfer, "primary")
    seeds = list(range(21000, 21016))
    plan = {
        "source_revision": args.expected_revision,
        "orchestrator_sha256": sha(__file__),
        "orchestrator_source": Path(__file__).read_text(),
        "transfer_manifest_sha256": sha(args.transfer_manifest),
        "existing": {},
        "missing": [],
        "original_stage_plans": {},
        "resources": {
            "backend": "cpu",
            "slurm_partition": "community",
            "cpus": 8,
            "memory_gib": 64,
            "execution": "one fresh process per missing cell, globally sequential",
        },
    }
    for trial in ("fsm-seed42", "fsm-seed100"):
        owner_id = read(args.study_root / "runs" / trial / "receipt.json")["attempt_id"]
        evaluator.resolve_training_attempt(records, trial, args.expected_revision, explicit=owner_id)
        checkpoints = evaluator.checkpoint_selection(records, evidence, trial, owner_id, target_steps=49968000)
        old_dir = args.study_root / "evaluations" / trial / "primary"
        old_plan = read(old_dir / "plan.json")
        if (
            old_plan["source_revision"] != args.expected_revision
            or old_plan["training_attempt"] != owner_id
            or old_plan["transfer_manifest_sha256"] != plan["transfer_manifest_sha256"]
        ):
            raise ValueError("Primary plan identity differs")
        plan["original_stage_plans"][trial] = {"path": str(old_dir / "plan.json"), "sha256": sha(old_dir / "plan.json")}
        for checkpoint in checkpoints:
            evaluator.validate_bundle(checkpoint["model"], checkpoint["sidecar"], checkpoint["steps"])
            for red in [None, *reds]:
                opponent = "fsm" if red is None else red["id"]
                variant = GameVariant(name="cc4_stock", cage4_enhanced_obs=True, resilience_roles=red is None)
                case = evaluator.reference_case(checkpoint, variant, snapshots, seeds, args.expected_revision, red)
                key = f"{trial}:{checkpoint['steps']}:{opponent}"
                receipt_path = old_dir / f"receipt-{checkpoint['steps']}-{opponent}.json"
                if receipt_path.exists():
                    receipt = validate_receipt(read(receipt_path), case, records, evidence)
                    plan["existing"][key] = {
                        "receipt_path": str(receipt_path),
                        "receipt_sha256": sha(receipt_path),
                        "receipt": receipt,
                    }
                    continue
                if checkpoint["steps"] != 49968000 or red is None:
                    raise ValueError("Unexpected missing initial or FSM cell; stop instead of broadening recovery")
                interrupted = []
                for record in records:
                    if record["kind"] != "evaluation" or record["status"] != "running":
                        continue
                    attempt = indexed[record["attempt_id"]]
                    roles = record["facts"]["input_artifacts"]["value"]
                    if (
                        attempt["trial_id"] == trial
                        and roles.get("blue checkpoint") == checkpoint["model_artifact"]
                        and indexed[roles["red checkpoint"]]["sha256"] == red["sha256"]
                    ):
                        if record["budgets"][0]["completed"] != 0:
                            raise ValueError(
                                "A missing cell contains completed episodes; requires narrower episode recovery"
                            )
                        interrupted.append(attempt["id"])
                if len(interrupted) > 1:
                    raise ValueError("Ambiguous interrupted predecessor")
                effective_path = old_dir / "effective-recipe.yaml"
                plan["missing"].append(
                    {
                        "key": key,
                        "trial": trial,
                        "opponent": opponent,
                        "checkpoint": checkpoint,
                        "red": red,
                        "case": case,
                        "topologies": snapshots,
                        "interrupted_predecessor": interrupted[0] if interrupted else None,
                        "effective_recipe": {"path": str(effective_path), "sha256": sha(effective_path)},
                    }
                )
    if len(plan["existing"]) != 14 or len(plan["missing"]) != 6:
        raise ValueError("Coverage differs from reviewed 14 completed plus 6 missing cells")
    write(args.recovery_root / "plan.json", plan)
    (args.recovery_root / "recover_fsm_primary.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "verified_existing_cells": 14,
                "missing_cells": 6,
                "missing_episodes": 384,
                "plan": str(args.recovery_root / "plan.json"),
            }
        ),
        flush=True,
    )


def cell(args):
    evaluator = setup(args)
    import yaml

    from jaxborg.evaluation.capture_store import load_records
    from jaxborg.evaluation.joint_capture import run_matchup
    from jaxborg.scenarios.cc4.game_variant import GameVariant

    plan = read(args.recovery_root / "plan.json")
    if sha(__file__) != plan["orchestrator_sha256"] or sha(args.transfer_manifest) != plan["transfer_manifest_sha256"]:
        raise ValueError("Recovery source or frozen manifest changed")
    item = next(item for item in plan["missing"] if item["key"] == args.cell)
    evidence = args.study_root / "evidence"
    records = load_records(evidence)
    checkpoint = item["checkpoint"]
    evaluator.validate_bundle(checkpoint["model"], checkpoint["sidecar"], 49968000)
    red = item["red"]
    blue_model = evaluator.checked_file(checkpoint["model"], checkpoint["model_sha256"])
    red_model = evaluator.checked_file(red["model"], red["sha256"])
    evaluator.checked_file(red["sidecar"], red["sidecar_sha256"])
    topologies = [evaluator.checked_file(value["path"], value["sha256"]) for value in item["topologies"]]
    recipe_path = evaluator.checked_file(item["effective_recipe"]["path"], item["effective_recipe"]["sha256"])
    recipe = yaml.safe_load(recipe_path.read_text())
    receipt_path = args.recovery_root / "cells" / f"{item['trial']}-{item['opponent']}.json"
    if receipt_path.exists():
        validate_receipt(read(receipt_path), item["case"], records, evidence)
        print(f"REUSE successful recovery cell {item['key']}", flush=True)
        return
    indexed = {record["id"]: record for record in records}
    matches = []
    for evaluation in records:
        if evaluation["kind"] != "evaluation":
            continue
        roles = evaluation["facts"].get("input_artifacts", {}).get("value", {})
        if (
            roles.get("blue checkpoint") != checkpoint["model_artifact"]
            or indexed.get(roles.get("red checkpoint"), {}).get("sha256") != red["sha256"]
        ):
            continue
        if evaluation["attempt_id"] == item["interrupted_predecessor"]:
            continue
        matches.append(evaluation)
    if matches:
        if len(matches) != 1 or matches[0]["status"] != "succeeded":
            raise RuntimeError("A recovery attempt exists without its receipt; inspect before another execution")
        evaluation = matches[0]
        mean = next(metric["value"] for metric in evaluation["metrics"] if metric["name"] == "blue_return")
        receipt = {
            "checkpoint_steps": 49968000,
            "opponent": item["opponent"],
            "episodes": 64,
            "mean_return": mean,
            "evaluation_id": evaluation["id"],
            "case": item["case"],
            "reused": True,
            "recovery_of": item["interrupted_predecessor"],
        }
        validate_receipt(receipt, item["case"], records, evidence)
        write(receipt_path, receipt)
        print(f"RECOVER successful recorded receipt {item['key']}", flush=True)
        return

    def condition_check(config):
        if (
            config["variant"] != item["case"]["variant"]
            or config["topology"]["sha256"] != item["case"]["topologies"]
            or config["seed_roles"]["episode_roots"] != item["case"]["seeds"]
            or config["deterministic"]
            or config["episodes_per_seed"] != 1
        ):
            raise ValueError("Resolved recovery case differs from primary protocol")

    capture, result = run_matchup(
        blue_model,
        red_model,
        root=evidence,
        protocol=read(evidence / "protocol.json"),
        trial_id=item["trial"],
        expected_revision=args.expected_revision,
        variant=GameVariant(**item["case"]["variant"]),
        seeds=item["case"]["seeds"],
        topology_path=topologies,
        condition_check=condition_check,
        input_artifacts={
            "blue checkpoint": checkpoint["model_artifact"],
            "blue sidecar": checkpoint["sidecar_artifact"],
        },
        recipe=recipe,
        retry_of=None,
        entrypoint=None,
        source_roots=(args.source_root / "studies/lstm-fsm-competence",),
        overrides={
            "recovery_plan_sha256": sha(args.recovery_root / "plan.json"),
            "external_orchestration": {
                "path": str(Path(__file__).resolve()),
                "sha256": plan["orchestrator_sha256"],
                "source": plan["orchestrator_source"],
            },
            "resources": plan["resources"],
            "recovery_cell": item["key"],
            "interrupted_predecessor": item["interrupted_predecessor"],
            "predecessor_status": "old attempt unsealed; scheduler reported signal 11 and LLVM allocation failure"
            if item["interrupted_predecessor"]
            else None,
        },
    )
    if len(result.blue_returns) != 64 or not all(math.isfinite(value) for value in result.blue_returns):
        raise ValueError("Recovery returned incomplete or nonfinite episodes")
    receipt = {
        "checkpoint_steps": 49968000,
        "opponent": item["opponent"],
        "episodes": 64,
        "mean_return": statistics.mean(result.blue_returns),
        "evaluation_id": capture.evaluation_id,
        "case": item["case"],
        "reused": False,
        "recovery_of": item["interrupted_predecessor"],
    }
    write(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


def run(args):
    import fcntl

    if (
        os.environ.get("SLURM_JOB_PARTITION") != "community"
        or not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_GPUS")
        or os.environ.get("SLURM_GPUS_ON_NODE")
    ):
        raise ValueError("Recovery requires a community CPU-only Slurm allocation")
    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get("JAXBORG_EVAL_BATCH_SIZE") != "1":
        raise ValueError("Recovery must preserve CPU batch size one")
    lock = (args.study_root / ".primary-recovery.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime(2026, 11, 9, 13, 0, tzinfo=timezone.utc).timestamp() - 300
    plan = read(args.recovery_root / "plan.json")
    if sha(__file__) != plan["orchestrator_sha256"]:
        raise ValueError("Recovery helper changed after planning")
    base = [
        str(args.source_root / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        "--source-root",
        str(args.source_root),
        "--study-root",
        str(args.study_root),
        "--recovery-root",
        str(args.recovery_root),
        "--transfer-manifest",
        str(args.transfer_manifest),
        "--expected-revision",
        args.expected_revision,
    ]
    for item in plan["missing"]:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Recovery resource window has ended")
        subprocess.run([*base, "cell", "--cell", item["key"]], check=True, timeout=remaining)
    subprocess.run([*base, "assemble"], check=True, timeout=max(1, deadline - time.time()))


def assemble(args):
    setup(args)
    from jaxborg.evaluation.capture_store import load_records

    plan = read(args.recovery_root / "plan.json")
    evidence = args.study_root / "evidence"
    records = load_records(evidence)
    selection = {key: item["receipt"]["evaluation_id"] for key, item in plan["existing"].items()}
    summaries = {key: item["receipt"] for key, item in plan["existing"].items()}
    for item in plan["existing"].values():
        if sha(item["receipt_path"]) != item["receipt_sha256"]:
            raise ValueError("Original completed receipt changed after recovery planning")
        validate_receipt(item["receipt"], item["receipt"]["case"], records, evidence)
    for item in plan["missing"]:
        receipt = read(args.recovery_root / "cells" / f"{item['trial']}-{item['opponent']}.json")
        validate_receipt(receipt, item["case"], records, evidence)
        selection[item["key"]] = receipt["evaluation_id"]
        summaries[item["key"]] = receipt
    write(args.recovery_root / "selection.json", selection)
    write(args.recovery_root / "summary.json", summaries)
    print(
        json.dumps({"recovery_root": str(args.recovery_root), "selected_cells": len(selection), "new_episodes": 384}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--transfer-manifest", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("phase", choices=("prepare", "run", "cell", "assemble"))
    parser.add_argument("--cell")
    args = parser.parse_args()
    if args.phase == "cell" and not args.cell:
        parser.error("--cell is required")
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
