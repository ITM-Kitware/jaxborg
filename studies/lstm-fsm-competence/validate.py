"""Accept a recorded smoke only after checking budgets, inputs and learned weights."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import yaml
from prepare import sha256, verify, write_json


def compare_parameters(initial, final):
    if set(initial) != set(final):
        raise ValueError("Initial and final checkpoint tensor keys differ")
    changed = []
    for name in initial:
        left, right = initial[name], final[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Checkpoint tensor contract changed: {name}")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        if not np.array_equal(left, right):
            changed.append(name)
    groups = {}
    for role in ("actor", "critic"):
        names = [
            name
            for name in initial
            if f"{role}_trunk/" in name and "OptimizedLSTMCell" in name and name.endswith("/kernel")
        ]
        updated = sorted(set(names) & set(changed))
        if not names or not updated:
            raise ValueError(f"No changed {role} LSTM kernel found")
        groups[role] = {"kernel_count": len(names), "changed_kernels": updated}
    return {"changed_tensor_count": len(changed), "total_tensor_count": len(initial), "recurrent_trunks": groups}


def read_checkpoint(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as stream:
        metadata = json.loads(stream.metadata()["jaxborg_bundle"])
        if metadata["schema_version"] != 1 or metadata["backend"] != "jax" or set(metadata["policies"]) != {"blue"}:
            raise ValueError("Expected a version-1 JAX Blue-only bundle")
        blue = metadata["policies"]["blue"]
        expected = {
            "name": "recurrent",
            "cell": "lstm",
            "trunk": "separate",
            "hidden_dim": 256,
            "hidden_layers": 1,
            "activation": "tanh",
        }
        if blue["arch"] != expected or not blue["trainable"] or blue["obs_dim"] != 450 or blue["action_dim"] != 242:
            raise ValueError("Checkpoint does not match the declared separate-LSTM Blue contract")
        tensors = {name: stream.get_tensor(name) for name in stream.keys()}
    if not tensors or any(not name.startswith("blue/") for name in tensors):
        raise ValueError("Unexpected checkpoint policy namespace")
    return metadata, tensors


def validate_smoke(root, expected_revision):
    from jaxborg.evaluation.capture_store import load_records

    root = Path(root).resolve()
    verify(root, expected_revision)
    run_dir = root / "runs/smoke-seed42"
    execution = json.loads((run_dir / "execution.json").read_text())
    receipt = json.loads((run_dir / "receipt.json").read_text())
    if execution["returncode"] != 0 or execution["interrupted"] or execution["receipt"] != receipt:
        raise ValueError("Smoke process did not complete successfully")
    evidence = root / "evidence"
    if Path(receipt["evidence_root"]).resolve() != evidence:
        raise ValueError("Unexpected smoke evidence root")
    records = load_records(evidence)
    indexed = {record["id"]: record for record in records}
    attempt = indexed[receipt["attempt_id"]]
    if (
        attempt["kind"] != "attempt"
        or attempt["execution_kind"] != "training"
        or attempt["status"] != "succeeded"
        or attempt["trial_id"] != "smoke-seed42"
    ):
        raise ValueError("Expected successful smoke training attempt")
    budgets = {item["unit"]: item["completed"] for item in attempt["budgets"]}
    if budgets != {
        "environment_steps": 144000,
        "joint_rollout_updates": 3,
        "blue_policy_updates": 3,
        "red_policy_updates": 0,
    }:
        raise ValueError(f"Unexpected completed smoke budgets: {budgets}")
    config = attempt["facts"]["resolved_config"]["value"]
    if config["trainable_teams"] != ["blue"] or config["opponents"] != {
        "red": {"kind": "scripted", "agent": "finite_state", "trainable": False}
    }:
        raise ValueError("Unexpected trainable teams or opponent")
    variant = config["variant"]
    for key, value in {
        "name": "cc4_stock",
        "red_agent": "finite_state",
        "num_steps": 500,
        "red_reward": "zero_sum",
        "blue_block_policy": "cc4",
        "cage4_enhanced_obs": True,
        "blue_observation_version": 2,
        "resilience_roles": False,
        "op_zone_servers": None,
    }.items():
        if variant[key] != value:
            raise ValueError(f"Unexpected training game contract: {key}")
    roles = attempt["facts"]["input_artifacts"]["value"]
    recipe = yaml.safe_load((root / "inputs/recipe-smoke.yaml").read_text())
    training_bank = recipe["train"]["topology_bank"]
    if len(training_bank) != 100 or sum(name.startswith("training topology ") for name in roles) != 100:
        raise ValueError("Expected the complete 100-topology training bank")
    for index, path in enumerate(training_bank):
        artifact = indexed[roles[f"training topology {index}"]]
        if artifact["sha256"] != sha256(path):
            raise ValueError("Captured training topology differs from prepared inputs")
    checkpoints = {}
    weights = []
    for label, fact_name, steps in (("initial", "initial_checkpoint", 0), ("final", "checkpoint_144000", 144000)):
        checkpoint = attempt["facts"][fact_name]["value"]
        if checkpoint["environment_steps"] != steps:
            raise ValueError("Unexpected checkpoint step count")
        artifact = indexed[checkpoint["model"]]
        if artifact["producer_id"] != attempt["id"]:
            raise ValueError("Checkpoint is not owned by the training attempt")
        metadata, tensors = read_checkpoint(evidence / artifact["uri"])
        if metadata["provenance"]["total_steps"] != steps or metadata["provenance"]["seed"] != 42:
            raise ValueError("Checkpoint provenance differs from the smoke")
        sidecar = indexed[checkpoint["sidecar"]]
        if sidecar["producer_id"] != attempt["id"]:
            raise ValueError("Checkpoint sidecar is not training-owned")
        checkpoints[label] = {
            "model_id": artifact["id"],
            "model_sha256": artifact["sha256"],
            "sidecar_id": sidecar["id"],
            "sidecar_sha256": sidecar["sha256"],
            "environment_steps": steps,
        }
        weights.append(tensors)
    parameter_check = compare_parameters(*weights)
    updates = sorted(
        (
            json.loads((evidence / item["uri"]).read_text())
            for item in records
            if item["kind"] == "artifact"
            and item["producer_id"] == attempt["id"]
            and item["role"] == "training update observation"
        ),
        key=lambda row: row["update_idx"],
    )
    if len(updates) != 3:
        raise ValueError("Missing smoke update rows")
    for row in updates:
        if any(isinstance(value, (float, int)) and not math.isfinite(value) for value in row.values()):
            raise ValueError("Nonfinite smoke training metric")
    warm = [updates[index]["wall_time_s"] - updates[index - 1]["wall_time_s"] for index in (1, 2)]
    if any(value <= 0 for value in warm):
        raise ValueError("Invalid synchronized update timing")
    median = statistics.median(warm)
    result = {
        "schema_version": 1,
        "passed": True,
        "source_revision": expected_revision,
        "manifest_sha256": sha256(root / "inputs/manifest.json"),
        "attempt_id": attempt["id"],
        "budgets": budgets,
        "checkpoints": checkpoints,
        "parameter_check": parameter_check,
        "timing": {
            "first_update_seconds_including_compilation": updates[0]["wall_time_s"],
            "warm_update_seconds": warm,
            "warm_median_seconds": median,
            "estimated_full_training_loop_seconds": updates[0]["wall_time_s"] + median * 1040,
            "limit": (
                "Two warm samples; estimate excludes preparation and additional production checkpoint retention. "
                "No performance-improvement requirement is imposed after three updates."
            ),
        },
    }
    target = run_dir / "acceptance.json"
    if target.exists():
        if json.loads(target.read_text()) != result:
            raise ValueError("Refusing to replace a different smoke acceptance result")
    else:
        write_json(target, result)
    print(json.dumps(result))
    return result


def validate_cpu_smoke(root, acceptance):
    from jaxborg.evaluation.capture_store import load_records

    root = Path(root).resolve()
    directory = root / "evaluations/smoke-seed42/smoke"
    plan = json.loads((directory / "plan.json").read_text())
    summaries = json.loads((directory / "summary.json").read_text())
    selection = json.loads((directory / "selection.json").read_text())
    if (
        plan["training_attempt"] != acceptance["attempt_id"]
        or plan["source_revision"] != acceptance["source_revision"]
        or plan["backend"] != "cpu"
        or plan["horizon"] != 500
        or plan["stage"] != "smoke"
        or set(selection) != {"0:fsm", "144000:fsm"}
        or len(summaries) != 2
    ):
        raise ValueError("CPU smoke does not match the accepted training attempt and cases")
    indexed = {item["id"]: item for item in load_records(root / "evidence")}
    seen = set()
    for summary in summaries:
        key = f"{summary['checkpoint_steps']}:{summary['opponent']}"
        if (
            key not in selection
            or key in seen
            or summary["evaluation_id"] != selection[key]
            or summary["episodes"] != 1
            or not math.isfinite(summary["mean_return"])
        ):
            raise ValueError("CPU smoke requires two finite initial/final FSM returns")
        seen.add(key)
        evaluation = indexed[summary["evaluation_id"]]
        if evaluation["kind"] != "evaluation" or evaluation["status"] != "succeeded":
            raise ValueError("CPU smoke evaluation did not complete successfully")
        if evaluation["budgets"][0]["completed"] != 1:
            raise ValueError("CPU smoke evaluation lacks its expected completed episode")
    ready = {
        "passed": True,
        "source_revision": acceptance["source_revision"],
        "manifest_sha256": acceptance["manifest_sha256"],
        "training_attempt": acceptance["attempt_id"],
        "cpu_smoke_selection": selection,
        "cpu_smoke_summary_sha256": sha256(directory / "summary.json"),
        "cpu_smoke_selection_sha256": sha256(directory / "selection.json"),
    }
    target = root / "runs/smoke-seed42/production-ready.json"
    if target.exists():
        if json.loads(target.read_text()) != ready:
            raise ValueError("Refusing to replace a different production acceptance result")
    else:
        write_json(target, ready)
    print(json.dumps(ready))
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--require-cpu-eval", action="store_true")
    args = parser.parse_args()
    acceptance = validate_smoke(args.study_root, args.expected_revision)
    if args.require_cpu_eval:
        validate_cpu_smoke(args.study_root, acceptance)


if __name__ == "__main__":
    main()
