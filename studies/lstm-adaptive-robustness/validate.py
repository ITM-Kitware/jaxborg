"""Verify frozen weights, exact continuation, and paired attacker initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from prepare import read_json, sha256, verify, write_json

SMOKE = "smoke-blue42-primary-red17"
RESUME = "smoke-resume-blue42-primary-red17"
TIMING = "timing-blue42-primary-red17"


def save_once(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Refusing to replace a different validation result: {path}")
    else:
        write_json(path, value)


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, array in sorted(tensors.items()):
        digest.update(json.dumps([name, array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def checkpoint_tensors(path, *, joint):
    from safetensors import safe_open

    from jaxborg.learned_red import RED_OBS_SIZE, RED_POLICY_ACTION_DIM

    with safe_open(str(path), framework="np") as stream:
        metadata = json.loads(stream.metadata()["jaxborg_bundle"])
        teams = {"blue", "red"} if joint else {"blue"}
        if metadata["schema_version"] != 1 or metadata["backend"] != "jax" or set(metadata["policies"]) != teams:
            raise ValueError("Unexpected checkpoint bundle contract")
        tensors = {
            team: {name: stream.get_tensor(name) for name in stream.keys() if name.startswith(team + "/")}
            for team in teams
        }
        if sum(map(len, tensors.values())) != len(stream.keys()):
            raise ValueError("Unexpected tensor namespace")
    if any(not group or any(not np.isfinite(array).all() for array in group.values()) for group in tensors.values()):
        raise ValueError("Missing or nonfinite policy tensors")
    blue = metadata["policies"]["blue"]
    if (
        blue["arch"]
        != {
            "name": "recurrent",
            "cell": "lstm",
            "trunk": "separate",
            "hidden_dim": 256,
            "hidden_layers": 1,
            "activation": "tanh",
        }
        or blue["obs_dim"] != 450
        or blue["action_dim"] != 242
    ):
        raise ValueError("Frozen Blue differs from its enhanced-observation LSTM contract")
    if joint:
        red = metadata["policies"]["red"]
        if red["obs_dim"] != RED_OBS_SIZE or red["action_dim"] != RED_POLICY_ACTION_DIM:
            raise ValueError("Red observation or action dimensions differ")
        if (
            blue["trainable"]
            or not red["trainable"]
            or red["arch"] != {"name": "shared", "hidden_dim": 256, "hidden_layers": 2, "activation": "tanh"}
        ):
            raise ValueError("Expected only the shared feedforward Red to be trainable")
    return metadata, tensors


def compare_policy_parameters(original, initial, final):
    for team in ("blue", "red"):
        if set(initial[team]) != set(final[team]):
            raise ValueError("Checkpoint tensor keys changed")
        for name in initial[team]:
            left, right = initial[team][name], final[team][name]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError("Checkpoint tensor shape or dtype changed")
    blue_hash = tensor_hash(original["blue"])
    if tensor_hash(initial["blue"]) != blue_hash or tensor_hash(final["blue"]) != blue_hash:
        raise ValueError("Frozen Blue parameters changed or differ from the selected source")
    initial_red, final_red = tensor_hash(initial["red"]), tensor_hash(final["red"])
    if initial_red == final_red:
        raise ValueError("Smoke did not update Red parameters")
    return {
        "frozen_blue_parameters_sha256": blue_hash,
        "initial_red_parameters_sha256": initial_red,
        "final_red_parameters_sha256": final_red,
    }


def owned_artifact(root, indexed, identity, owner):
    artifact = indexed[identity]
    path = (root / "evidence" / artifact["uri"]).resolve()
    if (
        artifact["kind"] != "artifact"
        or artifact["producer_id"] != owner
        or not path.is_relative_to(root / "evidence")
        or sha256(path) != artifact["sha256"]
    ):
        raise ValueError("Owned artifact identity or bytes differ")
    return path


def completed_attempt(root, indexed, run_id, trial):
    execution = read_json(root / "runs" / run_id / "execution.json")
    receipt = read_json(root / "runs" / run_id / "receipt.json")
    if execution["returncode"] != 0 or execution["interrupted"] or execution["receipt"] != receipt:
        raise ValueError("Training process did not complete successfully")
    if Path(receipt["evidence_root"]).resolve() != root / "evidence":
        raise ValueError("Training receipt names another store")
    attempt = indexed[receipt["attempt_id"]]
    if attempt["status"] != "succeeded" or attempt["execution_kind"] != "training" or attempt["trial_id"] != trial:
        raise ValueError("Training receipt does not select a successful attempt of this trial")
    return attempt


def checkpoint(root, indexed, attempt, steps):
    name = "initial_checkpoint" if steps == 0 else f"checkpoint_{steps}"
    value = attempt["facts"][name]["value"]
    if value["environment_steps"] != steps:
        raise ValueError("Checkpoint has another step identity")
    for role in ("model", "sidecar"):
        owned_artifact(root, indexed, value[role], attempt["id"])
    return value


def full_state(root, indexed, attempt, steps):
    from jaxborg.evaluation.resume_state import inspect_training_state

    selected = checkpoint(root, indexed, attempt, steps)
    state = indexed[selected["training_state"]]
    path = owned_artifact(root, indexed, state["id"], attempt["id"])
    result = inspect_training_state(path, expected_sha256=state["sha256"])
    if result["metadata"]["environment_steps"] != steps:
        raise ValueError("State metadata and selected milestone disagree")
    return {"artifact_id": state["id"], **result}


def observed_updates(root, indexed, attempt):
    rows = []
    for record in indexed.values():
        if (
            record["kind"] == "artifact"
            and record["producer_id"] == attempt["id"]
            and record["role"] == "training update observation"
        ):
            rows.append(read_json(owned_artifact(root, indexed, record["id"], attempt["id"])))
    return sorted(rows, key=lambda row: row["update_idx"])


def timing_summary(rows, expected_count):
    if len(rows) != expected_count:
        raise ValueError("Missing retained training updates")
    for row in rows:
        if any(isinstance(value, (float, int)) and not math.isfinite(value) for value in row.values()):
            raise ValueError("Nonfinite recorded training metric")
    durations = [later["wall_time_s"] - earlier["wall_time_s"] for earlier, later in zip(rows, rows[1:])]
    if not durations or any(value <= 0 for value in durations):
        raise ValueError("Invalid synchronized training timings")
    median = statistics.median(durations)
    return {
        "first_update_seconds": rows[0]["wall_time_s"],
        "warm_update_seconds": durations,
        "warm_median_seconds": median,
        "estimated_208_update_seconds": rows[0]["wall_time_s"] + median * 207,
        "limit": "Measured short-run projection; checkpoint delivery and changing contention can add time.",
    }


def validate_smoke(root, expected_revision, manifest, indexed):
    attempt = completed_attempt(root, indexed, SMOKE, SMOKE)
    budgets = {item["unit"]: item["completed"] for item in attempt["budgets"]}
    if budgets != {
        "environment_steps": 192000,
        "joint_rollout_updates": 4,
        "blue_policy_updates": 0,
        "red_policy_updates": 4,
    }:
        raise ValueError(f"Unexpected smoke budgets: {budgets}")
    config = attempt["facts"]["resolved_config"]["value"]
    if config["trainable_teams"] != ["red"] or attempt["facts"]["source_revision"]["value"] != expected_revision:
        raise ValueError("Smoke used another trainable team or source revision")
    expected_variant = {
        "name": "cc4_stock",
        "num_steps": 500,
        "red_reward": "zero_sum",
        "blue_block_policy": "cc4",
        "cage4_enhanced_obs": True,
        "blue_observation_version": 2,
        "resilience_roles": False,
        "op_zone_servers": None,
    }
    if any(config["variant"][key] != value for key, value in expected_variant.items()):
        raise ValueError("Smoke changed the stock game or observation contract")
    roles = attempt["facts"]["input_artifacts"]["value"]
    if sum(role.startswith("training topology ") for role in roles) != 100:
        raise ValueError("Smoke did not retain the 100-layout training bank")
    for index, topology in enumerate(manifest["training_topologies"]):
        if indexed[roles[f"training topology {index}"]]["sha256"] != topology["sha256"]:
            raise ValueError("Smoke trained on another topology bank")
    source = manifest["blue_checkpoints"]["blue42-primary"]
    _, original = checkpoint_tensors(source["model"], joint=False)
    initial = checkpoint(root, indexed, attempt, 0)
    final = checkpoint(root, indexed, attempt, 192000)
    initial_meta, initial_tensors = checkpoint_tensors(
        owned_artifact(root, indexed, initial["model"], attempt["id"]), joint=True
    )
    final_meta, final_tensors = checkpoint_tensors(
        owned_artifact(root, indexed, final["model"], attempt["id"]), joint=True
    )
    parameter_check = compare_policy_parameters(original, initial_tensors, final_tensors)
    for metadata, steps in ((initial_meta, 0), (final_meta, 192000)):
        if metadata["provenance"]["total_steps"] != steps or metadata["provenance"]["seed"] != 17:
            raise ValueError("Smoke policy metadata has another seed or step identity")
    captured = {indexed[value]["sha256"] for value in roles.values()}
    if not {source["model_sha256"], source["sidecar_sha256"]} <= captured:
        raise ValueError("Smoke did not retain the exact frozen opponent inputs")
    full_state(root, indexed, attempt, 96000)
    final_state = full_state(root, indexed, attempt, 192000)
    result = {
        "passed": True,
        "source_revision": expected_revision,
        "manifest_sha256": sha256(root / "inputs/manifest.json"),
        "attempt_id": attempt["id"],
        "budgets": budgets,
        "initial_checkpoint": initial,
        "final_checkpoint": final,
        **parameter_check,
        "final_state_sha256": final_state["state_sha256"],
        "timing": timing_summary(observed_updates(root, indexed, attempt), 4),
    }
    save_once(root / "checks/smoke.json", result)
    return result


def validate_resume(root, indexed, smoke):
    resumed = completed_attempt(root, indexed, RESUME, RESUME)
    budgets = {item["unit"]: item["completed"] for item in resumed["budgets"]}
    if budgets != {
        "environment_steps": 96000,
        "joint_rollout_updates": 2,
        "blue_policy_updates": 0,
        "red_policy_updates": 2,
    }:
        raise ValueError(f"Unexpected continuation budgets: {budgets}")
    original = indexed[smoke["attempt_id"]]
    parent = full_state(root, indexed, original, 96000)
    roles = resumed["facts"]["input_artifacts"]["value"]
    if indexed[roles["resume training state"]]["sha256"] != parent["sha256"]:
        raise ValueError("Resume smoke consumed another full state")
    continuation = full_state(root, indexed, resumed, 192000)
    if continuation["state_sha256"] != smoke["final_state_sha256"]:
        raise ValueError("Continuation differs from uninterrupted scientific numeric state")
    result = {
        "passed": True,
        "original_attempt": original["id"],
        "resumed_attempt": resumed["id"],
        "parent_state_sha256": parent["sha256"],
        "state_sha256": continuation["state_sha256"],
        "leaf_count": continuation["leaf_count"],
        "new_work_budgets": budgets,
    }
    save_once(root / "checks/resume.json", result)
    return result


def validate_fresh_compilation(root, indexed, smoke):
    original = indexed[smoke["attempt_id"]]
    timing = completed_attempt(root, indexed, TIMING, TIMING)
    budgets = {item["unit"]: item["completed"] for item in timing["budgets"]}
    if budgets != {
        "environment_steps": 480000,
        "joint_rollout_updates": 10,
        "blue_policy_updates": 0,
        "red_policy_updates": 10,
    }:
        raise ValueError("Fresh compilation check requires the complete ten-update timing trial")
    configs = [attempt["facts"]["resolved_config"]["value"] for attempt in (original, timing)]
    if any(attempt["facts"]["source_revision"]["value"] != smoke["source_revision"] for attempt in (original, timing)):
        raise ValueError("Fresh compilation comparison uses different source revisions")
    for name in ("trainable_teams", "variant", "seed_roles", "topology", "opponents"):
        if configs[0][name] != configs[1][name]:
            raise ValueError(f"Fresh compilation comparison changed scientific inputs: {name}")
    for config in configs:
        if config["initialization"]["kind"] != "fresh" or config.get("resume") is not None:
            raise ValueError("Fresh compilation comparison requires independently initialized runs")
        if any(team["ANNEAL_LR"] is not False for team in config["team_configs"].values()):
            raise ValueError("Different-horizon comparison requires constant learning rates")
    for team in ("blue", "red"):
        numerical = [
            {
                key: value
                for key, value in config["team_configs"][team].items()
                if key not in {"TOTAL_TIMESTEPS", "NUM_UPDATES"}
            }
            for config in configs
        ]
        if numerical[0] != numerical[1]:
            raise ValueError("Fresh compilation comparison changed learner settings")
    environments = [config["resume_contract"]["runtime_environment"] for config in configs]
    if environments[0] != environments[1]:
        raise ValueError("Fresh compilation comparison changed runtime settings")
    states = [full_state(root, indexed, attempt, 96000) for attempt in (original, timing)]
    if states[0]["state_sha256"] != states[1]["state_sha256"]:
        raise ValueError("Fresh compilation changed scientific numeric state at update 2")
    result = {
        "passed": True,
        "source_revision": smoke["source_revision"],
        "smoke_attempt_id": original["id"],
        "timing_attempt_id": timing["id"],
        "completed_updates": 2,
        "environment_steps": 96000,
        "smoke_state": {key: states[0][key] for key in ("artifact_id", "sha256", "state_sha256", "leaf_count")},
        "timing_state": {key: states[1][key] for key in ("artifact_id", "sha256", "state_sha256", "leaf_count")},
        "comparison": "Exact equality of all scientific numeric leaves from two fresh runs at update 2.",
    }
    save_once(root / "checks/fresh-compilation.json", result)
    return result


def validate_cpu(root, manifest, indexed, smoke):
    marker = read_json(root / "checks/cpu-evaluation.json")
    if (
        marker.get("passed") is not True
        or marker.get("source_revision") != smoke["source_revision"]
        or marker.get("manifest_sha256") != smoke["manifest_sha256"]
        or marker.get("training_attempt_id") != smoke["attempt_id"]
        or marker.get("episodes") != 1
    ):
        raise ValueError("CPU smoke marker differs from accepted training")
    evaluation = indexed[marker["evaluation_id"]]
    attempt = indexed[evaluation["attempt_id"]]
    if evaluation["kind"] != "evaluation" or evaluation["status"] != "succeeded" or attempt["status"] != "succeeded":
        raise ValueError("CPU smoke evaluation did not succeed")
    budget = next(item for item in evaluation["budgets"] if item["unit"] == "episodes")
    config = evaluation["facts"]["resolved_config"]["value"]
    if budget["completed"] != 1 or config["horizon"] != 500 or config["deterministic"] is not False:
        raise ValueError("CPU smoke changed the episode contract")
    roles = evaluation["facts"]["input_artifacts"]["value"]
    expected = {
        "red checkpoint": indexed[smoke["final_checkpoint"]["model"]]["sha256"],
        "blue checkpoint": manifest["blue_checkpoints"]["blue42-primary"]["model_sha256"],
    }
    if any(indexed[roles[role]]["sha256"] != value for role, value in expected.items()):
        raise ValueError("CPU smoke evaluated another policy")
    return marker


def validate_pairs(root, expected_revision):
    from jaxborg.evaluation.capture_store import load_records

    root = Path(root).resolve()
    verify(root, expected_revision)
    indexed = {record["id"]: record for record in load_records(root / "evidence")}
    result = {
        "source_revision": expected_revision,
        "manifest_sha256": sha256(root / "inputs/manifest.json"),
        "pairs": [],
    }
    for blue in (42, 100):
        for red in (17, 83, 149):
            trials = [f"br-blue{blue}-{age}-red{red}" for age in ("primary", "final")]
            if not all((root / "runs" / trial / "receipt.json").exists() for trial in trials):
                continue
            attempts, identities = [], []
            for trial in trials:
                attempt = completed_attempt(root, indexed, trial, trial)
                selected = checkpoint(root, indexed, attempt, 0)
                _, tensors = checkpoint_tensors(
                    owned_artifact(root, indexed, selected["model"], attempt["id"]), joint=True
                )
                attempts.append(attempt["id"])
                identities.append(tensor_hash(tensors["red"]))
            if identities[0] != identities[1]:
                raise ValueError(f"Initial Red parameters differ across checkpoint ages: blue{blue}, red{red}")
            result["pairs"].append(
                {
                    "blue_seed": blue,
                    "red_seed": red,
                    "attempts": attempts,
                    "initial_red_parameters_sha256": identities[0],
                }
            )
    result["complete_pairs"] = len(result["pairs"])
    return result


def main():
    from jaxborg.evaluation.capture_store import load_records

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--require-cpu-eval", action="store_true")
    parser.add_argument("--require-resume", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--pairs", action="store_true")
    args = parser.parse_args()
    root = args.study_root.resolve()
    if args.pairs:
        print(json.dumps(validate_pairs(root, args.expected_revision)))
        return
    manifest = verify(root, args.expected_revision)
    indexed = {record["id"]: record for record in load_records(root / "evidence")}
    if args.timing:
        attempt = completed_attempt(root, indexed, TIMING, TIMING)
        result = {
            "source_revision": args.expected_revision,
            "manifest_sha256": sha256(root / "inputs/manifest.json"),
            "attempt_id": attempt["id"],
            **timing_summary(observed_updates(root, indexed, attempt), 10),
        }
        save_once(root / "checks/timing.json", result)
    else:
        result = validate_smoke(root, args.expected_revision, manifest, indexed)
        if args.require_resume or args.require_cpu_eval:
            result = {**result, "resume_equivalence": validate_resume(root, indexed, result)}
        if args.require_cpu_eval:
            result = {
                **result,
                "fresh_compilation_equivalence": validate_fresh_compilation(root, indexed, result),
                "cpu_evaluation": validate_cpu(root, manifest, indexed, result),
            }
            save_once(root / "checks/production-ready.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
