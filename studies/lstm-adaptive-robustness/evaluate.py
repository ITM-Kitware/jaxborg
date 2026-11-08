"""Evaluate one owned response checkpoint against one fixed defender on CPU."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

from prepare import PROJECT, sha256, verify, write_json

HERE = Path(__file__).resolve().parent
STEPS = {"smoke": 192000, "full": 9984000}


def selected_response(root, run_id, trial, revision, target_steps):
    from jaxborg.evaluation.capture_store import load_records

    records = {row["id"]: row for row in load_records(root / "evidence")}
    receipt = json.loads((root / "runs" / run_id / "receipt.json").read_text())
    owner = records[receipt["attempt_id"]]
    if (
        owner.get("execution_kind") != "training"
        or owner.get("trial_id") != trial
        or owner.get("status") != "succeeded"
        or owner["facts"]["source_revision"]["value"] != revision
    ):
        raise ValueError("Select a successful, source-matched attempt of the declared trial")
    checkpoint = owner["facts"].get(f"checkpoint_{target_steps}", {}).get("value", {})
    if checkpoint.get("environment_steps") != target_steps:
        raise ValueError("The prespecified response checkpoint is missing")
    selected = {"attempt_id": owner["id"], "steps": target_steps}
    for role in ("model", "sidecar"):
        artifact = records[checkpoint[role]]
        path = (root / "evidence" / artifact["uri"]).resolve()
        if (
            artifact["kind"] != "artifact"
            or artifact["producer_id"] != owner["id"]
            or not path.is_relative_to(root / "evidence")
            or sha256(path) != artifact["sha256"]
        ):
            raise ValueError("Response checkpoint differs from its owned artifact")
        selected[role] = str(path)
        selected[f"{role}_artifact"] = artifact["id"]
        selected[f"{role}_sha256"] = artifact["sha256"]
    return selected


def evaluate(args):
    import jax

    from jaxborg.checkpoint import load_jax_bundle, sidecar_path
    from jaxborg.evaluation.joint_capture import run_matchup
    from jaxborg.scenarios.cc4.game_variant import GameVariant

    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("This evaluator requires JAX_PLATFORMS=cpu")
    root = args.study_root.resolve()
    verify(root, args.expected_revision)
    manifest = json.loads((root / "inputs/manifest.json").read_text())
    run_id = args.run_id or args.trial
    smoke = args.trial == "smoke-blue42-primary-red17"
    if smoke and args.blue != "blue42-primary":
        raise ValueError("Smoke evaluation uses the prespecified frozen primary defender")
    if not smoke and not args.trial.startswith("br-"):
        raise ValueError("Evaluation requires a production trial or the declared smoke")
    response = selected_response(root, run_id, args.trial, args.expected_revision, STEPS["smoke" if smoke else "full"])
    blue = manifest["blue_checkpoints"][args.blue]
    for role in ("model", "sidecar"):
        if sha256(blue[role]) != blue[f"{role}_sha256"]:
            raise ValueError("Frozen defender input changed")
    if sidecar_path(Path(response["model"])).resolve() != Path(response["sidecar"]).resolve():
        raise ValueError("Response bundle requires its exact adjacent sidecar")
    bundle = load_jax_bundle(response["model"])
    red = bundle.policies["red"]
    if (
        bundle.legacy
        or bundle.backend != "jax"
        or not red.trainable
        or red.arch != {"name": "shared", "hidden_dim": 256, "hidden_layers": 2, "activation": "tanh"}
        or bundle.policies["blue"].trainable
    ):
        raise ValueError("Response bundle has a different training or architecture contract")
    topologies = manifest["evaluation_topologies"][:1] if smoke else manifest["evaluation_topologies"]
    paths = []
    for item in topologies:
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError("Evaluation topology changed")
        paths.append(path)
    seeds = [990001] if smoke else list(range(21000, 21016))
    variant = GameVariant(name="cc4_stock", cage4_enhanced_obs=True)
    destination = root / "evaluations" / run_id / args.blue
    destination.mkdir(parents=True, exist_ok=False)
    plan = {
        "source_revision": args.expected_revision,
        "trial": args.trial,
        "run_id": run_id,
        "response": response,
        "blue": blue,
        "blue_id": args.blue,
        "topologies": topologies,
        "episode_roots": seeds,
        "variant": asdict(variant),
        "deterministic": False,
        "backend": "cpu",
    }
    write_json(destination / "plan.json", plan)

    def condition_check(config):
        if (
            config["variant"] != asdict(variant)
            or config["topology"]["sha256"] != [item["sha256"] for item in topologies]
            or config["seed_roles"]["episode_roots"] != seeds
            or config["episodes_per_seed"] != 1
            or config["deterministic"]
        ):
            raise ValueError("Resolved evaluation differs from the prespecified stock-game panel")

    capture, result = run_matchup(
        blue["model"],
        response["model"],
        root=root / "evidence",
        protocol=json.loads((root / "inputs/protocol.json").read_text()),
        trial_id=args.trial,
        expected_revision=args.expected_revision,
        variant=variant,
        seeds=seeds,
        topology_path=paths,
        condition_check=condition_check,
        input_artifacts={
            "red checkpoint": response["model_artifact"],
            "red sidecar": response["sidecar_artifact"],
        },
        entrypoint=Path(__file__),
        source_roots=(HERE, PROJECT / "studies/lstm-fsm-competence"),
        overrides={"plan_sha256": sha256(destination / "plan.json"), "blue_id": args.blue},
    )
    returns = list(result.blue_returns)
    if len(returns) != len(paths) * len(seeds) or not all(math.isfinite(value) for value in returns):
        raise ValueError("Evaluation returned missing or nonfinite episodes")
    receipt = {
        "source_revision": args.expected_revision,
        "manifest_sha256": sha256(root / "inputs/manifest.json"),
        "training_attempt_id": response["attempt_id"],
        "evaluation_id": capture.evaluation_id,
        "trial": args.trial,
        "run_id": run_id,
        "blue_id": args.blue,
        "response_environment_steps": response["steps"],
        "episodes": len(returns),
        "mean_blue_return": statistics.mean(returns),
        "blue_returns": returns,
        "passed": True,
    }
    write_json(destination / "receipt.json", receipt)
    if smoke:
        write_json(root / "checks/cpu-evaluation.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--blue", required=True, choices=("blue42-primary", "blue42-final", "blue100-primary", "blue100-final")
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
