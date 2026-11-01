"""Demonstrate a deliberately invalid checkpoint, explicit retry and retained trajectory."""

import argparse
import json
from pathlib import Path

from safetensors import SafetensorError

from jaxborg.evaluation.capture_store import load_records
from jaxborg.evaluation.joint_capture import outside_source, run_matchup
from jaxborg.evaluation.provenance import collect_provenance
from jaxborg.experiment_records import collate
from jaxborg.scenarios.cc4.game_variant import GameVariant

HERE = Path(__file__).resolve().parent


def protocol():
    return {
        "schema_version": 1,
        "kind": "protocol",
        "id": "deliberate-invalid-checkpoint-example",
        "study_id": "execution-recovery-example",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": [], "evaluation": []},
        "trials": [{"id": "retry", "condition_id": "fixture", "replicate_id": "1", "matched_unit_id": None}],
        "metric_definitions": [
            {
                "name": "blue_return",
                "unit": "score",
                "direction": "higher",
                "aggregation": "arithmetic mean",
                "sampling_unit": "episode conditional on fixed policies",
                "uncertainty": "none",
            }
        ],
    }


def source_inputs(source_store, evaluation_id):
    """Resolve exact baseline inputs without inventing a producer in the new store."""
    source_store = Path(source_store).resolve()
    records = {record["id"]: record for record in load_records(source_store)}
    evaluation = records.get(evaluation_id)
    if evaluation is None or evaluation["kind"] != "evaluation" or evaluation["status"] != "succeeded":
        raise ValueError("Select an exact successful baseline evaluation")
    config = evaluation["facts"]["resolved_config"]["value"]
    if (
        config.get("backend") != "jax"
        or config["builder"]["arguments"].get("offset") != 0
        or config["variant"]["blue_block_policy"] != "cc4"
        or config["variant"]["red_reward"] != "zero_sum"
    ):
        raise ValueError("Recovery example requires the zero-offset canonical baseline")
    identities = evaluation["facts"]["input_artifacts"]["value"]
    inputs = {role: source_store / records[identity]["uri"] for role, identity in identities.items()}
    return config, identities, inputs


def run_example(source_store, evaluation_id, output, *, expected_revision, allow_dirty=False):
    config, identities, inputs = source_inputs(source_store, evaluation_id)
    provenance = collect_provenance(
        expected_revision=expected_revision, allow_dirty=allow_dirty, entrypoint=Path(__file__), source_roots=(HERE,)
    )
    output = outside_source(output)
    output.mkdir(parents=True, exist_ok=False)
    invalid = output / "deliberately-invalid"
    invalid.mkdir()
    model = invalid / "model_invalid.safetensors"
    model.write_bytes(b"Deliberately invalid checkpoint for the recovery example.\n")
    (invalid / "recipe_invalid.yaml").write_bytes(inputs["blue sidecar"].read_bytes())
    common = {
        "root": output / "evidence",
        "protocol": protocol(),
        "trial_id": "retry",
        "expected_revision": expected_revision,
        "allow_dirty": allow_dirty,
        "variant": GameVariant(**config["variant"]),
        "seeds": config["seed_roles"]["episode_roots"][:1],
        "topology_path": [path for role, path in inputs.items() if role.startswith("topology ")],
        "recipe": config["recipe"],
        "source_roots": (HERE,),
        "entrypoint": Path(__file__),
        "overrides": {
            "fixture": "deliberately invalid input followed by an explicit valid retry",
            "imported_from_evaluation": evaluation_id,
            "imported_source_ids": identities,
        },
    }
    try:
        run_matchup(model, inputs["red checkpoint"], **common)
    except SafetensorError as error:
        failed = getattr(error, "_evaluation_capture", None)
        if failed is None:
            raise
    else:
        raise RuntimeError("Deliberately invalid checkpoint unexpectedly succeeded")
    from scripts.eval.plot_results import REPORT_CONFIGURATION, analysis_identity, write_collation_report

    analysis = {"source": analysis_identity(), "configuration": REPORT_CONFIGURATION}
    before = collate(
        load_records(common["root"]),
        protocol_id=protocol()["id"],
        selected_attempts={},
        selected_evaluations=[],
        analysis=analysis,
    )
    (output / "before-retry.json").write_text(json.dumps(before, indent=2, sort_keys=True) + "\n")
    retried, _ = run_matchup(
        inputs["blue checkpoint"],
        inputs["red checkpoint"],
        **common,
        retry_of=failed.attempt_id,
        correction={
            "evaluation_id": failed.evaluation_id,
            "reason": "Deliberate fixture: replace invalid checkpoint bytes with the selected baseline policy",
            "changes": {"blue_checkpoint": identities["blue checkpoint"]},
        },
        trajectories=True,
    )
    provenance.verify_sources()
    after = collate(
        load_records(common["root"]),
        protocol_id=protocol()["id"],
        selected_attempts={"retry": retried.attempt_id},
        selected_evaluations=[retried.evaluation_id],
        analysis=analysis,
    )
    collation = output / "collation.json"
    collation.write_text(json.dumps(after, indent=2, sort_keys=True) + "\n")
    write_collation_report(collation, common["root"], output / "report")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-store", required=True, type=Path)
    parser.add_argument("--source-evaluation", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    output = run_example(
        args.source_store,
        args.source_evaluation,
        args.output,
        expected_revision=args.expected_revision,
        allow_dirty=args.allow_dirty,
    )
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
