"""Record the maintained scripted opponent executor without changing its procedure."""

import hashlib
import json
from pathlib import Path

from jaxborg.evaluation.capture_store import (
    add_artifact,
    fail_evaluation,
    finish_evaluation,
    json_configuration,
    load_records,
    record_episode,
    start_evaluation,
)
from jaxborg.evaluation.joint_capture import fact, outside_source
from jaxborg.evaluation.provenance import collect_provenance, topology_evidence
from jaxborg.evaluation.tracking import destination_configured


def run_scripted(model, *, context, **kwargs):
    from jaxborg.evaluation.jax_scripted_red import (
        _normalise_reds,
        _parse_seeds,
        evaluate_jax_scripted_reds,
        write_results,
    )

    kwargs["seeds"] = _parse_seeds(kwargs["seeds"])
    kwargs["reds"] = _normalise_reds(kwargs["reds"])
    if type(kwargs["episodes_per_seed"]) is not int or kwargs["episodes_per_seed"] < 1:
        raise ValueError("episodes_per_seed must be positive")
    root = outside_source(context["root"])
    records = load_records(root)
    artifact = next(record for record in records if record["id"] == context["checkpoint"])
    owner = next(record for record in records if record["id"] == context["training_attempt_id"])
    if (
        artifact["kind"] != "artifact"
        or artifact["producer_id"] != owner["id"]
        or owner.get("execution_kind") != "training"
    ):
        raise ValueError("Checkpoint must belong to the declared training attempt")
    if hashlib.sha256(Path(model).read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("Requested model differs from owned checkpoint")
    project = Path(__file__).resolve().parents[3]
    provenance = collect_provenance(
        entrypoint=project / "scripts/eval/eval_scripted_reds_jax.py",
        source_roots=(project / "scripts/eval",),
        expected_revision=context["revision"],
        allow_dirty=context.get("allow_dirty", False),
    )
    inputs = {f"topology {i}": Path(path) for i, path in enumerate(kwargs["topology_paths"])}
    inputs["source recipe"] = Path(context["source_recipe"])
    inputs["effective recipe"] = Path(context["effective_recipe"])
    topology_count = len(kwargs["topology_paths"])
    identities = {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in inputs.items()}
    discovery = {
        "backend": "jax",
        "eval_env": "jax_fsm",
        "procedure": "jaxborg.evaluation.jax_scripted_red.evaluate_jax_scripted_reds",
        "topology": topology_evidence(
            kwargs["topology_paths"],
            [identities[f"topology {i}"] for i in range(topology_count)],
            sampling="exhaustive",
        ),
    }
    capture = start_evaluation(
        root,
        protocol=json.loads((root / "protocol.json").read_text()),
        trial_id=context["trial_id"],
        requested_episodes=(len(kwargs["reds"]) * len(kwargs["seeds"]) * kwargs["episodes_per_seed"] * topology_count),
        resolved_config=json_configuration({**kwargs, **discovery, "execution": provenance.configuration}),
        facts={**provenance.facts, **{name: fact(value) for name, value in discovery.items()}},
        inputs=inputs,
        retained_inputs={"blue checkpoint": artifact["id"]},
        source_files=provenance.source_files,
        track=destination_configured(),
    )
    try:
        provenance.verify_sources()
        kwargs["topology_paths"] = [capture.inputs[f"topology {i}"] for i in range(topology_count)]
        for role, identity in identities.items():
            if hashlib.sha256(capture.inputs[role].read_bytes()).hexdigest() != identity:
                raise ValueError("Evaluation input changed during retention")
        if json.loads((capture.directory / "source-files.json").read_text()) != provenance.hashes:
            raise ValueError("Retained source differs from preflight")
        index = 0

        def observe(row):
            nonlocal index
            row["evaluation_id"] = capture.evaluation_id
            row["training_checkpoint_artifact_id"] = artifact["id"]
            row["training_attempt_id"] = context["training_attempt_id"]
            for seed, reward in zip(row["per_episode_seeds"], row["per_episode"], strict=True):
                record_episode(capture, episode_index=index, seed=seed, reward=reward)
                index += 1
            output = write_results([row], capture.directory / f"scripted-{row['eval_red']}.jsonl")
            add_artifact(capture, output, role="scripted opponent episode results")
            if capture.tracking is not None:
                from jaxborg.evaluation.cia.reporting import cia_mlflow_metrics

                prefix = f"eval.scripted_red.{row['eval_red']}.blue"
                metrics = {
                    f"{prefix}.mean_reward": float(row["mean_reward"]),
                    f"{prefix}.std_reward": float(row["std_reward"]),
                    f"{prefix}.episodes": float(row["n_episodes"]),
                    **cia_mlflow_metrics(f"{prefix}.cia", row["cia_summary"]),
                }
                try:
                    capture.tracking.log_metrics(metrics)
                except Exception as error:
                    print(f"MLflow evaluation metrics warning: {error}", flush=True)

        rows = evaluate_jax_scripted_reds(capture.inputs["blue checkpoint"], on_row=observe, **kwargs)
        provenance.verify_sources()
        finish_evaluation(capture, status="succeeded")
        return rows
    except BaseException as error:
        fail_evaluation(capture, error)
        raise
