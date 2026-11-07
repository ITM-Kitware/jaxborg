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


def run_scripted(model, *, context, return_capture=False, **kwargs):
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
        entrypoint=context.get("entrypoint") or project / "scripts/eval/eval_scripted_reds_jax.py",
        source_roots=(project / "scripts/eval", *context.get("source_roots", ())),
        expected_revision=context["revision"],
        allow_dirty=context.get("allow_dirty", False),
    )
    capture, rows = _run_scripted(
        root=root,
        protocol=json.loads((root / "protocol.json").read_text()),
        trial_id=context["trial_id"],
        provenance=provenance,
        source_recipe=context["source_recipe"],
        effective_recipe=context["effective_recipe"],
        retained_inputs={"blue checkpoint": artifact["id"]},
        condition_check=context.get("condition_check"),
        training={"training_checkpoint_artifact_id": artifact["id"], "training_attempt_id": owner["id"]},
        kwargs=kwargs,
    )
    return (capture, rows) if return_capture else rows


def run_frozen_scripted(
    model,
    *,
    root,
    protocol,
    trial_id,
    expected_revision,
    source_recipe,
    effective_recipe,
    condition_check=None,
    retry_of=None,
    entrypoint=None,
    source_roots=(),
    allow_dirty=False,
    **kwargs,
):
    """Record the CIA scripted suite for an external frozen JAX checkpoint.

    The effective recipe supplies the evaluator settings. Imported weights and
    their adjacent sidecar have no asserted training parent. The maintained
    evaluator requires resilience roles; callers establish the game and
    topology contract for their comparison.
    Returns the capture receipt and the per-opponent result rows.
    """
    import yaml

    from jaxborg.checkpoint import sidecar_path

    if not isinstance(expected_revision, str) or len(expected_revision) != 40:
        raise ValueError("Recorded executions require an exact pinned source revision")
    model = Path(model).expanduser().resolve()
    if model.suffix != ".safetensors":
        raise ValueError("Frozen scripted capture requires a JAX .safetensors checkpoint")
    if any(name in kwargs for name in ("on_row", "policy_loader", "env_factory", "episode_runner")):
        raise ValueError("Frozen scripted capture uses the maintained executor without custom callbacks")
    effective = yaml.safe_load(Path(effective_recipe).read_text())
    if not isinstance(effective, dict):
        raise ValueError("Effective recipe must contain a mapping")
    if "recipe" in kwargs and kwargs["recipe"] != effective:
        raise ValueError("Supplied recipe differs from retained effective recipe")
    kwargs["recipe"] = effective
    kwargs.setdefault("reds", ("fsm", "cia_c", "cia_i", "cia_a"))
    kwargs.setdefault("seeds", "1000-1009")
    kwargs.setdefault("episodes_per_seed", 1)
    kwargs.setdefault("deterministic", False)
    kwargs.setdefault("progress", False)
    kwargs.setdefault("eval_name", None)
    provenance = collect_provenance(
        expected_revision=expected_revision,
        entrypoint=entrypoint,
        source_roots=source_roots,
        allow_dirty=allow_dirty,
    )
    return _run_scripted(
        root=outside_source(root),
        protocol=protocol,
        trial_id=trial_id,
        provenance=provenance,
        source_recipe=source_recipe,
        effective_recipe=effective_recipe,
        inputs={"blue checkpoint": model, "blue sidecar": sidecar_path(model)},
        condition_check=condition_check,
        retry_of=retry_of,
        kwargs=kwargs,
    )


def _run_scripted(
    *,
    root,
    protocol,
    trial_id,
    provenance,
    source_recipe,
    effective_recipe,
    kwargs,
    inputs=None,
    retained_inputs=None,
    training=None,
    condition_check=None,
    retry_of=None,
):
    from jaxborg.evaluation.jax_scripted_red import (
        _normalise_reds,
        _normalise_topology_paths,
        _parse_seeds,
        evaluate_jax_scripted_reds,
        write_results,
    )

    kwargs = dict(kwargs)
    kwargs["seeds"] = _parse_seeds(kwargs["seeds"])
    kwargs["reds"] = _normalise_reds(kwargs["reds"])
    kwargs["topology_paths"] = _normalise_topology_paths(kwargs["topology_paths"])
    if type(kwargs["episodes_per_seed"]) is not int or kwargs["episodes_per_seed"] < 1:
        raise ValueError("episodes_per_seed must be positive")
    external_inputs = dict(inputs or {})
    inputs = {f"topology {i}": Path(path) for i, path in enumerate(kwargs["topology_paths"])}
    inputs.update(external_inputs)
    inputs["source recipe"] = Path(source_recipe)
    inputs["effective recipe"] = Path(effective_recipe)
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
    configuration = json_configuration({**kwargs, **discovery, "execution": provenance.configuration})
    if condition_check is not None:
        condition_check(configuration)
    capture = start_evaluation(
        root,
        protocol=protocol,
        trial_id=trial_id,
        retry_of=retry_of,
        requested_episodes=(len(kwargs["reds"]) * len(kwargs["seeds"]) * kwargs["episodes_per_seed"] * topology_count),
        resolved_config=configuration,
        facts={**provenance.facts, **{name: fact(value) for name, value in discovery.items()}},
        inputs=inputs,
        retained_inputs=retained_inputs,
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
        if external_inputs:
            import yaml

            effective = yaml.safe_load(capture.inputs["effective recipe"].read_text())
            if effective != kwargs["recipe"]:
                raise ValueError("Effective recipe changed during retention")
            kwargs["recipe"] = effective
        index = 0

        def observe(row):
            nonlocal index
            row["evaluation_id"] = capture.evaluation_id
            if training:
                row.update(training)
            else:
                row["checkpoint_artifact_id"] = capture.input_ids["blue checkpoint"]
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
        return capture, rows
    except BaseException as error:
        fail_evaluation(capture, error)
        raise
