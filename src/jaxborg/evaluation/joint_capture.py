"""Retain the supported JAX joint matchup around its maintained executor."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from jaxborg.evaluation.capture_store import (
    add_artifact,
    finish_evaluation,
    json_configuration,
    load_records,
    record_episode,
    record_facts,
    start_evaluation,
)
from jaxborg.evaluation.provenance import builder_identity, collect_provenance


def fact(value, source="configuration.json"):
    return {"value": value, "source": source}


def outside_source(root):
    root = Path(root).expanduser().resolve()
    project = Path(__file__).resolve().parents[3]
    if root.is_relative_to(project):
        raise ValueError("Retained execution output must be outside the source checkout")
    return root


def retained_json(capture, name, value, role):
    with tempfile.TemporaryDirectory(prefix="execution-output-") as temporary:
        path = Path(temporary) / Path(name).name
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        return add_artifact(capture, path, role=role, name=name)


def run_matchup(
    blue_model,
    red_model,
    *,
    root,
    protocol,
    trial_id,
    expected_revision,
    variant,
    seeds,
    backend="jax",
    episodes_per_seed=1,
    deterministic=False,
    topology_path=None,
    env_factory=None,
    source_roots=(),
    condition_check=None,
    input_artifacts=None,
    retry_of=None,
    correction=None,
    trajectories=False,
    allow_dirty=False,
    recipe=None,
    supplied_recipe=None,
    overrides=None,
    entrypoint=None,
):
    """Execute one independent recorded evaluation with exact retained policy inputs.

    This supported path uses JAX policies, zero-sum payoffs and exhaustive fixed
    topology banks. It does not resume or reuse evaluations.
    Trajectories are observations/actions/payoffs from these same executed scans.
    """
    from jaxborg.checkpoint import load_jax_bundle, sidecar_path
    from jaxborg.evaluation.episode_seeds import expand_episode_seeds
    from jaxborg.evaluation.matchup_runner import evaluate_matchup

    if backend != "jax" or variant.red_reward != "zero_sum":
        raise ValueError("Recorded joint matchups require JAX policies and zero_sum reward")
    if not isinstance(expected_revision, str) or len(expected_revision) != 40:
        raise ValueError("Recorded executions require an exact pinned source revision")
    expanded = expand_episode_seeds(seeds, episodes_per_seed)
    if (
        not expanded
        or len(set(expanded)) != len(expanded)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in expanded)
    ):
        raise ValueError("Recorded matchups require unique uint32 episode seeds")
    root = outside_source(root)
    provenance = collect_provenance(
        allow_dirty=allow_dirty, expected_revision=expected_revision, entrypoint=entrypoint, source_roots=source_roots
    )
    builder = builder_identity(env_factory, provenance)
    topology_paths = (
        []
        if topology_path is None
        else (
            [Path(topology_path)] if isinstance(topology_path, (str, Path)) else [Path(path) for path in topology_path]
        )
    )
    if not topology_paths:
        raise ValueError("Recorded joint matchups require an explicit fixed topology bank")
    from jaxborg.scenarios.cc4.topology import load_topology_metadata

    models = {"blue": Path(blue_model).resolve(), "red": Path(red_model).resolve()}
    inputs = {f"{team} checkpoint": path for team, path in models.items()}
    inputs.update({f"{team} sidecar": sidecar_path(path) for team, path in models.items()})
    inputs.update({f"topology {index}": path for index, path in enumerate(topology_paths)})
    identities = {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in inputs.items()}
    retained = dict(input_artifacts or {})
    indexed = {record["id"]: record for record in load_records(root)} if retained else {}
    for role, identity in retained.items():
        artifact = indexed.get(identity)
        if (
            role not in inputs
            or artifact is None
            or artifact["kind"] != "artifact"
            or artifact["sha256"] != identities[role]
        ):
            raise ValueError("Owned policy input differs from the requested artifact identity")
    for team in ("blue", "red"):
        role = f"{team} checkpoint"
        sidecar_role = f"{team} sidecar"
        if sidecar_role in retained and role not in retained:
            raise ValueError("A retained sidecar requires its retained checkpoint")
        if role in retained:
            inputs[role] = root / indexed[retained[role]]["uri"]
            inputs[sidecar_role] = sidecar_path(inputs[role])
            candidates = [
                record["id"]
                for record in indexed.values()
                if record["kind"] == "artifact" and (root / record["uri"]).resolve() == inputs[sidecar_role].resolve()
            ]
            if len(candidates) != 1 or retained.get(sidecar_role, candidates[0]) != candidates[0]:
                raise ValueError("Retained checkpoint requires its exact indexed adjacent sidecar")
            retained[sidecar_role] = candidates[0]
            identities[sidecar_role] = hashlib.sha256(inputs[sidecar_role].read_bytes()).hexdigest()
    # Sidecars already owned by this store retain that ownership without a new import.
    for role, path in inputs.items():
        matches = [
            record
            for record in indexed.values()
            if record["kind"] == "artifact" and (root / record["uri"]).resolve() == path.resolve()
        ]
        if len(matches) == 1 and role not in retained:
            retained[role] = matches[0]["id"]
    seed_roles = {
        "episode_roots": list(seeds),
        "episode_seeds": expanded,
        "scheme": "seed * episodes_per_seed + replica",
        "policy_rng": "PRNGKey(episode_seed); split for reset, per-step policies and environment",
    }
    topology = {
        "sampling": "exhaustive" if topology_paths else "generative",
        "sha256": [identities[f"topology {index}"] for index in range(len(topology_paths))],
        "metadata": [load_topology_metadata(path) for path in topology_paths],
        "origin": "imported fixed topology snapshots; generator evidence is retained in snapshot metadata",
    }
    configuration = json_configuration(
        {
            "schema_version": 1,
            "variant": variant,
            "builder": builder,
            "backend": backend,
            "eval_env": "jax",
            "horizon": variant.num_steps,
            "seed_roles": seed_roles,
            "topology": topology,
            "episodes_per_seed": episodes_per_seed,
            "deterministic": deterministic,
            "procedure": "jaxborg.evaluation.matchup_runner.evaluate_matchup",
            "trajectories": trajectories,
            "recipe": recipe,
            "supplied_recipe": supplied_recipe,
            "overrides": overrides or {},
            "execution": provenance.configuration,
        }
    )
    if condition_check is not None:
        condition_check(configuration)
    facts = {
        **provenance.facts,
        **{
            name: fact(configuration[name])
            for name in ("backend", "eval_env", "builder", "seed_roles", "topology", "procedure")
        },
        "game_contract": fact(asdict(variant)),
        "seeds": fact(expanded),
        "step_accounting": fact("Actual completed transitions in each retained episode"),
    }
    from jaxborg.experiment_discovery.execution import destination_configured

    capture = start_evaluation(
        root,
        track=destination_configured(),
        protocol=protocol,
        trial_id=trial_id,
        retry_of=retry_of,
        requested_episodes=len(expanded) * max(1, len(topology_paths)),
        resolved_config=configuration,
        facts=facts,
        inputs={role: path for role, path in inputs.items() if role not in retained},
        retained_inputs=retained,
        source_files=provenance.source_files,
        correction=correction,
    )
    try:
        provenance.verify_sources()
        if json.loads((capture.directory / "source-files.json").read_text()) != provenance.hashes:
            raise ValueError("Retained source differs from the preflight identity")
        for role, path in capture.inputs.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != identities[role]:
                raise ValueError("Evaluation input changed during retention")
        contracts = {}
        for team in ("blue", "red"):
            bundle = load_jax_bundle(capture.inputs[f"{team} checkpoint"])
            entry = bundle.policies[team]
            contracts[team] = {"obs_dim": entry.obs_dim, "action_dim": entry.action_dim, "arch": entry.arch}
        import jax

        record_facts(
            capture,
            {
                "hardware": fact([str(device) for device in jax.devices()], "JAX device discovery"),
                "policies": fact(
                    {team: capture.input_ids[f"{team} checkpoint"] for team in ("blue", "red")}, "input-artifacts.json"
                ),
                "policy_contract": fact(contracts, "retained checkpoint bundle metadata"),
            },
        )

        def observe(index, seed, reward, steps):
            record_episode(capture, episode_index=index, seed=seed, policy_seed=seed, reward=reward, steps=steps)

        def trace(index, values):
            retained_json(
                capture,
                f"trajectories/{index}.json",
                {
                    "schema_version": 1,
                    "eval_env": "jax",
                    "episode_index": index,
                    "steps": values,
                    "meaning": "policy observations, actions and payoffs from the recorded evaluation",
                },
                "JAX policy trajectory",
            )

        result = evaluate_matchup(
            capture.inputs["blue checkpoint"],
            capture.inputs["red checkpoint"],
            backend=backend,
            variant=variant,
            seeds=list(seeds),
            episodes_per_seed=episodes_per_seed,
            deterministic=deterministic,
            progress=False,
            topology_path=[capture.inputs[f"topology {index}"] for index in range(len(topology_paths))] or None,
            topology_sampling="exhaustive",
            env_factory=env_factory,
            on_episode=observe,
            on_trajectory=trace if trajectories else None,
        )
        retained_json(capture, "result.json", asdict(result), "joint matchup result")
        provenance.verify_sources()
        finish_evaluation(capture, status="succeeded")
        return capture, result
    except BaseException as error:
        error._evaluation_capture = capture
        if not (capture.directory / "terminal.json").exists():
            finish_evaluation(
                capture, status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", error=error
            )
        raise
