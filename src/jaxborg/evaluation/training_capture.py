"""Evidence at the maintained JAX trainers' outer execution boundary."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from jaxborg.evaluation.capture_store import (
    add_artifact,
    finish_training,
    json_configuration,
    load_records,
    record_facts,
    start_training,
)
from jaxborg.evaluation.joint_capture import fact, outside_source
from jaxborg.evaluation.provenance import builder_identity, collect_provenance, topology_evidence
from jaxborg.recipe import resolve_train_opponents, team_recipe, training_teams


def _validate_resume_lineage(root, metadata, archive_sha256, *, joint=False):
    """Require an owned full state and its original zero-step policies in this store."""
    root = outside_source(root)
    if not (root / "protocol.json").is_file():
        raise ValueError("Resume requires a producing attempt in the same evidence store")
    records = {record["id"]: record for record in load_records(root)}
    producer = records.get(metadata["producer_attempt_id"])
    if producer is None or producer["kind"] != "attempt":
        raise ValueError(
            "Resume requires a producing attempt in the same evidence store; cross-store resume is unsupported"
        )
    checkpoint = producer["facts"].get(f"checkpoint_{metadata['environment_steps']}", {}).get("value", {})

    def owned_artifact(identity, owner, role):
        artifact = records.get(identity)
        if (
            artifact is None
            or artifact["kind"] != "artifact"
            or artifact["producer_id"] != owner
            or artifact["role"] != role
        ):
            raise ValueError("Resume lineage requires the exact owned checkpoint artifact")
        path = (root / artifact["uri"]).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]
        ):
            raise ValueError("Owned resume lineage artifact bytes differ from their recorded identity")
        return artifact

    state = owned_artifact(checkpoint.get("training_state"), producer["id"], "full numeric training state")
    if state["sha256"] != archive_sha256:
        raise ValueError("Resume input is not the selected producing attempt's owned full-state checkpoint")
    origin = metadata["training_origin"]
    original_attempt = records.get(origin["attempt_id"])
    if original_attempt is None or original_attempt["kind"] != "attempt":
        raise ValueError("Resume origin must belong to the same evidence store")
    original_checkpoint = original_attempt["facts"].get("initial_checkpoint", {}).get("value")
    if original_checkpoint != origin["checkpoint"] or original_checkpoint.get("environment_steps") != 0:
        raise ValueError("Resume origin differs from the owned original zero-step checkpoint")
    owned_artifact(
        original_checkpoint["model"], original_attempt["id"], "initial joint policy" if joint else "initial blue policy"
    )
    owned_artifact(original_checkpoint["sidecar"], original_attempt["id"], "checkpoint recipe sidecar")


def begin_training(
    *,
    root,
    protocol,
    trial_id,
    expected_revision,
    args,
    recipe,
    configs,
    env_factory=None,
    source_roots=(),
    condition_check=None,
    retry_of=None,
    allow_dirty=False,
    supplied_recipe=None,
    overrides=None,
    entrypoint=None,
):
    """Capture joint, Blue/FSM, or Red/frozen-Blue training and exact inputs."""
    if not isinstance(expected_revision, str) or len(expected_revision) != 40:
        raise ValueError("Recorded training requires an exact pinned source revision")
    resume_path = getattr(args, "resume_state", None)
    teams = set(training_teams(recipe))
    opponents = recipe.get("train", {}).get("opponents") or {}
    scripted_red = teams == {"blue"} and not opponents
    frozen_blue = teams == {"red"} and set(opponents) == {"blue"}
    if not (scripted_red or frozen_blue or (teams == {"blue", "red"} and not opponents)):
        raise ValueError("Recorded training supports fresh joint, Blue/FSM, or Red/frozen-Blue training only")
    if recipe.get("train", {}).get("initial_weights"):
        raise ValueError("Recorded training does not support weights-only initialization")
    if resume_path is not None and not (scripted_red or frozen_blue):
        raise ValueError("Full-state resume requires recorded Blue/FSM or Red/frozen-Blue training")
    if frozen_blue:
        if set(configs) != {"blue", "red"}:
            raise ValueError("Recorded Red/frozen-Blue training requires both policy configurations")
        if (
            any(config.get("ADAPTIVE_UPDATES") for config in configs.values())
            or configs["red"].get("UPDATE_EVERY", 1) != 1
        ):
            raise ValueError("Recorded Red/frozen-Blue training requires a Red update every rollout")
        if any(config["NUM_STEPS"] != config["TRAIN_VARIANT"].num_steps for config in configs.values()):
            raise ValueError("Red/frozen-Blue rollout length must equal the episode horizon")
    if scripted_red:
        if set(configs) != {"blue"} or configs["blue"]["TRAIN_VARIANT"].red_agent != "finite_state":
            raise ValueError("Recorded Blue-only training requires stock finite_state scripted Red")
        if configs["blue"].get("ADAPTIVE_UPDATES") or configs["blue"].get("UPDATE_EVERY", 1) != 1:
            raise ValueError("Recorded Blue-only training requires a Blue update every rollout")
        if configs["blue"].get("GRAD_CLIP_MODE", "global") != "global":
            raise ValueError("Recorded Blue-only training supports global gradient clipping only")
        if configs["blue"]["NUM_STEPS"] != configs["blue"]["TRAIN_VARIANT"].num_steps:
            raise ValueError("Blue-only rollout length must equal the episode horizon for recurrent reset alignment")
    if any(config["TRAIN_VARIANT"].red_reward != "zero_sum" for config in configs.values()):
        raise ValueError("Recorded joint training requires zero_sum reward")
    if any(
        config.get(key)
        for config in configs.values()
        for key in ("MISSION_BANK", "PHASE_BOUNDARY_BANK", "PHASE_REWARDS_BANK")
    ):
        raise ValueError("Recorded training supports topology snapshots without auxiliary mission banks")
    project = Path(__file__).resolve().parents[3]
    provenance = collect_provenance(
        allow_dirty=allow_dirty,
        expected_revision=expected_revision,
        entrypoint=entrypoint or project / "scripts/train/algorithms/ippo_jax.py",
        source_roots=(project / "scripts/train/algorithms", *source_roots),
    )
    if scripted_red:
        from jaxborg.evaluation.jax_env_factory import make_jax_env

        if env_factory is not None:
            raise ValueError("Recorded Blue/FSM training uses the maintained make_jax_env builder only")
        builder = builder_identity(make_jax_env, provenance)
    else:
        builder = builder_identity(env_factory, provenance)
    inputs = {}
    recipe_source = recipe.get("__source_path__")
    if recipe_source:
        inputs["supplied recipe"] = Path(recipe_source).resolve()
    topology_paths = [Path(path).resolve() for path in configs["blue"].get("TOPOLOGY_BANK") or ()]
    if not topology_paths:
        raise ValueError("Recorded joint training requires an explicit fixed topology bank")
    inputs.update({f"training topology {index}": path for index, path in enumerate(topology_paths)})
    if frozen_blue:
        from jaxborg.checkpoint import sidecar_path

        opponent = resolve_train_opponents(recipe, backend="jax", exp_dir=os.environ.get("JAXBORG_EXP_DIR"))["blue"]
        inputs["frozen blue policy"] = opponent.resolve()
        inputs["frozen blue sidecar"] = sidecar_path(opponent).resolve()
    if resume_path is not None:
        inputs["resume training state"] = Path(resume_path).resolve()
    input_hashes = {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in inputs.items()}
    topology = topology_evidence(
        topology_paths,
        [input_hashes[f"training topology {index}"] for index in range(len(topology_paths))],
        sampling="without replacement within each parallel reset batch",
    )
    generation = recipe.get("train", {}).get("topology_generation")
    if generation:
        topology["origin"] = "recipe-generated bank materialized by project_jax before training"
        topology["generation"] = generation
    measurement_units = {
        "env_steps": "new environment transitions executed by this attempt across all parallel environments",
        "cumulative_environment_steps": "total transitions through the selected full-state continuation lineage",
        "cumulative_update_idx": "total rollout updates through the selected full-state continuation lineage",
        "update_idx": "completed rollout batches (one-based); legacy budget unit joint_rollout_updates",
        "team.*.update_count": "completed team PPO rounds; optimizer steps divided by minibatches and epochs",
        "team.*.updated": "whether this rollout performed the team's PPO round",
        "wall_time_s": (
            "elapsed training loop time including first-update compilation, recording and checkpoint evaluation"
        ),
        "optimization_diagnostics": "skipped team updates may emit zero losses; use team.*.updated",
    }
    seed_roles = {
        "initialization_root": int(args.seed) + 1,
        "initialization_split": "split root for init_key; split init_key into ordered blue and red parameter keys",
        "environment_reset_root": int(args.seed),
        "policy_and_rollout_root": int(args.seed) + 1,
    }
    if scripted_red:
        seed_roles["initialization_split"] = "split root; second key initializes Blue parameters directly"
    requested_steps = int(configs["blue"]["TOTAL_TIMESTEPS"])
    rollout_steps = int(configs["blue"]["NUM_ENVS"]) * int(configs["blue"]["NUM_STEPS"])
    requested_updates = requested_steps // rollout_steps
    if requested_updates < 1:
        raise ValueError("Requested budget must cover at least one complete joint rollout")
    resume_contract = None
    resume_metadata = None
    completed_updates = 0
    opponent_contract = (
        {
            "blue": {
                "kind": "checkpoint",
                "trainable": False,
                "model_sha256": input_hashes["frozen blue policy"],
                "sidecar_sha256": input_hashes["frozen blue sidecar"],
            }
        }
        if frozen_blue
        else {}
    )
    if scripted_red or frozen_blue:
        # Paths and checkpoint cadence do not affect numerical continuation.
        # Everything affecting the learner, game, RNG or original horizon does.
        resume_contract = json_configuration(
            {
                "schema_version": 1,
                "source_revision": expected_revision,
                "source_identity": provenance.facts["source_identity"]["value"],
                "dependencies": provenance.configuration["dependencies"],
                "algorithm": recipe["algorithm"],
                "blue_arch": team_recipe(recipe, "blue")["arch"],
                "config": {
                    key: value
                    for key, value in configs["blue"].items()
                    if key not in {"TOPOLOGY_BANK", "CHECKPOINT_EVERY_UPDATES"}
                },
                "topology_sha256": [input_hashes[f"training topology {index}"] for index in range(len(topology_paths))],
                "runtime_environment": {
                    key: os.environ.get(key)
                    for key in (
                        "JAX_ENABLE_X64",
                        "JAX_DEFAULT_MATMUL_PRECISION",
                        "XLA_FLAGS",
                        "JAX_PLATFORMS",
                    )
                },
            }
        )
        if frozen_blue:
            resume_contract = json_configuration(
                {
                    "schema_version": 1,
                    "procedure": "red_frozen_blue",
                    "source_revision": expected_revision,
                    "source_identity": provenance.facts["source_identity"]["value"],
                    "dependencies": provenance.configuration["dependencies"],
                    "algorithm": recipe["algorithm"],
                    "red_arch": team_recipe(recipe, "red")["arch"],
                    "team_configs": {
                        team: {
                            key: value
                            for key, value in config.items()
                            if key not in {"TOPOLOGY_BANK", "CHECKPOINT_EVERY_UPDATES"}
                        }
                        for team, config in configs.items()
                    },
                    "opponents": opponent_contract,
                    "topology_sha256": [
                        input_hashes[f"training topology {index}"] for index in range(len(topology_paths))
                    ],
                    "runtime_environment": resume_contract["runtime_environment"],
                }
            )
        if resume_path is not None:
            from jaxborg.evaluation.resume_state import read_training_state_metadata

            resume_metadata = read_training_state_metadata(
                resume_path,
                expected_sha256=input_hashes["resume training state"],
            )
            if resume_metadata.get("contract") != resume_contract:
                raise ValueError("Resume checkpoint source/configuration/topology contract differs")
            completed_updates = resume_metadata.get("completed_updates")
            if type(completed_updates) is not int or not 0 < completed_updates < requested_updates:
                raise ValueError("Resume checkpoint must have completed work and remaining declared updates")
            if resume_metadata.get("environment_steps") != completed_updates * rollout_steps:
                raise ValueError("Resume checkpoint update and environment-step counts disagree")
            if (
                not isinstance(resume_metadata.get("producer_attempt_id"), str)
                or not resume_metadata["producer_attempt_id"]
            ):
                raise ValueError("Resume checkpoint must retain its producing attempt identity")
            origin = resume_metadata.get("training_origin")
            if (
                not isinstance(origin, dict)
                or not isinstance(origin.get("attempt_id"), str)
                or not isinstance(origin.get("checkpoint"), dict)
                or origin["checkpoint"].get("environment_steps") != 0
                or not all(isinstance(origin["checkpoint"].get(key), str) for key in ("model", "sidecar"))
            ):
                raise ValueError("Resume checkpoint must preserve its original zero-step policy lineage")
            _validate_resume_lineage(root, resume_metadata, input_hashes["resume training state"], joint=frozen_blue)
    config = json_configuration(
        {
            "schema_version": 1,
            "variant": configs["blue"]["TRAIN_VARIANT"],
            "builder": builder,
            "recipe": recipe,
            "supplied_recipe": supplied_recipe,
            "overrides": overrides or {},
            "team_configs": configs,
            "seed_roles": seed_roles,
            "measurement_units": measurement_units,
            "topology": topology,
            "planned_updates": requested_updates,
            "planned_environment_steps": requested_updates * rollout_steps,
            "planned_attempt_updates": requested_updates - completed_updates,
            "planned_attempt_environment_steps": (requested_updates - completed_updates) * rollout_steps,
            "resume_contract": resume_contract,
            "resume": resume_metadata,
            "resume_archive_sha256": input_hashes.get("resume training state"),
            "backend": "jax",
            "eval_env": "jax",
            "procedure": (
                "scripts.train.algorithms.ippo_jax._run_single_training"
                if scripted_red
                else "scripts.train.algorithms.ippo_jax._run_joint_training"
            ),
            "trainable_teams": sorted(teams),
            "opponents": {"red": {"kind": "scripted", "agent": "finite_state", "trainable": False}}
            if scripted_red
            else opponent_contract,
            "initialization": {
                "kind": "full_state_resume" if resume_metadata is not None else "fresh",
                "state_retention": "initial numeric optimizer, normalizer, RNG, environment and observation state",
                "seed_roots_retained": True,
                "checkpoint_continuation": "full numeric Blue/FSM state at aligned rollout boundaries"
                if scripted_red
                else "full numeric Red/frozen-Blue state at aligned rollout boundaries"
                if frozen_blue
                else "weights only; resume unsupported",
            },
            "execution": provenance.configuration,
        }
    )
    if condition_check is not None:
        condition_check(config)
    capture = start_training(
        outside_source(root),
        track=True,
        protocol=protocol,
        trial_id=trial_id,
        retry_of=retry_of,
        requested_steps=requested_steps - completed_updates * rollout_steps,
        requested_updates=requested_updates - completed_updates,
        resolved_config=config,
        inputs=inputs,
        source_files=provenance.source_files,
        facts={
            **provenance.facts,
            **{
                key: fact(config[key])
                for key in (
                    "backend",
                    "eval_env",
                    "procedure",
                    "builder",
                    "seed_roles",
                    "topology",
                    "initialization",
                    "measurement_units",
                    "trainable_teams",
                    "opponents",
                )
            },
            "resume": fact(resume_metadata)
            if resume_metadata is not None
            else {
                "value": None,
                "reason": "Fresh training has no continuation checkpoint",
                "source": "training invocation",
            },
            "game_contract": fact(asdict(configs["blue"]["TRAIN_VARIANT"])),
        },
    )
    try:
        if json.loads((capture.directory / "source-files.json").read_text()) != provenance.hashes:
            raise ValueError("Retained training source differs from its preflight identity")
        for role, path in capture.inputs.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != input_hashes[role]:
                raise ValueError("Training input changed while being retained")
        retained_bank = [str(capture.inputs[f"training topology {index}"]) for index in range(len(topology_paths))]
        if retained_bank:
            for config in configs.values():
                config["TOPOLOGY_BANK"] = tuple(retained_bank)
            record_facts(capture, {"materialized_topologies": fact(retained_bank, "input-artifacts.json")})
        provenance.verify_sources()
        return capture, provenance
    except BaseException as error:
        error._evaluation_capture = capture
        finish_training(capture, status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", error=error)
        raise


def retain_checkpoint(capture, model, sidecar, *, steps, initial=False, joint=True, training_state=None):
    """Retain exact checkpoint and sidecar bytes under this training attempt."""
    directory = "initial" if initial else f"checkpoints/{steps}"
    model_id = add_artifact(
        capture,
        model,
        role=("initial joint policy" if initial else "joint policy checkpoint")
        if joint
        else ("initial blue policy" if initial else "blue policy checkpoint"),
        name=f"{directory}/{Path(model).name}",
    )
    sidecar_id = add_artifact(
        capture, sidecar, role="checkpoint recipe sidecar", name=f"{directory}/{Path(sidecar).name}"
    )
    checkpoint = {"model": model_id, "sidecar": sidecar_id, "environment_steps": steps}
    if training_state is not None:
        checkpoint["training_state"] = add_artifact(
            capture,
            training_state,
            role="full numeric training state",
            name=f"{directory}/{Path(training_state).name}",
        )
    record_facts(
        capture,
        {
            "initial_checkpoint" if initial else f"checkpoint_{steps}": fact(
                checkpoint,
                "retained training-owned checkpoint and sidecar artifacts",
            )
        },
    )
    return checkpoint


def retain_initial_state(capture, *, train_states, reward_norm_states, rng, env_state, obs, adaptive_state=None):
    """Save numeric initial-state diagnostics without a pickle or resume format."""
    import tempfile

    import jax
    import numpy as np

    state = {
        "optimizer": {team: value.opt_state for team, value in train_states.items()},
        "optimizer_steps": {team: value.step for team, value in train_states.items()},
        "normalizer": reward_norm_states,
        "rng": rng,
        "environment": env_state,
        "observations": obs,
        "adaptive_schedule": adaptive_state,
    }
    arrays, manifest = {}, []
    for index, (path, leaf) in enumerate(jax.tree_util.tree_flatten_with_path(state)[0]):
        array = np.asarray(jax.device_get(leaf))
        if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
            raise ValueError(f"Initial state has an unsupported nonnumeric leaf: {jax.tree_util.keystr(path)}")
        key = f"leaf_{index}"
        arrays[key] = array
        manifest.append(
            {"key": key, "path": jax.tree_util.keystr(path), "dtype": str(array.dtype), "shape": list(array.shape)}
        )
    with tempfile.TemporaryDirectory(prefix="initial-state-") as temporary:
        snapshot = Path(temporary) / "initial-state.npz"
        np.savez_compressed(snapshot, **arrays)
        state_id = add_artifact(capture, snapshot, role="numeric initial training state")
        descriptor = Path(temporary) / "initial-state.json"
        descriptor.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "leaves": manifest,
                    "meaning": "numeric initial diagnostics; not an exact continuation format",
                },
                sort_keys=True,
            )
            + "\n"
        )
        manifest_id = add_artifact(capture, descriptor, role="numeric initial state manifest")
    record_facts(
        capture,
        {
            "initial_state": fact(
                {
                    "numeric_state_artifact": state_id,
                    "manifest_artifact": manifest_id,
                    "optimizer": True,
                    "normalizer": True,
                    "rng": True,
                    "environment": True,
                    "observations": True,
                    "adaptive_schedule": adaptive_state is not None,
                    "exact_continuation_supported": False,
                },
                "owned initial state snapshot and numeric leaf manifest",
            )
        },
    )
