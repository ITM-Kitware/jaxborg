"""Evidence at the maintained joint trainer's outer execution boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from jaxborg.evaluation.capture_store import (
    add_artifact,
    finish_training,
    json_configuration,
    record_facts,
    start_training,
)
from jaxborg.evaluation.joint_capture import fact, outside_source
from jaxborg.evaluation.provenance import builder_identity, collect_provenance
from jaxborg.recipe import training_teams


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
    """Capture only fresh, fixed-schedule joint training with canonical contracts."""
    from jaxborg.evaluation.checkpoint_scripted_reds import CheckpointScriptedRedsSettings
    from jaxborg.evaluation.cross_play import CrossPlaySettings
    from jaxborg.evaluation.cross_seed_play import CrossSeedPlaySettings
    from jaxborg.evaluation.play_priors import PlayPriorsSettings
    from jaxborg.evaluation.scripted_red import ScriptedRedEvalSettings

    if not isinstance(expected_revision, str) or len(expected_revision) != 40:
        raise ValueError("Recorded training requires an exact pinned source revision")
    if set(training_teams(recipe)) != {"blue", "red"} or recipe.get("train", {}).get("opponents"):
        raise ValueError(
            "Recorded training currently requires fresh Blue and Red initialization; "
            "resume and warm starts are unsupported"
        )
    if any(config.get("ADAPTIVE_UPDATES") for config in configs.values()):
        raise ValueError("Recorded training currently requires a fixed update schedule")
    if any(config["TRAIN_VARIANT"].red_reward != "zero_sum" for config in configs.values()):
        raise ValueError("Recorded joint training requires zero_sum reward")
    if any(
        config.get(key)
        for config in configs.values()
        for key in ("MISSION_BANK", "PHASE_BOUNDARY_BANK", "PHASE_REWARDS_BANK")
    ):
        raise ValueError("Recorded training supports topology snapshots without auxiliary mission banks")
    evaluation = recipe.get("eval", {})
    if (
        recipe.get("mlflow", {}).get("checkpoint_eval", {}).get("every_steps", 0)
        or evaluation.get("after_training")
        or ScriptedRedEvalSettings.from_recipe(recipe).after_training
        or any(
            settings.from_recipe(recipe).enabled
            for settings in (
                CrossPlaySettings,
                PlayPriorsSettings,
                CheckpointScriptedRedsSettings,
                CrossSeedPlaySettings,
            )
        )
    ):
        raise ValueError(
            "Recorded training requires separate recorded evaluations; disable automatic evaluation suites"
        )
    project = Path(__file__).resolve().parents[3]
    provenance = collect_provenance(
        allow_dirty=allow_dirty,
        expected_revision=expected_revision,
        entrypoint=entrypoint or project / "scripts/train/algorithms/ippo_jax.py",
        source_roots=(project / "scripts/train/algorithms", *source_roots),
    )
    builder = builder_identity(env_factory, provenance)
    inputs = {}
    recipe_source = recipe.get("__source_path__")
    if recipe_source:
        inputs["supplied recipe"] = Path(recipe_source).resolve()
    topology_paths = [Path(path).resolve() for path in configs["blue"].get("TOPOLOGY_BANK") or ()]
    if not topology_paths:
        raise ValueError("Recorded joint training requires an explicit fixed topology bank")
    from jaxborg.scenarios.cc4.topology import load_topology_metadata

    inputs.update({f"training topology {index}": path for index, path in enumerate(topology_paths)})
    input_hashes = {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in inputs.items()}
    topology = {
        "sampling": "without replacement within each parallel reset batch",
        "sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in topology_paths],
        "metadata": [load_topology_metadata(path) for path in topology_paths],
        "origin": "imported fixed topology snapshots; generator evidence is retained in snapshot metadata",
    }
    seed_roles = {
        "initialization_root": int(args.seed) + 1,
        "initialization_split": "split root for init_key; split init_key into ordered blue and red parameter keys",
        "environment_reset_root": int(args.seed),
        "policy_and_rollout_root": int(args.seed) + 1,
    }
    requested_steps = int(configs["blue"]["TOTAL_TIMESTEPS"])
    rollout_steps = int(configs["blue"]["NUM_ENVS"]) * int(configs["blue"]["NUM_STEPS"])
    requested_updates = requested_steps // rollout_steps
    if requested_updates < 1:
        raise ValueError("Requested budget must cover at least one complete joint rollout")
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
            "topology": topology,
            "planned_updates": requested_updates,
            "planned_environment_steps": requested_updates * rollout_steps,
            "backend": "jax",
            "eval_env": "jax",
            "procedure": "scripts.train.algorithms.ippo_jax._run_joint_training",
            "initialization": {
                "kind": "fresh",
                "state_retention": "initial numeric optimizer, normalizer, RNG, environment and observation state",
                "seed_roots_retained": True,
                "checkpoint_continuation": "weights only; resume unsupported",
            },
            "execution": provenance.configuration,
        }
    )
    if condition_check is not None:
        condition_check(config)
    capture = start_training(
        outside_source(root),
        protocol=protocol,
        trial_id=trial_id,
        retry_of=retry_of,
        requested_steps=int(configs["blue"]["TOTAL_TIMESTEPS"]),
        requested_updates=requested_updates,
        resolved_config=config,
        inputs=inputs,
        source_files=provenance.source_files,
        facts={
            **provenance.facts,
            **{
                key: fact(config[key])
                for key in ("backend", "eval_env", "procedure", "builder", "seed_roles", "topology", "initialization")
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


def retain_checkpoint(capture, model, sidecar, *, steps, initial=False):
    """Retain exact checkpoint and sidecar bytes under this training attempt."""
    directory = "initial" if initial else f"checkpoints/{steps}"
    model_id = add_artifact(
        capture,
        model,
        role="initial joint policy" if initial else "joint policy checkpoint",
        name=f"{directory}/{Path(model).name}",
    )
    sidecar_id = add_artifact(
        capture, sidecar, role="checkpoint recipe sidecar", name=f"{directory}/{Path(sidecar).name}"
    )
    record_facts(
        capture,
        {
            "initial_checkpoint" if initial else f"checkpoint_{steps}": fact(
                {"model": model_id, "sidecar": sidecar_id, "environment_steps": steps},
                "retained training-owned checkpoint and sidecar artifacts",
            )
        },
    )
    return {"model": model_id, "sidecar": sidecar_id}


def retain_initial_state(capture, *, train_states, reward_norm_states, rng, env_state, obs):
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
                    "exact_continuation_supported": False,
                },
                "owned initial state snapshot and numeric leaf manifest",
            )
        },
    )
