"""Execute a YAML-configured frozen-defender response campaign using existing trainers/evaluators."""

# ruff: noqa: E402

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
if not os.environ.get("SLURM_JOB_ID"):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Imported environment modules create JAX constants before checkpoint verification.
# Keep this long-lived coordinator from reserving a large GPU pool. Restore the
# caller's allocation setting for each real training/evaluation subprocess.
CHILD_PREALLOCATE = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import copy
import csv
import fcntl
import json
import shlex
import subprocess
import time
import uuid

import jax
import numpy as np
import yaml

from jaxborg.checkpoint import load_jax_bundle, read_sidecar
from jaxborg.evaluation.cia.fixed_topology import canonical_topology_fingerprint
from jaxborg.recipe import load as load_recipe
from jaxborg.recipe import team_recipe
from jaxborg.research_tracking import parameter_hash
from jaxborg.response_campaign import load_campaign
from jaxborg.response_oracle import (
    assert_recipe_contract,
    budget,
    gpu_lock_provenance,
    paired_gap,
    select_candidate,
    source_specific_recipe,
)
from jaxborg.scenarios.cc4.topology_cli import export_generated
from jaxborg.tracking import (
    Run,
    assigned_devices,
    dependency_snapshot,
    digest,
    experiment_root,
    file_hash,
    git,
    input_artifact,
    read_manifest,
    resolve_artifact,
    serializable,
    tracked_entrypoint,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(serializable(value), indent=2) + "\n")
    temporary.replace(path)


def dependency_identity():
    return dict(lockfile_sha256=file_hash(ROOT / "uv.lock"), installed_sha256=digest(dependency_snapshot()))


def response_teams(manifest):
    team = manifest.get("trainable_team", "red")
    if team not in ("blue", "red"):
        raise ValueError("trainable team must be blue or red")
    return team, "red" if team == "blue" else "blue"


def sidecar_path(source):
    stem = source.stem.removeprefix("model_")
    return next(
        source.with_name(f"recipe_{stem}.{suffix}")
        for suffix in ("yaml", "yml")
        if source.with_name(f"recipe_{stem}.{suffix}").is_file()
    )


def source_check(source, *, expected_steps=9600000, expected_seed=42):
    sidecar = read_sidecar(source)
    bundle = load_jax_bundle(source)
    if set(bundle.policies) != {"blue", "red"}:
        raise ValueError("source must contain both original policies")
    for team, dims in [("blue", (450, 242)), ("red", (706, 1106))]:
        entry = bundle.policies[team]
        if (entry.obs_dim, entry.action_dim) != dims:
            raise ValueError(f"incompatible {team} source contract")
    for provenance in (bundle.provenance, sidecar.get("run", {})):
        if provenance.get("total_steps") != expected_steps or provenance.get("seed") != expected_seed:
            raise ValueError("wrong original paired source provenance")
    return bundle, sidecar


@tracked_entrypoint
def prepare(args):
    # Input generation and import are CPU operations; research rollouts happen only in Slurm.
    config, defender, randomness = load_campaign(args.campaign, args.defender)
    team = config["challenger"].get("team", "red")
    frozen_team = "red" if team == "blue" else "blue"
    if experiment_root() != Path(config["tracking"]["root"]).resolve():
        raise ValueError("configured campaign store differs from JAXBORG_EXP_DIR")
    os.environ["JAXBORG_MLFLOW_EXPERIMENT"] = config["tracking"]["experiment"]
    args.source = defender["checkpoint"]
    args.source_steps, args.source_seed = defender["steps"], defender["seed"]
    args.report_dir = defender["report_dir"]
    args.challenger_template_model = config["challenger"]["template_model"]
    source = Path(args.source).resolve()
    bundle, sidecar = source_check(source, expected_steps=args.source_steps, expected_seed=args.source_seed)
    template_model = Path(args.challenger_template_model).resolve() if args.challenger_template_model else None
    challenger_source = source_check(template_model)[1] if template_model else None
    report_dir = Path(args.report_dir).resolve()
    if (report_dir / "manifest.json").exists():
        raise ValueError("report already has a campaign manifest; use it or choose a new --report-dir")
    campaign = f"{config['name']}-{args.defender}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    study = experiment_root() / "campaigns" / campaign
    study.mkdir(parents=True, exist_ok=False)
    original_sha = sidecar["run"]["git_commit"]
    generator_files = ["src/jaxborg/scenarios/cc4/topology.py", "src/jaxborg/scenarios/cc4/topology_cli.py"]
    for name in generator_files:
        if subprocess.check_output(["git", "show", f"{original_sha}:{name}"], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError(f"source generator differs: {name}")
    environment = gpu_lock_provenance(
        subprocess.check_output(["git", "show", f"{original_sha}:uv.lock"], cwd=ROOT),
        (ROOT / "uv.lock").read_bytes(),
    )
    topology = study / "topology-seed0.npz"
    export_generated(0, topology)
    fingerprint = canonical_topology_fingerprint(topology)
    owner = Run(
        {"meta": {"name": campaign}, "algorithm": "ippo"},
        backend="cpu",
        kind="input",
        inputs=[
            input_artifact(source, role="original Blue/Red pair"),
            input_artifact(topology, role="target topology"),
        ]
        + ([input_artifact(template_model, role="fixed IPPO challenger recipe source")] if template_model else []),
        config={
            "source_sha": original_sha,
            "topology_generation": {"generator": "jax", "seed": 0},
            "topology_fingerprint": fingerprint,
        },
    )
    source_sidecar_name = sidecar_path(source).name
    owner.publish(owner.input_path(0).with_name(source_sidecar_name), f"source/{source_sidecar_name}")
    source_ref = owner.publish(
        owner.input_path(0),
        f"source/{source.name}",
        sidecar=f"source/{source_sidecar_name}",
        step=args.source_steps,
    )
    topology_ref = owner.publish(topology, "topologies/topology-seed0.npz")
    source_path, topology_path = resolve_artifact(source_ref), resolve_artifact(topology_ref)
    campaign_reference = owner.publish(Path(args.campaign), "protocol/campaign.yaml")
    challenger_reference = None
    if template_model:
        template_sidecar_name = sidecar_path(template_model).name
        owner.publish(
            owner.input_path(2).with_name(template_sidecar_name), f"challenger-source/{template_sidecar_name}"
        )
        challenger_reference = owner.publish(
            owner.input_path(2),
            f"challenger-source/{template_model.name}",
            sidecar=f"challenger-source/{template_sidecar_name}",
            step=9600000,
        )
    recipe = source_specific_recipe(
        sidecar,
        source_path,
        topology_path,
        expected_source_steps=args.source_steps,
        expected_source_seed=args.source_seed,
        challenger_source=challenger_source,
        requested_steps=config["training"]["requested_steps"],
        trainable_team=team,
    )
    recipe_paths = {}
    for seed in config["training"]["seeds"]:
        path = study / f"recipe-{team}-seed{seed}.yaml"
        variant = copy.deepcopy(recipe)
        variant["meta"]["name"] = f"{campaign}-{team}-seed{seed}"
        path.write_text(yaml.safe_dump(serializable(variant), sort_keys=False))
        load_recipe(path)
        recipe_paths[str(seed)] = str(path)
        owner.publish(path, f"protocol/{path.name}")
    eval_recipe = study / "recipe-evaluation.yaml"
    eval_recipe.write_text(yaml.safe_dump(serializable(recipe), sort_keys=False))
    load_recipe(eval_recipe)
    owner.publish(eval_recipe, "protocol/recipe-evaluation.yaml")
    manifest = {
        "schema_version": 1,
        "campaign_configuration": config,
        "campaign_configuration_reference": campaign_reference,
        "tracking": config["tracking"],
        "resources": config["resources"],
        "smoke_configuration": config["smoke"],
        "campaign": campaign,
        "status": "prepared; research adapter GPU smoke unmet",
        "trainable_team": team,
        "frozen_team": frozen_team,
        "question": f"Can fresh {team} improve against original frozen {frozen_team}?",
        "source": {
            "original_path": str(source),
            "algorithm": sidecar["algorithm"],
            "checkpoint": source_ref,
            "sidecar": read_sidecar(source_path)["run"],
            "sha256": file_hash(source),
            "sidecar_sha256": file_hash(sidecar_path(source)),
            "sidecar_reference": f"runs:/{owner.run_id}/source/{source_sidecar_name}",
            "original_training_seed": args.source_seed,
            "original_training_steps": args.source_steps,
            "policies": {
                team: {
                    "architecture": entry.arch,
                    "obs_dim": entry.obs_dim,
                    "action_dim": entry.action_dim,
                    "parameter_sha256": parameter_hash(entry.weights),
                }
                for team, entry in bundle.policies.items()
            },
        },
        "challenger": {
            "algorithm": "ippo",
            "architecture": team_recipe(recipe, team)["arch"],
            "optimizer": team_recipe(recipe, team)["core"],
            "recipe_source": challenger_reference or source_ref,
            "recipe_source_sha256": file_hash(template_model or source),
            "initialization": f"fresh weights for every training seed; source {team} weights are never loaded",
        },
        "game": {
            "rules": "cc4_stock",
            "episode_length": 500,
            "enhanced_observations": True,
            "blue_observation_version": 2,
            "reward": "zero_sum",
            "policies": "stochastic throughout",
            "topology": topology_ref,
            "topology_sha256": file_hash(topology_path),
            "topology_fingerprint": fingerprint,
            "topology_generator": "jax",
            "topology_generator_seed": 0,
            "source_generator_matches": True,
            "source_generator_and_uv_lock_match": environment["identical_lockfile"],
            "layout_count": 1,
            "topology_reconstruction": (
                "regenerated with identical generator and unchanged existing dependencies; snapshot unavailable"
            ),
        },
        "randomness": randomness,
        "oracle_budget_per_attempt": budget(recipe),
        "training_seeds": list(config["training"]["seeds"]),
        "validation_episodes": len(randomness["validation_episode_roots"]),
        "final_test_episodes": len(randomness["final_test_episode_roots"]),
        "selection": {
            "rule": (
                f"{'highest' if team == 'blue' else 'lowest'} validation mean Blue return; original {team} fallback"
            ),
            "candidates": ["original", *[f"seed-{seed}" for seed in config["training"]["seeds"]]],
            "tie_break": ["original", *[f"seed-{seed}" for seed in config["training"]["seeds"]]],
            "checkpoints_per_attempt": "final only; no test scores inspected during selection",
        },
        "confidence_interval": {
            "method": "paired episode bootstrap percentile 95%",
            "unit": "aligned test episode",
            "resamples": config["bootstrap"]["resamples"],
            "seed": config["bootstrap"]["seed"],
        },
        "source_revision": git("rev-parse", "HEAD"),
        "research_base_revision": "2367e284d6e54ebadc277c692660c80219cbdc93",
        "logging_base_revision": "55ee8bd878bce4261aa24f3e199ed277bdcdda50",
        "dependencies": dependency_identity(),
        "source_environment_comparison": environment,
        "protocol_owner_run_id": owner.run_id,
        "study_dir": str(study),
        "report_dir": str(report_dir),
        "collection_readme": config["collection_readme"],
        "recipes": recipe_paths,
        "recipe_hashes": {seed: file_hash(path) for seed, path in recipe_paths.items()},
        "eval_recipe": str(eval_recipe),
        "eval_recipe_sha256": file_hash(eval_recipe),
    }
    protocol_ref = owner.write_json("protocol/manifest.json", manifest)
    owner.export("protocol/manifest.json", report_dir / "manifest.json")
    write_json(study / "manifest.json", manifest)
    print(f"Prepared manifest: {report_dir / 'manifest.json'}\nCanonical protocol: {protocol_ref}")
    return manifest


def load_protocol(path):
    manifest = json.loads(Path(path).read_text())
    if git("rev-parse", "HEAD") != manifest["source_revision"]:
        raise ValueError("campaign requires its pinned full source revision")
    if dependency_identity() != manifest["dependencies"]:
        raise ValueError("campaign dependency environment changed")
    if file_hash(resolve_artifact(manifest["source"]["checkpoint"])) != manifest["source"]["sha256"]:
        raise ValueError("source checkpoint hash changed")
    if file_hash(resolve_artifact(manifest["game"]["topology"])) != manifest["game"]["topology_sha256"]:
        raise ValueError("topology hash changed")
    for seed, path in manifest["recipes"].items():
        if file_hash(path) != manifest["recipe_hashes"][seed]:
            raise ValueError("prespecified training recipe changed")
        recipe = yaml.safe_load(Path(path).read_text())
        assert_recipe_contract(recipe)
        if recipe["train"]["teams"] != response_teams(manifest)[0]:
            raise ValueError("recipe response direction differs from manifest")
    if file_hash(manifest["eval_recipe"]) != manifest["eval_recipe_sha256"]:
        raise ValueError("evaluation recipe changed")
    return manifest


def command(*arguments):
    argv = [sys.executable, *map(str, arguments)]
    print("Executing: " + shlex.join(argv), flush=True)
    environment = dict(os.environ)
    if CHILD_PREALLOCATE is None:
        environment.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    else:
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = CHILD_PREALLOCATE
    subprocess.run(argv, cwd=ROOT, env=environment, check=True)


def evaluate(manifest, candidate, split, seeds, *, recipe=None, reuse=False):
    team, _ = response_teams(manifest)
    destination = Path(manifest["study_dir"]) / f"{split}-{candidate[0]}.json"
    argv = [
        "scripts/eval/eval_matchup.py",
        "--recipe",
        recipe or manifest["eval_recipe"],
        "--policy-backend",
        "jax",
        "--blue-path",
        candidate[1] if team == "blue" else manifest["source"]["checkpoint"],
        "--red-path",
        manifest["source"]["checkpoint"] if team == "blue" else candidate[1],
        "--episodes-per-seed",
        "1",
        "--seeds",
        ",".join(map(str, seeds)),
        "--name",
        f"{manifest['campaign']}-{split}-{candidate[0]}",
        "--output",
        destination,
    ]
    if reuse:
        argv += ["--reuse"]
    command(*argv)
    row = json.loads(destination.read_text())
    reference = f"runs:/{row['eval_id']}/evaluations/result.json"
    return checked_evaluation(reference, seeds), reference


def checked_evaluation(reference, seeds):
    path = resolve_artifact(reference)
    row = json.loads(path.read_text())
    owner = read_manifest(row["eval_id"])[0]
    if owner["status"] != "FINISHED" or owner["kind"] != "evaluation":
        raise ValueError("evaluation is incomplete")
    if row["per_episode_seeds"] != list(seeds) or row["n_episodes"] != len(seeds):
        raise ValueError("evaluation episode plan differs")
    values = np.asarray(row["per_episode_blue_returns"], dtype=float)
    if not np.all(np.isfinite(values)) or not np.isclose(values.mean(), row["blue_mean_return"]):
        raise ValueError("saved return summary is invalid")
    if not np.array_equal(-values, np.asarray(row["per_episode_red_returns"])):
        raise ValueError("evaluation is not zero sum")
    if row["variant"] != "cc4_stock" or not row["stochastic"]:
        raise ValueError("evaluation game differs from the protocol")
    return row


def train(manifest, seed, recipe, *, label=None):
    team, frozen_team = response_teams(manifest)
    name = label or f"seed-{seed}"
    destination = Path(manifest["study_dir"]) / f"training-{name}.json"
    command(
        "scripts/train/algorithms/ippo_jax.py",
        "--recipe",
        recipe,
        "--seed",
        seed,
        "--tag",
        f"{manifest['campaign']}-{name}",
        "--run-result",
        destination,
    )
    result = json.loads(destination.read_text())
    owner = read_manifest(result["run_id"])[0]
    if owner["status"] != "FINISHED":
        raise ValueError("training did not finish")
    # The controller remains alive between GPU subprocesses. Verify weights on
    # CPU so its JAX allocator does not reserve most of the next child's GPU.
    with jax.default_device(jax.devices("cpu")[0]):
        saved = load_jax_bundle(resolve_artifact(result["final_checkpoint"]))
    expected_frozen = manifest["source"]["policies"][frozen_team]["parameter_sha256"]
    if parameter_hash(saved.policies[frozen_team].weights) != expected_frozen:
        raise ValueError(f"saved {frozen_team} is not the original frozen source")
    checks = owner["parameter_checks"]
    if checks[frozen_team]["changed"] or not checks[team]["changed"]:
        raise ValueError(f"expected frozen {frozen_team} and updated {team}")
    if parameter_hash(saved.policies[team].weights) != checks[team]["current_sha256"]:
        raise ValueError(f"saved {team} differs from trained {team}")
    return result


@tracked_entrypoint
def smoke(args):
    manifest = load_protocol(args.manifest)
    team, frozen_team = response_teams(manifest)
    devices = assigned_devices()
    smoke_seeds = manifest["randomness"]["smoke_episode_roots"]
    config = manifest["smoke_configuration"]
    baseline, baseline_ref = evaluate(
        manifest, ("original", manifest["source"]["checkpoint"]), "smoke-baseline", smoke_seeds
    )
    recipe = yaml.safe_load(Path(manifest["recipes"][str(manifest["training_seeds"][0])]).read_text())
    recipe["train"]["total_timesteps"] = config["requested_steps"]
    recipe["jax"].update(
        {key: config[key] for key in ("num_envs", "num_minibatches", "update_epochs")}, checkpoint_every_updates=1
    )
    recipe["meta"]["name"] = manifest["campaign"] + "-smoke"
    path = Path(manifest["study_dir"]) / "recipe-smoke.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    result = train(manifest, config["training_seed"], path, label="smoke")
    trained, trained_ref = evaluate(manifest, ("trained", result["final_checkpoint"]), "smoke-trained", smoke_seeds)
    owner = Run(
        {"meta": {"name": manifest["campaign"] + "-smoke-check"}},
        backend="jax",
        kind="comparison",
        inputs=[
            input_artifact(baseline_ref, role="original baseline smoke"),
            input_artifact(result["final_checkpoint"], role=f"saved/reloaded frozen-{frozen_team} training smoke"),
            input_artifact(trained_ref, role="saved/reloaded trained matchup smoke"),
        ],
    )
    verification = {
        "verified": True,
        "canonical_reference": f"runs:/{owner.run_id}/verification/smoke.json",
        "source_revision": manifest["source_revision"],
        "dependencies": manifest["dependencies"],
        "manifest_hash": file_hash(args.manifest),
        "devices": devices,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "baseline_evaluation": baseline_ref,
        "training": result,
        "trained_evaluation": trained_ref,
        "original_blue_mean": baseline["blue_mean_return"],
        "trained_blue_mean": trained["blue_mean_return"],
        "checks": f"actual enhanced-v2 GPU rollout; exact {frozen_team}; changed {team}; both policies saved/reloaded",
        "trainable_team": team,
        "frozen_team": frozen_team,
    }
    owner.write_json("verification/smoke.json", verification)
    owner.export("verification/smoke.json", Path(manifest["report_dir"]) / "smoke.json")
    owner.export("verification/smoke.json", Path(manifest["study_dir"]) / "smoke.json")
    print("Research adapter GPU smoke verified.", flush=True)


def state_file(manifest):
    return Path(manifest["study_dir"]) / "state.json"


def require_smoke(manifest, manifest_path):
    verification = json.loads((Path(manifest["study_dir"]) / "smoke.json").read_text())
    if file_hash(resolve_artifact(verification["canonical_reference"])) != file_hash(
        Path(manifest["study_dir"]) / "smoke.json"
    ):
        raise ValueError("smoke export differs from its canonical completed artifact")
    if not verification["verified"] or verification["manifest_hash"] != file_hash(manifest_path):
        raise ValueError("verified smoke does not cover this exact protocol")
    if (
        verification["source_revision"] != manifest["source_revision"]
        or verification["dependencies"] != dependency_identity()
    ):
        raise ValueError("smoke source or environment differs")
    resolve_artifact(verification["training"]["final_checkpoint"])
    return verification


def dry_run(args):
    manifest = load_protocol(args.manifest)
    print(json.dumps(serializable(manifest), indent=2))
    print(
        f"GPU campaign: smoke, then {len(manifest['training_seeds'])} training attempts, validation and paired tests."
    )
    print(
        "Resources: "
        + json.dumps(manifest["resources"])
        + "; each attempt: "
        + json.dumps(manifest["oracle_budget_per_attempt"])
    )
    print("Smoke: " + shlex.join([sys.executable, str(Path(__file__).resolve()), "smoke", "--manifest", args.manifest]))
    print("Pilot: " + shlex.join([sys.executable, str(Path(__file__).resolve()), "run", "--manifest", args.manifest]))


@tracked_entrypoint
def execute(args):
    manifest = load_protocol(args.manifest)
    team, _ = response_teams(manifest)
    training_seeds = manifest["training_seeds"]
    validation_seeds = manifest["randomness"]["validation_episode_roots"]
    test_seeds = manifest["randomness"]["final_test_episode_roots"]
    require_smoke(manifest, args.manifest)
    assigned_devices()
    lock = (Path(manifest["study_dir"]) / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = state_file(manifest)
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "campaign": manifest["campaign"],
            "training": {},
            "validation": {},
            "test": {},
            "started_epoch": time.time(),
            "slurm_job_ids": [],
            "status": "running",
        }
    )
    if state["status"] == "complete":
        print("Pilot already complete; verifying and regenerating its report.")
        return aggregate(manifest, state)
    state["slurm_job_ids"].append(os.environ["SLURM_JOB_ID"])
    write_json(path, state)
    candidates = {"original": manifest["source"]["checkpoint"]}
    for seed in training_seeds:
        candidate = f"seed-{seed}"
        if candidate not in state["training"]:
            result = train(manifest, seed, manifest["recipes"][str(seed)])
            if result["actual_steps"] != manifest["oracle_budget_per_attempt"]["completed_steps"]:
                raise ValueError("training completed the wrong budget")
            state["training"][candidate] = result
            write_json(path, state)
        result = state["training"][candidate]
        resolve_artifact(result["final_checkpoint"])
        candidates[candidate] = result["final_checkpoint"]
    scores = {}
    for candidate, reference in candidates.items():
        if candidate not in state["validation"]:
            row, evaluation = evaluate(manifest, (candidate, reference), "validation", validation_seeds, reuse=True)
            state["validation"][candidate] = evaluation
            write_json(path, state)
        row = checked_evaluation(state["validation"][candidate], validation_seeds)
        scores[candidate] = row["blue_mean_return"]
    selected = select_candidate(scores, training_seeds, trainable_team=team)
    if "selection" not in state:
        state["selection"] = {
            "candidate": selected,
            "checkpoint": candidates[selected],
            "checkpoint_sha256": file_hash(resolve_artifact(candidates[selected])),
            "validation_scores": scores,
            "frozen_before_test_epoch": time.time(),
        }
        write_json(path, state)
    elif state["selection"]["candidate"] != selected:
        raise ValueError("frozen selection changed")
    # Final-test outcomes are accessed only after the selected candidate is persisted.
    for name, reference in [("original", candidates["original"]), ("selected", state["selection"]["checkpoint"])]:
        if name not in state["test"]:
            _, evaluation = evaluate(manifest, (name, reference), "final-test", test_seeds, reuse=True)
            state["test"][name] = evaluation
            write_json(path, state)
    state["status"] = "evaluations_finished"
    state["finished_epoch"] = time.time()
    write_json(path, state)
    aggregate(manifest, state)
    state["status"] = "complete"
    write_json(path, state)


def csv_file(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@tracked_entrypoint
def aggregate(manifest, state):
    team, frozen_team = response_teams(manifest)
    team_label, frozen_label = team.title(), frozen_team.title()
    gain_key = f"{team}_improvement"
    direction = 1 if team == "blue" else -1
    training_seeds = manifest["training_seeds"]
    validation_seeds = manifest["randomness"]["validation_episode_roots"]
    test_seeds = manifest["randomness"]["final_test_episode_roots"]
    bootstrap_samples = manifest["confidence_interval"]["resamples"]
    bootstrap_seed = manifest["confidence_interval"]["seed"]
    attempt_budget = manifest["oracle_budget_per_attempt"]
    baseline = checked_evaluation(state["test"]["original"], test_seeds)
    selected = checked_evaluation(state["test"]["selected"], test_seeds)
    # Policy/topology/contract evidence is checked independently from return alignment.
    for key in ("original", "selected"):
        evidence = read_manifest(json.loads(resolve_artifact(state["test"][key]).read_text())["eval_id"])[0]
        if evidence["inputs"][1 if team == "blue" else 0]["sha256"] != manifest["source"]["sha256"]:
            raise ValueError(f"final evaluation frozen {frozen_team} source differs")
        if evidence["inputs"][2]["sha256"] != manifest["game"]["topology_sha256"]:
            raise ValueError("final evaluation topology differs")
        if evidence["source"]["git_commit"] != manifest["source_revision"]:
            raise ValueError("final evaluator revision differs")
    scores = {
        name: checked_evaluation(reference, validation_seeds)["blue_mean_return"]
        for name, reference in state["validation"].items()
    }
    if select_candidate(scores, training_seeds, trainable_team=team) != state["selection"]["candidate"]:
        raise ValueError("report selection disagrees with validation-only rule")
    if set(state["training"]) != {f"seed-{seed}" for seed in training_seeds}:
        raise ValueError("report requires every prespecified training attempt")
    expected_challenger = {
        "original": manifest["source"]["sha256"],
        "selected": file_hash(resolve_artifact(state["selection"]["checkpoint"])),
    }
    for key in ("original", "selected"):
        row = json.loads(resolve_artifact(state["test"][key]).read_text())
        owner = read_manifest(row["eval_id"])[0]
        if owner["inputs"][0 if team == "blue" else 1]["sha256"] != expected_challenger[key]:
            raise ValueError(f"final evaluated {team} differs from its frozen selected identity")
    gap = paired_gap(
        baseline["per_episode_blue_returns"],
        selected["per_episode_blue_returns"],
        baseline["per_episode_seeds"],
        selected["per_episode_seeds"],
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        trainable_team=team,
    )
    inputs = [input_artifact(reference, role=f"final {name} matchup") for name, reference in state["test"].items()]
    inputs += [input_artifact(reference, role=f"validation {name}") for name, reference in state["validation"].items()]
    inputs += [input_artifact(result["final_checkpoint"], role=name) for name, result in state["training"].items()]
    report = Run(
        {"meta": {"name": manifest["campaign"] + "-report"}},
        backend="cpu",
        kind="comparison",
        inputs=inputs,
        config={"protocol": manifest, "selection": state["selection"], "statistic": gap},
    )
    report_dir = Path(manifest["report_dir"])
    report.write_json("results/summary.json", gap)
    report.write_json("results/state.json", state)
    rows = [
        {"episode_seed": seed, "baseline_blue_return": b, "selected_blue_return": r, gain_key: direction * (r - b)}
        for seed, b, r in zip(test_seeds, baseline["per_episode_blue_returns"], selected["per_episode_blue_returns"])
    ]
    csv_file(report.path("results/per-episode.csv"), rows)
    csv_file(
        report.path("results/summary.csv"),
        [
            dict(
                matchup="Original Blue vs Original Red",
                mean_blue_return=gap["baseline_blue_mean"],
                episodes=len(test_seeds),
            ),
            dict(
                matchup="Selected Blue vs Original Red" if team == "blue" else "Original Blue vs selected Red",
                mean_blue_return=gap["selected_blue_mean"],
                episodes=len(test_seeds),
            ),
        ],
    )
    validation_rows = []
    for name, reference in state["validation"].items():
        row = checked_evaluation(reference, validation_seeds)
        validation_rows.append(
            dict(
                candidate=name,
                mean_blue_return=row["blue_mean_return"],
                episodes=len(validation_seeds),
                evaluation=reference,
                selected=name == state["selection"]["candidate"],
            )
        )
    csv_file(report.path("results/validation.csv"), validation_rows)
    learning_rows = []
    for name, result in state["training"].items():
        metrics = resolve_artifact(f"runs:/{result['run_id']}/logs/metrics.jsonl")
        for line in metrics.read_text().splitlines():
            row = json.loads(line)
            learning_rows.append(
                {
                    "candidate": name,
                    **{
                        key: row[key]
                        for key in ("env_steps", "wall_time_s", "team.red.return", "team.blue.return", "throughput_sps")
                    },
                }
            )
    csv_file(report.path("results/learning-curves.csv"), learning_rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.bar(
        [f"Original {team_label}", f"Selected {team_label}"],
        [gap["baseline_blue_mean"], gap["selected_blue_mean"]],
        color=["#577590", "#b64a3b"],
    )
    axis.set_ylabel("Mean raw Blue episode return")
    axis.set_title(f"One frozen {frozen_label}, {len(test_seeds)} paired test episodes")
    figure.tight_layout()
    figure.savefig(report.path("plots/comparison.png"), dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    for name in state["training"]:
        curve = [row for row in learning_rows if row["candidate"] == name]
        axis.plot([row["env_steps"] / 1e6 for row in curve], [row[f"team.{team}.return"] for row in curve], label=name)
    axis.set_xlabel(f"Extra {team_label} training steps (millions)")
    axis.set_ylabel(f"Raw {team_label} rollout return (500 steps)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(report.path("plots/learning-curves.png"), dpi=160)
    plt.close(figure)
    for name in ("per-episode.csv", "summary.csv", "validation.csv", "learning-curves.csv"):
        report.publish(report.path("results/" + name), "results/" + name)
        report.export("results/" + name, report_dir / name)
    for name in ("comparison.png", "learning-curves.png"):
        report.publish(report.path("plots/" + name), "plots/" + name)
        report.export("plots/" + name, report_dir / name)
    report.export("results/summary.json", report_dir / "summary.json")
    report.export("results/state.json", report_dir / "state.json")
    tail_rows = []
    for name in state["training"]:
        curve = [row for row in learning_rows if row["candidate"] == name]
        previous = float(np.mean([row[f"team.{team}.return"] for row in curve[-40:-20]]))
        final = float(np.mean([row[f"team.{team}.return"] for row in curve[-20:]]))
        tail_rows.append(
            {
                "candidate": name,
                f"preceding_20_update_mean_{team}_return": previous,
                f"final_20_update_mean_{team}_return": final,
                "change": final - previous,
            }
        )
    csv_file(report.path("results/learning-tail.csv"), tail_rows)
    report.publish(report.path("results/learning-tail.csv"), "results/learning-tail.csv")
    report.export("results/learning-tail.csv", report_dir / "learning-tail.csv")
    tail_text = "; ".join(f"{row['candidate']}: {row['change']:+.2f}" for row in tail_rows)
    training_seconds = sum(
        read_manifest(x["run_id"])[0]["ended_utc"] is not None
        and json.loads(resolve_artifact(f"runs:/{x['run_id']}/logs/metrics.jsonl").read_text().splitlines()[-1])[
            "wall_time_s"
        ]
        for x in state["training"].values()
    )
    evaluation_seconds = sum(
        json.loads(resolve_artifact(ref).read_text())["wall_time_s"]
        for ref in list(state["validation"].values()) + list(state["test"].values())
    )
    if state["selection"]["candidate"] == "original":
        interpretation = (
            f"No fresh {team_label} candidate beat Original {team_label} on validation. "
            f"Original {team_label} was selected for the paired final test, so the zero gap compares "
            "the original policy with itself. Fresh candidates were not evaluated on final-test episodes."
        )
    else:
        interpretation = (
            f"Selected fresh {team_label} improved against the original frozen {frozen_label}."
            if gap[gain_key] > 0
            else f"Selected fresh {team_label} did not improve on the final-test episodes."
        )
    if gap["ci95"][0] <= 0 <= gap["ci95"][1] and gap[gain_key] != 0:
        interpretation += " The interval includes zero, so the observed change remains uncertain."
    text = f"""# Frozen {frozen_label} and {len(training_seeds)} fresh {team_label} challengers

{interpretation} The observed {team_label} improvement is **{gap[gain_key]:.2f} points**,
with a paired 95% bootstrap interval of **[{gap["ci95"][0]:.2f}, {gap["ci95"][1]:.2f}]**.
Higher Blue return is better for the defender.

| Matchup | Mean raw Blue return | Test episodes |
| --- | ---: | ---: |
| Original Blue vs Original Red | {gap["baseline_blue_mean"]:.2f} | {len(test_seeds)} |
| Selected {team_label} vs original {frozen_label} | {gap["selected_blue_mean"]:.2f} | {len(test_seeds)} |

![Final paired comparison](comparison.png)

Original {manifest["source"].get("algorithm", "ippo").upper()} training used seed
{manifest["source"]["original_training_seed"]} and
{manifest["source"]["original_training_steps"]:,} source steps. Each fresh IPPO {team_label} requested
{attempt_budget["requested_steps"]:,} extra steps and completed {attempt_budget["completed_steps"]:,}
({attempt_budget["updates"]} updates of {attempt_budget["steps_per_update"]:,} steps).
{frozen_label} stayed exactly unchanged. The source and every challenger use the same
stock CC4 game, enhanced-v2 observations, one JAX-generated topology (seed 0),
500-step episodes, zero-sum rewards and stochastic policy actions. This is
{len(test_seeds)} fresh episodes on one network. It is a different target game from Stage A's CIA/resilience evaluation.

Selection compared Original {team_label} and {len(training_seeds)} final {team_label} checkpoints
on {len(validation_seeds)} distinct validation episodes per candidate.
The {"highest" if team == "blue" else "lowest"} mean Blue return won, with exact
ties favoring Original {team_label} then the recorded seed order. The identity was saved
before testing. Final matchups share the {len(test_seeds)} episode roots, although differing
actions can produce different trajectories. The 95% percentile bootstrap samples
aligned episode differences {bootstrap_samples:,} times (seed {bootstrap_seed}).
It measures evaluation uncertainty conditional on these selected models from
one original training run. Negative observed gains are retained. This is a
one-sided response search and does not prove equilibrium or measure two-sided NashConv.

![Three challenger learning curves](learning-curves.png)

The budget bounds the search. Learning curves report actual raw 500-step rollout
returns. Changes from the preceding 20-update mean to the final 20-update mean
were {tail_text} {team_label} return points; [the exact tail table](learning-tail.csv)
records those descriptive comparisons. Continuing positive changes suggest the
search may still be improving; noisy flat tails cannot prove convergence.
If further improvement is plausible from those curves, the concrete
next experiment is another {attempt_budget["requested_steps"]:,} requested steps per fresh attempt against this
same {frozen_label} under a newly prespecified protocol; it is not included in this pilot.
Portable weights omit optimizer/PRNG/environment state, so a new launch is a
new attempt, not an optimizer resume.

Training update loops measured {training_seconds / 3600:.2f} hours including
first-update compilation; {len(state["validation"]) + len(state["test"])} planned
evaluation matchups measured {evaluation_seconds / 3600:.2f} hours including
compilation in their processes. Pipeline elapsed time was
{(state["finished_epoch"] - state["started_epoch"]) / 3600:.2f} hours; queue delay and
smoke are separate. No dollar tariff is available. Slurm job IDs: {state["slurm_job_ids"]}.
Code SHA: `{manifest["source_revision"]}`. [Resolved manifest](manifest.json),
[GPU smoke](smoke.json), [selection and run lineage](state.json),
[validation](validation.csv), [episode-level paired data](per-episode.csv),
[summary](summary.csv), [learning curves](learning-curves.csv).

Large models and logs remain under the explicit external experiment root.
Report owner: `runs:/{report.run_id}`. Source: `{manifest["source"]["checkpoint"]}`.
Selected {team_label}: `{state["selection"]["checkpoint"]}`.

This report covers one response direction at one source snapshot. A two-sided
response gap additionally needs the independently selected opposite direction
against the original source pair, on identical paired episode roots.
"""
    readme = report.path("reports/README.md")
    readme.write_text(text)
    report.publish(readme, "reports/README.md")
    report.export("reports/README.md", report_dir / "README.md")
    handoff = report.path("reports/handoff.md")
    handoff.write_text(f"""# Response search handoff

Response report: [README.md](README.md).

Reuse the pinned implementation `{manifest["source_revision"]}`,
`src/jaxborg/response_oracle.py`, the source-specific generator and serial controller,
canonical frozen-opponent loading, checkpoint completion and freezing assertions,
independent matchup lineage, explicit fingerprint reuse, and paired episode aggregation.
The exact protocol and seed domains are in [manifest.json](manifest.json).
The selected {team_label} is `{state["selection"]["checkpoint"]}`.

Stage C needs both independent response directions for each reported source pair;
do not use final-test seeds for candidate selection. Compare equal requested and
actual search budgets and preserve original source identities when combining results.
Only completed artifacts can be reused; portable weights do not provide optimizer resume.

Rerun report from the same checkout:

```bash
.venv/bin/python scripts/experiments/response_oracle.py aggregate \
  --manifest {report_dir / "manifest.json"}
```
from the unchanged clean launch checkout with the explicit experiment root and `JAX_PLATFORMS=cpu`.
""")
    report.publish(handoff, "reports/handoff.md")
    report.export("reports/handoff.md", report_dir / "handoff.md")
    # This collection is outside the source checkout and was explicitly requested by the task.
    collection = Path(manifest.get("collection_readme", report_dir.parent.parent / "README.md"))
    if collection.exists():
        current = collection.read_text()
        target = os.path.relpath(report_dir / "README.md", collection.parent)
        label = (
            f"{manifest['source'].get('algorithm', 'ippo').upper()} {frozen_label} at "
            f"{manifest['source']['original_training_steps']:,} source steps"
        )
        prefix = "Completed Blue response" if team == "blue" else "Completed Stage B pilot"
        link = f"\n{prefix}: [{label}]({target}).\n"
        if f"]({target})" not in current:
            collection.write_text(current + link)
    print(f"Completed pilot report: {report_dir / 'README.md'}", flush=True)
    return gap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--campaign", required=True, help="declarative response campaign YAML")
    prep.add_argument("--defender", required=True, help="checkpoint name within the campaign")
    for action in ("dry-run", "smoke", "run", "pilot", "aggregate"):
        command_parser = sub.add_parser(action)
        command_parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "dry-run":
        dry_run(args)
    elif args.action == "smoke":
        smoke(args)
    elif args.action == "run":
        execute(args)
    elif args.action == "pilot":
        smoke(args)
        execute(args)
    else:
        manifest = load_protocol(args.manifest)
        aggregate(manifest, json.loads(state_file(manifest).read_text()))


if __name__ == "__main__":
    main()
