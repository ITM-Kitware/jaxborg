"""Prepare, dry-run, smoke, execute, validate and report the bounded Stage B pilot."""

# ruff: noqa: E402

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
if not os.environ.get("SLURM_JOB_ID"):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

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
from jaxborg.oracle_stage_b import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    SMOKE_SEEDS,
    TEST_SEEDS,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    assert_recipe_contract,
    budget,
    gpu_lock_provenance,
    paired_gap,
    seed_protocol,
    select_candidate,
    source_specific_recipe,
)
from jaxborg.recipe import load as load_recipe
from jaxborg.research_tracking import parameter_hash
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

DEFAULT_SOURCE = Path(
    "/data/shared/jaxborg/jaxborg-harml-comparison/ippo_jax/"
    "cotraining_seed42_hmarl_comparison_50m/checkpoint_9600000.safetensors"
)
DEFAULT_REPORT = Path("/home/local/KHQ/paul.elliott/src/cyber/plans/jax/cc4/equilibrium-experiments/results/stage-b")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(serializable(value), indent=2) + "\n")
    temporary.replace(path)


def dependency_identity():
    return dict(lockfile_sha256=file_hash(ROOT / "uv.lock"), installed_sha256=digest(dependency_snapshot()))


def source_check(source):
    sidecar = read_sidecar(source)
    bundle = load_jax_bundle(source)
    if set(bundle.policies) != {"blue", "red"}:
        raise ValueError("source must contain both original policies")
    for team, dims in [("blue", (450, 242)), ("red", (706, 1106))]:
        entry = bundle.policies[team]
        if (entry.obs_dim, entry.action_dim) != dims:
            raise ValueError(f"incompatible {team} source contract")
    if bundle.provenance.get("total_steps") != 9600000 or bundle.provenance.get("seed") != 42:
        raise ValueError("wrong original paired source provenance")
    return bundle, sidecar


@tracked_entrypoint
def prepare(args):
    # Input generation and import are CPU operations; research rollouts happen only in Slurm.
    source = Path(args.source).resolve()
    bundle, sidecar = source_check(source)
    report_dir = Path(args.report_dir).resolve()
    if (report_dir / "manifest.json").exists():
        raise ValueError("report already has a campaign manifest; use it or choose a new --report-dir")
    campaign = f"stage-b-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
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
        ],
        config={
            "source_sha": original_sha,
            "topology_generation": {"generator": "jax", "seed": 0},
            "topology_fingerprint": fingerprint,
        },
    )
    owner.publish(
        owner.input_path(0).with_name("recipe_checkpoint_9600000.yaml"), "source/recipe_checkpoint_9600000.yaml"
    )
    source_ref = owner.publish(
        owner.input_path(0),
        "source/checkpoint_9600000.safetensors",
        sidecar="source/recipe_checkpoint_9600000.yaml",
        step=9600000,
    )
    topology_ref = owner.publish(topology, "topologies/topology-seed0.npz")
    source_path, topology_path = resolve_artifact(source_ref), resolve_artifact(topology_ref)
    recipe = source_specific_recipe(sidecar, source_path, topology_path)
    recipe_paths = {}
    for seed in TRAIN_SEEDS:
        path = study / f"recipe-red-seed{seed}.yaml"
        variant = copy.deepcopy(recipe)
        variant["meta"]["name"] = f"{campaign}-red-seed{seed}"
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
        "campaign": campaign,
        "status": "prepared; research adapter GPU smoke unmet",
        "question": "Can a fresh attacker reduce frozen Blue score more than its original training opponent?",
        "source": {
            "original_path": str(source),
            "checkpoint": source_ref,
            "sidecar": read_sidecar(source_path)["run"],
            "sha256": file_hash(source),
            "sidecar_sha256": file_hash(source.with_name("recipe_checkpoint_9600000.yaml")),
            "sidecar_reference": f"runs:/{owner.run_id}/source/recipe_checkpoint_9600000.yaml",
            "original_training_seed": 42,
            "original_training_steps": 9600000,
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
        "randomness": seed_protocol(),
        "oracle_budget_per_attempt": budget(recipe),
        "training_seeds": list(TRAIN_SEEDS),
        "validation_episodes": 100,
        "final_test_episodes": 600,
        "selection": {
            "rule": "lowest validation mean Blue episode return; original Red is fallback",
            "candidates": ["original", *[f"seed-{seed}" for seed in TRAIN_SEEDS]],
            "tie_break": ["original", *[f"seed-{seed}" for seed in TRAIN_SEEDS]],
            "checkpoints_per_attempt": "final only; no test scores inspected during selection",
        },
        "confidence_interval": {
            "method": "paired episode bootstrap percentile 95%",
            "unit": "aligned test episode",
            "resamples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "source_revision": git("rev-parse", "HEAD"),
        "research_base_revision": "2367e284d6e54ebadc277c692660c80219cbdc93",
        "logging_base_revision": "55ee8bd878bce4261aa24f3e199ed277bdcdda50",
        "dependencies": dependency_identity(),
        "source_environment_comparison": environment,
        "protocol_owner_run_id": owner.run_id,
        "study_dir": str(study),
        "report_dir": str(report_dir),
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
        assert_recipe_contract(yaml.safe_load(Path(path).read_text()))
    if file_hash(manifest["eval_recipe"]) != manifest["eval_recipe_sha256"]:
        raise ValueError("evaluation recipe changed")
    return manifest


def command(*arguments):
    argv = [sys.executable, *map(str, arguments)]
    print("Executing: " + shlex.join(argv), flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)


def evaluate(manifest, candidate, split, seeds, *, recipe=None, reuse=False):
    destination = Path(manifest["study_dir"]) / f"{split}-{candidate[0]}.json"
    argv = [
        "scripts/eval/eval_matchup.py",
        "--recipe",
        recipe or manifest["eval_recipe"],
        "--policy-backend",
        "jax",
        "--blue-path",
        manifest["source"]["checkpoint"],
        "--red-path",
        candidate[1],
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
    if parameter_hash(saved.policies["blue"].weights) != manifest["source"]["policies"]["blue"]["parameter_sha256"]:
        raise ValueError("saved Blue is not the original frozen source")
    checks = owner["parameter_checks"]
    if checks["blue"]["changed"] or not checks["red"]["changed"]:
        raise ValueError("expected frozen Blue and updated Red")
    if parameter_hash(saved.policies["red"].weights) != checks["red"]["current_sha256"]:
        raise ValueError("saved Red differs from trained Red")
    return result


@tracked_entrypoint
def smoke(args):
    manifest = load_protocol(args.manifest)
    devices = assigned_devices()
    baseline, baseline_ref = evaluate(
        manifest, ("original", manifest["source"]["checkpoint"]), "smoke-baseline", SMOKE_SEEDS
    )
    recipe = yaml.safe_load(Path(manifest["recipes"][str(TRAIN_SEEDS[0])]).read_text())
    recipe["train"]["total_timesteps"] = 4000
    recipe["jax"].update(num_envs=4, num_minibatches=4, update_epochs=1, checkpoint_every_updates=1)
    recipe["meta"]["name"] = manifest["campaign"] + "-smoke"
    path = Path(manifest["study_dir"]) / "recipe-smoke.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    result = train(manifest, 880001, path, label="smoke")
    trained, trained_ref = evaluate(manifest, ("trained", result["final_checkpoint"]), "smoke-trained", SMOKE_SEEDS)
    owner = Run(
        {"meta": {"name": manifest["campaign"] + "-smoke-check"}},
        backend="jax",
        kind="comparison",
        inputs=[
            input_artifact(baseline_ref, role="original baseline smoke"),
            input_artifact(result["final_checkpoint"], role="saved/reloaded frozen-Blue training smoke"),
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
        "checks": "actual enhanced-v2 GPU rollout; two Red updates; exact Blue; both policies saved/reloaded",
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
    print("Sequential GPU jobs: smoke, then 3 training attempts, validation and paired tests in one pipeline.")
    print(
        "Community partition; one GPU; no stage C. Each attempt: " + json.dumps(manifest["oracle_budget_per_attempt"])
    )
    print("Smoke: " + shlex.join([sys.executable, str(Path(__file__).resolve()), "smoke", "--manifest", args.manifest]))
    print("Pilot: " + shlex.join([sys.executable, str(Path(__file__).resolve()), "run", "--manifest", args.manifest]))


@tracked_entrypoint
def execute(args):
    manifest = load_protocol(args.manifest)
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
    for seed in TRAIN_SEEDS:
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
            row, evaluation = evaluate(manifest, (candidate, reference), "validation", VALIDATION_SEEDS, reuse=True)
            state["validation"][candidate] = evaluation
            write_json(path, state)
        row = checked_evaluation(state["validation"][candidate], VALIDATION_SEEDS)
        scores[candidate] = row["blue_mean_return"]
    selected = select_candidate(scores)
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
            _, evaluation = evaluate(manifest, (name, reference), "final-test", TEST_SEEDS, reuse=True)
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
    baseline = checked_evaluation(state["test"]["original"], TEST_SEEDS)
    selected = checked_evaluation(state["test"]["selected"], TEST_SEEDS)
    # Policy/topology/contract evidence is checked independently from return alignment.
    for key in ("original", "selected"):
        evidence = read_manifest(json.loads(resolve_artifact(state["test"][key]).read_text())["eval_id"])[0]
        if evidence["inputs"][0]["sha256"] != manifest["source"]["sha256"]:
            raise ValueError("final evaluation Blue source differs")
        if evidence["inputs"][2]["sha256"] != manifest["game"]["topology_sha256"]:
            raise ValueError("final evaluation topology differs")
        if evidence["source"]["git_commit"] != manifest["source_revision"]:
            raise ValueError("final evaluator revision differs")
    scores = {
        name: checked_evaluation(reference, VALIDATION_SEEDS)["blue_mean_return"]
        for name, reference in state["validation"].items()
    }
    if select_candidate(scores) != state["selection"]["candidate"]:
        raise ValueError("report selection disagrees with validation-only rule")
    if len(state["training"]) != 3:
        raise ValueError("report requires all three training attempts")
    expected_red = {
        "original": manifest["source"]["sha256"],
        "selected": file_hash(resolve_artifact(state["selection"]["checkpoint"])),
    }
    for key in ("original", "selected"):
        row = json.loads(resolve_artifact(state["test"][key]).read_text())
        owner = read_manifest(row["eval_id"])[0]
        if owner["inputs"][1]["sha256"] != expected_red[key]:
            raise ValueError("final evaluated Red differs from its frozen selected identity")
    gap = paired_gap(
        baseline["per_episode_blue_returns"],
        selected["per_episode_blue_returns"],
        baseline["per_episode_seeds"],
        selected["per_episode_seeds"],
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
        {"episode_seed": seed, "baseline_blue_return": b, "selected_blue_return": r, "red_improvement": b - r}
        for seed, b, r in zip(TEST_SEEDS, baseline["per_episode_blue_returns"], selected["per_episode_blue_returns"])
    ]
    csv_file(report.path("results/per-episode.csv"), rows)
    csv_file(
        report.path("results/summary.csv"),
        [
            dict(matchup="Original Blue vs Original Red", mean_blue_return=gap["baseline_blue_mean"], episodes=600),
            dict(matchup="Original Blue vs selected Red", mean_blue_return=gap["selected_blue_mean"], episodes=600),
        ],
    )
    validation_rows = []
    for name, reference in state["validation"].items():
        row = checked_evaluation(reference, VALIDATION_SEEDS)
        validation_rows.append(
            dict(
                candidate=name,
                mean_blue_return=row["blue_mean_return"],
                episodes=100,
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
        ["Original Red", "Selected Red"],
        [gap["baseline_blue_mean"], gap["selected_blue_mean"]],
        color=["#577590", "#b64a3b"],
    )
    axis.set_ylabel("Mean raw Blue episode return")
    axis.set_title("One frozen Blue, 600 paired test episodes")
    figure.tight_layout()
    figure.savefig(report.path("plots/comparison.png"), dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    for name in state["training"]:
        curve = [row for row in learning_rows if row["candidate"] == name]
        axis.plot([row["env_steps"] / 1e6 for row in curve], [row["team.red.return"] for row in curve], label=name)
    axis.set_xlabel("Extra Red training steps (millions)")
    axis.set_ylabel("Raw Red rollout return (500 steps)")
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
        previous = float(np.mean([row["team.red.return"] for row in curve[-40:-20]]))
        final = float(np.mean([row["team.red.return"] for row in curve[-20:]]))
        tail_rows.append(
            {
                "candidate": name,
                "preceding_20_update_mean_red_return": previous,
                "final_20_update_mean_red_return": final,
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
    interpretation = (
        "The search found an additional weakness in this defender."
        if gap["red_improvement"] > 0
        else "The search did not show additional exploitation on the final test episodes."
    )
    if gap["ci95"][0] <= 0 <= gap["ci95"][1] and gap["red_improvement"] != 0:
        interpretation += " The interval includes zero, so the observed change remains uncertain."
    text = f"""# Stage B: one frozen Blue and three fresh Red challengers

{interpretation} The observed Red improvement is **{gap["red_improvement"]:.2f} points**,
with a paired 95% bootstrap interval of **[{gap["ci95"][0]:.2f}, {gap["ci95"][1]:.2f}]**.
Higher Blue return is better for the defender.

| Matchup | Mean raw Blue return | Test episodes |
| --- | ---: | ---: |
| Original Blue vs Original Red | {gap["baseline_blue_mean"]:.2f} | 600 |
| Same Blue vs selected Red (`{state["selection"]["candidate"]}`) | {gap["selected_blue_mean"]:.2f} | 600 |

![Final paired comparison](comparison.png)

Original training used seed 42 and 9,600,000 source steps. Each new Red requested
10,000,000 extra steps and completed 9,984,000 (208 updates of 48,000 steps).
Blue stayed exactly unchanged. The source and every challenger use the same
stock CC4 game, enhanced-v2 observations, one JAX-generated topology (seed 0),
500-step episodes, zero-sum rewards and stochastic policy actions. This is
600 fresh episodes on one network. It is a different target game from Stage A's CIA/resilience evaluation.

Selection compared Original Red and three final Red checkpoints on 100 distinct
validation episodes per candidate. The lowest mean Blue return won, with exact
ties favoring Original Red then the recorded seed order. The identity was saved
before testing. Final matchups share the 600 episode roots, although differing
actions can produce different trajectories. The 95% percentile bootstrap samples
aligned episode differences {BOOTSTRAP_SAMPLES:,} times (seed {BOOTSTRAP_SEED}).
It measures evaluation uncertainty conditional on these selected models from
one original training run. Negative observed gains are retained. This is a
one-sided response search and does not prove equilibrium or measure two-sided NashConv.

![Three challenger learning curves](learning-curves.png)

The budget bounds the search. Learning curves report actual raw 500-step rollout
returns. Changes from the preceding 20-update mean to the final 20-update mean
were {tail_text} Red return points; [the exact tail table](learning-tail.csv)
records those descriptive comparisons. Continuing positive changes suggest the
search may still be improving; noisy flat tails cannot prove convergence.
If further improvement is plausible from those curves, the concrete
next experiment is another 10M requested steps per fresh attempt against this
same Blue under a newly prespecified protocol; it is not included in this pilot.
Portable weights omit optimizer/PRNG/environment state, so a new launch is a
new attempt, not an optimizer resume.

Training update loops measured {training_seconds / 3600:.2f} hours including
first-update compilation; six planned evaluation matchups measured {evaluation_seconds / 3600:.2f} hours including
compilation in their processes. Pipeline elapsed time was
{(state["finished_epoch"] - state["started_epoch"]) / 3600:.2f} hours; queue delay and
smoke are separate. No dollar tariff is available. Slurm job IDs: {state["slurm_job_ids"]}.
Code SHA: `{manifest["source_revision"]}`. [Resolved manifest](manifest.json),
[GPU smoke](smoke.json), [selection and run lineage](state.json),
[validation](validation.csv), [episode-level paired data](per-episode.csv),
[summary](summary.csv), [learning curves](learning-curves.csv).

Large models and logs remain under the explicit external experiment root.
Report owner: `runs:/{report.run_id}`. Source: `{manifest["source"]["checkpoint"]}`.
Selected Red: `{state["selection"]["checkpoint"]}`.

Stage C can reuse source-specific recipe generation, canonical policy loading,
freezing checks, launch pinning, disjoint stream roots, independent evaluation,
validation-only selection and paired aggregation. It needs new source checkpoints,
the reverse Blue-response direction and a separate manifest/budget. It has not been launched.
"""
    readme = report.path("reports/README.md")
    readme.write_text(text)
    report.publish(readme, "reports/README.md")
    report.export("reports/README.md", report_dir / "README.md")
    handoff = report.path("reports/handoff.md")
    handoff.write_text(f"""# Stage C handoff

Stage B report: [README.md](README.md).

Reuse the pinned implementation `{manifest["source_revision"]}`,
`src/jaxborg/oracle_stage_b.py`, the source-specific generator and serial controller,
canonical frozen-opponent loading, checkpoint completion and freezing assertions,
independent matchup lineage, explicit fingerprint reuse, and paired episode aggregation.
The exact Stage B protocol and seed domains are in [manifest.json](manifest.json).
The selected Stage B Red is `{state["selection"]["checkpoint"]}`.

Stage C needs a separately prespecified manifest, five new source pairs and
the reverse Blue response; do not reuse Stage B final seeds for candidate selection.
Re-estimate budgets from Stage B timings and learning tails before expansion.
Only completed artifacts can be reused; portable weights do not provide optimizer resume.
No Stage C execution is started.

Rerun report from the same checkout:

```bash
.venv/bin/python scripts/experiments/oracle_stage_b.py aggregate \
  --manifest {report_dir / "manifest.json"}
```
from the unchanged clean launch checkout with the explicit experiment root and `JAX_PLATFORMS=cpu`.
""")
    report.publish(handoff, "reports/handoff.md")
    report.export("reports/handoff.md", report_dir / "handoff.md")
    # This collection is outside the source checkout and was explicitly requested by the task.
    collection = report_dir.parent.parent / "README.md"
    if collection.exists():
        current = collection.read_text()
        link = "\nCompleted Stage B pilot: [one frozen Blue and three Red challengers](results/stage-b/README.md).\n"
        if "Completed Stage B pilot:" not in current:
            collection.write_text(current + link)
    print(f"Completed pilot report: {report_dir / 'README.md'}", flush=True)
    return gap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", default=str(DEFAULT_SOURCE))
    prep.add_argument("--report-dir", default=str(DEFAULT_REPORT))
    for action in ("dry-run", "smoke", "run", "aggregate"):
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
    else:
        manifest = load_protocol(args.manifest)
        aggregate(manifest, json.loads(state_file(manifest).read_text()))


if __name__ == "__main__":
    main()
