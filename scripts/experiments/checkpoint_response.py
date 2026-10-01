"""Evaluate YAML-prespecified saved Blue checkpoints using a pinned existing evaluator."""

# ruff: noqa: E402
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ["JAX_PLATFORMS"] = "cpu"  # Coordinator only; allocated evaluator children use CUDA.

import argparse
import csv
import json
import subprocess

import numpy as np

from jaxborg.checkpoint import load_jax_bundle
from jaxborg.checkpoint_diagnostic import candidates, confirmation_candidates, load_config, select_checkpoint
from jaxborg.research_tracking import parameter_hash
from jaxborg.response_oracle import assert_recipe_contract, paired_gap
from jaxborg.tracking import Run, file_hash, git, input_artifact, read_manifest, resolve_artifact


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(config_path):
    import yaml

    config = load_config(config_path)
    if Path(os.environ["JAXBORG_EXP_DIR"]).resolve() != Path(config["experiment_root"]).resolve():
        raise ValueError("explicit experiment root differs")
    checkout = Path(config["evaluation_checkout"])
    if git("rev-parse", "HEAD", repo=checkout) != config["evaluation_revision"] or git(
        "status", "--porcelain", repo=checkout
    ):
        raise ValueError("evaluator must be a clean checkout at the exact executed revision")
    manifest = {
        "config": config,
        "orchestration_revision": git("rev-parse", "HEAD"),
        "evaluator_lockfile_sha256": file_hash(checkout / "uv.lock"),
        "points": {},
    }
    for name, paths in config["points"].items():
        protocol, state = (json.loads(Path(paths[key]).read_text()) for key in ("manifest", "state"))
        if protocol["source_revision"] != config["evaluation_revision"]:
            raise ValueError("point execution revision differs")
        recipe = Path(protocol["eval_recipe"])
        if file_hash(recipe) != protocol["eval_recipe_sha256"]:
            raise ValueError("original evaluation recipe changed")
        assert_recipe_contract(yaml.safe_load(recipe.read_text()))
        old_domains = {
            seed
            for key, values in protocol["randomness"].items()
            if isinstance(values, list) and key.endswith("roots")
            for seed in values
        }
        for spec in config["episodes"].values():
            if old_domains & set(range(spec["seed_start"], spec["seed_start"] + spec["count"])):
                raise ValueError("diagnostic seeds overlap original experiment domains")
        source = resolve_artifact(protocol["source"]["checkpoint"])
        if file_hash(source) != protocol["source"]["sha256"]:
            raise ValueError("source model changed")
        topology = resolve_artifact(protocol["game"]["topology"])
        if file_hash(topology) != protocol["game"]["topology_sha256"]:
            raise ValueError("topology changed")
        frozen_hash = parameter_hash(load_jax_bundle(source).policies["red"].weights)
        candidate_list = candidates(protocol, state, config["checkpoint_steps"])
        for candidate in candidate_list:
            path = resolve_artifact(candidate["checkpoint"])
            bundle = load_jax_bundle(path)
            if parameter_hash(bundle.policies["red"].weights) != frozen_hash:
                raise ValueError("saved checkpoint frozen Red differs")
            if (bundle.policies["blue"].obs_dim, bundle.policies["blue"].action_dim) != (450, 242):
                raise ValueError("Blue contract differs")
            candidate["sha256"] = file_hash(path)
        manifest["points"][name] = {
            "protocol": protocol,
            "candidates": candidate_list,
            "frozen_red_parameter_hash": frozen_hash,
        }
    report = Path(config["report_dir"])
    with Run(
        {"meta": {"name": config["name"] + "-prepare"}}, backend="cpu", kind="comparison", config=manifest
    ) as owner:
        owner.write_json("protocol/diagnostic-manifest.json", manifest)
        owner.export("protocol/diagnostic-manifest.json", report / "manifest.json")
        owner.publish(config_path, "protocol/campaign.yaml")
        owner.export("protocol/campaign.yaml", report / "campaign.yaml")
    return manifest


def evaluate(manifest, point, candidate, split, output):
    config, protocol = manifest["config"], point["protocol"]
    checkout = Path(config["evaluation_checkout"])
    if git("rev-parse", "HEAD", repo=checkout) != config["evaluation_revision"] or git(
        "status", "--porcelain", repo=checkout
    ):
        raise ValueError("pinned evaluator source changed")
    if file_hash(checkout / "uv.lock") != manifest["evaluator_lockfile_sha256"]:
        raise ValueError("pinned evaluator lockfile changed")
    if file_hash(protocol["eval_recipe"]) != protocol["eval_recipe_sha256"]:
        raise ValueError("evaluation recipe changed")
    for reference, expected in [
        (candidate["checkpoint"], candidate["sha256"]),
        (protocol["source"]["checkpoint"], protocol["source"]["sha256"]),
        (protocol["game"]["topology"], protocol["game"]["topology_sha256"]),
    ]:
        if file_hash(resolve_artifact(reference)) != expected:
            raise ValueError("prespecified input artifact changed")
    spec = config["episodes"][split]
    seeds = list(range(spec["seed_start"], spec["seed_start"] + spec["count"]))
    argv = [
        str(checkout / ".venv/bin/python"),
        "scripts/eval/eval_matchup.py",
        "--recipe",
        protocol["eval_recipe"],
        "--policy-backend",
        "jax",
        "--blue-path",
        candidate["checkpoint"],
        "--red-path",
        protocol["source"]["checkpoint"],
        "--episodes-per-seed",
        "1",
        "--seeds",
        ",".join(map(str, seeds)),
        "--name",
        f"{config['name']}-{protocol['source']['original_training_steps']}-{split}-{candidate['name']}",
        "--output",
        str(output),
        "--reuse",
    ]
    environment = dict(os.environ, JAX_PLATFORMS="cuda", JAXBORG_EXPECTED_SHA=config["evaluation_revision"])
    environment.pop("JAXBORG_LAUNCH_RECORD", None)
    subprocess.run(argv, cwd=checkout, env=environment, check=True)
    result = json.loads(output.read_text())
    evidence = read_manifest(result["eval_id"])[0]
    if evidence["status"] != "FINISHED" or evidence["source"]["git_commit"] != config["evaluation_revision"]:
        raise ValueError("evaluation is incomplete or has wrong source")
    if [row["sha256"] for row in evidence["inputs"][:3]] != [
        candidate["sha256"],
        protocol["source"]["sha256"],
        protocol["game"]["topology_sha256"],
    ]:
        raise ValueError("evaluation loaded different policies or topology")
    blue = np.asarray(result["per_episode_blue_returns"])
    if (
        result["per_episode_seeds"] != seeds
        or not np.array_equal(-blue, result["per_episode_red_returns"])
        or not np.all(np.isfinite(blue))
        or not np.isclose(blue.mean(), result["blue_mean_return"])
    ):
        raise ValueError("evaluation episode contract differs")
    if result["variant"] != "cc4_stock" or not result["stochastic"]:
        raise ValueError("evaluation game differs")
    return result


def run(manifest_path, point_name):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "community") != "community":
        raise ValueError("GPU evaluations require a community Slurm allocation")
    manifest = json.loads(Path(manifest_path).read_text())
    config, point = manifest["config"], manifest["points"][point_name]
    if git("rev-parse", "HEAD") != manifest["orchestration_revision"]:
        raise ValueError("orchestrator revision changed")
    output_dir = Path(config["report_dir"]) / point_name
    output_dir.mkdir(parents=True, exist_ok=True)
    study = Path(config["experiment_root"]) / "campaigns" / config["name"] / point_name
    study.mkdir(parents=True, exist_ok=True)
    by_name = {c["name"]: c for c in point["candidates"]}
    results = {"validation": {}, "confirmation": {}}
    scores = {}
    for name, candidate in by_name.items():
        results["validation"][name] = evaluate(
            manifest, point, candidate, "validation", study / f"validation-{name}.json"
        )
        scores[name] = results["validation"][name]["blue_mean_return"]
    selected = select_checkpoint(scores, list(by_name))
    selection = {
        "selected": selected,
        "validation_scores": scores,
        "candidate_order": list(by_name),
        "rule": "maximum validation mean Blue return; original then prescribed seed/checkpoint order for exact ties",
    }
    write_json(study / "selection.json", selection)  # Persist before any confirmation outcomes are accessed.
    for name in confirmation_candidates(point["candidates"], selected):
        results["confirmation"][name] = evaluate(
            manifest, point, by_name[name], "confirmation", study / f"confirmation-{name}.json"
        )
    write_json(study / "evaluations.json", results)
    # The behavioral pass uses the same seeds/policies as confirmation and must
    # reproduce the original evaluator's per-episode returns exactly.
    trace_env = dict(os.environ, JAX_PLATFORMS="cuda")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jaxborg.checkpoint_behavior",
            "--manifest",
            manifest_path,
            "--point",
            point_name,
            "--evaluations",
            str(study / "evaluations.json"),
            "--output",
            str(output_dir / "behavior.json"),
        ],
        cwd=ROOT,
        env=trace_env,
        check=True,
    )
    bootstrap = config["bootstrap"]

    def compare(baseline, challenger):
        b, c = (results["confirmation"][name] for name in (baseline, challenger))
        return paired_gap(
            b["per_episode_blue_returns"],
            c["per_episode_blue_returns"],
            b["per_episode_seeds"],
            c["per_episode_seeds"],
            trainable_team="blue",
            samples=bootstrap["samples"],
            seed=bootstrap["seed"],
        )

    summary = {
        "selection": selection,
        "selected_vs_original": compare("original", selected),
        "early_vs_final": {},
        "source_steps": point["protocol"]["source"]["original_training_steps"],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "evaluation_revision": config["evaluation_revision"],
    }
    comparison_rows = []
    for seed in point["protocol"]["training_seeds"]:
        final, early = f"seed-{seed}-step-final", f"seed-{seed}-step-1920000"
        gap = compare(final, early)
        summary["early_vs_final"][str(seed)] = gap
        comparison_rows.append(
            {
                "training_seed": seed,
                "early_minus_final": gap["blue_improvement"],
                "ci_low": gap["ci95"][0],
                "ci_high": gap["ci95"][1],
                "episodes": gap["episodes"],
            }
        )
    episode_rows, evaluation_rows = [], []
    for split, outcomes in results.items():
        for name, result in outcomes.items():
            reference = f"runs:/{result['eval_id']}/evaluations/result.json"
            evaluation_rows.append(
                {
                    "split": split,
                    "candidate": name,
                    "blue_mean_return": result["blue_mean_return"],
                    "episodes": result["n_episodes"],
                    "reference": reference,
                }
            )
            for seed, blue in zip(result["per_episode_seeds"], result["per_episode_blue_returns"]):
                episode_rows.append(
                    {
                        "split": split,
                        "candidate": name,
                        "episode_seed": seed,
                        "blue_return": blue,
                        "eval_id": result["eval_id"],
                    }
                )
    inputs = [
        input_artifact(f"runs:/{r['eval_id']}/evaluations/result.json", role=f"{split} {name}")
        for split, outcomes in results.items()
        for name, r in outcomes.items()
    ]
    with Run(
        {"meta": {"name": config["name"] + "-" + point_name + "-report"}},
        backend="cpu",
        kind="comparison",
        inputs=inputs,
        config=manifest,
    ) as owner:
        summary["canonical_owner"] = owner.run_id
        owner.write_json("results/summary.json", summary)
        owner.export("results/summary.json", output_dir / "summary.json")
        owner.write_json("results/evaluations.json", results)
        owner.export("results/evaluations.json", output_dir / "evaluations.json")
        for filename, rows in [
            ("comparisons.csv", comparison_rows),
            ("episodes.csv", episode_rows),
            ("evaluations.csv", evaluation_rows),
        ]:
            write_csv(owner.path("results/" + filename), rows)
            owner.publish(owner.path("results/" + filename), "results/" + filename)
            owner.export("results/" + filename, output_dir / filename)
    write_json(
        output_dir / "completed.json",
        {"status": "complete", "owner": summary["canonical_owner"], "slurm_job_id": os.environ["SLURM_JOB_ID"]},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--manifest", required=True)
    execute.add_argument("--point", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.config)
    else:
        run(args.manifest, args.point)


if __name__ == "__main__":
    main()
