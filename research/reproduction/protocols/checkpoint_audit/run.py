"""Evaluate the sealed 150M checkpoint audit under its Slurm allocation."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def validate(m):
    assert not subprocess.check_output(
        ["git", "-C", m["checkout"], "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    assert (
        subprocess.check_output(["git", "-C", m["checkout"], "rev-parse", "HEAD"], text=True).strip()
        == m["source_revision"]
    )
    for name, expected in m["launcher_hashes"].items():
        assert sha(ROOT / name) == expected, name
    for item in m["inputs"].values():
        assert sha(item["path"]) == item["sha256"]
        assert sha(item["sidecar_path"]) == item["sidecar_sha256"]
    for key in ("recipe", "topology", "source_manifest"):
        assert sha(m[key]["path"]) == m[key]["sha256"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    m = json.loads((ROOT / "manifest.json").read_text())
    validate(m)
    if args.validate_only:
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "community"
        assert os.environ.get("JAX_PLATFORMS") == "cuda" and os.environ.get("JAXBORG_EVAL_BATCH_SIZE") == "64"
    os.chdir(m["checkout"])
    sys.path[:0] = [m["checkout"] + "/src", m["checkout"]]
    if args.validate_only:
        import jax

        from jaxborg.evaluation.matchup_runner import MatchupEvaluationContext
        from jaxborg.recipe import eval_variant, load

        recipe = load(m["recipe"]["path"])
        variant = eval_variant(recipe)
        assert variant.name == "cc4_stock" and variant.num_steps == 500
        assert all(d.platform == "cpu" for d in jax.devices())
        context = MatchupEvaluationContext()
        loaded = []
        for steps, item in m["inputs"].items():
            for team in ("blue", "red"):
                p = context.load_policy(item["path"], team=team, backend="jax").source
                assert p["bundle_provenance"]["seed"] == 43 and p["bundle_provenance"]["total_steps"] == int(steps)
                assert p["train_run_id"] == m["source_training_run_id"]
                loaded.append([team, int(steps), p["observation_dim"], p["action_dim"]])
        context.environment(variant, [Path(m["topology"]["path"])])
        print(json.dumps({"status": "validated_no_rollouts", "loaded_policies": loaded}))
        return
    child_env = {**os.environ, "JAX_PLATFORMS": "cuda"}
    os.environ["JAX_PLATFORMS"] = "cpu"
    save(
        ROOT / "provenance/started.json",
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": os.environ["SLURM_JOB_ID"],
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "manifest_sha256": sha(ROOT / "manifest.json"),
        },
    )
    for index, spec in enumerate(m["cells"], 1):
        output = ROOT / "eval" / (spec["name"] + ".json")
        assert not output.exists()
        command = [
            m["python"],
            "scripts/eval/eval_matchup.py",
            "--recipe",
            m["recipe"]["path"],
            "--policy-backend",
            "jax",
            "--blue-path",
            m["inputs"][str(spec["blue_steps"])]["canonical_reference"],
            "--red-path",
            m["inputs"][str(spec["red_steps"])]["canonical_reference"],
            "--episodes-per-seed",
            "1",
            "--seeds",
            f"{m['episode_seed_range'][0]}-{m['episode_seed_range'][1]}",
            "--topology-path",
            m["topology"]["path"],
            "--topology-sampling",
            "exhaustive",
            "--name",
            m["study"] + "-" + spec["name"],
            "--output",
            str(output),
        ]
        print("START_CELL", index, len(m["cells"]), spec["name"], flush=True)
        subprocess.run(command, check=True, env=child_env)
        print("COMPLETED_CELL", index, len(m["cells"]), spec["name"], flush=True)
    from analyze import analyze

    report = analyze(ROOT)
    from jaxborg.recipe import eval_variant, load
    from jaxborg.tracking import Run, input_artifact

    recipe = load(m["recipe"]["path"])
    records = [input_artifact(ROOT / "manifest.json", role="registered checkpoint audit protocol")]
    records += [input_artifact(ROOT / "eval" / (spec["name"] + ".json"), role=spec["name"]) for spec in m["cells"]]
    with Run(
        recipe,
        backend="cpu",
        kind="analysis",
        name=m["study"] + "-analysis",
        config={**m["protocol"], "variant": eval_variant(recipe)},
        inputs=records,
    ) as run:
        run.publish(ROOT / "analysis/audit-report.json", "analysis/audit-report.json")
        run.publish(ROOT / "analysis/audit-contrasts.csv", "analysis/audit-contrasts.csv")
        for name in m["launcher_hashes"]:
            run.publish(ROOT / name, "protocol/" + name)
        analysis_reference = f"runs:/{run.run_id}/analysis/audit-report.json"
    save(
        ROOT / "provenance/completion.json",
        {
            "complete": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": os.environ["SLURM_JOB_ID"],
            "analysis_reference": analysis_reference,
            "evidence_against_final_pair_equilibrium": report["evidence_against_final_pair_equilibrium"],
        },
    )
    print(
        json.dumps(
            {"status": report["status"], "contrasts": report["contrasts"], "analysis_reference": analysis_reference}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
