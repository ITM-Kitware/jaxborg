"""Evaluate only the four missing cells with a CPU coordinator."""

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


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "community"
    assert os.environ.get("JAX_PLATFORMS") == "cuda"
    child_env = {**os.environ, "JAX_PLATFORMS": "cuda"}
    os.environ["JAX_PLATFORMS"] = "cpu"
    m = json.loads((ROOT / "manifest.json").read_text())
    assert not (ROOT / "provenance/started.json").exists()
    assert not subprocess.check_output(
        ["git", "-C", m["checkout"], "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    assert (
        subprocess.check_output(["git", "-C", m["checkout"], "rev-parse", "HEAD"], text=True).strip()
        == m["source_revision"]
    )
    for name, expected in m["launcher_hashes"].items():
        assert sha(ROOT / name) == expected
    for key in ["topology", "recipe"]:
        assert sha(m[key]["path"]) == m[key]["sha256"]
    for item in m["inputs"].values():
        assert sha(item["path"]) == item["sha256"] and sha(item["sidecar_path"]) == item["sidecar_sha256"]
    for item in m["reused_cells"].values():
        assert sha(item["path"]) == item["sha256"]
    os.chdir(m["checkout"])
    sys.path[:0] = [m["checkout"] + "/src", m["checkout"]]
    with (ROOT / "provenance/started.json").open("x") as f:
        json.dump(
            {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "job_id": os.environ["SLURM_JOB_ID"],
                "manifest_sha256": sha(ROOT / "manifest.json"),
            },
            f,
            indent=2,
        )
    new = [c for c in m["cells"] if not c["reused"]]
    for index, spec in enumerate(new, 1):
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
            "51000000-51004095",
            "--topology-path",
            m["topology"]["path"],
            "--topology-sampling",
            "exhaustive",
            "--name",
            m["study"] + "-" + spec["name"],
            "--output",
            str(output),
        ]
        print("START_NEW_CELL", index, 4, spec["name"], flush=True)
        subprocess.run(command, check=True, env=child_env)
        print("COMPLETED_NEW_CELL", index, 4, spec["name"], flush=True)
    from analyze import analyze

    report = analyze(ROOT)
    from jaxborg.recipe import eval_variant, load
    from jaxborg.tracking import Run, input_artifact

    recipe = load(m["recipe"]["path"])
    variant = eval_variant(recipe)
    records = [input_artifact(ROOT / "manifest.json", role="registered exploratory crossplay protocol")]
    records += [input_artifact(ROOT / "eval" / (c["name"] + ".json"), role=c["name"]) for c in m["cells"]]
    with Run(
        recipe,
        backend="cpu",
        kind="analysis",
        name=m["study"] + "-analysis",
        config={"variant": variant, "crossplay_protocol": m["protocol"]},
        inputs=records,
    ) as run:
        for name in ["crossplay-report.json", "crossplay-contrasts.csv", "crossplay-cells.csv"]:
            run.publish(ROOT / "analysis" / name, "analysis/" + name)
        for name in m["launcher_hashes"]:
            run.publish(ROOT / name, "protocol/" + name)
        reference = f"runs:/{run.run_id}/analysis/crossplay-report.json"
    with (ROOT / "provenance/completion.json").open("x") as f:
        json.dump(
            {
                "complete": True,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "job_id": os.environ["SLURM_JOB_ID"],
                "reused_cells": 5,
                "new_cells": 4,
                "analysis_reference": reference,
            },
            f,
            indent=2,
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checkpoint_pair_findings": report["checkpoint_pair_findings"],
                "supported_better_response_cycle": report["supported_better_response_cycle"],
                "analysis_reference": reference,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
