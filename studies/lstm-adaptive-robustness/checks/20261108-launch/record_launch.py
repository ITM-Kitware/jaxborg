"""Snapshot declared inputs, accepted checks, and submitted jobs without changing them."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from jaxborg.evaluation.capture_store import load_records

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--study-root", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
root = args.study_root.resolve()


def read(relative):
    return json.loads((root / relative).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


manifest = read("inputs/manifest.json")
checks = read("checks/production-ready.json")
jobs = [json.loads(line) for line in (root / "submissions/production-jobs.jsonl").read_text().splitlines()]
records = load_records(root / "evidence")
attempts = []
for record in records:
    if (
        record["kind"] != "attempt"
        or record.get("execution_kind") != "training"
        or not record["trial_id"].startswith("br-")
    ):
        continue
    attempts.append(
        {
            "id": record["id"],
            "trial": record["trial_id"],
            "status": record["status"],
            "budgets": record["budgets"],
            "initialization": record["facts"].get("policy_initialization_hashes", {}).get("value"),
        }
    )
files = (
    "inputs/manifest.json",
    "inputs/protocol.json",
    "checks/production-ready.json",
    "checks/timing.json",
    "checks/fresh-compilation.json",
    "submissions/production-plan.json",
    "submissions/production-jobs.jsonl",
)
result = {
    "observed_at": datetime.now(timezone.utc).isoformat(),
    "study_root": str(root),
    "source_revision": manifest["source_revision"],
    "primary_environment_steps": manifest["primary_updates"] * 48000,
    "main_final_environment_steps": manifest["production_updates"] * 48000,
    "gpu_deadline_utc": manifest["deadline_utc"],
    "retained_files": {relative: sha(root / relative) for relative in files},
    "blue_checkpoints": {
        name: {
            key: item[key] for key in ("model_sha256", "sidecar_sha256", "steps", "training_attempt", "source_revision")
        }
        for name, item in manifest["blue_checkpoints"].items()
    },
    "validation": {
        "passed": checks["passed"],
        "resume": checks["resume_equivalence"],
        "fresh_compilation": checks["fresh_compilation_equivalence"],
        "cpu_evaluation": checks["cpu_evaluation"],
    },
    "timing": read("checks/timing.json"),
    "submission_plan": read("submissions/production-plan.json"),
    "jobs": [{key: row[key] for key in ("job_id", "kind", "trial", "dependencies")} for row in jobs],
    "observed_attempts": attempts,
    "squeue": subprocess.check_output(
        ["squeue", "-h", "-j", ",".join(str(row["job_id"]) for row in jobs), "-o", "%i|%T|%M|%R"], text=True
    ).splitlines(),
    "gpu_use": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], text=True
    ).splitlines(),
    "recorder_sha256": sha(__file__),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(
    json.dumps(
        {
            "output": str(args.output),
            "sha256": sha(args.output),
            "observed_attempts": attempts,
            "gpu_use": result["gpu_use"],
        }
    )
)
