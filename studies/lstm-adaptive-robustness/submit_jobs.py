"""Queue the prespecified response searches using retained validation and timing."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from prepare import PROJECT, sha256, verify, write_json

DEADLINE = datetime(2026, 11, 9, 13, tzinfo=timezone.utc)
SMOKE = "smoke-blue42-primary-red17"
TIMING = "timing-blue42-primary-red17"


def projected_run_seconds(rows):
    if len(rows) < 8:
        raise ValueError("Timing requires at least eight completed production-shaped updates")
    elapsed = [float(row["wall_time_s"]) for row in rows]
    deltas = [right - left for left, right in zip(elapsed, elapsed[1:])]
    if any(not math.isfinite(value) or value <= 0 for value in deltas) or elapsed[0] <= 0:
        raise ValueError("Timing measurements must be finite and increasing")
    steady = statistics.median(deltas[-5:])
    expected = elapsed[0] + 207 * steady + 300
    upper = expected * 1.3 + 300
    return {
        "seconds_per_update": steady,
        "first_update_seconds": elapsed[0],
        "expected_seconds": expected,
        "reserved_seconds": upper,
    }


def block_count(reserved_seconds, now):
    available = (DEADLINE - now).total_seconds() - 3600
    if reserved_seconds <= 0 or not math.isfinite(reserved_seconds):
        raise ValueError("Invalid projected duration")
    if reserved_seconds > 6 * 3600 or 2 * reserved_seconds > available:
        raise ValueError("The first complete paired block does not fit the remaining window")
    return 2 if 4 * reserved_seconds <= available else 1


def timing_projection(root):
    from jaxborg.evaluation.capture_store import load_records

    receipt = json.loads((root / "runs" / TIMING / "receipt.json").read_text())
    records = {record["id"]: record for record in load_records(root / "evidence")}
    owner = records[receipt["attempt_id"]]
    if owner.get("status") != "succeeded" or owner.get("trial_id") != TIMING:
        raise ValueError("Timing requires its successful recorded attempt")
    manifest = json.loads((root / "inputs/manifest.json").read_text())
    if owner["facts"]["source_revision"]["value"] != manifest["source_revision"]:
        raise ValueError("Timing was produced by a different source revision")
    rows = []
    for artifact in records.values():
        if artifact.get("producer_id") != owner["id"] or artifact.get("role") != "training update observation":
            continue
        path = (root / "evidence" / artifact["uri"]).resolve()
        if not path.is_relative_to(root / "evidence") or sha256(path) != artifact["sha256"]:
            raise ValueError("Retained timing observations changed")
        rows.append(json.loads(path.read_text()))
    rows.sort(key=lambda row: row["update_idx"])
    if [row["update_idx"] for row in rows] != list(range(1, 11)):
        raise ValueError("Timing must retain all ten consecutive updates")
    return {**projected_run_seconds(rows), "attempt_id": owner["id"]}


def submit(root, revision, kind, trial, minutes, dependencies):
    command = [
        "sbatch",
        "--parsable",
        "--partition=community",
        "--cpus-per-task=8",
        "--mem=32G" if kind == "train" else "--mem=24G",
        f"--time={minutes}",
        "--deadline=2026-11-09T08:00:00",
        "--signal=INT@120",
        f"--job-name=adaptive-{kind}-{trial}",
        f"--chdir={PROJECT}",
        f"--output={root}/logs/%j.out",
        f"--error={root}/logs/%j.err",
    ]
    if kind == "train":
        command.append("--gres=gpu:1")
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(str(job) for job in dependencies))
    command.extend(
        [str(PROJECT / "studies/lstm-adaptive-robustness/allocated.sh"), str(root), revision, kind, trial, str(PROJECT)]
    )
    response = subprocess.check_output(command, text=True).strip()
    job = int(response.split(";", 1)[0])
    return {"job_id": job, "kind": kind, "trial": trial, "dependencies": dependencies, "command": command}


def queue(args):
    root = args.study_root.resolve()
    verify(root, args.expected_revision)
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=PROJECT, text=True):
        raise ValueError("Submit only from clean pinned source")
    directory = root / "submissions"
    directory.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = directory / f"{args.phase}-plan.json"
        plan = {
            "source_revision": args.expected_revision,
            "manifest_sha256": sha256(root / "inputs/manifest.json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": args.phase,
        }
        if args.phase == "production":
            acceptance = json.loads((root / "checks/production-ready.json").read_text())
            if (
                acceptance.get("passed") is not True
                or acceptance.get("source_revision") != args.expected_revision
                or acceptance.get("manifest_sha256") != plan["manifest_sha256"]
            ):
                raise ValueError("Production requires passing validation of these exact inputs and source")
            projection = timing_projection(root)
            blocks = block_count(projection["reserved_seconds"], datetime.now(timezone.utc))
            plan.update({"projection": projection, "blocks": blocks})
        write_json(plan_path, plan)
        ledger = directory / f"{args.phase}-jobs.jsonl"

        def add(kind, trial, minutes, dependencies=()):
            if datetime.now(timezone.utc) >= DEADLINE:
                raise ValueError("The experiment window has ended")
            entry = submit(root, args.expected_revision, kind, trial, minutes, list(dependencies))
            with ledger.open("a") as stream:
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
                stream.flush()
            print(json.dumps(entry), flush=True)
            return entry["job_id"]

        if args.phase == "validation":
            smoke = add("train", SMOKE, 60)
            add("train", "smoke-resume-blue42-primary-red17", 60, [smoke])
            add("train", TIMING, 60)
            add("eval", SMOKE, 60, [smoke])
        else:
            protocol = json.loads((root / "inputs/protocol.json").read_text())
            lanes = protocol["extensions"]["lstm_adaptive_robustness"]["values"]["lanes"]
            minutes = math.ceil(plan["projection"]["reserved_seconds"] / 60)
            for lane in ("17", "83"):
                prior_train, prior_eval = [], []
                for trial in lanes[lane][: 2 * plan["blocks"]]:
                    training = add("train", trial, minutes, prior_train)
                    evaluation = add("eval", trial, 120, [training, *prior_eval])
                    prior_train, prior_eval = [training], [evaluation]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--phase", required=True, choices=("validation", "production"))
    queue(parser.parse_args())


if __name__ == "__main__":
    main()
