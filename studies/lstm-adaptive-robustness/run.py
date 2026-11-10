"""Execute one recorded adaptive-response trial with immutable run receipts."""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
from pathlib import Path

from prepare import PROJECT, read_json, sha256, trial_settings, verify, write_json


def require_production_acceptance(root, expected_revision):
    acceptance = read_json(root / "checks/production-ready.json")
    if (
        acceptance.get("passed") is not True
        or acceptance.get("source_revision") != expected_revision
        or acceptance.get("manifest_sha256") != sha256(root / "inputs/manifest.json")
        or acceptance.get("resume_equivalence", {}).get("passed") is not True
        or acceptance.get("cpu_evaluation", {}).get("passed") is not True
        or acceptance.get("fresh_compilation_equivalence", {}).get("passed") is not True
    ):
        raise ValueError("Production requires accepted smoke, resume and CPU evaluation from these exact inputs")


def smoke_resume_state(root):
    from jaxborg.evaluation.capture_store import load_records

    receipt = read_json(root / "runs/smoke-blue42-primary-red17/receipt.json")
    records = {record["id"]: record for record in load_records(root / "evidence")}
    attempt = records[receipt["attempt_id"]]
    if attempt["status"] != "succeeded" or attempt["trial_id"] != "smoke-blue42-primary-red17":
        raise ValueError("Resume smoke requires the successful uninterrupted smoke")
    checkpoint = attempt["facts"]["checkpoint_96000"]["value"]
    state = records[checkpoint["training_state"]]
    if state["producer_id"] != attempt["id"]:
        raise ValueError("Resume checkpoint must belong to the selected original smoke")
    path = (root / "evidence" / state["uri"]).resolve()
    if not path.is_relative_to(root / "evidence") or sha256(path) != state["sha256"]:
        raise ValueError("Resume checkpoint bytes changed")
    return path


def execute(root, expected_revision, trial, *, resume_state=None, run_id=None, retry_of=None):
    root = Path(root).resolve()
    manifest = verify(root, expected_revision)
    settings = trial_settings(trial, manifest["production_updates"])
    if settings["phase"] == "full":
        require_production_acceptance(root, expected_revision)
    if settings["phase"] == "smoke-resume":
        if resume_state is not None:
            raise ValueError("Resume smoke always selects the original smoke update2")
        resume_state = smoke_resume_state(root)
    if resume_state is not None and settings["phase"] == "full" and run_id is None:
        raise ValueError("Continuation requires an explicit fresh --run-id")
    run_id = run_id or trial
    if re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("Run identity must be a simple directory name")
    directory = root / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(PROJECT / "scripts/train/algorithms/ippo_jax.py"),
        "--recipe",
        manifest["recipes"][trial],
        "--seed",
        str(settings["red_seed"]),
        "--tag",
        f"{root.name}-{run_id}",
        "--evidence-root",
        str(root / "evidence"),
        "--protocol",
        str(root / "inputs/protocol.json"),
        "--trial",
        trial,
        "--expected-revision",
        expected_revision,
    ]
    if settings["phase"] == "full":
        command.extend(["--checkpoint-update", str(manifest["primary_updates"])])
    else:
        command.extend(["--checkpoint-update", "2"])
    if settings["phase"] == "timing":
        command.extend(["--checkpoint-update", "5"])
    if resume_state is not None:
        command.extend(["--resume-state", str(Path(resume_state).resolve())])
    if retry_of is not None:
        command.extend(["--retry-of", retry_of])
    write_json(directory / "command.json", command)
    receipt, interrupted = None, False
    process = subprocess.Popen(
        command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        if process.poll() is None:
            process.send_signal(signal.SIGINT)

    previous_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (directory / "training.log").open("x", buffering=1) as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                try:
                    candidate = json.loads(line)
                except ValueError:
                    continue
                if isinstance(candidate, dict) and {"attempt_id", "evidence_root"} <= candidate.keys():
                    if receipt is not None and candidate != receipt:
                        raise ValueError("Trainer emitted inconsistent execution receipts")
                    if receipt is None:
                        receipt = candidate
                        write_json(directory / "receipt.json", receipt)
        code = process.wait()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    write_json(
        directory / "execution.json",
        {"returncode": code, "interrupted": interrupted, "receipt": receipt, "trial_id": trial, "run_id": run_id},
    )
    if code != 0 or receipt is None:
        raise SystemExit(code if code else 1)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--retry-of")
    args = parser.parse_args()
    execute(
        args.study_root,
        args.expected_revision,
        args.trial,
        resume_state=args.resume_state,
        run_id=args.run_id,
        retry_of=args.retry_of,
    )


if __name__ == "__main__":
    main()
