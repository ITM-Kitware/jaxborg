"""Execute one recorded training trial and retain its machine-readable receipt."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path

from prepare import PROJECT, sha256, verify, write_json


def trial_name(phase, seed):
    if phase == "smoke" and seed == 42:
        return "smoke-seed42"
    if phase == "smoke-resume" and seed == 42:
        return "smoke-resume-seed42"
    if phase == "full" and seed in (42, 100):
        return f"fsm-seed{seed}"
    raise ValueError("Expected smoke/smoke-resume seed42 or full seed42/100")


def execute(root, expected_revision, phase, seed):
    root = Path(root).resolve()
    verify(root, expected_revision)
    trial = trial_name(phase, seed)
    if phase == "full":
        acceptance = json.loads((root / "runs/smoke-seed42/production-ready.json").read_text())
        if (
            acceptance.get("passed") is not True
            or acceptance.get("resume_equivalence", {}).get("passed") is not True
            or acceptance.get("source_revision") != expected_revision
            or acceptance.get("manifest_sha256") != sha256(root / "inputs/manifest.json")
        ):
            raise ValueError("Production requires accepted smoke evidence from these exact inputs and source")
    recipe_phase = "smoke" if phase.startswith("smoke") else "full"
    run_dir = root / "runs" / trial
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(PROJECT / "scripts/train/algorithms/ippo_jax.py"),
        "--recipe",
        str(root / "inputs" / f"recipe-{recipe_phase}.yaml"),
        "--seed",
        str(seed),
        "--tag",
        f"{root.name}-{trial}",
        "--evidence-root",
        str(root / "evidence"),
        "--protocol",
        str(root / "inputs/protocol.json"),
        "--trial",
        trial,
        "--expected-revision",
        expected_revision,
    ]
    command.extend(["--checkpoint-update", "1041" if phase == "full" else "2"])
    if phase == "smoke-resume":
        from jaxborg.evaluation.capture_store import load_records

        parent = json.loads((root / "runs/smoke-seed42/receipt.json").read_text())
        indexed = {record["id"]: record for record in load_records(root / "evidence")}
        original = indexed[parent["attempt_id"]]
        if original["status"] != "succeeded" or original["trial_id"] != "smoke-seed42":
            raise ValueError("Resume smoke requires a completed uninterrupted parent smoke")
        state_id = original["facts"]["checkpoint_96000"]["value"]["training_state"]
        state = indexed[state_id]
        if state["producer_id"] != original["id"]:
            raise ValueError("Resume state must be owned by the uninterrupted smoke")
        command.extend(["--resume-state", str(root / "evidence" / state["uri"])])
    write_json(run_dir / "command.json", command)
    receipt = None
    interrupted = False
    process = subprocess.Popen(
        command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        if process.poll() is None:
            process.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    with (run_dir / "training.log").open("w", buffering=1) as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if isinstance(candidate, dict) and {"attempt_id", "evidence_root"} <= candidate.keys():
                receipt = candidate
                write_json(run_dir / "receipt.json", receipt)
    code = process.wait()
    write_json(
        run_dir / "execution.json",
        {"returncode": code, "interrupted": interrupted, "receipt": receipt, "trial_id": trial},
    )
    if code != 0 or receipt is None:
        raise SystemExit(code if code else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--phase", choices=("smoke", "smoke-resume", "full"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    execute(args.study_root, args.expected_revision, args.phase, args.seed)


if __name__ == "__main__":
    main()
