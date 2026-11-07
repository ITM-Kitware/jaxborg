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
    if phase == "full" and seed in (42, 100):
        return f"fsm-seed{seed}"
    raise ValueError("Expected smoke seed 42 or full seed 42/100")


def execute(root, expected_revision, phase, seed):
    root = Path(root).resolve()
    verify(root, expected_revision)
    trial = trial_name(phase, seed)
    if phase == "full":
        acceptance = json.loads((root / "runs/smoke-seed42/production-ready.json").read_text())
        if (
            acceptance.get("passed") is not True
            or acceptance.get("source_revision") != expected_revision
            or acceptance.get("manifest_sha256") != sha256(root / "inputs/manifest.json")
        ):
            raise ValueError("Production requires accepted smoke evidence from these exact inputs and source")
    run_dir = root / "runs" / trial
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(PROJECT / "scripts/train/algorithms/ippo_jax.py"),
        "--recipe",
        str(root / "inputs" / f"recipe-{phase}.yaml"),
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
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    execute(args.study_root, args.expected_revision, args.phase, args.seed)


if __name__ == "__main__":
    main()
