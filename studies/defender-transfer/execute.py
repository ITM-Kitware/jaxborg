"""Prepare and execute declared cells serially, retaining an explicit progressive selection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
from prepare import protocol_for

from jaxborg.evaluation.capture_store import load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "final", "early"), required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "community" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Run through the community GPU launcher")
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU runtime preflight failed: {devices}")
    args.study_root.mkdir(parents=True, exist_ok=True)
    runtime = {
        "devices": [str(device) for device in devices],
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "source_revision": args.expected_revision,
        "stage": args.stage,
        "command": sys.argv,
    }
    (args.study_root / f"runtime-{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(runtime, indent=2) + "\n")
    here = Path(__file__).resolve().parent
    inputs = args.study_root / "inputs"
    manifest_path = inputs / "manifest.json"
    if not manifest_path.exists():
        subprocess.run(
            [
                sys.executable,
                str(here / "prepare.py"),
                "--archive-root",
                str(args.archive_root),
                "--output",
                str(inputs),
                "--source-revision",
                args.expected_revision,
            ],
            check=True,
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_revision"] != args.expected_revision:
        raise ValueError("Prepared inputs pin another source revision; use a new study root")
    protocol = json.loads((inputs / "protocol.json").read_text())
    if protocol != protocol_for(manifest):
        raise ValueError("Prepared manifest differs from its immutable protocol")
    evidence = args.study_root / ("smoke-evidence" if args.stage == "smoke" else "evidence")
    selection_path = args.study_root / ("smoke-selection.json" if args.stage == "smoke" else "selection.json")
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    if selection:
        records = {record["id"]: record for record in load_records(evidence)}
        for trial, identity in selection.items():
            evaluation = records.get(identity, {})
            attempt = records.get(evaluation.get("attempt_id"), {})
            if (
                evaluation.get("kind") != "evaluation"
                or evaluation.get("status") != "succeeded"
                or attempt.get("status") != "succeeded"
                or attempt.get("protocol_id") != protocol["id"]
                or attempt.get("trial_id") != trial
            ):
                raise ValueError(f"Selection is not a completed evaluation of its declared trial: {trial}")
    cells = [cell for cell in manifest["cells"] if cell["stage"] == ("final" if args.stage == "smoke" else args.stage)]
    if args.stage == "smoke":
        cells = [
            next(cell for cell in cells if cell["kind"] == kind and cell["bank_id"] == "anchor")
            for kind in ("scripted", "learned")
        ]
    for cell in cells:
        if not cell.get("executable", True):
            print(f"BLOCKED {cell['trial_id']}: {cell['blocked_reason']}", flush=True)
            continue
        if cell["trial_id"] in selection:
            print(f"Already explicitly selected: {cell['trial_id']}", flush=True)
            continue
        print(f"START {cell['trial_id']}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(here / "run.py"),
                "--manifest",
                str(manifest_path),
                "--root",
                str(evidence),
                "--trial",
                cell["trial_id"],
                "--expected-revision",
                args.expected_revision,
            ],
            check=True,
        )
        receipts = [json.loads(path.read_text()) for path in (evidence / "receipts").glob("*.json")]
        matching = [item for item in receipts if item["trial_id"] == cell["trial_id"]]
        if len(matching) != 1:
            raise ValueError("Multiple successful receipts require a manual explicit selection")
        selection[cell["trial_id"]] = matching[0]["evaluation_id"]
        temp = selection_path.with_suffix(".tmp")
        temp.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        temp.replace(selection_path)
    if args.stage != "smoke":
        subprocess.run(
            [
                sys.executable,
                str(here / "analyze.py"),
                "--manifest",
                str(manifest_path),
                "--root",
                str(evidence),
                "--selection",
                str(selection_path),
                "--output",
                str(args.study_root / f"analysis-{os.environ['SLURM_JOB_ID']}"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
