"""Pin launch source/dependencies before scheduling and verify inside the job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from jaxborg.tracking import (
    assigned_devices,
    dependency_snapshot,
    digest,
    experiment_root,
    file_hash,
    git,
    verify_launch,
)


def pin():
    repo = Path(git("rev-parse", "--show-toplevel")).resolve()
    sha, _ = verify_launch(os.environ.get("JAXBORG_EXPECTED_SHA"), repo=repo)
    root = experiment_root()
    record = {
        "schema_version": 1,
        "checkout": str(repo),
        "sha": sha,
        "python": sys.executable,
        "dependencies": dependency_snapshot(),
        "lockfile_hash": file_hash(repo / "uv.lock"),
    }
    # Retain revisions independently of branch movement.
    subprocess.run(["git", "-C", str(repo), "update-ref", f"refs/jaxborg/launches/{sha}", sha], check=True)
    dest = root / "launches" / f"{sha}-{digest(record)[:16]}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_text(json.dumps(record, indent=2) + "\n")
    return dest


def verify(record_path, *, gpu=False):
    record = json.loads(Path(record_path).read_text())
    repo = Path(git("rev-parse", "--show-toplevel")).resolve()
    if record["checkout"] != str(repo):
        raise ValueError("Launch checkout differs from the checkout pinned at submission")
    expected = os.environ.get("JAXBORG_EXPECTED_SHA", record["sha"])
    if expected != record["sha"]:
        raise ValueError("Launch record SHA differs from the expected submission SHA")
    verify_launch(expected, repo=repo)
    if record["dependencies"] != dependency_snapshot() or record["lockfile_hash"] != file_hash(repo / "uv.lock"):
        raise ValueError("Launch dependencies or lockfile changed after submission")
    if record["python"] != sys.executable:
        raise ValueError("Launch must use the retained isolated Python environment")
    experiment_root()
    if gpu:
        return assigned_devices()
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("pin")
    check = sub.add_parser("verify")
    check.add_argument("--gpu", action="store_true")
    check.add_argument("record")
    args = parser.parse_args()
    if args.action == "pin":
        print(pin())
    else:
        print(verify(args.record, gpu=args.gpu))


if __name__ == "__main__":
    main()
