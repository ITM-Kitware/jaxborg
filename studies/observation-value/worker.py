"""Execute a declared study stage within its existing Slurm allocation."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def allocated_affinity():
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-d"], text=True)
    match = re.search(r"CPU_IDs=([0-9,-]+)", allocation)
    if match is None:
        raise ValueError("Slurm did not expose allocated CPUs")
    cpus = set()
    for part in match.group(1).split(","):
        bounds = list(map(int, part.split("-")))
        cpus.update(range(bounds[0], bounds[-1] + 1))
    os.sched_setaffinity(0, cpus)
    print("Allocated CPU affinity:", sorted(os.sched_getaffinity(0)), flush=True)


def main(args):
    allocated_affinity()
    from compare import analyze, calibrate, fit, read, replay, run, write
    from verify import verify

    if args.phase == "pilot":
        for blue in read(args.root / "inputs/manifest.json")["blues"]:
            args.blue = blue["id"]
            replay(args)
        write(args.root / "checks/pilot-validation.json", verify(args.root, pilot=True))
    elif args.phase == "calibration":
        if not (args.root / "checks/pilot-validation.json").is_file():
            raise ValueError("Pilot and tracking download must pass before calibration")
        calibrate(args)
        fit(args)
    elif args.phase == "production":
        if not (args.root / "checks/pilot-validation.json").is_file():
            raise ValueError("Pilot and tracking download must pass before production")
        run(args)
    else:
        validation = verify(args.root)
        analyze(args)
        write(args.root / "checks/completion-validation.json", validation)
        from mlflow import MlflowClient

        client = MlflowClient()
        result = read(next((args.root / "results").glob("*__blind.json")))
        runs = client.search_runs(
            [validation["experiment_id"]],
            filter_string=f"tags.`jaxborg.discovery.evaluation_id` = '{result['evaluation_id']}'",
        )
        if len(runs) != 1:
            raise ValueError("Report destination run is not unique")
        client.log_artifacts(runs[0].info.run_id, str(args.root / "report"), artifact_path="study-report")
        client.log_artifact(runs[0].info.run_id, str(args.root / "checks/completion-validation.json"), "study-report")
        print(json.dumps({"report": str(args.root / "report"), "run_id": runs[0].info.run_id}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "calibration", "production", "analysis"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blue")
    main(parser.parse_args())
