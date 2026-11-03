"""Apply an exclusive worker CPU group before importing the evaluator runtime."""

import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--allocation", type=Path, required=True)
parser.add_argument("--slot", type=int, required=True)
parser.add_argument("--record", type=Path, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
plan = json.loads(args.allocation.read_text())
assert plan["slurm_job_id"] == os.environ["SLURM_JOB_ID"]
assert os.environ["JAX_PLATFORMS"] == "cpu" and os.environ["JAXBORG_ALLOW_CPU"] == "1"
cpus = set(plan["worker_groups"][args.slot])
assert len(cpus) == 8 and cpus <= set(plan["linux_cpu_ids"])
os.sched_setaffinity(0, cpus)
assert os.sched_getaffinity(0) == cpus
record = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "step_id": os.environ.get("SLURM_STEP_ID"),
    "pid": os.getpid(),
    "linux_cpu_ids": sorted(cpus),
}
with args.record.open("x") as f:
    json.dump(record, f, indent=2)
    f.write("\n")
print("CPU_AFFINITY " + json.dumps(record), flush=True)
command = args.command[1:] if args.command and args.command[0] == "--" else args.command
assert command
os.execv(command[0], command)
