"""Map allocated Slurm threads to Linux CPU IDs on the verified node topology."""

import csv
import io
import re
import subprocess


def expand_ranges(text):
    result = []
    for part in text.split(","):
        bounds = [int(v) for v in part.split("-")]
        result.extend(range(bounds[0], bounds[-1] + 1))
    return result


def allocation_plan(job_id, cpus_per_worker):
    allocation = subprocess.check_output(["scontrol", "show", "job", "--details", str(job_id)], text=True)
    node = subprocess.check_output(["scontrol", "show", "node", "itm"], text=True)
    topology = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True)
    matches = re.findall(r"Nodes=itm CPU_IDs=([0-9,\-]+)", allocation)
    assert len(matches) == 1, "A single known ITM node allocation is required"
    assert re.search(r"Sockets=1\b", node) and re.search(r"ThreadsPerCore=2\b", node)
    rows = [[int(v) for v in row] for row in csv.reader(io.StringIO(topology)) if row and not row[0].startswith("#")]
    assert len(rows) == 64 and len({(r[2], r[1]) for r in rows}) == 32
    ordered = sorted(rows, key=lambda row: (row[2], row[1], row[0]))
    abstract = expand_ranges(matches[0])
    native = [ordered[index][0] for index in abstract]
    assert len(native) == len(set(native)) and len(native) % cpus_per_worker == 0
    groups = [native[i : i + cpus_per_worker] for i in range(0, len(native), cpus_per_worker)]
    return {
        "slurm_job_id": str(job_id),
        "slurm_abstract_cpu_ids": abstract,
        "linux_cpu_ids": native,
        "worker_groups": groups,
        "slurm_allocation": allocation,
        "linux_topology": topology,
        "numbering_reference": "https://slurm.schedmd.com/cpu_management.html#cpu-numbering",
    }
