"""Execute a fixed CPU cross-play matrix under a CPU-only Slurm allocation."""

import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def immutable(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def state_save(state):
    temporary = ROOT / "state.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(ROOT / "state.json")


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "community"
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert not os.environ.get("SLURM_JOB_GPUS") and not os.environ.get("SLURM_STEP_GPUS")
    m = read(ROOT / "manifest.json")
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) >= m["resources"]["cpus"]
    assert not subprocess.check_output(
        ["git", "-C", m["checkout"], "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    assert (
        subprocess.check_output(["git", "-C", m["checkout"], "rev-parse", "HEAD"], text=True).strip()
        == m["source_revision"]
    )
    for name, digest in m["launcher_hashes"].items():
        assert sha(ROOT / name) == digest, name
    for field in ["recipe", "topology"]:
        assert sha(m[field]["path"]) == m[field]["sha256"]
    for item in m["inputs"].values():
        assert sha(item["path"]) == item["sha256"] and sha(item["sidecar_path"]) == item["sidecar_sha256"]
    assert sha(m["parent_manifest"]["path"]) == m["parent_manifest"]["sha256"]
    for item in m["gpu_reference_cells"].values():
        assert sha(item["path"]) == item["sha256"]
    os.chdir(m["checkout"])
    sys.path[:0] = [m["checkout"] + "/src", m["checkout"]]
    from cpu_affinity import allocation_plan

    allocation = allocation_plan(os.environ["SLURM_JOB_ID"], m["resources"]["cpus_per_evaluator"])
    os.sched_setaffinity(0, set(allocation["linux_cpu_ids"]))
    import jax

    from jaxborg.tracking import read_manifest

    assert all(d.platform == "cpu" for d in jax.devices())
    attempt = ROOT / "provenance" / ("attempt-" + os.environ["SLURM_JOB_ID"])
    attempt.mkdir()
    immutable(attempt / "cpu-allocation.json", allocation)
    immutable(
        attempt / "started.json",
        {
            "job_id": os.environ["SLURM_JOB_ID"],
            "started_utc": now(),
            "manifest_sha256": sha(ROOT / "manifest.json"),
            "accelerator": [str(d) for d in jax.devices()],
        },
    )
    state = (
        read(ROOT / "state.json")
        if (ROOT / "state.json").exists()
        else {"manifest_sha256": sha(ROOT / "manifest.json"), "cells": {}, "attempts": []}
    )
    assert state["manifest_sha256"] == sha(ROOT / "manifest.json")
    assert state.get("status") != "complete"
    state["attempts"].append({"job_id": os.environ["SLURM_JOB_ID"], "started_utc": now()})
    state.update(status="running", phase="benchmark", job_id=os.environ["SLURM_JOB_ID"])
    state.pop("last_error", None)
    state.pop("error_utc", None)
    state_save(state)
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    abort = threading.Event()
    worker_local = threading.local()
    worker_slots = queue.Queue()
    for slot in range(m["resources"]["parallel_evaluators"]):
        worker_slots.put(slot)

    def initialize_worker():
        worker_local.slot = worker_slots.get_nowait()

    def verify(output, spec, bounds):
        d = read(output)
        seeds = list(range(bounds[0], bounds[1] + 1))
        assert d["seeds"] == d["per_episode_seeds"] == seeds
        assert d["n_episodes"] == len(seeds) and len(d["per_episode_blue_returns"]) == len(seeds)
        assert d["policy_backend"] == "jax" and d["git_commit"] == m["source_revision"]
        assert d["stochastic"] and d["topology_sampling"] == "exhaustive"
        assert all(
            math.isfinite(b) and b == -r for b, r in zip(d["per_episode_blue_returns"], d["per_episode_red_returns"])
        )
        owner, artifacts = read_manifest(d["eval_id"])
        assert owner["status"] == "FINISHED"
        devices = owner["effective_config"]["accelerator"]
        assert devices and all(x["platform"] == "cpu" for x in devices)
        assert owner["source"]["git_commit"] == m["source_revision"] and not owner["source"]["dirty"]
        assert sha(artifacts / "evaluations/result.json") == sha(output)
        for team in ["blue", "red"]:
            expected = m["inputs"][str(spec[team + "_steps"])]
            assert sha(d["policies"][team]["path"]) == expected["sha256"]
        assert all(sha(p) == m["topology"]["sha256"] for p in set(d["per_episode_topology_paths"]))
        return {
            "path": str(output),
            "sha256": sha(output),
            "eval_id": d["eval_id"],
            "wall_time_s": d["wall_time_s"],
            "blue_mean": d["blue_mean_return"],
            "accelerator": "cpu",
        }

    def evaluate(spec, bounds, output, label, timeout=None):
        if output.exists():
            return verify(output, spec, bounds)
        if abort.is_set():
            raise RuntimeError("An evaluation failed; remaining unstarted cells are retained for a later continuation")
        log = attempt / (label + ".log")
        rss = attempt / (label + "-rss-kib.txt")
        progress = attempt / (label + "-progress.jsonl")
        command = [
            "srun",
            "--exclusive",
            "--exact",
            "--ntasks=1",
            "--cpus-per-task=" + str(m["resources"]["cpus_per_evaluator"]),
            "--mem=" + str(m["resources"]["memory_gib_per_evaluator"]) + "G",
            "--cpu-bind=none",
            "--kill-on-bad-exit=1",
            "/usr/bin/time",
            "--format=%M",
            "--output=" + str(rss),
            m["python"],
            str(ROOT / "cpu_worker.py"),
            "--allocation",
            str(attempt / "cpu-allocation.json"),
            "--slot",
            str(getattr(worker_local, "slot", 0)),
            "--record",
            str(attempt / (label + "-affinity.json")),
            "--",
            m["python"],
            "scripts/eval/eval_matchup.py",
            "--recipe",
            m["recipe"]["path"],
            "--policy-backend",
            "jax",
            "--blue-path",
            m["inputs"][str(spec["blue_steps"])]["canonical_reference"],
            "--red-path",
            m["inputs"][str(spec["red_steps"])]["canonical_reference"],
            "--episodes-per-seed",
            "1",
            "--seeds",
            str(bounds[0]) + "-" + str(bounds[1]),
            "--topology-path",
            m["topology"]["path"],
            "--topology-sampling",
            "exhaustive",
            "--name",
            m["study"] + "-" + label,
            "--output",
            str(output),
        ]
        immutable(attempt / (label + "-command.json"), {"started_utc": now(), "command": command})
        started = time.monotonic()
        events = []
        print("START", label, now(), flush=True)
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def capture():
            with log.open("x") as f, progress.open("x") as p:
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    match = re.search(r"episodes (\d+)/(\d+)", line)
                    if match:
                        event = {
                            "utc": now(),
                            "elapsed_s": time.monotonic() - started,
                            "episodes": int(match[1]),
                            "total": int(match[2]),
                        }
                        events.append(event)
                        p.write(json.dumps(event) + "\n")
                        p.flush()

        reader = threading.Thread(target=capture, daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            reader.join()
            abort.set()
            raise RuntimeError(label + " exceeded its operational timeout; attempt logs are retained")
        reader.join()
        if code != 0:
            abort.set()
            raise RuntimeError(label + " failed with exit " + str(code) + "; inspect " + str(log))
        record = verify(output, spec, bounds)
        record.update(
            elapsed_s=time.monotonic() - started, progress_events=events, max_rss_kib=int(rss.read_text().strip())
        )
        immutable(attempt / (label + "-completion.json"), record)
        print("COMPLETE", label, now(), flush=True)
        return record

    benchmark_spec = next(x for x in m["cells"] if x["name"] == m["timing_preflight"]["cell"])
    benchmark_path = ROOT / "benchmark/result.json"
    benchmark = evaluate(
        benchmark_spec,
        m["timing_preflight"]["episode_seed_range"],
        benchmark_path,
        "timing-preflight",
        m["timing_preflight"]["maximum_minutes"] * 60,
    )
    previous_summary = ROOT / "benchmark/summary.json"
    if previous_summary.exists():
        timing = read(previous_summary)
    else:
        events = benchmark["progress_events"]
        assert len(events) == 2 and [x["episodes"] for x in events] == [64, 128]
        steady_batch = events[1]["elapsed_s"] - events[0]["elapsed_s"]
        host_bytes = m["resources"]["memory_gib"] * 1024**3
        peak_bytes = benchmark["max_rss_kib"] * 1024
        workers = min(m["resources"]["parallel_evaluators"], int((host_bytes - 2 * 1024**3) / (peak_bytes * 1.4)))
        assert workers >= 1, "Measured evaluator memory exceeds the reserved host-memory margin"
        waves = math.ceil(len(m["cells"]) / workers)
        estimate = waves * (steady_batch * 64 + events[0]["elapsed_s"]) + 600
        timing = {
            "completed_utc": now(),
            "benchmark": benchmark,
            "steady_seconds_per_64_episodes": steady_batch,
            "parallel_evaluators": workers,
            "estimated_matrix_seconds": estimate,
            "estimated_completion_utc": (datetime.now(timezone.utc) + timedelta(seconds=estimate)).isoformat(),
            "caveat": (
                "Timing estimate only; concurrent CPU throughput may differ. Checkpoints, roots "
                "and sample sizes are fixed."
            ),
        }
        immutable(previous_summary, timing)
    state.update(phase="matrix", benchmark=timing, parallel_evaluators=timing["parallel_evaluators"])
    state_save(state)
    print("TIMING", json.dumps(timing), flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=timing["parallel_evaluators"], initializer=initialize_worker) as executor:
        futures = {
            executor.submit(
                evaluate, spec, m["episode_seed_range"], ROOT / "eval" / (spec["name"] + ".json"), spec["name"]
            ): spec
            for spec in m["cells"]
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                state["cells"][spec["name"]] = future.result()
            except Exception as exc:
                errors.append({"cell": spec["name"], "error": str(exc)})
                abort.set()
            state_save(state)
    if errors:
        state.update(status="incomplete_evaluation_error", errors=errors)
        state_save(state)
        immutable(attempt / "failure.json", {"errors": errors, "ended_utc": now()})
        raise RuntimeError("Some fixed cells are incomplete; completed evaluations have been retained")
    assert len(state["cells"]) == 9
    state.update(phase="analysis")
    state_save(state)
    from analyze import analyze

    report_path = ROOT / "analysis/crossplay-report.json"
    if not report_path.exists():
        analyze(ROOT)
    bridge = []
    for name, item in m["gpu_reference_cells"].items():
        gpu, cpu = read(item["path"]), read(ROOT / "eval" / (name + ".json"))
        assert gpu["per_episode_seeds"] == cpu["per_episode_seeds"]
        differences = [c - g for c, g in zip(cpu["per_episode_blue_returns"], gpu["per_episode_blue_returns"])]
        bridge.append(
            {
                "cell": name,
                "roots_with_different_returns": sum(x != 0 for x in differences),
                "mean_cpu_minus_gpu_blue_return": sum(differences) / len(differences),
                "descriptive_only": True,
            }
        )
    cpu_record = ROOT / "analysis/cpu-provenance.json"
    if not cpu_record.exists():
        immutable(
            cpu_record,
            {
                "platform": "cpu",
                "all_nine_cells_new_cpu_evaluations": True,
                "original_gpu_outputs_preserved": True,
                "parent_manifest_sha256": m["parent_manifest"]["sha256"],
                "gpu_reference_descriptive_comparison": bridge,
            },
        )
    from jaxborg.recipe import eval_variant, load
    from jaxborg.tracking import Run, input_artifact

    recipe = load(m["recipe"]["path"])
    records = [input_artifact(ROOT / "manifest.json", role="fixed CPU crossplay protocol")]
    records += [input_artifact(ROOT / "eval" / (c["name"] + ".json"), role=c["name"]) for c in m["cells"]]
    with Run(
        recipe,
        backend="cpu",
        kind="analysis",
        name=m["study"] + "-analysis",
        config={"variant": eval_variant(recipe), "crossplay_protocol": m["protocol"], "execution_platform": "cpu"},
        inputs=records,
    ) as run:
        for name in ["crossplay-report.json", "crossplay-contrasts.csv", "crossplay-cells.csv", "cpu-provenance.json"]:
            run.publish(ROOT / "analysis" / name, "analysis/" + name)
        for name in m["launcher_hashes"]:
            run.publish(ROOT / name, "protocol/" + name)
        reference = f"runs:/{run.run_id}/analysis/crossplay-report.json"
    completion = {
        "complete": True,
        "completed_utc": now(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "new_cpu_cells": 9,
        "gpu_cells_reused_in_matrix": 0,
        "analysis_reference": reference,
        "manifest_sha256": sha(ROOT / "manifest.json"),
    }
    immutable(ROOT / "provenance/completion.json", completion)
    state.update(status="complete", phase="complete", completion=completion)
    state_save(state)
    print("STUDY_COMPLETE", json.dumps(completion), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if (ROOT / "state.json").exists():
            state = read(ROOT / "state.json")
            if state.get("status") != "complete":
                state.update(status="incomplete_error", last_error=str(exc), error_utc=now())
                state_save(state)
        raise
