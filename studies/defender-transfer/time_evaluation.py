"""Recorded scalar CPU/GPU episode timing; replays are not new research replicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from benchmark import (
    HERE,
    REPO,
    assert_source,
    checked_path,
    digest,
    encoded,
    read_json,
    retain,
    validate_episode,
    write_json,
)

POLICY = "blue-single-42-final"
REPETITIONS = 4


def protocol_for(plan):
    identity = hashlib.sha256(encoded(plan)).hexdigest()
    return {
        "schema_version": 1,
        "kind": "protocol",
        "id": f"scalar-eval-timing:{identity}",
        "study_id": "defender-eval-throughput",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {
            "attempt": ["resolved_config", "source_identity"],
            "evaluation": ["resolved_config", "source_identity"],
        },
        "trials": [
            {
                "id": backend,
                "condition_id": backend,
                "replicate_id": "one-process",
                "matched_unit_id": "same-policy-case-seeds",
            }
            for backend in ("cpu", "cuda")
        ],
        "metric_definitions": [
            {
                "name": "blue_return",
                "unit": "score",
                "direction": "higher",
                "aggregation": "arithmetic mean",
                "sampling_unit": "identical timing replay",
                "uncertainty": "No independent policy or episode replication",
            }
        ],
    }


def prepare(root, benchmark_manifest, revision):
    assert_source(revision)
    root, benchmark_manifest = Path(root).resolve(), Path(benchmark_manifest).resolve()
    if root.is_relative_to(REPO) or root.is_relative_to(benchmark_manifest.parent):
        raise ValueError("Timing evidence must be outside source and original inputs")
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    original = read_json(benchmark_manifest)
    policy = next(p for p in original["policies"] if p["id"] == POLICY)
    case = original["cases"]["0"]
    plan = {
        "schema_version": 1,
        "source_revision": revision,
        "benchmark_source_revision": original["source_revision"],
        "policy_id": POLICY,
        "case_id": case["case_id"],
        "policy_seed": case["policy_seed"],
        "dynamics_seed": case["dynamics_seed"],
        "horizon": 500,
        "repetitions": REPETITIONS,
        "worker_order": ["cpu", "cuda"],
        "cpu_slots": 8,
        "measurement": "end-to-end scalar run_episode latency, including synchronized host results",
        "limits": (
            "One cold call and three identical warm replays; excludes capture writes. "
            "Not batched or CPU-worker throughput."
        ),
        "inputs": {"benchmark_manifest": retain(benchmark_manifest, inputs, "benchmark-manifest.json")},
    }
    for name in ("model", "sidecar"):
        item = policy[name]
        source = checked_path(benchmark_manifest.parent, item)
        plan["inputs"][name] = retain(source, inputs, f"policy/{source.name}", item["sha256"])
    source = checked_path(benchmark_manifest.parent, case["snapshot"])
    plan["inputs"]["snapshot"] = retain(source, inputs, source.name, case["snapshot"]["sha256"])
    write_json(inputs / "plan.json", plan)
    write_json(inputs / "protocol.json", protocol_for(plan))
    subprocess.run(
        ["git", "-C", str(REPO), "archive", "--format=tar.gz", f"--output={root / 'source.tar.gz'}", revision],
        check=True,
    )
    return inputs / "plan.json"


def worker(args):
    started = time.perf_counter()
    assert_source(args.expected_revision)
    plan = read_json(args.plan)
    if plan["source_revision"] != args.expected_revision or read_json(
        args.plan.parent / "protocol.json"
    ) != protocol_for(plan):
        raise ValueError("Timing plan differs from pinned source/protocol")
    if os.environ.get("JAX_PLATFORMS") != args.backend:
        raise ValueError("Backend must be selected before importing JAX")
    before_affinity = sorted(os.sched_getaffinity(0))
    if len(before_affinity) < plan["cpu_slots"]:
        raise ValueError("Timing worker has fewer CPU slots than declared")
    os.sched_setaffinity(0, before_affinity[: plan["cpu_slots"]])
    cache = Path(os.environ["JAX_COMPILATION_CACHE_DIR"])
    if not cache.is_dir() or any(cache.iterdir()):
        raise ValueError("Each backend needs its own new, empty compilation cache")

    import jax
    from benchmark_rollout import load_policy, run_episode

    from jaxborg.evaluation.capture_store import (
        add_artifact,
        fail_evaluation,
        finish_evaluation,
        record_episode,
        start_evaluation,
    )
    from jaxborg.evaluation.provenance import collect_provenance
    from jaxborg.evaluation.tracking import destination_configured

    devices = jax.devices()
    expected_platform = "gpu" if args.backend == "cuda" else "cpu"
    if len(devices) != 1 or devices[0].platform != expected_platform:
        raise ValueError(f"Unexpected timing devices: {devices}")
    startup = time.perf_counter() - started
    if not destination_configured():
        raise ValueError("Timing capture requires the shared MLflow destination")
    provenance = collect_provenance(expected_revision=args.expected_revision, entrypoint=__file__, source_roots=(HERE,))
    inputs = {name: checked_path(args.plan.parent, item) for name, item in plan["inputs"].items()}
    inputs["plan"] = args.plan
    runtime = {
        "backend": args.backend,
        "devices": [str(d) for d in devices],
        "cpu_affinity_before": before_affinity,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "SLURM_CPUS_PER_TASK",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
                "JAX_PLATFORMS",
                "JAX_COMPILATION_CACHE_DIR",
            )
        },
        "execution": provenance.configuration,
    }
    capture = start_evaluation(
        args.root / "evidence",
        protocol=protocol_for(plan),
        trial_id=args.backend,
        requested_episodes=REPETITIONS,
        resolved_config={"plan": plan, "runtime": runtime},
        facts=provenance.facts,
        inputs=inputs,
        source_files=provenance.source_files,
        track=True,
    )
    try:
        provenance.verify_sources()
        for name, item in plan["inputs"].items():
            if digest(capture.inputs[name]) != item["sha256"]:
                raise ValueError(f"Captured timing input differs from plan: {name}")
        load_started = time.perf_counter()
        policy = load_policy(capture.inputs["model"], kind="lstm")
        policy = replace(policy, weights=jax.device_put(policy.weights))
        jax.block_until_ready(policy.weights)
        load_seconds = time.perf_counter() - load_started
        rows = []
        for index in range(REPETITIONS):
            wall, cpu = time.perf_counter(), time.process_time()
            result = run_episode(
                policy,
                snapshot_path=capture.inputs["snapshot"],
                policy_seed=plan["policy_seed"],
                dynamics_seed=plan["dynamics_seed"],
                num_steps=plan["horizon"],
            )
            cpu_seconds, wall_seconds = time.process_time() - cpu, time.perf_counter() - wall
            validate_episode(result, plan, plan["inputs"]["snapshot"], plan["horizon"])
            row = {
                "repetition": index,
                "phase": "cold" if index == 0 else "warm",
                "wall_seconds": wall_seconds,
                "process_cpu_seconds": cpu_seconds,
                "reward": result["reward"],
                "actions_sha256": hashlib.sha256(encoded(result["per_step_actions"])).hexdigest(),
            }
            rows.append(row)
            path = capture.directory / f"replay-{index}.json"
            write_json(path, {"timing": row, "trajectory": result})
            add_artifact(capture, path, role="timed scalar replay")
            record_episode(
                capture,
                episode_index=index,
                seed=plan["dynamics_seed"],
                policy_seed=plan["policy_seed"],
                reward=result["reward"],
                steps=result["steps"],
            )
            print(json.dumps({"backend": args.backend, **row}), flush=True)
        warm = [row["wall_seconds"] for row in rows[1:]]
        summary = {
            "backend": args.backend,
            "source_revision": args.expected_revision,
            "plan_sha256": digest(args.plan),
            "runtime": runtime,
            "startup_seconds": startup,
            "model_load_and_sync_seconds": load_seconds,
            "replays": rows,
            "warm_median_seconds": statistics.median(warm),
            "warm_min_seconds": min(warm),
            "warm_max_seconds": max(warm),
            "warm_episodes_per_second": 1 / statistics.median(warm),
            "within_backend_identical": len({(row["reward"], row["actions_sha256"]) for row in rows}) == 1,
        }
        path = capture.directory / "timing.json"
        write_json(path, summary)
        add_artifact(capture, path, role="scalar timing summary")
        provenance.verify_sources()
        finish_evaluation(capture, status="succeeded")
        write_json(
            args.root / f"receipt-{args.backend}.json",
            {
                "evaluation_id": capture.evaluation_id,
                "attempt_id": capture.attempt_id,
                "summary": str(path),
                "summary_sha256": digest(path),
            },
        )
    except BaseException as error:
        fail_evaluation(capture, error)
        raise


def execute(args):
    assert_source(args.expected_revision)
    if os.environ.get("SLURM_JOB_PARTITION") != "community" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Use the one-GPU community launcher")
    plan_path = prepare(args.root, args.benchmark_manifest, args.expected_revision)
    summaries, process_walls, selection = {}, {}, {}
    for backend in ("cpu", "cuda"):
        cache = args.root / f"cache-{backend}"
        cache.mkdir(exist_ok=False)
        environment = {**os.environ, "JAX_PLATFORMS": backend, "JAX_COMPILATION_CACHE_DIR": str(cache)}
        started = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--root",
                str(args.root),
                "--plan",
                str(plan_path),
                "--backend",
                backend,
                "--expected-revision",
                args.expected_revision,
            ],
            env=environment,
            check=True,
        )
        process_walls[backend] = time.perf_counter() - started
        receipt = read_json(args.root / f"receipt-{backend}.json")
        if digest(receipt["summary"]) != receipt["summary_sha256"]:
            raise ValueError("Retained timing summary bytes changed")
        summaries[backend] = read_json(receipt["summary"])
        selection[backend] = receipt["evaluation_id"]
        write_json(args.root / f"selection-through-{backend}.json", selection)
    from jaxborg.evaluation.capture_store import load_records

    records = {row["id"]: row for row in load_records(args.root / "evidence", verify=True)}
    if any(records[identity]["status"] != "succeeded" for identity in selection.values()):
        raise ValueError("Timing summary selection is incomplete")
    cpu, gpu = summaries["cpu"], summaries["cuda"]
    if cpu["plan_sha256"] != gpu["plan_sha256"] or cpu["plan_sha256"] != digest(plan_path):
        raise ValueError("Timing workers used different input plans")
    if cpu["runtime"]["cpu_affinity"] != gpu["runtime"]["cpu_affinity"]:
        raise ValueError("Backend workers received different CPU affinity")
    summary = {
        "selection": selection,
        "source_revision": args.expected_revision,
        "process_wall_seconds": process_walls,
        "backends": summaries,
        "cpu_over_gpu_warm_latency": cpu["warm_median_seconds"] / gpu["warm_median_seconds"],
        "cross_backend_reward_difference_gpu_minus_cpu": gpu["replays"][0]["reward"] - cpu["replays"][0]["reward"],
        "cross_backend_actions_equal": gpu["replays"][0]["actions_sha256"] == cpu["replays"][0]["actions_sha256"],
        "limits": read_json(plan_path)["limits"],
    }
    write_json(args.root / "selection.json", selection)
    write_json(args.root / "summary.json", summary)
    print(
        json.dumps(
            {
                "cpu_warm_seconds": cpu["warm_median_seconds"],
                "gpu_warm_seconds": gpu["warm_median_seconds"],
                "cpu_over_gpu": summary["cpu_over_gpu_warm_latency"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--benchmark-manifest", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    if args.mode == "execute":
        if args.benchmark_manifest is None:
            parser.error("execute requires --benchmark-manifest")
        execute(args)
    else:
        if args.plan is None or args.backend is None:
            parser.error("worker requires --plan and --backend")
        args.plan = args.plan.expanduser().resolve()
        worker(args)
