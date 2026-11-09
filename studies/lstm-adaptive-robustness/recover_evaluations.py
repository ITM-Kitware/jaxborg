"""Recover missing adaptive-study evaluation cells without changing pinned source."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

BLUES = ("blue42-primary", "blue42-final", "blue100-primary", "blue100-final")
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}
CPU_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAXBORG_EVAL_BATCH_SIZE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def pinned(plan):
    source = Path(plan["source_root"])
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "studies/lstm-adaptive-robustness"))
    spec = importlib.util.spec_from_file_location(
        "recovery_pinned_evaluate", source / "studies/lstm-adaptive-robustness/evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.verify(Path(plan["study_root"]), plan["expected_revision"])
    if digest(Path(plan["study_root"]) / "inputs/manifest.json") != plan["manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    from dataclasses import asdict

    from jaxborg.scenarios.cc4.game_variant import GameVariant

    manifest["_expected_variant"] = asdict(GameVariant(name="cc4_stock", cage4_enhanced_obs=True))
    return module, manifest


def records_for(plan):
    from jaxborg.evaluation.capture_store import load_records

    return {row["id"]: row for row in load_records(Path(plan["study_root"]) / "evidence")}


def scheduler_state(job):
    job = str(int(job))
    active = subprocess.check_output(["squeue", "--noheader", "--format=%i"], text=True).split()
    if job in active or any(item.startswith(job + "_") for item in active):
        return "ACTIVE"
    text = subprocess.check_output(
        ["sacct", "--noheader", "--allocations", "--parsable2", "--jobs", job, "--format=JobIDRaw,State,ExitCode"],
        text=True,
    )
    rows = [line.split("|") for line in text.splitlines() if line.split("|")[0] == job]
    if len(rows) != 1:
        raise ValueError(f"Cannot establish terminal state for original job {job}")
    state = rows[0][1].split()[0].rstrip("+")
    return state if state in TERMINAL else "ACTIVE"


def require_terminal(job):
    state = scheduler_state(job)
    if state not in TERMINAL:
        raise ValueError(f"Original evaluator {job} is still active")
    return state


def bind_training(plan, cell, evaluator, records):
    root = Path(plan["study_root"])
    directory = root / "runs" / cell["trial"]
    receipt_path, execution_path = directory / "receipt.json", directory / "execution.json"
    if not receipt_path.exists() or not execution_path.exists():
        raise ValueError("Original training has no completed receipt/execution")
    receipt, execution = read(receipt_path), read(execution_path)
    if (
        execution.get("returncode") != 0
        or execution.get("interrupted") is not False
        or execution.get("trial_id") != cell["trial"]
        or execution.get("run_id") != cell["trial"]
        or execution.get("receipt") != receipt
        or Path(receipt.get("evidence_root", "")).resolve() != root / "evidence"
    ):
        raise ValueError("Original training did not successfully complete this declared run")
    attempt = records.get(receipt.get("attempt_id"), {})
    if (
        attempt.get("kind") != "attempt"
        or attempt.get("execution_kind") != "training"
        or attempt.get("status") != "succeeded"
        or attempt.get("trial_id") != cell["trial"]
        or attempt.get("facts", {}).get("source_revision", {}).get("value") != plan["expected_revision"]
    ):
        raise ValueError("Selected original training attempt is not successful and source-matched")
    selection = cell.get("planned_training")
    if selection and (
        receipt["attempt_id"] != selection["attempt_id"] or digest(receipt_path) != selection["receipt_sha256"]
    ):
        raise ValueError("Original training receipt changed since planning")
    response = evaluator.selected_response(root, cell["trial"], cell["trial"], plan["expected_revision"], cell["steps"])
    if response["attempt_id"] != receipt["attempt_id"]:
        raise ValueError("Checkpoint selection changed training owner")
    return {
        "attempt_id": receipt["attempt_id"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": digest(receipt_path),
        "execution_path": str(execution_path),
        "execution_sha256": digest(execution_path),
        "response": response,
    }


def verified_success(path, plan, cell, binding, records, manifest):
    """An existing receipt is either a fully verified matching success or an error."""
    path = Path(path)
    receipt = read(path)
    expected = {
        "passed": True,
        "trial": cell["trial"],
        "stage": cell["stage"],
        "blue_id": cell["blue"],
        "episodes": 64,
        "training_attempt_id": binding["attempt_id"],
        "response_environment_steps": cell["steps"],
        "source_revision": plan["expected_revision"],
        "manifest_sha256": plan["manifest_sha256"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing receipt does not match the selected cell: {path}")
    returns = receipt.get("blue_returns", [])
    if (
        len(returns) != 64
        or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            for value in returns
        )
        or receipt.get("mean_blue_return") != statistics.mean(returns)
    ):
        raise ValueError(f"Existing receipt has incomplete or inconsistent returns: {path}")
    evaluation = records.get(receipt.get("evaluation_id"), {})
    attempt = records.get(evaluation.get("attempt_id"), {})
    if (
        evaluation.get("kind") != "evaluation"
        or evaluation.get("status") != "succeeded"
        or attempt.get("status") != "succeeded"
        or attempt.get("execution_kind") != "evaluation"
        or attempt.get("trial_id") != cell["trial"]
        or attempt.get("facts", {}).get("source_revision", {}).get("value") != plan["expected_revision"]
    ):
        raise ValueError(f"Receipt lacks its successful source-matched evidence: {path}")
    metrics = [row["value"] for row in evaluation.get("metrics", []) if row.get("name") == "blue_return"]
    results = [
        row
        for row in records.values()
        if row.get("producer_id") == evaluation["id"] and row.get("role") == "joint matchup result"
    ]
    if (metrics and metrics != [receipt["mean_blue_return"]]) or len(results) != 1:
        raise ValueError("Receipt differs from recorded evaluation results")
    result_path = Path(plan["study_root"]) / "evidence" / results[0]["uri"]
    if digest(result_path) != results[0]["sha256"] or read(result_path).get("blue_returns") != returns:
        raise ValueError("Receipt returns differ from retained result artifact")
    budgets = evaluation.get("budgets", [])
    if not budgets or budgets[0].get("requested") != 64 or budgets[0].get("completed") != 64:
        raise ValueError("Evidence has an incomplete episode budget")
    evaluation_plan_path = path.with_name("plan.json")
    evaluation_plan = read(evaluation_plan_path)
    if (
        evaluation_plan.get("trial") != cell["trial"]
        or evaluation_plan.get("stage") != cell["stage"]
        or evaluation_plan.get("blue_id") != cell["blue"]
        or evaluation_plan.get("run_id") != receipt.get("run_id")
        or evaluation_plan.get("response") != binding["response"]
        or evaluation_plan.get("blue") != manifest["blue_checkpoints"][cell["blue"]]
        or evaluation_plan.get("topologies") != manifest["evaluation_topologies"]
        or evaluation_plan.get("episode_roots") != list(range(21000, 21016))
        or evaluation_plan.get("deterministic") is not False
        or evaluation_plan.get("backend") != "cpu"
    ):
        raise ValueError("Existing evaluation plan differs from the selected panel")
    configs = [
        row
        for row in records.values()
        if row.get("producer_id") == attempt["id"] and row.get("role") == "resolved evaluation configuration"
    ]
    if len(configs) != 1:
        raise ValueError("Missing unique retained evaluation configuration")
    config_path = Path(plan["study_root"]) / "evidence" / configs[0]["uri"]
    if digest(config_path) != configs[0]["sha256"]:
        raise ValueError("Retained configuration changed")
    config = read(config_path)
    if (
        config.get("overrides", {}).get("plan_sha256") != digest(evaluation_plan_path)
        or config.get("overrides", {}).get("blue_id") != cell["blue"]
        or config.get("horizon") != 500
        or config.get("episodes_per_seed") != 1
        or config.get("seed_roles", {}).get("episode_roots") != list(range(21000, 21016))
        or config.get("topology", {}).get("sha256") != [row["sha256"] for row in manifest["evaluation_topologies"]]
        or config.get("variant") != evaluation_plan.get("variant")
        or config.get("variant") != manifest["_expected_variant"]
        or config.get("deterministic") is not False
    ):
        raise ValueError("Successful evidence does not own this evaluation plan")
    inputs = [records[item] for item in evaluation.get("inputs", []) if item in records]
    for role in ("model", "sidecar"):
        blue_hash = manifest["blue_checkpoints"][cell["blue"]][role + "_sha256"]
        if not any(row.get("kind") == "artifact" and row.get("sha256") == blue_hash for row in inputs):
            raise ValueError("Successful evidence does not reference the selected Blue input")
        if binding["response"][role + "_artifact"] not in evaluation.get("inputs", []):
            raise ValueError("Successful evidence does not reference the selected checkpoint")
    return {"receipt": str(path), "receipt_sha256": digest(path), "evaluation_id": receipt["evaluation_id"]}


def registry(root):
    return Path(root) / "evaluation-recovery"


def cell_key(cell):
    return hashlib.sha256(f"{cell['trial']}/{cell['stage']}/{cell['blue']}".encode()).hexdigest()


def completed(plan, cell, binding, records, manifest):
    root = Path(plan["study_root"])
    paths = [root / "evaluations" / cell["trial"] / cell["stage"] / cell["blue"] / "receipt.json"]
    for path in sorted((registry(root) / "aliases").glob("*.json")):
        alias = read(path)
        if alias["cell_key"] == cell_key(cell) and alias["training_attempt_id"] == binding["attempt_id"]:
            receipt = root / "runs" / alias["run_id"] / "receipt.json"
            if digest(receipt) != alias["training_receipt_sha256"]:
                raise ValueError("Registered recovery selection alias changed")
            candidate = root / "evaluations" / alias["run_id"] / cell["stage"] / cell["blue"] / "receipt.json"
            if not candidate.exists():
                start_path = Path(alias["execution_directory"]) / "start.json"
                job = read(start_path).get("slurm_job_id")
                if not job:
                    raise ValueError("Partial recovery alias lacks its original job owner")
                require_terminal(job)
            paths.append(candidate)
    for path in paths:
        if not path.exists() and path.with_name("plan.json").exists():
            plan_sha = digest(path.with_name("plan.json"))
            for record in records.values():
                if record.get("role") != "resolved evaluation configuration":
                    continue
                config = read(root / "evidence" / record["uri"])
                if config.get("overrides", {}).get("plan_sha256") == plan_sha:
                    owner = records.get(record["producer_id"], {})
                    evaluations = [
                        row
                        for row in records.values()
                        if row.get("kind") == "evaluation" and row.get("attempt_id") == owner.get("id")
                    ]
                    if owner.get("status") == "succeeded" or any(
                        row.get("status") == "succeeded" for row in evaluations
                    ):
                        raise ValueError("Completed evidence lacks its receipt; reconcile without rerunning")
    successes = [verified_success(path, plan, cell, binding, records, manifest) for path in paths if path.exists()]
    return successes[0] if successes else None


def build_plan(args):
    root, source = args.study_root.resolve(), args.source_root.resolve()
    if root == source or root.is_relative_to(source):
        raise ValueError("Use the external production study root")
    plan = {
        "schema_version": 1,
        "created_at": now(),
        "study_root": str(root),
        "source_root": str(source),
        "expected_revision": args.expected_revision,
        "manifest_sha256": digest(root / "inputs/manifest.json"),
        "helper_sha256": digest(__file__),
        "recovery_root": str(args.recovery_root.resolve()),
        "meaning": "CPU evaluation recovery; receipt aliases select existing training, not new training",
        "submission_ledger_sha256": digest(root / "submissions/production-jobs.jsonl"),
    }
    evaluator, manifest = pinned(plan)
    rows = [
        json.loads(line)
        for line in (root / "submissions/production-jobs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    jobs = {}
    for row in rows:
        key = (row["kind"], row["trial"])
        if key in jobs:
            raise ValueError("Ambiguous original submission ledger")
        jobs[key] = int(row["job_id"])
    lanes = read(root / "inputs/protocol.json")["extensions"]["lstm_adaptive_robustness"]["values"]["lanes"]
    main = [trial for pair in zip(lanes["17"], lanes["83"], strict=True) for trial in pair]
    optional = [
        "br-blue42-primary-red149",
        "br-blue42-final-red149",
        "br-blue100-final-red149",
        "br-blue100-primary-red149",
    ]
    cells = []
    for stage, trials in (("primary", main), ("final", main), ("primary", optional)):
        for offset in range(0, len(trials), 2):
            for blue in BLUES:
                for trial in trials[offset : offset + 2]:
                    if ("eval", trial) not in jobs:
                        continue
                    updates = manifest["primary_updates"] if stage == "primary" else manifest["trial_updates"][trial]
                    if stage == "final" and updates == manifest["primary_updates"]:
                        continue
                    receipt = root / "runs" / trial / "receipt.json"
                    selected = None
                    if receipt.exists():
                        selected = {"attempt_id": read(receipt)["attempt_id"], "receipt_sha256": digest(receipt)}
                    cells.append(
                        {
                            "id": f"c{len(cells):03d}",
                            "trial": trial,
                            "stage": stage,
                            "blue": blue,
                            "steps": updates * 48000,
                            "training_job": jobs["train", trial],
                            "original_eval_job": jobs["eval", trial],
                            "planned_training": selected,
                        }
                    )
    if not cells or len(cells) > 80:
        raise ValueError("Unexpected recovery panel size")
    plan["cells"] = cells
    return plan


def create_plan(args):
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return plan
    recovery = args.recovery_root.resolve()
    if recovery.is_relative_to(Path(plan["source_root"])) or recovery == Path(plan["study_root"]):
        raise ValueError("Recovery records need their own external directory")
    recovery.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, recovery / "recover_evaluations.py")
    (recovery / "logs").mkdir()
    write(recovery / "plan.json", plan)
    print(json.dumps({"plan": str(recovery / "plan.json"), "cells": len(plan["cells"])}))
    return plan


def load_plan(recovery):
    plan = read(recovery / "plan.json")
    if (
        str(recovery.resolve()) != plan["recovery_root"]
        or digest(recovery / "recover_evaluations.py") != plan["helper_sha256"]
    ):
        raise ValueError("Recovery helper or plan location changed")
    if digest(__file__) != plan["helper_sha256"]:
        raise ValueError("Execute the exact retained recovery helper")
    if digest(Path(plan["study_root"]) / "submissions/production-jobs.jsonl") != plan["submission_ledger_sha256"]:
        raise ValueError("Original submission ledger changed")
    return plan


def sbatch_command(plan, cell, dependencies, recovery):
    source = Path(plan["source_root"])
    command = [
        "sbatch",
        "--parsable",
        "--partition=community",
        "--cpus-per-task=8",
        "--mem=24G",
        "--time=120",
        "--signal=B:INT@180",
        f"--job-name=adaptive-recovery-{cell['id']}",
        f"--chdir={source}",
        f"--output={recovery}/logs/%j.out",
        f"--error={recovery}/logs/%j.err",
    ]
    if dependencies:
        command.append("--dependency=afterany:" + ":".join(str(value) for value in dict.fromkeys(dependencies)))
    invocation = [
        str(source / ".venv/bin/python"),
        str(recovery / "recover_evaluations.py"),
        "run-cell",
        "--recovery-root",
        str(recovery),
        "--cell-id",
        cell["id"],
        "--plan-sha256",
        digest(recovery / "plan.json"),
    ]
    command.extend(["--wrap", "exec " + shlex.join(invocation)])
    return command


def submit(args):
    recovery = args.recovery_root.resolve()
    plan = load_plan(recovery)
    evaluator, manifest = pinned(plan)
    records = records_for(plan)
    ledger = recovery / "submissions.jsonl"
    context = locked(recovery / ".submission.lock") if not args.dry_run else null_lock()
    with context:
        prior_rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        intents = {row["cell_id"] for row in prior_rows if row["event"] == "intent"}
        results = {row["cell_id"]: row for row in prior_rows if row["event"] in {"submitted", "skipped"}}
        if intents - results.keys():
            raise ValueError("Uncertain prior submission: inspect Slurm and reconcile; do not resubmit")
        previous = None
        for cell in plan["cells"]:
            if cell["id"] in results:
                previous = results[cell["id"]].get("job_id", previous)
                continue
            success = None
            execution = Path(plan["study_root"]) / "runs" / cell["trial"] / "execution.json"
            if execution.exists() and read(execution).get("returncode") == 0:
                binding = bind_training(plan, cell, evaluator, records)
                success = completed(plan, cell, binding, records, manifest)
            if success:
                entry = {"event": "skipped", "cell_id": cell["id"], "success": success, "at": now()}
                if not args.dry_run:
                    append(ledger, entry)
                print(json.dumps(entry), flush=True)
                continue
            state = scheduler_state(cell["original_eval_job"])
            dependencies = [] if state in TERMINAL else [cell["original_eval_job"]]
            if previous is not None and (args.dry_run or scheduler_state(previous) not in TERMINAL):
                dependencies.append(previous)
            command = sbatch_command(plan, cell, dependencies, recovery)
            if args.dry_run:
                print(json.dumps({"cell": cell, "command": command, "previous_recovery": previous}))
                previous = 999000 + int(cell["id"][1:])
                continue
            append(
                ledger,
                {
                    "event": "intent",
                    "at": now(),
                    "cell_id": cell["id"],
                    "command": command,
                    "original_eval_state": state,
                    "plan_sha256": digest(recovery / "plan.json"),
                },
            )
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            append(
                ledger,
                {
                    "event": "response",
                    "at": now(),
                    "cell_id": cell["id"],
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            )
            match = re.fullmatch(r"([0-9]+)(?:;[^\s]+)?\s*", result.stdout)
            if result.returncode != 0 or not match:
                raise ValueError("Submission outcome uncertain: stop and reconcile recorded sbatch response")
            previous = int(match.group(1))
            entry = {
                "event": "submitted",
                "at": now(),
                "cell_id": cell["id"],
                "job_id": previous,
                "dependencies": dependencies,
            }
            append(ledger, entry)
            print(json.dumps(entry), flush=True)


@contextmanager
def null_lock():
    yield


def make_alias(root, cell, binding, recovery, execution):
    run_id = "eval-recovery-" + uuid.uuid4().hex
    directory = root / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(binding["receipt_path"])
    if digest(source) != binding["receipt_sha256"]:
        raise ValueError("Original receipt changed before alias creation")
    shutil.copyfile(source, directory / "receipt.json")
    if digest(directory / "receipt.json") != binding["receipt_sha256"]:
        raise ValueError("Receipt selection alias differs from retained original")
    alias = {
        "meaning": "Selection alias for existing training; not a new training attempt",
        "run_id": run_id,
        "cell_key": cell_key(cell),
        "cell": cell,
        "training_attempt_id": binding["attempt_id"],
        "training_receipt_sha256": binding["receipt_sha256"],
        "original_receipt": str(source),
        "recovery_root": str(recovery),
        "execution_directory": str(execution),
        "plan_sha256": digest(recovery / "plan.json"),
        "created_at": now(),
    }
    write(directory / "selection-alias.json", alias)
    write(registry(root) / "aliases" / f"{run_id}.json", alias)
    return run_id


def partial_evidence(root, cell, records):
    directory = root / "evaluations" / cell["trial"] / cell["stage"] / cell["blue"]
    result = {"directory": str(directory), "exists": directory.exists(), "evaluation_ids": []}
    if (directory / "plan.json").exists():
        result["plan_sha256"] = digest(directory / "plan.json")
        for record in records.values():
            if record.get("role") == "resolved evaluation configuration":
                config = read(root / "evidence" / record["uri"])
                if config.get("overrides", {}).get("plan_sha256") == result["plan_sha256"]:
                    result["evaluation_ids"].extend(
                        row["id"]
                        for row in records.values()
                        if row.get("kind") == "evaluation" and row.get("attempt_id") == record["producer_id"]
                    )
    return result


def evaluation_command(plan, cell, run_id):
    source = Path(plan["source_root"])
    return [
        str(source / ".venv/bin/python"),
        str(source / "studies/lstm-adaptive-robustness/evaluate.py"),
        "--study-root",
        plan["study_root"],
        "--expected-revision",
        plan["expected_revision"],
        "--trial",
        cell["trial"],
        "--run-id",
        run_id,
        "--stage",
        cell["stage"],
        "--blue",
        cell["blue"],
    ]


def launch_cell(plan, cell, run_id, directory):
    source = Path(plan["source_root"])
    env = os.environ.copy()
    env.update(CPU_ENV)
    env["JAX_COMPILATION_CACHE_DIR"] = str(Path(plan["study_root"]) / "compilation-cache/eval")
    env["MLFLOW_EXPERIMENT_NAME"] = env["JAXBORG_MLFLOW_EXPERIMENT"] = "lstm-adaptive-robustness"
    setup = 'source "$HOME/jaxborg-exp/shared/store.env"\n'
    setup += 'export MLFLOW_TRACKING_URI="sqlite:///$JAXBORG_EXP_DIR/mlflow.db"\n'
    setup += "source scripts/jax_env.sh\nexec " + shlex.join(evaluation_command(plan, cell, run_id))
    command = ["timeout", "--signal=INT", "--kill-after=90", "6600", "bash", "-c", setup]
    recorded_env = {
        name: env[name]
        for name in (*CPU_ENV, "JAX_COMPILATION_CACHE_DIR", "MLFLOW_EXPERIMENT_NAME", "JAXBORG_MLFLOW_EXPERIMENT")
    }
    write(
        directory / "command.json",
        {
            "command": command,
            "cwd": str(source),
            "environment": recorded_env,
            "environment_sources": ["~/jaxborg-exp/shared/store.env", "scripts/jax_env.sh"],
        },
    )
    with (directory / "evaluation.log").open("x") as output:
        process = subprocess.Popen(
            command, cwd=source, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )

        def interrupt(_signum, _frame):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)

        handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            return process.wait()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def require_cpu_allocation():
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or os.environ.get("SLURM_JOB_PARTITION") != "community":
        raise ValueError("run-cell requires a community Slurm CPU allocation")
    allocation = subprocess.check_output(["scontrol", "show", "job", "--oneliner", job], text=True)
    fields = dict(field.split("=", 1) for field in allocation.split() if "=" in field)
    resources = fields.get("AllocTRES", fields.get("TRES"))
    if (
        resources is None
        or re.search(r"gres[/=:]gpu|Gres=gpu", allocation)
        or fields.get("JobState") != "RUNNING"
        or fields.get("Partition") != "community"
    ):
        raise ValueError("run-cell requires a running CPU-only allocation")


def run_cell(args):
    require_cpu_allocation()
    recovery = args.recovery_root.resolve()
    plan = load_plan(recovery)
    if digest(recovery / "plan.json") != args.plan_sha256:
        raise ValueError("Recovery plan changed since submission")
    cell = next(row for row in plan["cells"] if row["id"] == args.cell_id)
    root = Path(plan["study_root"])
    with locked(registry(root) / "locks" / (cell_key(cell) + ".lock")):
        state = require_terminal(cell["original_eval_job"])
        evaluator, manifest = pinned(plan)
        records = records_for(plan)
        directory = recovery / "cells" / cell["id"] / uuid.uuid4().hex
        directory.mkdir(parents=True)
        write(
            directory / "start.json",
            {
                "cell": cell,
                "at": now(),
                "original_eval_state": state,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "plan_sha256": digest(recovery / "plan.json"),
            },
        )
        try:
            if require_terminal(cell["training_job"]) != "COMPLETED":
                raise ValueError("Original training job did not complete successfully")
            binding = bind_training(plan, cell, evaluator, records)
            selected = {key: binding[key] for key in ("attempt_id", "receipt_sha256", "execution_sha256")}
            selection_path = registry(root) / "training-selections" / (cell["trial"] + ".json")
            with locked(registry(root) / "locks" / ("training-" + cell["trial"] + ".lock")):
                if selection_path.exists():
                    if read(selection_path) != selected:
                        raise ValueError("Original training selection differs from an earlier recovery cell")
                else:
                    write(selection_path, selected)
            write(directory / "selection.json", binding)
            shutil.copyfile(binding["receipt_path"], directory / "training-receipt.json")
            success = completed(plan, cell, binding, records, manifest)
            if success:
                write(directory / "result.json", {"status": "skipped", "success": success, "at": now()})
                return
            write(directory / "original-evidence.json", partial_evidence(root, cell, records))
            run_id = make_alias(root, cell, binding, recovery, directory)
            code = launch_cell(plan, cell, run_id, directory)
            if code != 0:
                raise RuntimeError(f"Evaluation exited {code}; partial evidence retained under {run_id}")
            success = verified_success(
                root / "evaluations" / run_id / cell["stage"] / cell["blue"] / "receipt.json",
                plan,
                cell,
                binding,
                records_for(plan),
                manifest,
            )
            write(directory / "result.json", {"status": "succeeded", "success": success, "run_id": run_id, "at": now()})
        except BaseException as error:
            write(directory / "result.json", {"status": "failed", "error": str(error), "at": now()})
            raise


def main():
    os.environ.update(CPU_ENV)
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    create = sub.add_parser("plan")
    create.add_argument("--study-root", type=Path, required=True)
    create.add_argument("--source-root", type=Path, required=True)
    create.add_argument("--expected-revision", required=True)
    for command in (create, sub.add_parser("submit"), sub.add_parser("run-cell")):
        command.add_argument("--recovery-root", type=Path, required=True)
        if command.prog.endswith("run-cell"):
            command.add_argument("--cell-id", required=True)
            command.add_argument("--plan-sha256", required=True)
        else:
            command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    {"plan": create_plan, "submit": submit, "run-cell": run_cell}[args.mode](args)


if __name__ == "__main__":
    main()
