"""Recovery preserves successful cells and never changes original executions."""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("adaptive_recovery", Path(__file__).with_name("recover_evaluations.py"))
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def success(tmp_path):
    plan = {"study_root": str(tmp_path), "expected_revision": "a" * 40, "manifest_sha256": "manifest"}
    cell = {
        "id": "c000",
        "trial": "br-blue42-primary-red17",
        "stage": "primary",
        "blue": "blue42-primary",
        "steps": 9984000,
        "original_eval_job": 4159,
        "training_job": 4158,
    }
    response = {
        "attempt_id": "training",
        "steps": cell["steps"],
        "model_artifact": "red-model",
        "sidecar_artifact": "red-sidecar",
        "model_sha256": "red",
        "sidecar_sha256": "red-side",
    }
    binding = {"attempt_id": "training", "response": response}
    manifest = {
        "blue_checkpoints": {cell["blue"]: {"model_sha256": "blue", "sidecar_sha256": "blue-side"}},
        "evaluation_topologies": [{"sha256": f"topology-{index}"} for index in range(4)],
    }
    manifest["_expected_variant"] = {"name": "cc4_stock"}
    path = tmp_path / "evaluations" / cell["trial"] / cell["stage"] / cell["blue"] / "receipt.json"
    evaluation_plan = {
        "trial": cell["trial"],
        "stage": cell["stage"],
        "blue_id": cell["blue"],
        "run_id": cell["trial"],
        "response": response,
        "blue": manifest["blue_checkpoints"][cell["blue"]],
        "topologies": manifest["evaluation_topologies"],
        "episode_roots": list(range(21000, 21016)),
        "variant": {"name": "cc4_stock"},
        "deterministic": False,
        "backend": "cpu",
    }
    put(path.with_name("plan.json"), evaluation_plan)
    receipt = {
        "passed": True,
        "trial": cell["trial"],
        "stage": cell["stage"],
        "blue_id": cell["blue"],
        "episodes": 64,
        "training_attempt_id": "training",
        "response_environment_steps": cell["steps"],
        "source_revision": plan["expected_revision"],
        "manifest_sha256": "manifest",
        "blue_returns": [-1.0] * 64,
        "mean_blue_return": -1.0,
        "evaluation_id": "evaluation",
        "run_id": cell["trial"],
    }
    put(path, receipt)
    config = {
        "overrides": {"plan_sha256": recovery.digest(path.with_name("plan.json")), "blue_id": cell["blue"]},
        "horizon": 500,
        "episodes_per_seed": 1,
        "seed_roles": {"episode_roots": list(range(21000, 21016))},
        "topology": {"sha256": [row["sha256"] for row in manifest["evaluation_topologies"]]},
        "variant": evaluation_plan["variant"],
        "deterministic": False,
    }
    config_path = put(tmp_path / "evidence/configuration.json", config)
    records = {
        "evaluation": {
            "id": "evaluation",
            "kind": "evaluation",
            "status": "succeeded",
            "attempt_id": "eval-attempt",
            "budgets": [{"requested": 64, "completed": 64}],
            "inputs": ["red-model", "red-sidecar", "blue-model", "blue-sidecar"],
        },
        "eval-attempt": {
            "id": "eval-attempt",
            "kind": "attempt",
            "status": "succeeded",
            "execution_kind": "evaluation",
            "trial_id": cell["trial"],
            "facts": {"source_revision": {"value": plan["expected_revision"]}},
        },
        "config": {
            "id": "config",
            "producer_id": "eval-attempt",
            "role": "resolved evaluation configuration",
            "uri": "configuration.json",
            "sha256": recovery.digest(config_path),
        },
        "red-model": {"kind": "artifact", "sha256": "red"},
        "red-sidecar": {"kind": "artifact", "sha256": "red-side"},
        "blue-model": {"kind": "artifact", "sha256": "blue"},
        "blue-sidecar": {"kind": "artifact", "sha256": "blue-side"},
    }
    result_path = put(tmp_path / "evidence/result.json", {"blue_returns": receipt["blue_returns"]})
    records["evaluation"]["metrics"] = []
    records["result"] = {
        "id": "result",
        "producer_id": "evaluation",
        "role": "joint matchup result",
        "uri": "result.json",
        "sha256": recovery.digest(result_path),
    }
    return SimpleNamespace(
        root=tmp_path,
        path=path,
        plan=plan,
        cell=cell,
        binding=binding,
        manifest=manifest,
        records=records,
        receipt=receipt,
        config=config,
        config_path=config_path,
    )


def test_verified_completed_cell_is_skipped(success):
    result = recovery.completed(success.plan, success.cell, success.binding, success.records, success.manifest)
    assert result["evaluation_id"] == "evaluation"


@pytest.mark.parametrize(
    "change",
    [
        {"training_attempt_id": "different-training"},
        {"stage": "final"},
        {"episodes": 63},
        {"passed": False},
        {"response_environment_steps": 29952000},
        {"blue_id": "blue100-final"},
        {"blue_returns": [-1.0] * 63},
    ],
)
def test_invalid_existing_success_fails_closed(success, change):
    put(success.path, {**success.receipt, **change})
    with pytest.raises(ValueError):
        recovery.completed(success.plan, success.cell, success.binding, success.records, success.manifest)


@pytest.mark.parametrize("status", ["running", "failed", "cancelled"])
def test_receipt_cannot_override_unsuccessful_evidence(success, status):
    success.records["evaluation"]["status"] = status
    with pytest.raises(ValueError, match="successful"):
        recovery.verified_success(
            success.path, success.plan, success.cell, success.binding, success.records, success.manifest
        )


@pytest.mark.parametrize("changed", ["red-input", "blue-input", "horizon", "seed"])
def test_actual_evidence_must_match_cell(success, changed):
    if changed == "red-input":
        success.records["evaluation"]["inputs"].remove("red-model")
    elif changed == "blue-input":
        success.records["blue-model"]["sha256"] = "another-blue"
    else:
        if changed == "horizon":
            success.config["horizon"] = 499
        else:
            success.config["seed_roles"]["episode_roots"] = [123]
        put(success.config_path, success.config)
        success.records["config"]["sha256"] = recovery.digest(success.config_path)
    with pytest.raises(ValueError):
        recovery.completed(success.plan, success.cell, success.binding, success.records, success.manifest)


def test_completed_evidence_without_receipt_refuses_rerun(success):
    success.path.unlink()
    with pytest.raises(ValueError, match="reconcile without rerunning"):
        recovery.completed(success.plan, success.cell, success.binding, success.records, success.manifest)


def test_partial_evidence_is_preserved_and_can_be_recovered(success):
    success.path.unlink()
    success.records["evaluation"]["status"] = "cancelled"
    success.records["eval-attempt"]["status"] = "cancelled"
    assert recovery.completed(success.plan, success.cell, success.binding, success.records, success.manifest) is None
    assert success.path.with_name("plan.json").exists()


def test_fresh_alias_does_not_collide_with_original_output(success):
    source = put(success.root / "runs" / success.cell["trial"] / "receipt.json", {"attempt_id": "training"})
    binding = {**success.binding, "receipt_path": str(source), "receipt_sha256": recovery.digest(source)}
    external = success.root / "recovery"
    put(external / "plan.json", success.plan)
    alias = recovery.make_alias(success.root, success.cell, binding, external, external / "cells/c000")
    assert alias != success.cell["trial"]
    assert (success.root / "runs" / alias / "receipt.json").read_bytes() == source.read_bytes()
    assert success.path.exists()
    assert not (success.root / "evaluations" / alias).exists()
    registration = recovery.read(success.root / "evaluation-recovery/aliases" / (alias + ".json"))
    assert registration["training_attempt_id"] == "training"


def test_active_original_refuses_before_alias_or_execution(tmp_path, monkeypatch):
    plan = {
        "study_root": str(tmp_path / "production"),
        "cells": [
            {"id": "c000", "trial": "trial", "stage": "primary", "blue": "blue42-primary", "original_eval_job": 4159}
        ],
    }
    path = put(tmp_path / "recovery/plan.json", plan)
    monkeypatch.setattr(recovery, "load_plan", lambda _: plan)
    monkeypatch.setattr(recovery, "require_cpu_allocation", lambda: None)
    monkeypatch.setattr(recovery, "scheduler_state", lambda _: "ACTIVE")
    with pytest.raises(ValueError, match="still active"):
        recovery.run_cell(Namespace(recovery_root=path.parent, cell_id="c000", plan_sha256=recovery.digest(path)))
    assert not (path.parent / "cells").exists()
    assert not (tmp_path / "production/runs").exists()


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_failed_training_never_selects_checkpoint(tmp_path, missing):
    cell = {"trial": "trial"}
    if not missing:
        receipt = {"attempt_id": "training", "evidence_root": str(tmp_path / "evidence")}
        put(tmp_path / "runs/trial/receipt.json", receipt)
        put(
            tmp_path / "runs/trial/execution.json",
            {"returncode": 1, "interrupted": False, "trial_id": "trial", "run_id": "trial", "receipt": receipt},
        )

    def unexpected(*_):
        pytest.fail("Checkpoint selection called for unsuccessful training")

    with pytest.raises(ValueError):
        recovery.bind_training({"study_root": str(tmp_path)}, cell, SimpleNamespace(selected_response=unexpected), {})


def test_serial_cpu_submission_has_afterany_dependencies(tmp_path):
    put(tmp_path / "plan.json", {})
    command = recovery.sbatch_command({"source_root": "/pinned"}, {"id": "c003"}, [4159, 4202, 4159], tmp_path)
    assert "--dependency=afterany:4159:4202" in command
    assert "--cpus-per-task=8" in command and "--mem=24G" in command and "--time=120" in command
    assert not any("gpu" in item or "deadline" in item for item in command)
    assert "--plan-sha256" in command[-1]


def test_old_original_job_uses_accounting_after_no_active_queue_entry(monkeypatch):
    def output(command, **_):
        return "4160\n" if command[0] == "squeue" else "4159|TIMEOUT|0:9\n"

    monkeypatch.setattr(recovery.subprocess, "check_output", output)
    assert recovery.require_terminal(4159) == "TIMEOUT"


def test_uncertain_submission_ledger_stops_without_sbatch(tmp_path, monkeypatch):
    put(tmp_path / "plan.json", {})
    (tmp_path / "submissions.jsonl").write_text(json.dumps({"event": "intent", "cell_id": "c000"}) + "\n")
    monkeypatch.setattr(recovery, "load_plan", lambda _: {"cells": []})
    monkeypatch.setattr(recovery, "pinned", lambda _: (None, {}))
    monkeypatch.setattr(recovery, "records_for", lambda _: {})
    monkeypatch.setattr(recovery.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Resubmission attempted"))
    with pytest.raises(ValueError, match="Uncertain"):
        recovery.submit(Namespace(recovery_root=tmp_path, dry_run=False))


def test_submission_chain_survives_skips_and_omits_old_terminal_dependencies(tmp_path, monkeypatch):
    cells = [
        {"id": f"c{index:03d}", "trial": f"trial-{index}", "original_eval_job": 4159 + index} for index in range(3)
    ]
    plan = {"study_root": str(tmp_path / "production"), "source_root": "/pinned", "cells": cells}
    put(tmp_path / "plan.json", plan)
    monkeypatch.setattr(recovery, "load_plan", lambda _: plan)
    monkeypatch.setattr(recovery, "pinned", lambda _: (None, {}))
    monkeypatch.setattr(recovery, "records_for", lambda _: {})
    monkeypatch.setattr(recovery, "scheduler_state", lambda job: "TIMEOUT" if job == 4159 else "ACTIVE")
    calls = []

    def submit(command, **_):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=str(4200 + len(calls)) + "\n", stderr="")

    monkeypatch.setattr(recovery.subprocess, "run", submit)
    recovery.submit(Namespace(recovery_root=tmp_path, dry_run=False))
    assert not any(arg.startswith("--dependency") for arg in calls[0])
    assert "--dependency=afterany:4160:4201" in calls[1]
    assert "--dependency=afterany:4161:4202" in calls[2]
    before = len(calls)
    recovery.submit(Namespace(recovery_root=tmp_path, dry_run=False))
    assert len(calls) == before


def test_run_cell_refuses_login_node(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm CPU allocation"):
        recovery.require_cpu_allocation()


def test_run_cell_refuses_gpu_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4201")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "community")
    monkeypatch.setattr(
        recovery.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "JobState=RUNNING Partition=community AllocTRES=cpu=8,mem=24G,gres/gpu=1",
    )
    with pytest.raises(ValueError, match="CPU-only"):
        recovery.require_cpu_allocation()


@pytest.mark.parametrize("field", ["TRES", "AllocTRES"])
def test_cpu_guard_accepts_itm_and_new_slurm_resource_fields(monkeypatch, field):
    monkeypatch.setenv("SLURM_JOB_ID", "4201")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "community")
    monkeypatch.setattr(
        recovery.subprocess,
        "check_output",
        lambda *_args, **_kwargs: f"JobState=RUNNING Partition=community {field}=cpu=8,mem=24G,node=1,billing=8",
    )
    recovery.require_cpu_allocation()


def test_partial_registered_recovery_must_have_terminal_job(success, monkeypatch):
    success.path.unlink()
    success.records["evaluation"]["status"] = "cancelled"
    success.records["eval-attempt"]["status"] = "cancelled"
    source = put(success.root / "runs" / success.cell["trial"] / "receipt.json", {"attempt_id": "training"})
    binding = {**success.binding, "receipt_path": str(source), "receipt_sha256": recovery.digest(source)}
    external = success.root / "recovery"
    put(external / "plan.json", success.plan)
    execution = external / "cells/c000"
    put(execution / "start.json", {"slurm_job_id": "4201"})
    recovery.make_alias(success.root, success.cell, binding, external, execution)
    monkeypatch.setattr(recovery, "scheduler_state", lambda _: "ACTIVE")
    with pytest.raises(ValueError, match="still active"):
        recovery.completed(success.plan, success.cell, binding, success.records, success.manifest)


def test_cpu_guard_rejects_itm_gpu_tres_per_node(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4160")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "community")
    monkeypatch.setattr(
        recovery.subprocess,
        "check_output",
        lambda *_args, **_kwargs: (
            "JobState=RUNNING Partition=community TRES=cpu=8,mem=32G,node=1,billing=8 TresPerNode=gres:gpu:1"
        ),
    )
    with pytest.raises(ValueError, match="CPU-only"):
        recovery.require_cpu_allocation()


def test_consistent_but_altered_receipt_returns_are_rejected(success):
    put(success.path, {**success.receipt, "blue_returns": [-2.0] * 64, "mean_blue_return": -2.0})
    with pytest.raises(ValueError, match="returns differ from retained result"):
        recovery.completed(success.plan, success.cell, success.binding, success.records, success.manifest)


@pytest.mark.parametrize("value", [-1.0, -2.0])
def test_optional_blue_return_metric_must_match_if_present(success, value):
    success.records["evaluation"]["metrics"] = [{"name": "blue_return", "value": value}]
    if value == -1.0:
        assert recovery.completed(success.plan, success.cell, success.binding, success.records, success.manifest)
    else:
        with pytest.raises(ValueError, match="recorded evaluation results"):
            recovery.completed(success.plan, success.cell, success.binding, success.records, success.manifest)
