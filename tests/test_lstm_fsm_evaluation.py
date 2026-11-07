"""Guard checkpoint ownership and frozen evaluation membership without rollouts."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def study_module():
    path = Path(__file__).resolve().parents[1] / "studies/lstm-fsm-competence/evaluate.py"
    spec = importlib.util.spec_from_file_location("lstm_fsm_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def owned(tmp_path):
    owner = {
        "id": "training-42",
        "kind": "attempt",
        "execution_kind": "training",
        "status": "succeeded",
        "trial_id": "fsm-seed42",
        "budgets": [{"unit": "environment_steps", "requested": 50000000, "completed": 49968000}],
        "facts": {},
    }
    records = [owner]
    for steps, key in ((0, "initial_checkpoint"), (49968000, "checkpoint_49968000")):
        value = {"environment_steps": steps}
        for role in ("model", "sidecar"):
            path = tmp_path / f"{steps}-{role}"
            path.write_text(f"{role}-{steps}")
            identity = f"artifact:{steps}:{role}"
            records.append(
                {
                    "kind": "artifact",
                    "id": identity,
                    "producer_id": owner["id"],
                    "uri": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            value[role] = identity
        owner["facts"][key] = {"value": value}
    return tmp_path, records


def test_checkpoint_selection_uses_owned_final_budget_and_checks_bytes(owned):
    root, records = owned
    study = study_module()
    selected = study.checkpoint_selection(records, root, "fsm-seed42", "training-42")
    assert [item["steps"] for item in selected] == [0, 49968000]
    Path(selected[1]["model"]).write_text("changed")
    with pytest.raises(ValueError, match="hash differs"):
        study.checkpoint_selection(records, root, "fsm-seed42", "training-42")


@pytest.mark.parametrize("mutation", ["failed", "wrong_trial", "foreign_sidecar", "missing_final"])
def test_checkpoint_selection_rejects_incomplete_or_foreign_inputs(owned, mutation):
    root, records = owned
    if mutation == "failed":
        records[0]["status"] = "failed"
    elif mutation == "wrong_trial":
        records[0]["trial_id"] = "fsm-seed100"
    elif mutation == "foreign_sidecar":
        records[-1]["producer_id"] = "training-100"
    else:
        del records[0]["facts"]["checkpoint_49968000"]
    with pytest.raises(ValueError):
        study_module().checkpoint_selection(records, root, "fsm-seed42", "training-42")


def test_live_training_selects_exact_published_milestone_not_latest(owned):
    root, records = owned
    records[0]["status"] = "running"
    records[0]["budgets"][0]["completed"] = 72000000
    selected = study_module().checkpoint_selection(
        records, root, "fsm-seed42", "training-42", target_steps=49968000, allow_running=True
    )
    assert [item["steps"] for item in selected] == [0, 49968000]
    with pytest.raises(ValueError, match="eligible training"):
        study_module().checkpoint_selection(records, root, "fsm-seed42", "training-42", target_steps=49968000)
    with pytest.raises(ValueError, match="not yet completed"):
        study_module().checkpoint_selection(
            records, root, "fsm-seed42", "training-42", target_steps=124992000, allow_running=True
        )
    with pytest.raises(ValueError, match="Missing declared"):
        study_module().checkpoint_selection(
            records, root, "fsm-seed42", "training-42", target_steps=72000000, allow_running=True
        )


def test_resumed_training_uses_original_zero_step_owner(owned):
    import copy

    root, records = owned
    original = records[0]
    resumed = copy.deepcopy(original)
    resumed["id"] = "resumed-42"
    resumed["facts"]["initial_checkpoint"] = copy.deepcopy(original["facts"]["checkpoint_49968000"])
    resumed["facts"]["training_origin"] = {
        "value": {"attempt_id": original["id"], "checkpoint": original["facts"]["initial_checkpoint"]["value"]}
    }
    for artifact in records[3:]:
        artifact["producer_id"] = resumed["id"]
    records.append(resumed)
    selected = study_module().checkpoint_selection(records, root, "fsm-seed42", "resumed-42")
    assert selected[0]["training_attempt"] == "training-42"
    assert selected[1]["training_attempt"] == "resumed-42"
    resumed["facts"]["training_origin"]["value"]["attempt_id"] = "missing-origin"
    with pytest.raises(ValueError, match="verified original"):
        study_module().checkpoint_selection(records, root, "fsm-seed42", "resumed-42")


def test_panel_never_substitutes_or_drops_reserved_topologies(tmp_path):
    study = study_module()
    inputs = []
    for seed in (200, 201, 202, 203):
        path = tmp_path / str(seed)
        path.write_text(str(seed))
        inputs.append({"source_seed": seed, "path": str(path), "sha256": study.digest(path)})
    reds = [
        {
            "id": identity,
            "model": inputs[0]["path"],
            "sidecar": inputs[1]["path"],
            "sha256": inputs[0]["sha256"],
            "sidecar_sha256": inputs[1]["sha256"],
        }
        for identity in study.RED_IDS
    ]
    manifest = {"topology_banks": {"unseen": {"snapshots": inputs}}, "policies": {"red": reds}}
    assert len(study.panel_from_manifest(manifest, "primary")[0]) == 4
    assert len(study.panel_from_manifest(manifest, "smoke")[0]) == 1
    assert study.panel_from_manifest(manifest, "interim")[2] == []
    assert len(study.panel_from_manifest(manifest, "primary")[2]) == 4
    manifest["topology_banks"]["unseen"]["snapshots"][0]["source_seed"] = 100
    with pytest.raises(ValueError, match="200..203"):
        study.panel_from_manifest(manifest, "primary")


def test_initial_reference_reuse_checks_records_inputs_and_raw_rewards(tmp_path):
    study = study_module()
    case = {
        "checkpoint": "blue",
        "checkpoint_sha256": "blue-hash",
        "sidecar_sha256": "sidecar-hash",
        "variant": {"name": "stock"},
        "topologies": ["topo-hash"],
        "seeds": [21000, 21001],
        "source_revision": "a" * 40,
        "red_id": "fsm",
    }
    receipt = {"case": case, "checkpoint_steps": 0, "evaluation_id": "evaluation:1", "episodes": 2, "mean_return": -2.0}
    directory = tmp_path / "attempts/1"
    directory.mkdir(parents=True)
    config = {
        "base_variant": case["variant"],
        "topology": {"sha256": case["topologies"]},
        "seeds": case["seeds"],
        "execution": {"source_revision": case["source_revision"]},
        "deterministic": False,
        "episodes_per_seed": 1,
        "reds": ["fsm"],
    }
    (directory / "configuration.json").write_text(json.dumps(config))
    (directory / "input-artifacts.json").write_text(json.dumps({"source recipe": "sidecar"}))
    records = [
        {
            "id": "evaluation:1",
            "kind": "evaluation",
            "attempt_id": "attempt:1",
            "status": "succeeded",
            "inputs": ["blue"],
        },
        {"id": "attempt:1", "kind": "attempt", "status": "succeeded"},
        {"id": "blue", "kind": "artifact", "sha256": "blue-hash"},
        {"id": "sidecar", "kind": "artifact", "sha256": "sidecar-hash"},
    ]
    for index, reward in enumerate((-1.0, -3.0)):
        path = directory / f"episode{index}.json"
        path.write_text(json.dumps({"reward": reward}))
        records.append(
            {
                "id": f"episode:{index}",
                "kind": "artifact",
                "producer_id": "evaluation:1",
                "role": "evaluation episode observation",
                "uri": str(path.relative_to(tmp_path)),
                "sha256": study.digest(path),
            }
        )
    assert study.validate_reference(receipt, case, records, tmp_path)["reused"] is True
    receipt["mean_return"] = 5.0
    with pytest.raises(ValueError, match="summary differs"):
        study.validate_reference(receipt, case, records, tmp_path)
    receipt["mean_return"] = -2.0
    config["reds"] = ["cia_a"]
    (directory / "configuration.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="another scripted"):
        study.validate_reference(receipt, case, records, tmp_path)


def test_owned_scripted_capture_retains_caller_closure_and_condition_check(tmp_path, monkeypatch):
    from jaxborg.evaluation import scripted_capture

    model = tmp_path / "model.safetensors"
    model.write_bytes(b"checkpoint")
    (tmp_path / "protocol.json").write_text(json.dumps({"id": "protocol"}))
    records = [
        {
            "id": "model",
            "kind": "artifact",
            "producer_id": "training",
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        },
        {"id": "training", "execution_kind": "training"},
    ]
    provenance_calls, captured = [], []
    provenance = SimpleNamespace()
    monkeypatch.setattr(scripted_capture, "load_records", lambda root: records)
    monkeypatch.setattr(
        scripted_capture, "collect_provenance", lambda **kwargs: provenance_calls.append(kwargs) or provenance
    )
    monkeypatch.setattr(
        scripted_capture, "_run_scripted", lambda **kwargs: captured.append(kwargs) or ("receipt", ["row"])
    )

    def check(config):
        return config

    context = {
        "root": tmp_path,
        "checkpoint": "model",
        "training_attempt_id": "training",
        "revision": "a" * 40,
        "trial_id": "trial",
        "source_recipe": "source",
        "effective_recipe": "effective",
        "entrypoint": Path(__file__),
        "source_roots": (tmp_path,),
        "condition_check": check,
    }
    assert scripted_capture.run_scripted(model, context=context, return_capture=True) == ("receipt", ["row"])
    assert provenance_calls[0]["entrypoint"] == Path(__file__)
    assert tmp_path in provenance_calls[0]["source_roots"]
    assert captured[0]["condition_check"] is check
    assert captured[0]["retained_inputs"] == {"blue checkpoint": "model"}
    assert captured[0]["training"]["training_attempt_id"] == "training"
    assert scripted_capture.run_scripted(model, context=context) == ["row"]


def test_cpu_launcher_uses_explicit_checkout_after_slurm_spooling(tmp_path):
    import os
    import shutil
    import subprocess

    if os.name != "posix" or shutil.which("bash") is None:
        pytest.skip("Slurm launcher requires Linux")
    script = Path(__file__).resolve().parents[1] / "studies/lstm-fsm-competence/eval_launch.sh"
    spool = tmp_path / "slurm_script"
    spool.write_bytes(script.read_bytes())
    checkout = tmp_path / "checkout"
    (checkout / ".venv/bin").mkdir(parents=True)
    stub = checkout / ".venv/bin/python"
    stub.write_text('#!/usr/bin/env bash\npwd > "$CALL_LOG"\nprintf "%s\\n" "$@" >> "$CALL_LOG"\n')
    stub.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    revision = "a" * 40
    git = fake_bin / "git"
    git.write_text(f'#!/usr/bin/env bash\nif [ "$1" = rev-parse ]; then echo {revision}; fi\n')
    git.chmod(0o755)
    date = fake_bin / "date"
    date.write_text('#!/usr/bin/env bash\nif [ "$1" = -d ]; then echo 7200; else echo 1000; fi\n')
    date.chmod(0o755)
    home = tmp_path / "home"
    shared = home / "jaxborg-exp/shared"
    shared.mkdir(parents=True)
    (shared / "store.env").write_text(f'export JAXBORG_EXP_DIR="{shared}"\n')
    study = tmp_path / "study"
    study.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    output = tmp_path / "invocation.txt"
    env = dict(
        os.environ,
        HOME=str(home),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        CALL_LOG=str(output),
        SLURM_JOB_ID="123",
        SLURM_JOB_PARTITION="community",
    )
    subprocess.run(
        ["bash", str(spool), "allocated", str(study), "smoke-seed42", "smoke", str(manifest), revision, str(checkout)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = output.read_text().splitlines()
    assert lines[0] == str(checkout)
    assert lines[1] == "studies/lstm-fsm-competence/evaluate.py"
    assert lines[-1] == revision


def test_evaluation_can_finish_with_owned_checkpoint_from_running_training(tmp_path, monkeypatch):
    from jaxborg.evaluation.capture_store import (
        add_artifact,
        finish_evaluation,
        load_records,
        record_episode,
        start_evaluation,
        start_training,
    )

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    protocol = {
        "schema_version": 1,
        "kind": "protocol",
        "id": "live-checkpoint",
        "study_id": "live-checkpoint-test",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": [], "evaluation": []},
        "metric_definitions": [],
        "trials": [{"id": "trial", "condition_id": "fsm", "replicate_id": "42", "matched_unit_id": None}],
    }
    root = tmp_path / "evidence"
    training = start_training(
        root,
        protocol=protocol,
        trial_id="trial",
        requested_steps=100,
        requested_updates=1,
        resolved_config={},
        inputs={},
        source_files={},
    )
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"immutable owned checkpoint")
    checkpoint = add_artifact(training, model, role="policy checkpoint")
    evaluation = start_evaluation(
        root,
        protocol=protocol,
        trial_id="trial",
        requested_episodes=1,
        resolved_config={},
        inputs={},
        source_files={},
        retained_inputs={"blue checkpoint": checkpoint},
    )
    record_episode(evaluation, episode_index=0, seed=21000, reward=-10.0, steps=500)
    finish_evaluation(evaluation, status="succeeded")
    records = {record["id"]: record for record in load_records(root)}
    assert records[training.attempt_id]["status"] == "running"
    assert records[evaluation.evaluation_id]["status"] == "succeeded"
    assert checkpoint in records[evaluation.evaluation_id]["inputs"]


def test_running_attempt_resolves_without_a_terminal_receipt(owned):
    root, records = owned
    records[0]["status"] = "running"
    records[0]["facts"]["source_revision"] = {"value": "a" * 40}
    missing = root / "runs/fsm-seed42/receipt.json"
    assert not missing.exists()
    assert (
        study_module().resolve_training_attempt(records, "fsm-seed42", "a" * 40, receipt_path=missing) == "training-42"
    )


def test_attempt_resolver_rejects_ambiguity_and_wrong_source(owned):
    import copy

    root, records = owned
    owner = records[0]
    owner["facts"]["source_revision"] = {"value": "a" * 40}
    retry = copy.deepcopy(owner)
    retry["id"] = "retry-42"
    records.append(retry)
    study = study_module()
    with pytest.raises(ValueError, match="multiple training"):
        study.resolve_training_attempt(records, "fsm-seed42", "a" * 40)
    assert study.resolve_training_attempt(records, "fsm-seed42", "a" * 40, explicit="retry-42") == "retry-42"
    with pytest.raises(ValueError, match="different source"):
        study.resolve_training_attempt(records, "fsm-seed42", "b" * 40, explicit="retry-42")
    with pytest.raises(ValueError, match="requested trial"):
        study.resolve_training_attempt(records, "fsm-seed100", "a" * 40, explicit="retry-42")
