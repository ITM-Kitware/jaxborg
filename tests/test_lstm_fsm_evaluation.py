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
    assert len(study.panel_from_manifest(manifest, "full")[0]) == 4
    assert len(study.panel_from_manifest(manifest, "smoke")[0]) == 1
    manifest["topology_banks"]["unseen"]["snapshots"][0]["source_seed"] = 100
    with pytest.raises(ValueError, match="200..203"):
        study.panel_from_manifest(manifest, "full")


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
