"""Protect the declared intervention and immutable preparation inputs."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]
STUDY = PROJECT / "studies/lstm-fsm-competence"
SPEC = importlib.util.spec_from_file_location("lstm_fsm_prepare", STUDY / "prepare.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def test_smoke_changes_only_budget_and_preserves_complete_recurrent_episodes(tmp_path):
    recipe = yaml.safe_load((STUDY / "recipe.yaml").read_text())
    original = copy.deepcopy(recipe)
    full, smoke = prepare.recipes_with_banks(recipe, [tmp_path / "train.npz"], [tmp_path / "eval.npz"])
    assert recipe == original
    assert smoke["train"]["total_timesteps"] == 144000
    smoke["train"]["total_timesteps"] = full["train"]["total_timesteps"]
    assert smoke == full
    assert full["train"]["episode_length"] == 500
    assert full["jax"]["num_envs"] == 96
    assert full["jax"]["num_minibatches"] == 16
    assert full["jax"]["update_epochs"] == 4
    assert full["arch"]["cell"] == "lstm"
    assert full["arch"]["trunk"] == "separate"
    assert full["cage4_enhanced_obs"] is True
    assert full["mlflow"]["checkpoint_eval"]["every_steps"] == 0
    assert full["eval"]["after_training"] == []


def test_protocol_declares_actual_budget_and_distinct_training_replicates():
    from jaxborg.experiment_records.core import validate_records

    protocol = json.loads((STUDY / "protocol.json").read_text())
    validate_records([protocol])
    recipe = yaml.safe_load((STUDY / "recipe.yaml").read_text())
    settings = protocol["extensions"]["lstm_fsm_competence"]["values"]
    stride = recipe["train"]["episode_length"] * recipe["jax"]["num_envs"]
    updates = recipe["train"]["total_timesteps"] // stride
    assert settings["rollout_updates_per_seed"] == updates == 1041
    assert settings["completed_environment_steps_per_seed"] == updates * stride == 49968000
    trials = [trial for trial in protocol["trials"] if trial["condition_id"] == "fixed-stock-fsm"]
    assert {trial["replicate_id"] for trial in trials} == {"training-42", "training-100"}
    assert len(protocol["trials"]) == 3


def test_frozen_input_verification_rejects_changed_topology(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "source_identity", lambda revision: revision)
    root = tmp_path / "study"
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    topology = inputs / "topology.npz"
    topology.write_bytes(b"original topology fixture")
    revision = "a" * 40
    prepare.write_json(
        inputs / "manifest.json",
        {
            "source_revision": revision,
            "input_sha256": {"inputs/topology.npz": prepare.sha256(topology)},
        },
    )
    prepare.verify(root, revision)
    topology.write_bytes(b"changed topology fixture")
    with pytest.raises(ValueError, match="Changed or invalid input"):
        prepare.verify(root, revision)
    with pytest.raises(ValueError, match="different source revision"):
        prepare.verify(root, "b" * 40)


def test_stock_environment_and_reserved_bank_are_preserved():
    recipe = yaml.safe_load((STUDY / "recipe.yaml").read_text())
    train = recipe["train"]
    assert train["teams"] == "blue"
    assert train["variant"] == recipe["eval"]["variant"] == "cc4_stock"
    assert not any(train.get(key) for key in ("opponents", "initial_weights", "variant_overrides", "mission_bank"))
    assert train["topology_generation"]["count"] == 100
    assert train["topology_generation"]["seed_start"] == 0
    assert recipe["eval"]["topology_generation"]["seed_start"] == 200
    assert recipe["eval"]["topology_generation"]["count"] == 4
    assert "op_zone_servers" not in recipe["eval"]["topology_generation"]


@pytest.fixture
def validation(monkeypatch):
    monkeypatch.syspath_prepend(str(STUDY))
    spec = importlib.util.spec_from_file_location("lstm_fsm_validate", STUDY / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tensors():
    import numpy as np

    names = [f"blue/params/{role}_trunk/ScannedRNN_0/OptimizedLSTMCell_0/hf/kernel" for role in ("actor", "critic")]
    return {name: np.zeros((2, 2), dtype=np.float32) for name in names}


def test_acceptance_requires_both_recurrent_trunks_to_change(validation, tensors):
    updated = {name: value + 1 for name, value in tensors.items()}
    result = validation.compare_parameters(tensors, updated)
    assert result["changed_tensor_count"] == 2
    critic = next(name for name in tensors if "critic_trunk" in name)
    updated[critic] = tensors[critic].copy()
    with pytest.raises(ValueError, match="No changed critic LSTM"):
        validation.compare_parameters(tensors, updated)


def test_acceptance_rejects_nonfinite_weights(validation, tensors):
    import numpy as np

    updated = {name: value + 1 for name, value in tensors.items()}
    next(iter(updated.values()))[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite checkpoint tensor"):
        validation.compare_parameters(tensors, updated)


def test_acceptance_rejects_changed_parameter_contract(validation, tensors):
    updated = {name: value + 1 for name, value in tensors.items()}
    name = next(iter(updated))
    updated[name] = updated[name][:1]
    with pytest.raises(ValueError, match="tensor contract changed"):
        validation.compare_parameters(tensors, updated)


def test_cpu_gate_is_idempotent_and_refuses_changed_acceptance(tmp_path, monkeypatch, validation):
    from jaxborg.evaluation import capture_store

    directory = tmp_path / "evaluations/smoke-seed42/smoke"
    directory.mkdir(parents=True)
    target = tmp_path / "runs/smoke-seed42/production-ready.json"
    target.parent.mkdir(parents=True)
    acceptance = {"attempt_id": "training-42", "source_revision": "a" * 40, "manifest_sha256": "b" * 64}
    plan = {
        "training_attempt": "training-42",
        "source_revision": "a" * 40,
        "backend": "cpu",
        "horizon": 500,
        "stage": "smoke",
    }
    selection = {"0:fsm": "eval-initial", "144000:fsm": "eval-final"}
    summaries = [
        {"checkpoint_steps": steps, "opponent": "fsm", "evaluation_id": identity, "episodes": 1, "mean_return": -1.0}
        for steps, identity in ((0, "eval-initial"), (144000, "eval-final"))
    ]
    for name, value in (("plan", plan), ("selection", selection), ("summary", summaries)):
        (directory / f"{name}.json").write_text(json.dumps(value))
    records = [
        {"id": identity, "kind": "evaluation", "status": "succeeded", "budgets": [{"completed": 1}]}
        for identity in selection.values()
    ]
    monkeypatch.setattr(capture_store, "load_records", lambda root: records)
    writes = []
    original_writer = validation.write_json

    def track_write(path, value):
        writes.append(path)
        original_writer(path, value)

    monkeypatch.setattr(validation, "write_json", track_write)
    first = validation.validate_cpu_smoke(tmp_path, acceptance)
    before = target.read_bytes()
    second = validation.validate_cpu_smoke(tmp_path, acceptance)
    assert first == second
    assert writes == [target]
    assert target.read_bytes() == before
    with pytest.raises(ValueError, match="different production acceptance"):
        validation.validate_cpu_smoke(tmp_path, {**acceptance, "manifest_sha256": "c" * 64})
    assert target.read_bytes() == before
