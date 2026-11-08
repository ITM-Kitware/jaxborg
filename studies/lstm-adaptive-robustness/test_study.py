"""Scientific-input, pairing and validation failures for the adaptive study."""

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prepare = module("adaptive_prepare_test", HERE / "prepare.py")
validate = module("adaptive_validate_test", HERE / "validate.py")


def test_protocol_crossed_design_equal_budget_and_counterbalanced_order():
    from jaxborg.experiment_records.core import validate_records

    protocol = prepare.read_json(HERE / "protocol.json")
    validate_records([protocol])
    values = protocol["extensions"]["lstm_adaptive_robustness"]["values"]
    full = [trial for trial in protocol["trials"] if trial["id"].startswith("br-")]
    assert len(full) == 8
    assert (
        values["completed_environment_steps"]
        == values["rollout_environment_steps"] * values["rollout_updates"]
        == 9984000
    )
    for blue in (42, 100):
        for red in (17, 83):
            pair = [trial for trial in full if trial["matched_unit_id"] == f"blue{blue}-red{red}"]
            assert {trial["id"] for trial in pair} == {
                f"br-blue{blue}-primary-red{red}",
                f"br-blue{blue}-final-red{red}",
            }
            lane = [trial for trial in values["lanes"][str(red)] if f"blue{blue}-" in trial]
            assert [prepare.trial_settings(trial)["age"] for trial in lane] == (
                ["primary", "final"] if red == 17 else ["final", "primary"]
            )
    assert len(values["evaluation_topology_seeds"]) * len(values["evaluation_episode_roots"]) == 64


def test_paired_recipes_change_only_frozen_checkpoint_and_leave_template_untouched(tmp_path):
    base = yaml.safe_load((HERE / "recipe.yaml").read_text())
    original = copy.deepcopy(base)
    checkpoints = {f"blue42-{age}": {"model": str(tmp_path / f"{age}.safetensors")} for age in ("primary", "final")}
    training = [tmp_path / f"train-{index}.npz" for index in range(100)]
    evaluation = [tmp_path / f"eval-{index}.npz" for index in range(4)]
    early = prepare.recipe_for_trial(base, "br-blue42-primary-red17", checkpoints, training, evaluation)
    late = prepare.recipe_for_trial(base, "br-blue42-final-red17", checkpoints, training, evaluation)
    assert base == original
    assert early["train"].pop("opponents") != late["train"].pop("opponents")
    assert early == late
    smoke = prepare.recipe_for_trial(base, "smoke-blue42-primary-red17", checkpoints, training, evaluation)
    resumed = prepare.recipe_for_trial(base, "smoke-resume-blue42-primary-red17", checkpoints, training, evaluation)
    assert smoke == resumed
    assert smoke["train"]["total_timesteps"] == 192000


@pytest.fixture
def owned_baseline(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    owner = {
        "id": "attempt",
        "kind": "attempt",
        "execution_kind": "training",
        "status": "succeeded",
        "trial_id": "fsm-seed42",
        "budgets": [{"unit": "environment_steps", "completed": 124992000}],
        "facts": {"source_revision": {"value": prepare.BASELINE_REVISION}},
    }
    records = [owner]
    for steps in (0, 49968000):
        checkpoint = {"environment_steps": steps}
        for role in ("model", "sidecar"):
            path = evidence / f"{role}-{steps}"
            path.write_bytes(f"{role}:{steps}".encode())
            identity = f"{role}-{steps}"
            records.append(
                {
                    "id": identity,
                    "kind": "artifact",
                    "producer_id": "attempt",
                    "uri": path.name,
                    "sha256": prepare.sha256(path),
                }
            )
            checkpoint[role] = identity
        owner["facts"]["initial_checkpoint" if steps == 0 else f"checkpoint_{steps}"] = {"value": checkpoint}
    return evidence, records


@pytest.mark.parametrize("tamper", ["bytes", "owner", "source", "milestone"])
def test_baseline_copy_rejects_wrong_source_owner_steps_or_changed_bytes(owned_baseline, tamper):
    evidence, records = owned_baseline
    selected = prepare.verified_owned_checkpoint(records, evidence, "attempt", "fsm-seed42", "primary")
    assert selected["steps"] == 49968000
    if tamper == "bytes":
        Path(selected["model"]).write_bytes(b"changed")
    elif tamper == "owner":
        next(record for record in records if record["id"] == selected["model_artifact"])["producer_id"] = "other"
    elif tamper == "source":
        records[0]["facts"]["source_revision"]["value"] = "0" * 40
    else:
        records[0]["facts"]["checkpoint_49968000"]["value"]["environment_steps"] = 48000000
    with pytest.raises(ValueError):
        prepare.verified_owned_checkpoint(records, evidence, "attempt", "fsm-seed42", "primary")


def test_parameter_check_rejects_frozen_drift_and_no_learning():
    original = {"blue": {"blue/params/x": np.array([1, 2], dtype=np.float32)}}
    initial = {**copy.deepcopy(original), "red": {"red/params/x": np.array([3, 4], dtype=np.float32)}}
    final = copy.deepcopy(initial)
    final["red"]["red/params/x"][0] += 1
    result = validate.compare_policy_parameters(original, initial, final)
    assert result["initial_red_parameters_sha256"] != result["final_red_parameters_sha256"]
    final["blue"]["blue/params/x"][0] += 1
    with pytest.raises(ValueError, match="Frozen Blue"):
        validate.compare_policy_parameters(original, initial, final)
    with pytest.raises(ValueError, match="did not update"):
        validate.compare_policy_parameters(original, initial, initial)


def test_timing_excludes_compile_cost_and_rejects_missing_or_nonfinite_updates():
    rows = [
        {"update_idx": index + 1, "wall_time_s": duration} for index, duration in enumerate([1000, 1002, 1004, 1006])
    ]
    result = validate.timing_summary(rows, 4)
    assert result["warm_median_seconds"] == 2
    assert result["estimated_208_update_seconds"] == 1414
    with pytest.raises(ValueError, match="Missing"):
        validate.timing_summary(rows[:3], 4)
    rows[-1]["loss"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        validate.timing_summary(rows, 4)


def test_frozen_input_manifest_detects_mutation_and_path_escape(tmp_path):
    path = tmp_path / "inputs/topology.npz"
    path.parent.mkdir()
    path.write_bytes(b"original")
    manifest = {"input_sha256": {"inputs/topology.npz": prepare.sha256(path)}}
    prepare.BASELINE_PREPARE.verify_input_files(tmp_path, manifest)
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="Changed or invalid"):
        prepare.BASELINE_PREPARE.verify_input_files(tmp_path, manifest)
    manifest["input_sha256"] = {"../escape": "0" * 64}
    with pytest.raises(ValueError, match="Changed or invalid"):
        prepare.BASELINE_PREPARE.verify_input_files(tmp_path, manifest)
