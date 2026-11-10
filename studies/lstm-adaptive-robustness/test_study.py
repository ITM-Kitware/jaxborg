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
    full = [
        trial
        for trial in protocol["trials"]
        if trial["id"].startswith("br-") and prepare.trial_settings(trial["id"])["red_seed"] in (17, 83)
    ]
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


@pytest.mark.parametrize("updates", [208, 416, 520, 624])
def test_extended_horizon_preserves_primary_and_all_crossed_cells(tmp_path, updates):
    from jaxborg.experiment_records.core import validate_records

    template = prepare.read_json(HERE / "protocol.json")
    original = copy.deepcopy(template)
    protocol = prepare.materialized_protocol(template, updates)
    validate_records([protocol])
    assert template == original
    assert protocol["trials"] == template["trials"]
    values = protocol["extensions"]["lstm_adaptive_robustness"]["values"]
    assert values["primary_updates"] == 208
    assert values["primary_environment_steps"] == 9984000
    assert values["production_updates"] == values["rollout_updates"] == updates
    assert values["completed_environment_steps"] == values["requested_environment_steps"] == updates * 48000
    assert values["secondary_environment_steps"] == (updates * 48000 if updates > 208 else None)
    base = yaml.safe_load((HERE / "recipe.yaml").read_text())
    checkpoints = {"blue42-primary": {"model": str(tmp_path / "blue.safetensors")}}
    full = prepare.recipe_for_trial(base, "br-blue42-primary-red17", checkpoints, [], [], updates)
    smoke = prepare.recipe_for_trial(base, "smoke-blue42-primary-red17", checkpoints, [], [], updates)
    timing = prepare.recipe_for_trial(base, "timing-blue42-primary-red17", checkpoints, [], [], updates)
    assert full["train"]["total_timesteps"] == updates * 48000
    assert smoke["train"]["total_timesteps"] == 4 * 48000
    assert timing["train"]["total_timesteps"] == 10 * 48000


def test_undeclared_production_horizon_rejected():
    with pytest.raises(ValueError, match="Production horizon"):
        prepare.materialized_protocol(prepare.read_json(HERE / "protocol.json"), 209)
    with pytest.raises(ValueError, match="Production horizon"):
        prepare.trial_settings("br-blue42-primary-red17", 209)


@pytest.mark.parametrize("production_updates", [208, 416, 520, 624])
def test_optional_replication_stays_at_primary_budget_and_covers_paired_ages(tmp_path, production_updates):
    protocol = prepare.materialized_protocol(prepare.read_json(HERE / "protocol.json"), production_updates)
    supplementary = protocol["extensions"]["lstm_adaptive_robustness"]["values"]["optional_replication"]
    assert supplementary["primary_only"] is True
    assert supplementary["updates"] == 208
    assert supplementary["environment_steps"] == 9984000
    declared = {trial["id"] for trial in protocol["trials"]}
    assert set(supplementary["trials"]) <= declared
    assert len(supplementary["trials"]) == 4
    assert set(supplementary["lanes"]["a"] + supplementary["lanes"]["b"]) == set(supplementary["trials"])
    for index, blue in enumerate((42, 100)):
        same_wave = [
            prepare.trial_settings(supplementary["lanes"][lane][index], production_updates) for lane in ("a", "b")
        ]
        assert {item["blue_seed"] for item in same_wave} == {blue}
        assert {item["age"] for item in same_wave} == {"primary", "final"}
    base = yaml.safe_load((HERE / "recipe.yaml").read_text())
    checkpoints = {
        f"blue{blue}-{age}": {"model": str(tmp_path / f"blue{blue}-{age}.safetensors")}
        for blue in (42, 100)
        for age in ("primary", "final")
    }
    for trial in supplementary["trials"]:
        assert prepare.trial_settings(trial, production_updates)["updates"] == 208
        recipe = prepare.recipe_for_trial(base, trial, checkpoints, [], [], production_updates)
        assert recipe["train"]["total_timesteps"] == 9984000


@pytest.fixture
def fresh_compilation_records(tmp_path):
    from jaxborg.evaluation.resume_state import save_training_state

    root = tmp_path.resolve()
    evidence = root / "evidence"
    evidence.mkdir()
    indexed = {}
    for label, trial, updates in (("smoke", validate.SMOKE, 4), ("timing", validate.TIMING, 10)):
        config = {
            "trainable_teams": ["red"],
            "variant": {"name": "cc4_stock"},
            "seed_roles": {"initialization_root": 18},
            "topology": {"sha256": ["fixture"]},
            "opponents": {"blue": {"model_sha256": "frozen"}},
            "initialization": {"kind": "fresh"},
            "resume": None,
            "team_configs": {
                team: {"ANNEAL_LR": False, "LR": 0.0003, "TOTAL_TIMESTEPS": updates * 48000} for team in ("blue", "red")
            },
            "resume_contract": {"runtime_environment": {"JAX_PLATFORMS": "cpu"}},
        }
        owner = {
            "id": label,
            "status": "succeeded",
            "execution_kind": "training",
            "trial_id": trial,
            "facts": {"resolved_config": {"value": config}, "source_revision": {"value": "1" * 40}},
            "budgets": [
                {"unit": unit, "completed": value}
                for unit, value in {
                    "environment_steps": updates * 48000,
                    "joint_rollout_updates": updates,
                    "blue_policy_updates": 0,
                    "red_policy_updates": updates,
                }.items()
            ],
        }
        indexed[label] = owner
        selected = {"environment_steps": 96000}
        for role in ("model", "sidecar", "training_state"):
            identity = f"{label}-{role}"
            path = evidence / identity
            if role == "training_state":
                save_training_state(
                    path,
                    state={"weight": np.array([1, 2], dtype=np.float32), "rng": np.array([17, 18], dtype=np.uint32)},
                    metadata={"environment_steps": 96000, "producer_attempt_id": label},
                )
            else:
                path.write_bytes(role.encode())
            indexed[identity] = {
                "id": identity,
                "kind": "artifact",
                "producer_id": label,
                "uri": path.name,
                "sha256": prepare.sha256(path),
            }
            selected[role] = identity
        owner["facts"]["checkpoint_96000"] = {"value": selected}
        receipt = {"attempt_id": label, "evidence_root": str(evidence)}
        prepare.write_json(root / "runs" / trial / "receipt.json", receipt)
        prepare.write_json(
            root / "runs" / trial / "execution.json", {"returncode": 0, "interrupted": False, "receipt": receipt}
        )
    smoke = {"attempt_id": "smoke", "source_revision": "1" * 40}
    return root, indexed, smoke


def test_fresh_compilation_requires_all_numeric_state_leaves_to_match(fresh_compilation_records):
    from jaxborg.evaluation.resume_state import save_training_state

    root, indexed, smoke = fresh_compilation_records
    result = validate.validate_fresh_compilation(root, indexed, smoke)
    assert result["passed"] is True
    assert result["smoke_state"]["sha256"] != result["timing_state"]["sha256"]
    assert result["smoke_state"]["state_sha256"] == result["timing_state"]["state_sha256"]
    changed = root / "evidence/changed-timing-state"
    save_training_state(
        changed,
        state={"weight": np.array([1, 2], dtype=np.float32), "rng": np.array([17, 19], dtype=np.uint32)},
        metadata={"environment_steps": 96000, "producer_attempt_id": "timing"},
    )
    artifact = indexed["timing-training_state"]
    artifact.update(uri=changed.name, sha256=prepare.sha256(changed))
    with pytest.raises(ValueError, match="scientific numeric state"):
        validate.validate_fresh_compilation(root, indexed, smoke)


@pytest.mark.parametrize("change", ["annealing", "learning_rate", "source", "unfinished"])
def test_fresh_compilation_rejects_noncomparable_or_incomplete_runs(fresh_compilation_records, change):
    root, indexed, smoke = fresh_compilation_records
    timing = indexed["timing"]
    config = timing["facts"]["resolved_config"]["value"]
    if change == "annealing":
        config["team_configs"]["red"]["ANNEAL_LR"] = True
    elif change == "learning_rate":
        config["team_configs"]["red"]["LR"] = 0.01
    elif change == "source":
        timing["facts"]["source_revision"]["value"] = "2" * 40
    else:
        timing["status"] = "running"
    with pytest.raises(ValueError):
        validate.validate_fresh_compilation(root, indexed, smoke)
