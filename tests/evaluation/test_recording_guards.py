"""Reject unsupported execution settings before producing scientific work."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaxborg.evaluation import joint_capture
from jaxborg.evaluation.capture_store import add_artifact, load_records, start_training
from jaxborg.evaluation.training_capture import begin_training
from scripts.train.algorithms import ippo_jax as trainer


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("cross_play", {}),
        ("play_priors", {}),
        ("checkpoint_scripted_reds", {}),
        ("cross_seed_play", {}),
        ("scripted_red", {"after_training": True}),
    ],
)
def test_implicit_automatic_evaluation_is_rejected_before_attempt(tmp_path, setting, value):
    with pytest.raises(ValueError, match="separate recorded evaluations"):
        begin_training(
            root=tmp_path / "evidence",
            protocol={},
            trial_id="trial",
            expected_revision="a" * 40,
            args=SimpleNamespace(seed=42),
            recipe={"train": {"teams": "both"}, "eval": {setting: value}},
            configs={"blue": {"TRAIN_VARIANT": SimpleNamespace(red_reward="zero_sum")}},
        )
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize("case", ["sidecar_only", "unindexed", "wrong_adjacent"])
def test_owned_policy_selection_requires_exact_adjacent_sidecar(tmp_path, monkeypatch, case):
    root = tmp_path / "evidence"
    protocol = {
        "schema_version": 1,
        "kind": "protocol",
        "id": "p",
        "study_id": "fixture",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": [], "evaluation": []},
        "metric_definitions": [],
        "trials": [{"id": "trial", "condition_id": "control", "replicate_id": "1", "matched_unit_id": None}],
    }
    training = start_training(
        root,
        protocol=protocol,
        trial_id="trial",
        requested_steps=1,
        requested_updates=1,
        resolved_config={},
        inputs={},
        source_files={},
    )
    model = tmp_path / "model_fixture.safetensors"
    model.write_bytes(b"fixture checkpoint")
    sidecar = tmp_path / "recipe_fixture.yaml"
    sidecar.write_text("{}\n")
    model_id = add_artifact(training, model, role="joint policy checkpoint")
    if case == "unindexed":
        artifact = next(record for record in load_records(root) if record["id"] == model_id)
        (root / artifact["uri"]).with_name(sidecar.name).write_bytes(sidecar.read_bytes())
        retained = {"blue checkpoint": model_id}
    else:
        sidecar_id = add_artifact(training, sidecar, role="checkpoint recipe sidecar")
        if case == "sidecar_only":
            retained = {"blue sidecar": sidecar_id}
        else:
            other_id = add_artifact(training, sidecar, role="checkpoint recipe sidecar", name="other.yaml")
            retained = {"blue checkpoint": model_id, "blue sidecar": other_id}
    topology = tmp_path / "topology.npz"
    topology.write_bytes(b"unused preflight topology")
    monkeypatch.setattr(joint_capture, "collect_provenance", lambda **kwargs: None)
    monkeypatch.setattr(joint_capture, "builder_identity", lambda *args: {})
    with pytest.raises(ValueError, match="retained checkpoint|indexed adjacent sidecar"):
        joint_capture.run_matchup(
            model,
            model,
            root=root,
            protocol=protocol,
            trial_id="trial",
            expected_revision="a" * 40,
            variant=SimpleNamespace(red_reward="zero_sum"),
            seeds=[1],
            topology_path=topology,
            input_artifacts=retained,
        )
    assert not any(record["kind"] == "evaluation" for record in load_records(root))


@pytest.mark.parametrize("failed", [False, True])
def test_training_cli_prints_attempt_receipt_for_retry(tmp_path, monkeypatch, capsys, failed):
    project = Path(__file__).resolve().parents[2]
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n")
    capture = SimpleNamespace(root=tmp_path / "evidence", attempt_id="fixture:attempt:1")

    def execute(*args, **kwargs):
        if failed:
            error = RuntimeError("deliberate execution failure")
            error._evaluation_capture = capture
            raise error
        return capture

    monkeypatch.setattr(trainer, "_run_joint_training", execute)
    monkeypatch.setattr(trainer, "EXP_DIR", tmp_path / "runtime")
    monkeypatch.setattr(
        trainer.sys,
        "argv",
        [
            "ippo_jax.py",
            "--recipe",
            str(project / "recipes/cotraining/cotraining.yaml"),
            "--evidence-root",
            str(capture.root),
            "--protocol",
            str(protocol),
            "--trial",
            "trial",
            "--expected-revision",
            "a" * 40,
        ],
    )
    if failed:
        with pytest.raises(RuntimeError, match="deliberate execution failure"):
            trainer.main()
    else:
        trainer.main()
    output = capsys.readouterr()
    receipt = json.loads((output.err if failed else output.out).splitlines()[-1])
    assert receipt == {"evidence_root": str(capture.root), "attempt_id": capture.attempt_id}
    assert not (tmp_path / "runtime").exists()
