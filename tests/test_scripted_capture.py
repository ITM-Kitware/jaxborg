import hashlib
import json
from types import SimpleNamespace

import pytest

from jaxborg.evaluation import scripted_capture as recording
from jaxborg.evaluation.capture_store import load_records


def test_completed_opponent_survives_later_failure(tmp_path, monkeypatch):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"policy")
    topology = tmp_path / "topology.json"
    topology.write_text("{}")
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "source-files.json").write_text("{}")
    records = [
        {
            "id": "checkpoint",
            "kind": "artifact",
            "producer_id": "training",
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        },
        {"id": "training", "execution_kind": "training"},
    ]
    monkeypatch.setattr(recording, "load_records", lambda root: records)
    monkeypatch.setattr(
        recording,
        "collect_provenance",
        lambda **kw: SimpleNamespace(
            configuration={"command": ["evaluation"]}, facts={}, source_files={}, hashes={}, verify_sources=lambda: None
        ),
    )
    monkeypatch.setattr(recording, "destination_configured", lambda: False)
    monkeypatch.setattr(
        recording,
        "topology_evidence",
        lambda paths, hashes, **kw: {"sha256": hashes, "metadata": [{"generator": "retained"}]},
    )
    metric_calls = []
    capture = SimpleNamespace(
        tracking=SimpleNamespace(log_metrics=lambda values: metric_calls.append(values)),
        directory=tmp_path,
        evaluation_id="evaluation",
        inputs={
            "blue checkpoint": model,
            "topology 0": topology,
            "source recipe": topology,
            "effective recipe": topology,
        },
    )
    configurations = []

    def start(*args, **kwargs):
        configurations.append(kwargs)
        return capture

    monkeypatch.setattr(recording, "start_evaluation", start)
    episodes, artifacts, failures = [], [], []
    monkeypatch.setattr(recording, "record_episode", lambda *a, **kw: episodes.append(kw))
    monkeypatch.setattr(recording, "add_artifact", lambda c, path, **kw: artifacts.append(json.loads(path.read_text())))
    monkeypatch.setattr(recording, "fail_evaluation", lambda c, error: failures.append(error))
    from jaxborg.evaluation import jax_scripted_red

    def execute(model, *, on_row, **kwargs):
        assert kwargs["seeds"] == (1, 2)
        on_row(
            {
                "eval_red": "fsm",
                "per_episode_seeds": [1, 2],
                "per_episode": [3.0, 4.0],
                "mean_reward": 3.5,
                "std_reward": 0.5,
                "n_episodes": 2,
                "cia_summary": {"n": 2, **{key: {"mean": 0.75, "std": 0.1} for key in ("c", "i", "a")}},
            }
        )
        raise RuntimeError("later opponent failed")

    monkeypatch.setattr(jax_scripted_red, "evaluate_jax_scripted_reds", execute)
    context = dict(
        root=str(tmp_path),
        checkpoint="checkpoint",
        training_attempt_id="training",
        revision="a" * 40,
        trial_id="trial",
        source_recipe=str(topology),
        effective_recipe=str(topology),
    )
    with pytest.raises(RuntimeError, match="later opponent"):
        recording.run_scripted(
            model, context=context, topology_paths=[topology], reds=["fsm", "cia_c"], seeds="1-2", episodes_per_seed=1
        )
    assert [episode["reward"] for episode in episodes] == [3.0, 4.0]
    assert artifacts[0]["evaluation_id"] == "evaluation"
    assert artifacts[0]["training_checkpoint_artifact_id"] == "checkpoint"
    assert len(failures) == 1
    assert configurations[0]["facts"]["backend"]["value"] == "jax"
    assert configurations[0]["facts"]["topology"]["value"]["sha256"] == [
        hashlib.sha256(topology.read_bytes()).hexdigest()
    ]
    assert configurations[0]["facts"]["topology"]["value"]["metadata"] == [{"generator": "retained"}]
    assert metric_calls[0]["eval.scripted_red.fsm.blue.mean_reward"] == 3.5
    assert metric_calls[0]["eval.scripted_red.fsm.blue.episodes"] == 2.0
    assert metric_calls[0]["eval.scripted_red.fsm.blue.cia.c.mean"] == 0.75
    assert metric_calls[0]["eval.scripted_red.fsm.blue.cia.i.std"] == 0.1
    assert metric_calls[0]["eval.scripted_red.fsm.blue.cia.a.mean_plus_std"] == 0.85


@pytest.mark.parametrize("recorded", [False, True])
def test_cli_has_one_tracking_owner(tmp_path, monkeypatch, recorded):
    from jaxborg import checkpoint
    from jaxborg import recipe as recipes
    from jaxborg.evaluation import jax_scripted_red as evaluator

    variant = SimpleNamespace(resilience_roles=True)
    monkeypatch.setattr(checkpoint, "read_sidecar", lambda path: {})
    monkeypatch.setattr(evaluator, "_cia_recipe_configuration", lambda recipe: None)
    monkeypatch.setattr(recipes, "eval_variant", lambda recipe: variant)
    monkeypatch.setattr(recipes, "project_eval", lambda *a, **kw: {"TOPOLOGY_BANK": [tmp_path]})
    rows = [{"train_run_id": "training"}]
    monkeypatch.setattr(evaluator, "evaluate_jax_scripted_reds", lambda *a, **kw: rows)
    monkeypatch.setattr(recording, "run_scripted", lambda *a, **kw: rows)
    monkeypatch.setattr(evaluator, "write_results", lambda *a: tmp_path / "results.jsonl")
    attaches = []
    monkeypatch.setattr(evaluator, "attach_results_to_mlflow", lambda rows: attaches.append(rows))
    if recorded:
        monkeypatch.setenv("JAXBORG_SCRIPTED_CAPTURE", "{}")
    else:
        monkeypatch.delenv("JAXBORG_SCRIPTED_CAPTURE", raising=False)
    evaluator.main(["--model", "model.safetensors"])
    assert len(attaches) == (0 if recorded else 1)


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    model = tmp_path / "model_fixture.safetensors"
    model.write_bytes(b"external policy")
    sidecar = tmp_path / "recipe_fixture.yaml"
    sidecar.write_text("run:\n  train_run_id: historical-external-run\n")
    effective = tmp_path / "effective.yaml"
    effective.write_text("eval:\n  cia:\n    enabled: true\n")
    topology = tmp_path / "topology.npz"
    topology.write_bytes(b"fixed topology bytes")
    monkeypatch.setattr(
        recording,
        "collect_provenance",
        lambda **kw: SimpleNamespace(
            configuration={"source_revision": kw["expected_revision"]},
            facts={},
            source_files={},
            hashes={},
            verify_sources=lambda: None,
        ),
    )
    monkeypatch.setattr(recording, "destination_configured", lambda: False)
    monkeypatch.setattr(
        recording, "topology_evidence", lambda paths, hashes, **kw: {"sha256": hashes, "sampling": kw["sampling"]}
    )
    protocol = {
        "schema_version": 1,
        "kind": "protocol",
        "id": "frozen:protocol:1",
        "study_id": "frozen",
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": ["resolved_config"], "evaluation": ["resolved_config"]},
        "trials": [{"id": "trial", "condition_id": "external", "replicate_id": "42", "matched_unit_id": None}],
        "metric_definitions": [
            {
                "name": "blue_return",
                "unit": "score",
                "direction": "higher",
                "aggregation": "arithmetic mean",
                "sampling_unit": "episode conditional on fixed policy",
                "uncertainty": "none",
            }
        ],
    }
    return model, {
        "root": tmp_path / "evidence",
        "protocol": protocol,
        "trial_id": "trial",
        "expected_revision": "a" * 40,
        "source_recipe": sidecar,
        "effective_recipe": effective,
        "base_variant": {"resilience_roles": True, "num_steps": 500},
        "topology_paths": [topology],
        "reds": ["fsm", "cia_c"],
        "seeds": "17-18",
        "episodes_per_seed": 1,
    }


@pytest.mark.parametrize("fail_later", [False, True])
def test_frozen_capture_retains_external_inputs_and_completed_opponents(frozen_inputs, monkeypatch, fail_later):
    from jaxborg.checkpoint import sidecar_path
    from jaxborg.evaluation import jax_scripted_red

    model, settings = frozen_inputs
    checked = []

    def execute(retained_model, *, on_row, **kwargs):
        assert retained_model != model
        assert retained_model.read_bytes() == model.read_bytes()
        assert sidecar_path(retained_model).read_bytes() == sidecar_path(model).read_bytes()
        assert kwargs["topology_paths"][0] != settings["topology_paths"][0]
        assert kwargs["topology_paths"][0].read_bytes() == settings["topology_paths"][0].read_bytes()
        assert kwargs["recipe"] == {"eval": {"cia": {"enabled": True}}}
        rows = []
        for opponent in kwargs["reds"]:
            if fail_later and rows:
                raise RuntimeError("second opponent failed")
            row = {
                "eval_red": opponent,
                "per_episode_seeds": [17, 18],
                "per_episode": [-2.0, -4.0],
                "per_episode_cia": [{"c": 0.8}, {"c": 0.7}],
                "per_episode_topology_fingerprints": ["canonical-topology", "canonical-topology"],
                "episode_role_map_ids": ["fixed-roles", "fixed-roles"],
                "mean_reward": -3.0,
                "std_reward": 2**0.5,
                "n_episodes": 2,
                "cia_summary": {},
            }
            on_row(row)
            rows.append(row)
        return rows

    monkeypatch.setattr(jax_scripted_red, "evaluate_jax_scripted_reds", execute)
    settings["condition_check"] = checked.append
    if fail_later:
        with pytest.raises(RuntimeError, match="second opponent failed"):
            recording.run_frozen_scripted(model, **settings)
    else:
        capture, rows = recording.run_frozen_scripted(model, **settings)
        assert all(row["evaluation_id"] == capture.evaluation_id for row in rows)
    assert checked[0]["topology"]["sampling"] == "exhaustive"
    records = load_records(settings["root"])
    attempts = [r for r in records if r["kind"] == "attempt"]
    assert len(attempts) == 1 and attempts[0]["execution_kind"] == "evaluation"
    evaluation = next(r for r in records if r["kind"] == "evaluation")
    assert evaluation["status"] == ("failed" if fail_later else "succeeded")
    assert evaluation["budgets"][0] == {"unit": "episodes", "requested": 4, "completed": 2 if fail_later else 4}
    checkpoint = next(r for r in records if r["kind"] == "artifact" and r["role"] == "blue checkpoint")
    assert checkpoint["producer_id"] is None
    artifacts = [r for r in records if r["kind"] == "artifact" and r["role"] == "scripted opponent episode results"]
    assert len(artifacts) == (1 if fail_later else 2)
    for artifact in artifacts:
        row = json.loads((settings["root"] / artifact["uri"]).read_text())
        assert row["checkpoint_artifact_id"] == checkpoint["id"]
        assert "training_attempt_id" not in row and "training_checkpoint_artifact_id" not in row
        assert row["per_episode_topology_fingerprints"] == ["canonical-topology", "canonical-topology"]
        assert row["per_episode_cia"] == [{"c": 0.8}, {"c": 0.7}]


def test_frozen_condition_guard_rejects_before_start(frozen_inputs):
    model, settings = frozen_inputs

    def reject(configuration):
        raise ValueError("wrong declared scientific cell")

    with pytest.raises(ValueError, match="wrong declared scientific cell"):
        recording.run_frozen_scripted(model, condition_check=reject, **settings)
    assert not settings["root"].exists()


def test_frozen_recipe_mismatch_rejects_before_start(frozen_inputs):
    model, settings = frozen_inputs
    with pytest.raises(ValueError, match="differs from retained effective recipe"):
        recording.run_frozen_scripted(model, recipe={"eval": {}}, **settings)
    assert not settings["root"].exists()
