import hashlib
import json
from types import SimpleNamespace

import pytest

from jaxborg.evaluation import scripted_capture as recording


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
