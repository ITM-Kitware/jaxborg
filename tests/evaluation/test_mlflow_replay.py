import json
from pathlib import Path

import pytest
import torch
import yaml
from mlflow import MlflowClient

from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.evaluation.capture import run_evaluation
from jaxborg.evaluation.replay import rebuild_blue_return
from jaxborg.policies import make_torch_policy


@pytest.mark.parametrize("training_simulator", ["cyborg", None])
def test_saved_blue_evaluation_survives_producer_relocation(tmp_path, monkeypatch, training_simulator):
    uri = f"sqlite:///{tmp_path}/tracking.db"
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment("recorded-blue", artifact_location=(tmp_path / "artifacts").as_uri())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "recorded-blue")
    model = tmp_path / "model_tiny.pt"
    policy = make_torch_policy("shared", obs_dim=BLUE_OBS_SIZE, action_dim=242, hidden_dim=8, hidden_layers=1)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    torch.save(policy.state_dict(), model)
    sidecar = tmp_path / "recipe_tiny.yaml"
    recipe = {"arch": {"name": "shared", "hidden_dim": 8, "hidden_layers": 1}}
    if training_simulator:
        recipe["run"] = {"backend": training_simulator, "checkpoint_format": "pytorch_state_dict"}
    sidecar.write_text(yaml.safe_dump(recipe))
    protocol = Path(__file__).resolve().parents[2] / "docs/examples/blue-evaluation-protocol.json"
    store = tmp_path / "original"
    run = run_evaluation(
        root=store,
        protocol=json.loads(protocol.read_text()),
        trial_id="blue-1",
        model=model,
        seeds=[42, 43],
        horizon=6,
        policy_seed=700,
        allow_dirty=True,
    )
    tags = client.get_run(run.tracking.run_id).data.tags
    assert tags["jaxborg.discovery.eval_env"] == "cyborg"
    assert tags["jaxborg.discovery.checkpoint_format"] == "pytorch_state_dict"
    assert tags.get("jaxborg.discovery.trained_backend") == training_simulator
    expected = [json.loads(p.read_text())["reward"] for p in (run.directory / "episodes").glob("*.json")]
    assert len(expected) == 2
    store.rename(tmp_path / "moved")
    model.unlink()
    sidecar.unlink()
    result = rebuild_blue_return(client, run.tracking.run_id, tmp_path / "retrieved")
    assert result["evaluation_id"] == run.evaluation_id
    assert result["episodes"] == 2
    assert result["blue_return"] == sum(expected) / 2
