"""Prevent cached evidence from being reused in an unmatched credit ablation."""

import copy
import json

import pytest
import yaml

from scripts.experiments.blue_learning_mechanism import controlled_override


def control(tmp_path):
    configs = {team: {"GAE_LAMBDA": 0.95, "GAMMA": 0.99, "LR": 0.0003} for team in ("blue", "red")}
    protocol = {
        "source_steps": 9600000,
        "training_seed": 11001,
        "warm_updates": 20,
        "validation_episodes": 32,
        "seeds": {"validation_start": 6200000, "warm_rollout": 6100001},
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "FINISHED"}))
    (tmp_path / "effective-config.json").write_text(json.dumps(configs))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(protocol))
    return configs, {**protocol, "core_override": {"gae_lambda": 1.0}}


def test_controlled_credit_override_changes_only_lambda_and_rejects_unmatched_learning_rate(tmp_path):
    configs, protocol = control(tmp_path)
    before = copy.deepcopy(configs)
    recipe = {"core": {"gae_lambda": 0.95, "gamma": 0.99, "lr": 0.0003}}
    assert controlled_override(recipe, configs, protocol, tmp_path) == tmp_path
    for team in configs:
        assert configs[team] == {**before[team], "GAE_LAMBDA": 1.0}
    assert recipe["core"] == {"gae_lambda": 1.0, "gamma": 0.99, "lr": 0.0003}
    before["blue"]["LR"] = 0.0001
    with pytest.raises(ValueError, match="different settings"):
        controlled_override(recipe, before, protocol, tmp_path)


def test_credit_ablation_rejects_failed_control_and_nonfresh_resume(tmp_path):
    configs, protocol = control(tmp_path)
    with pytest.raises(ValueError, match="only the controlled"):
        controlled_override({}, configs, protocol, tmp_path, resume=True)
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "FAILED"}))
    with pytest.raises(ValueError, match="completed"):
        controlled_override({}, configs, protocol, tmp_path)
