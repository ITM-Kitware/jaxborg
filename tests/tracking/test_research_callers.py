"""Real research trainer publication/freezing and independent evaluator lineage."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import mlflow
import pytest
import yaml
from mlflow import MlflowClient

from jaxborg import tracking as t
from jaxborg.checkpoint import PolicyBundleEntry, load_jax_bundle, save_jax_bundle, write_sidecar
from jaxborg.policies import init_policy_params, policy_from_arch
from jaxborg.recipe import load
from jaxborg.research_tracking import parameter_hash
from scripts.train.algorithms import ippo_jax_joint as joint

REPO = Path(__file__).resolve().parents[2]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("JAXBORG_ALLOW_DIRTY", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAXBORG_SKIP_POST_TRAINING_EVAL", "1")
    monkeypatch.delenv("JAXBORG_EXPECTED_SHA", raising=False)
    monkeypatch.delenv("JAXBORG_MLFLOW_EXPERIMENT", raising=False)
    recipe = copy.deepcopy(load("cotraining/cotraining"))
    recipe["cage4_enhanced_obs"] = True
    recipe["arch"].update(hidden_dim=8, hidden_layers=1)
    recipe["train"].update(episode_length=2, total_timesteps=4)
    recipe["train"].pop("topology_generation")
    recipe["eval"] = dict(variant="cc4_stock", after_training=[])
    recipe["jax"].update(num_envs=1, num_minibatches=1, update_epochs=1, checkpoint_every_updates=1)
    recipe["mlflow"] = dict(checkpoint_eval=dict(every_steps=0))
    policies = {}
    for team, obs_dim, actions in [("blue", 450, 242), ("red", 706, 1106)]:
        network = policy_from_arch(recipe["arch"], action_dim=actions)
        weights = init_policy_params(network, jax.random.PRNGKey(3), obs_dim)
        policies[team] = PolicyBundleEntry(weights, team, obs_dim, actions, recipe["arch"])
    model = tmp_path / "source/model_test.safetensors"
    save_jax_bundle(model, policies)
    write_sidecar(model.with_name("recipe_test.yaml"), recipe, seed=42, total_steps=9600000, backend="jax")
    yield model, recipe
    if t._current:
        t._current.finish("FAILED")
    mlflow.end_run()


def runs(kind):
    t.configure()
    client = MlflowClient()
    owners = client.search_runs([e.experiment_id for e in client.search_experiments()], f"tags.`run.kind` = '{kind}'")
    return [t.read_manifest(owner.info.run_id)[0] for owner in owners]


@pytest.mark.parametrize("cancel", [False, True])
def test_frozen_enhanced_blue_and_changed_red_survive_publication_and_cancel(source, tmp_path, monkeypatch, cancel):
    model, recipe = source
    tiny = module_at("research_tiny_env", REPO / "tests/cotraining/test_jax_joint_trainer.py")
    env = tiny._TinyJointEnv(blue_obs_dim=450, red_obs_dim=706, blue_actions=242, red_actions=1106)
    monkeypatch.setattr(joint, "make_joint_jax_env", lambda *a, **kw: env)
    trainer = module_at("research_trainer", REPO / "scripts/train/algorithms/ippo_jax.py")
    recipe["train"].update(teams="red", opponents=dict(blue=dict(path=str(model))))
    path = tmp_path / "red.yaml"
    path.write_text(yaml.safe_dump(t.serializable(recipe)))
    real_make = trainer.make_joint_train

    def make(*args, **kwargs):
        env, obs, state, init, update = real_make(*args, **kwargs)
        count = 0

        def controlled(*args):
            nonlocal count
            count += 1
            if cancel and count == 2:
                raise KeyboardInterrupt("after first completed joint update")
            return update(*args)

        return env, obs, state, init, controlled

    monkeypatch.setattr(trainer, "make_joint_train", make)
    monkeypatch.setattr(sys, "argv", ["train", "--recipe", str(path), "--seed", "11001"])
    if cancel:
        with pytest.raises(KeyboardInterrupt):
            trainer.main()
    else:
        trainer.main()
    owner = runs("training")[0]
    assert owner["status"] == ("KILLED" if cancel else "FINISHED")
    assert owner["actual_steps"] == (2 if cancel else 4)
    assert owner["parameter_checks"]["blue"]["changed"] is False
    assert owner["parameter_checks"]["red"]["changed"] is True
    assert owner["policy_contract"]["blue"]["obs_dim"] == 450
    checkpoint = t.resolve_artifact(f"runs:/{owner['run_id']}/checkpoints/checkpoint_2.safetensors")
    saved = load_jax_bundle(checkpoint)
    assert parameter_hash(saved.policies["blue"].weights) == parameter_hash(
        load_jax_bundle(model).policies["blue"].weights
    )
    assert owner["inputs"][0]["sha256"] == t.file_hash(model)
    assert owner["inputs"][0]["retained_reference"].startswith("runs:/")
    assert saved.policies["blue"].trainable is False
    assert saved.policies["red"].trainable is True


def test_matchup_owns_both_inputs_and_only_reuses_validated_fingerprint(source, tmp_path, monkeypatch):
    model, recipe = source
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(t.serializable(recipe)))
    evaluator = module_at("research_evaluator", REPO / "scripts/eval/eval_matchup.py")
    calls = []

    def evaluate(blue, red, **kwargs):
        calls.append((blue, red, kwargs))
        return SimpleNamespace(
            blue_returns=[-2.0, -4.0],
            red_returns=[2.0, 4.0],
            episode_seeds=[1, 2],
            policies=dict(blue=dict(path=str(blue)), red=dict(path=str(red))),
            topology_paths=[],
            topology_sampling="generative",
            episode_topology_paths=[None, None],
        )

    monkeypatch.setattr(evaluator, "evaluate_matchup", evaluate)
    argv = [
        "eval",
        "--recipe",
        str(path),
        "--policy-backend",
        "jax",
        "--blue-path",
        str(model),
        "--red-path",
        str(model),
        "--episodes-per-seed",
        "1",
        "--seeds",
        "1,2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    first = evaluator.main()
    second = evaluator.main()
    assert first != second and len(calls) == 2
    monkeypatch.setattr(sys, "argv", argv + ["--reuse"])
    assert evaluator.main() == second and len(calls) == 2
    owner = t.read_manifest(second)[0]
    assert [x["role"] for x in owner["inputs"]] == ["blue policy", "red policy"]
    result = json.loads(t.resolve_artifact(f"runs:/{second}/evaluations/result.json").read_text())
    assert result["eval_id"] == second and result["blue_mean_return"] == -3
    monkeypatch.setattr(sys, "argv", argv + ["--reuse", "--supersedes-eval-run-id", first, "--bug-reference", "test"])
    corrected = evaluator.main()
    assert corrected not in (first, second) and len(calls) == 3
    assert t.read_manifest(corrected)[0]["supersedes_eval_run_id"] == first
