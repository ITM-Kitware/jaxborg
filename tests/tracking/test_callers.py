"""Exercise real caller adapters while substituting expensive episode rollouts."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
import torch
from mlflow import MlflowClient

from jaxborg import tracking as t
from jaxborg.checkpoint import read_sidecar, write_sidecar
from jaxborg.policies import make_torch_policy
from jaxborg.recipe import load

REPO = Path(__file__).resolve().parents[2]


def script(name):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def consumer(tmp_path, monkeypatch):
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("JAXBORG_ALLOW_DIRTY", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("JAXBORG_EXPECTED_SHA", raising=False)
    monkeypatch.delenv("JAXBORG_MLFLOW_EXPERIMENT", raising=False)
    monkeypatch.syspath_prepend(str(REPO / "scripts/eval"))
    torch.set_num_threads(1)
    recipe = load("default")
    model = tmp_path / "legacy/model_test.pt"
    model.parent.mkdir()
    torch.save(make_torch_policy("shared", obs_dim=210, action_dim=242).state_dict(), model)
    write_sidecar(model.with_name("recipe_test.yaml"), recipe, seed=42, total_steps=1, backend="cyborg")
    yield model
    if t._current:
        t._current.finish("FAILED")
    mlflow.end_run()


def manifests(kind):
    t.configure()
    client = MlflowClient()
    runs = client.search_runs([e.experiment_id for e in client.search_experiments()], f"tags.`run.kind` = '{kind}'")
    return [t.read_manifest(r.info.run_id)[0] for r in runs]


def test_default_evaluations_are_new_and_reuse_returns_original(consumer, tmp_path, monkeypatch):
    from jaxborg.evaluation import cyborg_runner

    calls = []

    def rollout(*args, **kwargs):
        calls.append(kwargs)
        return [-2.0, -1.0], [42, 43]

    monkeypatch.setattr(cyborg_runner, "evaluate_on_cyborg", rollout)
    module = script("eval/eval_recipe.py")
    argv = ["eval_recipe.py", "--model", str(consumer), "--episodes", "2", "--seeds", "42", "--workers", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    first = module.main()
    second = module.main()
    assert first != second and len(calls) == 2
    monkeypatch.setattr(sys, "argv", argv + ["--reuse", "--output", str(tmp_path / "export.jsonl")])
    assert module.main() == second
    assert len(calls) == 2 and len(manifests("evaluation")) == 2
    assert json.loads((tmp_path / "export.jsonl").read_text())["eval_id"] == second
    old = t.read_manifest(first)[0]
    monkeypatch.setattr(sys, "argv", argv + ["--supersedes-eval-run-id", first, "--bug-reference", "example"])
    corrected = module.main()
    assert corrected not in (first, second) and len(calls) == 3
    assert t.read_manifest(corrected)[0]["supersedes_eval_run_id"] == first
    assert t.read_manifest(first)[0] == old


@pytest.mark.parametrize("filename", ["export_trajectory.py", "generate_cynex_trajectories.py"])
def test_cynex_exports_keep_format_and_run_ownership(consumer, tmp_path, monkeypatch, filename):
    module = script(f"eval/{filename}")
    payload = {
        "format_version": "2.0",
        "challenge": "cc4",
        "seed": 42,
        "network_topology": {},
        "agent_actions": {},
        "step_states": [{"cumulative_reward": {"blue_agent_0": -2}}],
    }
    runner = "run_episode_policy" if filename == "export_trajectory.py" else "run_episode_torch"
    monkeypatch.setattr(module, runner, lambda *args, **kwargs: payload)
    model_flag = "--model" if filename == "export_trajectory.py" else "--model-pt"
    exports = tmp_path / "cynex"
    monkeypatch.setattr(
        sys,
        "argv",
        [filename, model_flag, str(consumer), "--tag", "test", "--num-episodes", "1", "--output-dir", str(exports)],
    )
    module.main()
    module.main()
    assert len(manifests("trajectory")) == 2
    for m in manifests("trajectory"):
        name = "trajectories/cc4-test-seed42-E0.json"
        canonical = t.resolve_artifact(f"runs:/{m['run_id']}/{name}")
        assert json.loads(canonical.read_text()) == payload
        assert t.file_hash(canonical) == t.file_hash(exports / "cc4-test-seed42-E0.json")
        assert m["exports"][0]["reference"] == f"runs:/{m['run_id']}/{name}"


def test_jsonl_trajectory_and_separate_scoring_lineage(consumer, tmp_path, monkeypatch):
    module = script("eval/cc4_trajectory_eval.py")

    def episode(*args):
        args[-1].write_text('{"type":"footer","total_reward":-2,"steps":500}\n')
        return -2, 500

    monkeypatch.setattr(module, "rollout_episode", episode)
    monkeypatch.setattr(module, "make_cyborg_env", lambda *args, **kwargs: None)
    module.evaluate(str(consumer), 1, 42, False, str(tmp_path / "exports"), "test")
    trajectory = manifests("trajectory")[0]
    scorer = script("eval/score_trajectories.py")
    score = SimpleNamespace(
        steps=500,
        total_reward=-2,
        C_mean=1,
        I_mean=1,
        A_mean=1,
        R_mean=1,
        C_min=1,
        I_min=1,
        A_min=1,
        R_min=1,
        impact_counts={},
    )
    monkeypatch.setattr(scorer, "get_cia_scorer", lambda cfg: lambda path: score)
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys, "argv", ["score", f"runs:/{trajectory['run_id']}/trajectories", "--summary-json", str(summary)]
    )
    scorer.main()
    comparison = manifests("comparison")[0]
    assert comparison["inputs"][0]["source_run_id"] == trajectory["run_id"]
    assert comparison["inputs"][0]["sha256"] == trajectory["outputs"]["trajectories/test_seed42.jsonl"]["sha256"]
    assert json.loads(summary.read_text())["reward_mean"] == -2
    assert t.resolve_artifact(f"runs:/{comparison['run_id']}/evaluations/per_episode.json").exists()


@pytest.mark.parametrize("backend", ["cyborg", "jax"])
def test_baseline_callers_own_evaluations(consumer, monkeypatch, backend):
    module = script(f"eval/baselines_{backend}.py")
    monkeypatch.setattr(module, "run_sleep_episode", lambda *args: -1)
    if backend == "jax":
        monkeypatch.setattr(module, "make_jax_env", lambda *args, **kwargs: None)
        module.evaluate("sleep", 42, 2)
    else:
        monkeypatch.setattr(module, "make_env", lambda *args: None)
        module.evaluate("sleep", 42, 2)
    m = manifests("evaluation")[0]
    assert m["effective_config"]["episode_seeds"] == [42, 43]
    assert m["effective_config"]["variant"]["num_steps"] == 500
    assert json.loads(t.resolve_artifact(f"runs:/{m['run_id']}/evaluations/result.json").read_text())[
        "per_episode"
    ] == [-1, -1]


def test_cyborg_periodic_checkpoints_load_and_overrides_are_captured(consumer, tmp_path, monkeypatch):
    module = script("train/algorithms/ippo_cyborg.py")
    recipe = load("default")
    recipe["cleanrl"].update(rollout_length=2, num_minibatches=1, num_epochs=1)
    recipe["train"]["buffer_size"] = 2
    recipe_path = tmp_path / "tiny.yaml"
    import yaml

    recipe_path.write_text(yaml.safe_dump(t.serializable(recipe)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--recipe",
            str(recipe_path),
            "--num-envs",
            "1",
            "--total-timesteps",
            "4",
            "--num-rollouts-per-update",
            "1",
            "--checkpoint-every-updates",
            "1",
        ],
    )
    module.main()
    m = manifests("training")[0]
    assert m["recipe"]["train"]["total_timesteps"] == 4
    assert m["recipe"]["cleanrl"]["num_envs"] == 1
    assert m["effective_config"]["num_rollouts_per_update"] == 1
    from jaxborg.evaluation.cyborg_runner import load_torch_policy

    for steps in (2, 4):
        path = t.resolve_artifact(f"runs:/{m['run_id']}/checkpoints/checkpoint_{steps}.pt")
        agent, sidecar = load_torch_policy(path)
        assert sidecar["run"]["total_steps"] == steps
        assert sidecar["run"]["train_run_id"] == m["run_id"]
        assert read_sidecar(path)["train"]["total_timesteps"] == 4
        assert agent is not None


@pytest.mark.parametrize("cancel", [False, True])
def test_jax_periodic_checkpoints_have_loadable_sidecars(consumer, tmp_path, monkeypatch, cancel):
    import jax
    import jax.numpy as jnp
    import yaml

    from jaxborg.evaluation.jax_runner import load_jax_checkpoint

    module = script("train/algorithms/ippo_jax.py")
    recipe = load("default")
    recipe["train"].update(episode_length=2, total_timesteps=4)
    recipe["jax"].update(num_envs=1, num_minibatches=1, checkpoint_every_updates=1)
    recipe_path = tmp_path / "tiny_jax.yaml"
    recipe_path.write_text(yaml.safe_dump(t.serializable(recipe)))
    calls = []

    def train(config, network):
        config.update(NUM_UPDATES=2, NUM_ACTORS=5, MINIBATCH_SIZE=10)

        def init(key):
            return SimpleNamespace(params=network.init(key, jnp.zeros((210,))))

        def collect(state, env, obs, rng, norm):
            calls.append(1)
            if cancel and len(calls) == 2:
                raise KeyboardInterrupt("simulated cancellation after completed update")
            metric = {
                k: jnp.array(0.0)
                for k in (
                    "actor_loss",
                    "critic_loss",
                    "entropy",
                    "total_loss",
                    "approx_kl",
                    "clip_frac",
                    "explained_var",
                    "raw_rollout_return",
                    "grad_norm",
                    "pre_clip_grad_norm",
                    "mean_rollout_return",
                )
            }
            return state, env, obs, rng, norm, metric

        return None, None, None, init, collect

    monkeypatch.setattr(module, "make_train", train)
    monkeypatch.setattr(
        sys, "argv", ["train", "--recipe", str(recipe_path), "--total-timesteps", "4", "--num-envs", "1"]
    )
    if cancel:
        with pytest.raises(KeyboardInterrupt):
            module.main()
    else:
        module.main()
    m = manifests("training")[0]
    assert m["status"] == ("KILLED" if cancel else "FINISHED")
    assert m["effective_config"]["NUM_UPDATES"] == 2
    assert m["recipe"]["train"]["total_timesteps"] == 4
    path = t.resolve_artifact(f"runs:/{m['run_id']}/checkpoints/checkpoint_2.safetensors")
    policy, params, sidecar = load_jax_checkpoint(path)
    pi, value = policy.apply(params, jnp.zeros((210,)))
    from jaxborg.checkpoint import load_jax_params

    assert pi.logits.shape == (load_jax_params(path)[1],)
    assert jax.device_get(value).size == 1
    assert sidecar["run"]["train_run_id"] == m["run_id"]
