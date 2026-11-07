"""Blue/FSM recording lifecycle and policy-contract regressions."""

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.blue_observation_contract import recipe_blue_obs_size
from jaxborg.checkpoint import load_jax_policy, read_sidecar
from jaxborg.evaluation.capture_store import load_records
from jaxborg.evaluation.training_capture import begin_training
from jaxborg.recipe import load, project_jax, team_recipe
from scripts.train.algorithms import ippo_jax as trainer

PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def single_training(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(tmp_path / "tracking"))
    recipe = load(str(PROJECT / "recipes/cotraining/separate/cotraining_lstm_separate.yaml"))
    topology = tmp_path / "topology.npz"
    np.savez(topology, fixture=np.asarray([1]))
    recipe["train"].update(
        teams="blue",
        episode_length=500,
        total_timesteps=1001,
        topology_bank=[str(topology)],
    )
    recipe["train"].pop("topology_generation", None)
    recipe["arch"].update(hidden_dim=4, trunk="shared")
    recipe["train"]["team_overrides"] = {"blue": {"arch": {"trunk": "separate"}}}
    recipe["jax"] = {"num_envs": 1, "num_minibatches": 1, "update_epochs": 1, "checkpoint_every_updates": 1}
    recipe["eval"] = {"variant": "cc4_stock", "policy_backend": "jax"}
    recipe["mlflow"] = {"checkpoint_eval": {"every_steps": 0}}
    recording = {
        "root": tmp_path / "evidence",
        "protocol": {
            "schema_version": 1,
            "kind": "protocol",
            "id": "fsm-training",
            "study_id": "fsm-training",
            "revision": "1",
            "selection_rule": "explicit",
            "required_facts": {"attempt": [], "evaluation": []},
            "metric_definitions": [],
            "trials": [{"id": "trial", "condition_id": "fsm", "replicate_id": "42", "matched_unit_id": None}],
        },
        "trial_id": "trial",
        "expected_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "allow_dirty": True,
        "entrypoint": Path(__file__),
        "supplied_recipe": copy.deepcopy(recipe),
    }
    return recipe, recording


@pytest.mark.parametrize("failure", [None, "initialization", "second_update", "nonfinite", "tracking"])
def test_single_training_records_owned_state_budget_and_failures(tmp_path, monkeypatch, single_training, failure):
    recipe, recording = single_training
    calls = []
    observed = []

    def make_train(config, network):
        active = next(r for r in load_records(recording["root"]) if r["kind"] == "attempt")
        observed.append(active["id"])
        assert active["status"] == "running"
        assert active["budgets"][0]["completed"] == 0
        # The effective team override must reach the actual constructed policy.
        assert network.trunk == "separate"
        assert all(str(recording["root"]) in path for path in config["TOPOLOGY_BANK"])
        if failure == "initialization":
            raise RuntimeError("initialization failed")
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"] * config["NUM_STEPS"])
        env = SimpleNamespace(
            agents=["blue_agent_0"],
            observation_space=lambda agent: SimpleNamespace(shape=(recipe_blue_obs_size(recipe),)),
            action_space=lambda agent: SimpleNamespace(n=BLUE_ALLOW_TRAFFIC_END),
        )
        state = SimpleNamespace(params={"weight": np.zeros(1, np.float32)}, opt_state={"count": 0}, step=0)

        def collect(state, env_state, obs, rng, normalizer):
            if failure == "second_update" and calls:
                raise RuntimeError("second update failed")
            calls.append(len(calls))
            new_state = SimpleNamespace(
                params={"weight": np.full(1, len(calls), np.float32)}, opt_state={"count": len(calls)}, step=len(calls)
            )
            metrics = dict.fromkeys(
                [
                    "actor_loss",
                    "critic_loss",
                    "entropy",
                    "total_loss",
                    "approx_kl",
                    "clip_frac",
                    "explained_var",
                    "grad_norm",
                    "pre_clip_grad_norm",
                    "raw_rollout_return",
                    "mean_rollout_return",
                ],
                1.0,
            )
            metrics.update(
                parameters_finite=True,
                optimizer_state_finite=True,
                reward_normalizer_finite=True,
                rollout_boundary_time=0,
            )
            if failure == "nonfinite":
                metrics["total_loss"] = float("nan")
            return new_state, env_state, obs, rng, normalizer, metrics

        return (
            env,
            {"blue_agent_0": np.zeros((1, recipe_blue_obs_size(recipe)))},
            {"time": 0},
            lambda key: state,
            collect,
        )

    monkeypatch.setattr(trainer, "make_train", make_train)
    if failure == "tracking":
        from jaxborg.evaluation import tracking

        def unavailable():
            raise OSError("tracking unavailable")

        monkeypatch.setattr(tracking, "tracking_client", unavailable)

    def unexpected(*args, **kwargs):
        pytest.fail("Recorded single-team training must not open a fluent tracking run")

    monkeypatch.setattr(trainer, "start_run", unexpected)
    output = tmp_path / "outputs"
    if failure not in (None, "tracking"):
        with pytest.raises((RuntimeError, ValueError)) as error:
            trainer._run_single_training(SimpleNamespace(seed=42), recipe, "test", output, recording=recording)
        assert error.value._evaluation_capture.attempt_id == observed[0]
    else:
        capture = trainer._run_single_training(SimpleNamespace(seed=42), recipe, "test", output, recording=recording)
        assert capture.attempt_id == observed[0]

    records = load_records(recording["root"])
    attempt = next(record for record in records if record["kind"] == "attempt")
    expected_updates = 2 if failure in (None, "tracking") else 1 if failure == "second_update" else 0
    assert attempt["status"] == ("succeeded" if expected_updates == 2 else "failed")
    budget = {item["unit"]: item["completed"] for item in attempt["budgets"]}
    assert budget["environment_steps"] == expected_updates * 500
    assert budget["joint_rollout_updates"] == expected_updates
    if expected_updates:
        assert budget["blue_policy_updates"] == expected_updates
        assert budget["red_policy_updates"] == 0
    configuration = json.loads(next(recording["root"].glob("attempts/*/configuration.json")).read_text())
    assert configuration["planned_environment_steps"] == 1000
    assert configuration["trainable_teams"] == ["blue"]
    assert configuration["opponents"]["red"] == {"kind": "scripted", "agent": "finite_state", "trainable": False}
    assert configuration["procedure"].endswith("._run_single_training")
    assert configuration["builder"]["function"] == "make_jax_env"
    if failure != "initialization":
        assert "initial_state" in attempt["facts"]
        initial = load_jax_policy(output / "initial.safetensors", team="blue")
        sidecar = read_sidecar(output / "initial.safetensors")
        assert initial.arch["trunk"] == team_recipe(sidecar, "blue")["arch"]["trunk"] == "separate"
        assert initial.obs_dim == recipe_blue_obs_size(sidecar) == recipe_blue_obs_size(recipe)
    if expected_updates == 2:
        final = load_jax_policy(output / "model_test.safetensors", team="blue")
        assert final.arch == initial.arch
        assert final.weights["weight"].item() == 2
        assert initial.weights["weight"].item() == 0
        assert "checkpoint_500" in attempt["facts"] and "checkpoint_1000" in attempt["facts"]
    assert trainer.mlflow.active_run() is None


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("NUM_STEPS", 5, "episode horizon"),
        ("UPDATE_EVERY", 2, "every rollout"),
        ("ADAPTIVE_UPDATES", {"enabled": True}, "every rollout"),
        ("GRAD_CLIP_MODE", "per_leaf", "global gradient"),
    ],
)
def test_recorded_single_team_rejects_ignored_or_misaligned_settings(tmp_path, single_training, key, value, message):
    recipe, recording = single_training
    config = project_jax(recipe, team="blue")
    config[key] = value
    with pytest.raises(ValueError, match=message):
        begin_training(args=SimpleNamespace(seed=42), recipe=recipe, configs={"blue": config}, **recording)
    assert not recording["root"].exists()


@pytest.mark.parametrize("failed", [False, True])
def test_single_team_cli_retains_receipt_and_selects_single_path(
    tmp_path, monkeypatch, single_training, capsys, failed
):
    recipe, _ = single_training
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n")
    capture = SimpleNamespace(root=tmp_path / "evidence", attempt_id="fsm:attempt:1")

    def execute(args, actual_recipe, tag, save_dir, *, recording):
        assert actual_recipe["train"]["teams"] == "blue"
        if failed:
            error = RuntimeError("deliberate execution failure")
            error._evaluation_capture = capture
            raise error
        return capture

    monkeypatch.setattr(trainer, "load_recipe", lambda path: recipe)
    monkeypatch.setattr(trainer, "_run_single_training", execute)
    monkeypatch.setattr(
        trainer.sys,
        "argv",
        [
            "ippo_jax.py",
            "--recipe",
            "fixture",
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
    assert json.loads((output.err if failed else output.out).splitlines()[-1]) == {
        "evidence_root": str(capture.root),
        "attempt_id": capture.attempt_id,
    }
