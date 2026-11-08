"""Recorded frozen-Blue response contracts, including exact continuation."""

import copy
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.checkpoint import PolicyBundleEntry, load_jax_policy, save_jax_bundle, write_sidecar
from jaxborg.evaluation.capture_store import load_records
from jaxborg.evaluation.resume_state import inspect_training_state
from jaxborg.learned_red import RED_OBS_SIZE, RED_POLICY_ACTION_DIM
from jaxborg.policies import init_policy_params, policy_from_arch
from jaxborg.recipe import VARIANTS, load
from scripts.train.algorithms import ippo_jax as trainer
from scripts.train.algorithms import ippo_jax_joint as joint

PROJECT = Path(__file__).resolve().parents[2]


@struct.dataclass
class _Simulation:
    time: jax.Array
    blue_pending_ticks: jax.Array
    red_pending_ticks: jax.Array
    red_agent_active: jax.Array


@struct.dataclass
class _Environment:
    state: _Simulation


class _TinyResponseEnv:
    blue_agents = ("blue_0",)
    red_agents = ("red_0",)

    def observation_space(self, agent):
        return SimpleNamespace(shape=(450 if agent == "blue_0" else RED_OBS_SIZE,))

    def _observations(self, state):
        value = 0.1 + state.state.time.astype(jnp.float32) / 10
        return {
            agent: jnp.full(self.observation_space(agent).shape, value, dtype=jnp.float32)
            for agent in self.blue_agents + self.red_agents
        }

    def reset(self, key):
        del key
        state = _Environment(
            _Simulation(
                time=jnp.array(0, jnp.int32),
                blue_pending_ticks=jnp.zeros(1, jnp.int32),
                red_pending_ticks=jnp.zeros(1, jnp.int32),
                red_agent_active=jnp.ones(1, jnp.bool_),
            )
        )
        return self._observations(state), state

    def get_avail_actions(self, state):
        del state
        return {
            "blue_0": jnp.ones(BLUE_ALLOW_TRAFFIC_END, jnp.bool_),
            "red_0": jnp.ones(RED_POLICY_ACTION_DIM, jnp.bool_),
        }

    def step(self, key, state, actions):
        del key
        time = state.state.time + 1
        done = time == 2
        state = state.replace(state=state.state.replace(time=jnp.where(done, 0, time)))
        blue = -1.0 - 0.01 * actions["red_0"].astype(jnp.float32)
        blue = blue + 0.001 * actions["blue_0"].astype(jnp.float32)
        zero = jnp.array(0.0, jnp.float32)
        return (
            self._observations(state),
            state,
            {"blue_0": blue, "red_0": -blue},
            {"blue_0": done, "red_0": done, "__all__": done},
            {
                "reward_ria": blue,
                "reward_lwf": zero,
                "reward_asf": zero,
                "action_cost": zero,
                "impact_count": zero,
                "green_lwf_count": zero,
                "green_asf_count": zero,
            },
        )


def _tiny_response_env(variant, *, topology_mode, training_mode, topology_path):
    assert variant.num_steps == 2
    assert topology_mode == "generative"
    assert training_mode
    assert topology_path
    return _TinyResponseEnv()


def _same_tree(left, right):
    left_leaves, left_def = jax.tree.flatten(left)
    right_leaves, right_def = jax.tree.flatten(right)
    return left_def == right_def and all(
        np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(left_leaves, right_leaves, strict=True)
    )


@pytest.fixture
def response(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(tmp_path / "tracking"))
    monkeypatch.setitem(VARIANTS, "cc4_stock", replace(VARIANTS["cc4_stock"], num_steps=2))
    monkeypatch.setattr(joint, "make_joint_jax_env", _tiny_response_env)
    recipe = load(str(PROJECT / "recipes/cotraining/br_red_vs_cotraining.yaml"))
    recipe["cage4_enhanced_obs"] = True
    recipe["arch"] = {"name": "shared", "hidden_dim": 4, "hidden_layers": 1, "activation": "tanh"}
    topology = tmp_path / "topology.npz"
    np.savez(topology, fixture=np.asarray([1]))
    recipe["train"] = {
        "teams": "red",
        "variant": "cc4_stock",
        "episode_length": 2,
        "total_timesteps": 8,
        "topology_bank": [str(topology)],
    }
    recipe["core"].update(anneal_lr=False, norm_rewards=True)
    recipe["jax"] = {
        "num_envs": 1,
        "num_minibatches": 1,
        "update_epochs": 1,
        "checkpoint_every_updates": 2,
    }
    recipe["eval"] = {"variant": "cc4_stock", "after_training": []}
    recipe["mlflow"] = {"checkpoint_eval": {"every_steps": 0}}
    blue_arch = {
        "name": "recurrent",
        "hidden_dim": 4,
        "hidden_layers": 1,
        "activation": "tanh",
        "cell": "lstm",
        "trunk": "separate",
    }
    blue_network = policy_from_arch(blue_arch, action_dim=BLUE_ALLOW_TRAFFIC_END)
    blue_params = init_policy_params(blue_network, jax.random.PRNGKey(99), 450)

    def save_blue(name, *, shift=0, obs_dim=450):
        path = tmp_path / f"model_{name}.safetensors"
        weights = jax.tree.map(lambda x: x + shift, blue_params)
        save_jax_bundle(
            path,
            {
                "blue": PolicyBundleEntry(
                    weights=weights,
                    team="blue",
                    obs_dim=obs_dim,
                    action_dim=BLUE_ALLOW_TRAFFIC_END,
                    arch=blue_arch,
                    trainable=True,
                    source={"kind": "fixture", "checkpoint": name},
                )
            },
            provenance={"total_steps": 49_968_000 if not shift else 124_992_000},
        )
        source_recipe = copy.deepcopy(recipe)
        source_recipe["arch"] = blue_arch
        source_recipe["train"]["teams"] = "blue"
        write_sidecar(
            path.with_name(f"recipe_{name}.yaml"),
            source_recipe,
            seed=42,
            total_steps=49_968_000 if not shift else 124_992_000,
            backend="jax",
        )
        return path, weights

    blue_path, _ = save_blue("early")
    recipe["train"]["opponents"] = {"blue": {"path": str(blue_path)}}
    recording = {
        "root": tmp_path / "evidence",
        "protocol": {
            "schema_version": 1,
            "kind": "protocol",
            "id": "frozen-response",
            "study_id": "frozen-response",
            "revision": "1",
            "selection_rule": "explicit",
            "required_facts": {"attempt": [], "evaluation": []},
            "metric_definitions": [],
            "trials": [{"id": "trial", "condition_id": "response", "replicate_id": "17", "matched_unit_id": None}],
        },
        "trial_id": "trial",
        "expected_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "allow_dirty": True,
        "entrypoint": Path(__file__),
        "supplied_recipe": copy.deepcopy(recipe),
    }
    return recipe, recording, save_blue


def _run(tmp_path, recipe, recording, *, tag="response", resume_state=None):
    return trainer._run_joint_training(
        SimpleNamespace(seed=17, resume_state=resume_state, checkpoint_update=[2]),
        recipe,
        tag,
        tmp_path / tag,
        env_factory=None,
        recording=recording,
    )


def _attempt(root, identity):
    return next(item for item in load_records(root) if item["id"] == identity)


def _checkpoint_path(root, attempt, steps, key):
    artifact_id = attempt["facts"][f"checkpoint_{steps}"]["value"][key]
    artifact = _attempt(root, artifact_id)
    return root / artifact["uri"]


def test_frozen_response_consumes_retained_inputs_and_only_updates_red(tmp_path, monkeypatch, response):
    recipe, recording, _ = response
    original = Path(recipe["train"]["opponents"]["blue"]["path"])
    expected_blue = load_jax_policy(original, team="blue").weights
    actual_loader = trainer.load_jax_policy
    consumed = []

    def load_retained(path, **kwargs):
        path = Path(path)
        assert path.is_relative_to(recording["root"])
        consumed.append(path)
        original.write_bytes(b"changed after retention")
        original.with_name("recipe_early.yaml").write_text("changed after retention")
        return actual_loader(path, **kwargs)

    monkeypatch.setattr(trainer, "load_jax_policy", load_retained)
    capture = _run(tmp_path, recipe, recording)
    assert consumed
    initial = tmp_path / "response/initial.safetensors"
    final = tmp_path / "response/model_response.safetensors"
    assert _same_tree(expected_blue, load_jax_policy(final, team="blue").weights)
    assert not _same_tree(
        load_jax_policy(initial, team="red").weights,
        load_jax_policy(final, team="red").weights,
    )
    attempt = _attempt(recording["root"], capture.attempt_id)
    assert attempt["status"] == "succeeded"
    assert attempt["facts"]["trainable_teams"]["value"] == ["red"]
    assert {item["unit"]: item["completed"] for item in attempt["budgets"]} == {
        "environment_steps": 8,
        "joint_rollout_updates": 4,
        "blue_policy_updates": 0,
        "red_policy_updates": 4,
    }
    config = json.loads((capture.directory / "configuration.json").read_text())
    assert config["opponents"]["blue"]["trainable"] is False
    contract = attempt["facts"]["policy_contract"]["value"]
    assert contract["blue"]["obs_dim"] == 450
    assert contract["blue"]["arch"]["cell"] == "lstm"
    assert contract["red"]["arch"]["name"] == "shared"
    rows = [json.loads(line) for line in (tmp_path / "response/metrics.jsonl").read_text().splitlines()]
    assert all(row["team.blue.update_count"] == 0 for row in rows)
    assert all(row["team.red.updated"] == 1 for row in rows)


def test_red_initialization_is_identical_across_frozen_blue_ages(tmp_path, response):
    recipe, recording, save_blue = response
    _run(tmp_path, recipe, recording, tag="early")
    late, _ = save_blue("late", shift=0.01)
    recipe["train"]["opponents"]["blue"]["path"] = str(late)
    _run(tmp_path, recipe, recording, tag="late")
    early_model = tmp_path / "early/initial.safetensors"
    late_model = tmp_path / "late/initial.safetensors"
    assert _same_tree(
        load_jax_policy(early_model, team="red").weights,
        load_jax_policy(late_model, team="red").weights,
    )
    assert not _same_tree(
        load_jax_policy(early_model, team="blue").weights,
        load_jax_policy(late_model, team="blue").weights,
    )


def test_frozen_response_resume_reproduces_every_numeric_leaf(tmp_path, response):
    recipe, recording, _ = response
    uninterrupted = _run(tmp_path, recipe, recording, tag="uninterrupted")
    parent = _attempt(recording["root"], uninterrupted.attempt_id)
    intermediate = _checkpoint_path(recording["root"], parent, 4, "training_state")
    final = _checkpoint_path(recording["root"], parent, 8, "training_state")
    resumed = _run(tmp_path, recipe, recording, tag="resumed", resume_state=intermediate)
    child = _attempt(recording["root"], resumed.attempt_id)
    resumed_final = _checkpoint_path(recording["root"], child, 8, "training_state")
    assert inspect_training_state(final)["state_sha256"] == inspect_training_state(resumed_final)["state_sha256"]
    assert {item["unit"]: item["completed"] for item in child["budgets"]} == {
        "environment_steps": 4,
        "joint_rollout_updates": 2,
        "blue_policy_updates": 0,
        "red_policy_updates": 2,
    }


def test_resume_rejects_a_different_frozen_defender(tmp_path, response):
    recipe, recording, save_blue = response
    capture = _run(tmp_path, recipe, recording)
    attempt = _attempt(recording["root"], capture.attempt_id)
    intermediate = _checkpoint_path(recording["root"], attempt, 4, "training_state")
    late, _ = save_blue("late", shift=0.01)
    recipe["train"]["opponents"]["blue"]["path"] = str(late)
    with pytest.raises(ValueError, match="contract differs"):
        _run(tmp_path, recipe, recording, tag="retargeted", resume_state=intermediate)


@pytest.mark.parametrize("failure", ["initialization", "dimensions"])
def test_frozen_response_failure_retains_attempt_and_opponent(tmp_path, monkeypatch, response, failure):
    recipe, recording, save_blue = response

    def unexpected(*args, **kwargs):
        raise RuntimeError("deliberate initialization failure")

    monkeypatch.setattr(trainer, "make_joint_train", unexpected)
    if failure == "dimensions":
        incompatible, _ = save_blue("incompatible", obs_dim=449)
        recipe["train"]["opponents"]["blue"]["path"] = str(incompatible)
        expected = ValueError
    else:
        expected = RuntimeError
    with pytest.raises(expected) as caught:
        _run(tmp_path, recipe, recording)
    capture = caught.value._evaluation_capture
    attempt = _attempt(recording["root"], capture.attempt_id)
    assert attempt["status"] == "failed"
    assert all(budget["completed"] == 0 for budget in attempt["budgets"])
    assert capture.inputs["frozen blue policy"].is_file()
    assert capture.inputs["frozen blue sidecar"].is_file()


@pytest.mark.parametrize("fault", ["frozen_parameters", "nonfinite_metric"])
def test_response_rejects_corrupted_rollout_and_preserves_failure(tmp_path, monkeypatch, response, fault):
    recipe, recording, _ = response
    actual_builder = trainer.make_joint_train

    def corrupt_builder(*args, **kwargs):
        env, obs, env_state, init, collect = actual_builder(*args, **kwargs)

        def corrupt_rollout(*args, **kwargs):
            result = list(collect(*args, **kwargs))
            if fault == "frozen_parameters":
                states = dict(result[0])
                states["blue"] = states["blue"].replace(
                    params=jax.tree.map(lambda value: value + 1, states["blue"].params)
                )
                result[0] = states
            else:
                result[5] = {
                    **result[5],
                    "red": {**result[5]["red"], "total_loss": jnp.asarray(float("nan"))},
                }
            return tuple(result)

        return env, obs, env_state, init, corrupt_rollout

    monkeypatch.setattr(trainer, "make_joint_train", corrupt_builder)
    expected = "Frozen Blue parameters or optimizer changed" if fault == "frozen_parameters" else "Nonfinite"
    with pytest.raises(ValueError, match=expected) as caught:
        _run(tmp_path, recipe, recording)
    capture = caught.value._evaluation_capture
    attempt = _attempt(recording["root"], capture.attempt_id)
    assert attempt["status"] == "failed"
    assert "checkpoint_4" not in attempt["facts"]
    assert "checkpoint_8" not in attempt["facts"]
