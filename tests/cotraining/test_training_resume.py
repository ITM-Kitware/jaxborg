import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from jaxborg.training_resume import (
    LATEST_CHECKPOINT,
    preserve_resume_metrics,
    resume_request,
    save_training_checkpoint,
    select_training_resume,
)


@pytest.fixture
def recipe():
    return {
        "meta": {"name": "resume_test"},
        "algorithm": "ippo",
        "core": {"lr": 0.01},
        "arch": {"name": "shared"},
        "train": {"teams": "both", "total_timesteps": 100},
        "jax": {"num_envs": 2},
    }


def save(path, recipe, **kwargs):
    settings = dict(recipe=recipe, seed=42, completed_updates=2, rollout_steps=10, state={"x": jnp.ones(2)})
    settings.update(kwargs)
    return save_training_checkpoint(path, **settings)


def select(path, recipe, **kwargs):
    settings = dict(seed=42, rollout_steps=10, total_updates=10, requested="latest")
    settings.update(kwargs)
    return select_training_resume(path, recipe, **settings)


@pytest.mark.parametrize(
    "value, expected", [(None, None), (False, None), (True, "latest"), ("latest", "latest"), ("a.msgpack", "a.msgpack")]
)
def test_yaml_resume_options(value, expected):
    recipe = {"train": {"resume": value}}
    assert resume_request(recipe) == expected
    assert resume_request(recipe, "override.msgpack") == "override.msgpack"


@pytest.mark.parametrize("value", [0, 1, [], {}, "", " "])
def test_invalid_yaml_resume_options(value):
    with pytest.raises(ValueError, match="train.resume"):
        resume_request({"train": {"resume": value}})


def test_restore_optimizer_and_schedule_matches_uninterrupted_updates(tmp_path, recipe):
    tx = optax.adam(optax.linear_schedule(0.01, 0.001, 10))
    template = TrainState.create(apply_fn=lambda: None, params={"w": jnp.array([1.0, 2.0])}, tx=tx)

    def update(state, rng):
        rng, key = jax.random.split(rng)
        return state.apply_gradients(grads={"w": jax.random.normal(key, (2,))}), rng

    state, rng = template, jax.random.PRNGKey(2)
    for _ in range(2):
        state, rng = update(state, rng)
    save(tmp_path / LATEST_CHECKPOINT, recipe, state=(state, rng))
    restored, restored_rng = select(tmp_path, recipe).restore((template, jax.random.PRNGKey(99)))
    for _ in range(3):
        state, rng = update(state, rng)
        restored, restored_rng = update(restored, restored_rng)
    for expected, actual in zip(jax.tree.leaves((state, rng)), jax.tree.leaves((restored, restored_rng))):
        np.testing.assert_array_equal(actual, expected)


def test_latest_and_explicit_checkpoint_are_self_contained(tmp_path, recipe):
    path = save(tmp_path / LATEST_CHECKPOINT, recipe)
    recipe["train"]["resume"] = "latest"
    recipe["eval"] = {"after_training": []}
    recipe["jax"]["checkpoint_every_updates"] = 1
    for requested in ("latest", str(path)):
        checkpoint = select(tmp_path, recipe, requested=requested)
        assert checkpoint.completed_updates == 2
        np.testing.assert_array_equal(checkpoint.restore({"x": jnp.zeros(2)})["x"], np.ones(2))


@pytest.mark.parametrize("change", ["seed", "recipe", "rollout", "budget", "architecture", "reward"])
def test_incompatible_resume_fails(tmp_path, recipe, change):
    save(tmp_path / LATEST_CHECKPOINT, recipe)
    kwargs = {}
    if change == "seed":
        kwargs["seed"] = 43
    elif change == "recipe":
        recipe["meta"]["name"] = "another"
    elif change == "rollout":
        kwargs["rollout_steps"] = 5
    elif change == "budget":
        recipe["train"]["total_timesteps"] = 10
    elif change == "architecture":
        recipe["arch"]["hidden_dim"] = 4
    else:
        recipe["core"]["norm_rewards"] = True
    with pytest.raises(ValueError):
        select(tmp_path, recipe, **kwargs)


def test_budget_extension_requires_constant_learning_rate(tmp_path, recipe):
    save(tmp_path / LATEST_CHECKPOINT, recipe)
    extended = copy.deepcopy(recipe)
    extended["train"]["total_timesteps"] = 200
    assert select(tmp_path, extended, total_updates=20).completed_updates == 2
    recipe["train"]["team_overrides"] = {"red": {"core": {"anneal_lr": True}}}
    save(tmp_path / LATEST_CHECKPOINT, recipe)
    assert select(tmp_path, recipe).completed_updates == 2
    recipe["train"]["total_timesteps"] = 200
    with pytest.raises(ValueError, match="anneal_lr"):
        select(tmp_path, recipe, total_updates=20)


def test_failed_atomic_write_leaves_previous_checkpoint_intact(tmp_path, recipe, monkeypatch):
    path = save(tmp_path / LATEST_CHECKPOINT, recipe)
    original = path.read_bytes()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr("jaxborg.training_resume.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        save(path, recipe, completed_updates=4)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    assert select(tmp_path, recipe).completed_updates == 2


def test_damaged_or_policy_only_progress_does_not_restart(tmp_path, recipe):
    assert select(tmp_path, recipe) is None
    (tmp_path / "metrics.jsonl").touch()
    assert select(tmp_path, recipe) is None
    path = tmp_path / LATEST_CHECKPOINT
    path.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="checkpoint"):
        select(tmp_path, recipe)
    path.unlink()
    (tmp_path / "checkpoint_20.safetensors").touch()
    with pytest.raises(ValueError, match="No full training checkpoint"):
        select(tmp_path, recipe)
    with pytest.raises(ValueError, match="policy-only"):
        select(tmp_path, recipe, requested=str(tmp_path / "checkpoint_20.safetensors"))
    (tmp_path / "checkpoint_20.safetensors").unlink()
    (tmp_path / "metrics.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="No full training checkpoint"):
        select(tmp_path, recipe)
    with pytest.raises(FileNotFoundError):
        select(tmp_path, recipe, requested=str(path))


@pytest.mark.parametrize("template", [{"y": jnp.ones(2)}, {"x": jnp.ones(3)}, {"x": jnp.ones(2, dtype=jnp.int32)}])
def test_incompatible_state_structure_shape_or_dtype_is_rejected(tmp_path, recipe, template):
    save(tmp_path / LATEST_CHECKPOINT, recipe)
    with pytest.raises(ValueError, match="Resume training state"):
        select(tmp_path, recipe).restore(template)


def test_resume_metrics_archives_original_and_removes_uncheckpointed_progress(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    original = "\n".join(json.dumps({"env_steps": step}) for step in [10, 20, 30]) + '\n{"env_steps":'
    metrics.write_text(original)
    preserve_resume_metrics(metrics, 20)
    assert [json.loads(line)["env_steps"] for line in metrics.read_text().splitlines()] == [10, 20]
    assert next(tmp_path.glob("metrics_before_resume_*.jsonl")).read_text() == original
