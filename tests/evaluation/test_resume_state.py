"""Full dynamic state survives a numeric checkpoint and the next compiled update."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.evaluation.resume_state import (
    inspect_training_state,
    load_training_state,
    read_training_state_metadata,
    save_training_state,
)
from jaxborg.policies import policy_from_arch
from scripts.train.algorithms import ippo_jax


def _example():
    return {"step": 0, "weights": jnp.arange(6, dtype=jnp.float32).reshape(2, 3), "rng": jax.random.PRNGKey(43)}


def test_numeric_checkpoint_is_immutable_and_metadata_does_not_change_state_identity(tmp_path):
    path = tmp_path / "state.npz"
    state = _example()
    metadata = {"contract": {"source": "a" * 40}, "completed_updates": 2}
    descriptor = save_training_state(path, state=state, metadata=metadata)
    assert read_training_state_metadata(path, expected_sha256=descriptor["sha256"]) == metadata
    restored, actual_metadata = load_training_state(
        path,
        template=_example(),
        expected_metadata={"contract": metadata["contract"]},
        expected_sha256=descriptor["sha256"],
    )
    assert actual_metadata == metadata
    assert bool(restored["step"].weak_type)
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_training_state(path, state=state, metadata=metadata)
    assert path.read_bytes() == before
    second = save_training_state(tmp_path / "other.npz", state=restored, metadata={**metadata, "producer": "different"})
    assert descriptor["state_sha256"] == second["state_sha256"]
    assert descriptor["sha256"] != second["sha256"]


@pytest.mark.parametrize("change", ["shape", "dtype", "path", "metadata", "hash"])
def test_resume_rejects_incompatible_templates_and_contracts(tmp_path, change):
    path = tmp_path / "state.npz"
    metadata = {"contract": {"source": "a" * 40}}
    save_training_state(path, state=_example(), metadata=metadata)
    template = _example()
    kwargs = {"expected_metadata": metadata}
    if change == "shape":
        template["weights"] = jnp.zeros(7, dtype=jnp.float32)
    elif change == "dtype":
        template["weights"] = np.zeros((2, 3), dtype=np.float64)
    elif change == "path":
        template["other"] = template.pop("weights")
    elif change == "metadata":
        kwargs["expected_metadata"] = {"contract": {"source": "b" * 40}}
    elif change == "hash":
        kwargs["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        load_training_state(path, template=template, **kwargs)


def test_nonfinite_and_changed_array_bytes_are_rejected(tmp_path):
    path = tmp_path / "state.npz"
    bad = _example()
    bad["weights"] = jnp.asarray([float("nan")])
    with pytest.raises(ValueError, match="nonfinite"):
        save_training_state(path, state=bad, metadata={})
    assert not path.exists()
    save_training_state(path, state=_example(), metadata={})
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays["leaf_0"].flat[0] += 1
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    with pytest.raises(ValueError, match="differs from its manifest"):
        inspect_training_state(path)


def _tiny_environment():
    path = Path(__file__).parents[1] / "cotraining/test_fsm_recurrent_jit.py"
    spec = importlib.util.spec_from_file_location("resume_test_environment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._EpisodeEnv


def _config():
    return {
        "SEED": 42,
        "NUM_ENVS": 4,
        "NUM_STEPS": 4,
        "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 2,
        "TOTAL_TIMESTEPS": 48,
        "LR": 1e-2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "MAX_GRAD_NORM": 0.5,
        "CLIP_VALUE_LOSS": False,
        "ANNEAL_LR": False,
        "NORM_REWARDS": True,
        "REWARD_SCALE": 1.0,
        "TRAIN_VARIANT": SimpleNamespace(blue_block_policy="cc4", num_steps=4),
        "TOPOLOGY_BANK": tuple(f"test-topology-{index}.npz" for index in range(4)),
    }


def _initialize(seed):
    network = policy_from_arch(
        {"name": "recurrent", "hidden_dim": 8, "hidden_layers": 1, "cell": "lstm", "trunk": "separate"},
        action_dim=3,
    )
    _, obs, env, initialize, update = ippo_jax.make_train(_config(), network)
    state = {
        "train_state": initialize(jax.random.PRNGKey(seed)),
        "env_state": env,
        "obs": obs,
        "rng": jax.random.PRNGKey(seed + 1),
        "reward_norm_state": ippo_jax.RewardNormState(
            returns=jnp.zeros(4), mean=jnp.zeros(()), var=jnp.ones(()), count=jnp.array(1e-4)
        ),
    }
    return state, update


def _update(state, update):
    train_state, env_state, obs, rng, reward_norm_state, metrics = update(
        state["train_state"], state["env_state"], state["obs"], state["rng"], state["reward_norm_state"]
    )
    return {
        "train_state": train_state,
        "env_state": env_state,
        "obs": obs,
        "rng": rng,
        "reward_norm_state": reward_norm_state,
    }, metrics


def test_fresh_template_resume_matches_uninterrupted_next_compiled_lstm_update(tmp_path, monkeypatch):
    environment = _tiny_environment()
    monkeypatch.setattr(ippo_jax, "make_jax_env", lambda *_args, **_kwargs: environment())
    monkeypatch.setattr(ippo_jax, "compute_blue_action_mask", lambda *_args, **_kwargs: jnp.ones(3, dtype=jnp.bool_))
    with jax.disable_jit(False):
        initial, update = _initialize(43)
        after_one, _ = _update(initial, update)
        checkpoint = tmp_path / "after-one.npz"
        saved = save_training_state(checkpoint, state=after_one, metadata={"contract": "same", "completed_updates": 1})
        expected, expected_metrics = _update(after_one, update)
        # Recreate every static optimizer/network function and use deliberately
        # different initial parameters/RNG. Restore all dynamic state from disk.
        fresh_template, fresh_update = _initialize(999)
        restored, metadata = load_training_state(
            checkpoint,
            template=fresh_template,
            expected_metadata={"contract": "same"},
            expected_sha256=saved["sha256"],
        )
        assert metadata["completed_updates"] == 1
        actual, actual_metrics = _update(restored, fresh_update)
        for expected_leaf, actual_leaf in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            np.testing.assert_array_equal(np.asarray(expected_leaf), np.asarray(actual_leaf))
        for key in expected_metrics:
            np.testing.assert_array_equal(np.asarray(expected_metrics[key]), np.asarray(actual_metrics[key]))
        uninterrupted = save_training_state(
            tmp_path / "uninterrupted.npz", state=expected, metadata={"producer": "original"}
        )
        resumed = save_training_state(tmp_path / "resumed.npz", state=actual, metadata={"producer": "resumed"})
        assert uninterrupted["state_sha256"] == resumed["state_sha256"]
        assert int(actual["train_state"].step) == 8
        assert np.all(np.asarray(actual["env_state"].env_state.state.time) == 0)
