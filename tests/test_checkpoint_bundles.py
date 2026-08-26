from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jaxborg.checkpoint import (
    BUNDLE_SCHEMA_VERSION,
    load_jax_bundle,
    load_jax_params,
    load_jax_policy,
    load_torch_bundle,
    load_torch_policy,
    save_jax_bundle,
    save_jax_params,
    save_torch_bundle,
)

ARCH = {"name": "shared", "hidden_dim": 4, "hidden_layers": 1, "activation": "tanh"}


def _jax_params(value: float):
    return {
        "params": {
            "Dense_0": {
                "kernel": jnp.full((3, 4), value),
                "bias": jnp.full((4,), value),
            }
        }
    }


def _entry(weights, team: str, *, trainable: bool = True, source=None):
    return {
        "weights": weights,
        "team": team,
        "obs_dim": 706 if team == "red" else 892,
        "action_dim": 1106 if team == "red" else 319,
        "arch": ARCH,
        "trainable": trainable,
        "source": source,
    }


def test_jax_bundle_round_trip_multiple_teams(tmp_path):
    path = tmp_path / "model_joint.safetensors"
    save_jax_bundle(
        path,
        {
            "blue": _entry(_jax_params(1.0), "blue"),
            "red": _entry(_jax_params(2.0), "red", trainable=False, source={"experiment": "red_base"}),
        },
        provenance={"run_id": "run-123", "recipe": Path("recipes/cotraining.yaml")},
    )

    bundle = load_jax_bundle(path)
    assert bundle.schema_version == BUNDLE_SCHEMA_VERSION
    assert bundle.backend == "jax"
    assert not bundle.legacy
    assert set(bundle.policies) == {"blue", "red"}
    assert bundle.policies["blue"].trainable
    assert not bundle.policies["red"].trainable
    assert bundle.policies["red"].source == {"experiment": "red_base"}
    assert bundle.provenance["recipe"] == "recipes/cotraining.yaml"
    np.testing.assert_array_equal(bundle.policies["blue"].weights["params"]["Dense_0"]["kernel"], 1.0)
    np.testing.assert_array_equal(bundle.policies["red"].weights["params"]["Dense_0"]["kernel"], 2.0)

    blue_params, action_dim = load_jax_params(path)
    assert action_dim == 319
    np.testing.assert_array_equal(blue_params["params"]["Dense_0"]["bias"], 1.0)


def test_jax_bundle_validates_team_and_dimensions(tmp_path):
    path = tmp_path / "model_red.safetensors"
    save_jax_bundle(path, {"red": _entry(_jax_params(2.0), "red")})

    assert load_jax_policy(path, "red", expected_obs_dim=706, expected_action_dim=1106).team == "red"
    with pytest.raises(ValueError, match="no 'blue' policy"):
        load_jax_policy(path)
    with pytest.raises(ValueError, match="observation dimension mismatch"):
        load_jax_policy(path, "red", expected_obs_dim=705)
    with pytest.raises(ValueError, match="action dimension mismatch"):
        load_jax_policy(path, "red", expected_action_dim=1105)


def test_legacy_jax_checkpoint_is_blue_only(tmp_path):
    path = tmp_path / "model_legacy.safetensors"
    save_jax_params(path, _jax_params(3.0), action_dim=319)

    bundle = load_jax_bundle(path)
    assert bundle.legacy
    assert bundle.schema_version == 0
    assert set(bundle.policies) == {"blue"}
    assert bundle.policies["blue"].action_dim == 319
    assert load_jax_policy(path, expected_action_dim=319).team == "blue"


def _torch_state(value: float):
    return {
        "features.0.weight": torch.full((4, 3), value),
        "actor.weight": torch.full((5, 4), value),
    }


def test_torch_bundle_round_trip_multiple_teams(tmp_path):
    path = tmp_path / "model_joint.pt"
    save_torch_bundle(
        path,
        {
            "blue": _entry(_torch_state(1.0), "blue", trainable=False, source={"path": Path("blue.pt")}),
            "red": {**_entry(_torch_state(2.0), "red"), "weights": None, "state_dict": _torch_state(2.0)},
        },
        provenance={"seed": 42},
    )

    bundle = load_torch_bundle(path)
    assert bundle.schema_version == BUNDLE_SCHEMA_VERSION
    assert bundle.backend == "cyborg"
    assert set(bundle.policies) == {"blue", "red"}
    assert bundle.policies["blue"].source == {"path": "blue.pt"}
    assert bundle.provenance == {"seed": 42}
    torch.testing.assert_close(bundle.policies["red"].state_dict["actor.weight"], torch.full((5, 4), 2.0))
    assert load_torch_policy(path, "blue", expected_obs_dim=892, expected_action_dim=319).team == "blue"


def test_legacy_torch_state_dict_is_blue_only(tmp_path):
    path = tmp_path / "model_legacy.pt"
    torch.save(_torch_state(3.0), path)

    bundle = load_torch_bundle(path)
    assert bundle.legacy
    assert bundle.schema_version == 0
    assert set(bundle.policies) == {"blue"}
    torch.testing.assert_close(load_torch_policy(path).weights["actor.weight"], torch.full((5, 4), 3.0))
    with pytest.raises(ValueError, match="no 'red' policy"):
        load_torch_policy(path, "red")


@pytest.mark.parametrize("save", [save_jax_bundle, save_torch_bundle])
def test_bundle_requires_complete_policy_metadata(tmp_path, save):
    suffix = ".safetensors" if save is save_jax_bundle else ".pt"
    weights = _jax_params(1.0) if save is save_jax_bundle else _torch_state(1.0)
    with pytest.raises(ValueError, match="missing bundle metadata"):
        save(tmp_path / f"bad{suffix}", {"blue": {"weights": weights, "action_dim": 2}})
