from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jaxborg.evaluation.training_checkpoint import evaluate_training_checkpoint


def _recipe(*, teams: str = "blue", eval_name: str = "cia_c") -> dict:
    return {
        "meta": {"name": "checkpoint-eval-test"},
        "algorithm": "ippo",
        "core": {"lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95},
        "arch": {"name": "shared", "hidden_dim": 8, "hidden_layers": 1},
        "train": {
            "teams": teams,
            "episode_length": 10,
            "buffer_size": 20,
            "total_timesteps": 100,
            "variant": "cc4_stock",
        },
        "eval": {"variant": eval_name},
    }


@pytest.mark.parametrize(
    ("backend", "expected_backend"),
    [
        ("jax", "jax"),
        ("cyborg", "cyborg"),
        ("torch", "cyborg"),
        ("cleanrl", "cyborg"),
    ],
)
def test_joint_checkpoint_eval_uses_one_bundle_and_ten_episodes(
    monkeypatch,
    backend,
    expected_backend,
):
    from jaxborg.evaluation import matchup_runner

    calls = []

    def fake_evaluate_matchup(blue_model, red_model, **kwargs):
        calls.append((blue_model, red_model, kwargs))
        return SimpleNamespace(
            blue_returns=[1.0, 3.0, 8.0],
            red_returns=[-1.0, -3.0, -8.0],
        )

    monkeypatch.setattr(matchup_runner, "evaluate_matchup", fake_evaluate_matchup)
    recipe = _recipe(teams="both")
    recipe["mlflow"] = {
        "checkpoint_eval": {
            "seed": 777,
            "deterministic": True,
        }
    }
    checkpoint = Path("checkpoints/checkpoint_500.pt")

    rewards = evaluate_training_checkpoint(
        checkpoint,
        backend=backend,
        recipe=recipe,
        seed=12,
    )

    assert rewards == {
        "blue": pytest.approx(4.0),
        "red": pytest.approx(-4.0),
    }
    assert len(calls) == 1
    blue_model, red_model, kwargs = calls[0]
    assert blue_model == checkpoint
    assert red_model == checkpoint
    assert kwargs["backend"] == expected_backend
    assert kwargs["variant"].name == "cia_c"
    assert kwargs["seeds"] == [777]
    assert kwargs["episodes_per_seed"] == 10
    assert kwargs["deterministic"] is True
    assert kwargs["progress"] is False


@pytest.mark.parametrize(
    ("teams", "expected"),
    [
        ("blue", {"blue": pytest.approx(4.0)}),
        ("red", {"red": pytest.approx(-4.0)}),
    ],
)
def test_learned_matchup_returns_only_trained_team(monkeypatch, teams, expected):
    from jaxborg.evaluation import matchup_runner

    monkeypatch.setattr(
        matchup_runner,
        "evaluate_matchup",
        lambda *_args, **_kwargs: SimpleNamespace(
            blue_returns=[1.0, 7.0],
            red_returns=[-1.0, -7.0],
        ),
    )
    recipe = _recipe(teams=teams)
    if teams == "blue":
        recipe["train"]["opponents"] = {"red": "frozen-red.safetensors"}

    assert (
        evaluate_training_checkpoint(
            "checkpoint.safetensors",
            backend="jax",
            recipe=recipe,
            seed=1,
        )
        == expected
    )


def test_legacy_jax_checkpoint_eval_dispatches_to_cyborg_contract(monkeypatch):
    from jaxborg.evaluation import jax_runner

    calls = []

    def fake_evaluate(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return [2.0, 4.0, 9.0], [100_005, 100_006, 100_007], {"meta": {"name": "ignored"}}

    monkeypatch.setattr(jax_runner, "evaluate_jax_on_cyborg", fake_evaluate)
    checkpoint = Path("checkpoint_250.safetensors")

    rewards = evaluate_training_checkpoint(
        checkpoint,
        backend="jax",
        recipe=_recipe(eval_name="cia_i"),
        seed=5,
    )

    assert rewards == {"blue": pytest.approx(5.0)}
    assert len(calls) == 1
    called_checkpoint, kwargs = calls[0]
    assert called_checkpoint == checkpoint
    assert kwargs["variant"].name == "cia_i"
    assert kwargs["seeds"] == [100_005]
    assert kwargs["episodes_per_seed"] == 10
    assert kwargs["deterministic"] is False
    assert kwargs["workers"] == 1
    assert kwargs["progress"] is False


def test_legacy_cyborg_checkpoint_eval_dispatches_to_cyborg_runner(monkeypatch):
    from jaxborg.evaluation import cyborg_runner

    calls = []

    def fake_evaluate(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return [-6.0, -2.0], [100_009, 100_010]

    monkeypatch.setattr(cyborg_runner, "evaluate_on_cyborg", fake_evaluate)
    checkpoint = Path("checkpoint_250.pt")

    rewards = evaluate_training_checkpoint(
        checkpoint,
        backend="cleanrl",
        recipe=_recipe(eval_name="cia_a"),
        seed=9,
    )

    assert rewards == {"blue": pytest.approx(-4.0)}
    assert len(calls) == 1
    called_checkpoint, kwargs = calls[0]
    assert called_checkpoint == checkpoint
    assert kwargs["variant"].name == "cia_a"
    assert kwargs["seeds"] == [100_009]
    assert kwargs["episodes_per_seed"] == 10
    assert kwargs["deterministic"] is False
    assert kwargs["workers"] == 1
    assert kwargs["progress"] is False


@pytest.mark.parametrize("episodes", [0, -1])
def test_checkpoint_eval_rejects_non_positive_episode_count(episodes):
    with pytest.raises(ValueError, match="episodes must be positive"):
        evaluate_training_checkpoint(
            "checkpoint.safetensors",
            backend="jax",
            recipe=_recipe(),
            seed=1,
            episodes=episodes,
        )


def test_checkpoint_eval_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be 'jax' or 'cyborg'"):
        evaluate_training_checkpoint(
            "checkpoint.bin",
            backend="unknown",
            recipe=_recipe(),
            seed=1,
        )


def test_cyborg_checkpoint_eval_restores_python_numpy_and_torch_rng(monkeypatch):
    import torch

    from jaxborg.evaluation import matchup_runner

    def fake_evaluate(*_args, **_kwargs):
        random.seed(901)
        np.random.seed(902)
        torch.manual_seed(903)
        random.random()
        np.random.random()
        torch.rand(1)
        return SimpleNamespace(blue_returns=[1.0], red_returns=[-1.0])

    monkeypatch.setattr(matchup_runner, "evaluate_matchup", fake_evaluate)
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_torch = torch.random.get_rng_state()
    try:
        random.seed(101)
        np.random.seed(102)
        torch.manual_seed(103)
        expected_python = random.getstate()
        expected_numpy = np.random.get_state()
        expected_torch = torch.random.get_rng_state().clone()

        evaluate_training_checkpoint(
            "checkpoint.pt",
            backend="torch",
            recipe=_recipe(teams="both"),
            seed=1,
        )

        assert random.getstate() == expected_python
        actual_numpy = np.random.get_state()
        assert actual_numpy[0] == expected_numpy[0]
        np.testing.assert_array_equal(actual_numpy[1], expected_numpy[1])
        assert actual_numpy[2:] == expected_numpy[2:]
        torch.testing.assert_close(torch.random.get_rng_state(), expected_torch)
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.random.set_rng_state(original_torch)
