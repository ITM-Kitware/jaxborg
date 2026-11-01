"""Exercise shared native evaluation through real episodes and observable outputs."""

import copy
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from jaxborg.checkpoint import read_sidecar
from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.evaluation.cyborg_runner import (
    evaluate_on_cyborg,
    resolve_torch_evaluation_settings,
    run_torch_episodes,
)
from jaxborg.policies import make_torch_policy
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK

PROJECT = Path(__file__).resolve().parents[2]
SHORT_VARIANT = replace(CC4_STOCK, num_steps=2)


@pytest.fixture
def tiny_model(tmp_path):
    model = tmp_path / "model_native.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        policy = make_torch_policy("shared", obs_dim=BLUE_OBS_SIZE, action_dim=242, hidden_dim=8, hidden_layers=1)
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.zero_()
        torch.save(policy.state_dict(), model)
    model.with_name("recipe_native.yaml").write_text(
        yaml.safe_dump(
            {"arch": {"name": "shared", "hidden_dim": 8, "hidden_layers": 1}, "train": {"variant": "cc4_stock"}}
        )
    )
    return model


def test_recipe_precedence_horizon_and_policy_settings_leave_input_unchanged(tiny_model):
    recipe = read_sidecar(tiny_model)
    before = copy.deepcopy(recipe)
    inherited = resolve_torch_evaluation_settings(recipe)
    assert inherited.variant == CC4_STOCK
    recipe["eval"] = {"variant": "cia_c"}
    before = copy.deepcopy(recipe)
    resolved = resolve_torch_evaluation_settings(recipe, deterministic=True, horizon=2)
    assert resolved.variant.name == "cia_c"
    assert resolved.variant.num_steps == 2
    assert resolved.variant.resilience_roles is True
    assert resolved.deterministic is True
    assert resolved.policy == {
        "name": "shared",
        "obs_dim": BLUE_OBS_SIZE,
        "action_dim": 242,
        "hidden_dim": 8,
        "hidden_layers": 1,
    }
    assert recipe == before


@pytest.mark.parametrize("arguments", [{"horizon": 0}, {"horizon": True}, {"horizon": 501}, {"deterministic": "yes"}])
def test_invalid_settings_are_rejected(tiny_model, arguments):
    with pytest.raises(ValueError):
        resolve_torch_evaluation_settings(read_sidecar(tiny_model), **arguments)


def test_expanded_repeated_seeds_and_real_spawn_workers_keep_global_episode_order(tiny_model):
    arguments = dict(variant=SHORT_VARIANT, seeds=[17, 18], episodes_per_seed=2, policy_seed=100, progress=False)
    serial = evaluate_on_cyborg(tiny_model, workers=1, **arguments)
    parallel = evaluate_on_cyborg(tiny_model, workers=2, **arguments)
    assert serial == parallel
    assert serial[1] == [17, 18, 18, 19]
    completed = []
    explicit = run_torch_episodes(
        tiny_model,
        variant=SHORT_VARIANT,
        episode_seeds=[17, 18, 18, 19],
        policy_seeds=[100, 101, 102, 103],
        on_complete=lambda *values: completed.append(values),
    )
    assert explicit == serial
    assert [(row[0], row[1], row[2]) for row in completed] == [(0, 17, 100), (1, 18, 101), (2, 18, 102), (3, 19, 103)]
    more_workers = evaluate_on_cyborg(
        tiny_model,
        variant=SHORT_VARIANT,
        seeds=[17, 18],
        episodes_per_seed=1,
        policy_seed=100,
        workers=8,
        progress=False,
    )
    assert more_workers == (serial[0][:2], [17, 18])


def test_seeded_stochastic_execution_restores_caller_draws_and_ignores_ambient_seed(tiny_model):
    arguments = dict(variant=SHORT_VARIANT, seeds=[17, 18], episodes_per_seed=1, policy_seed=100, progress=False)
    torch.manual_seed(312)
    expected_draws = torch.rand(8, generator=torch.Generator().manual_seed(312))
    first = evaluate_on_cyborg(tiny_model, **arguments)
    assert torch.equal(torch.rand(8), expected_draws)
    torch.manual_seed(999)
    second = evaluate_on_cyborg(tiny_model, **arguments)
    assert first == second
    assert torch.equal(torch.rand(8), torch.rand(8, generator=torch.Generator().manual_seed(999)))


def test_seeded_model_load_failure_restores_caller_draws(tmp_path):
    torch.manual_seed(312)
    with pytest.raises(FileNotFoundError):
        run_torch_episodes(tmp_path / "model_missing.pt", variant=SHORT_VARIANT, episode_seeds=[17], policy_seeds=[100])
    assert torch.equal(torch.rand(8), torch.rand(8, generator=torch.Generator().manual_seed(312)))


def test_observers_see_loaded_policy_reset_and_completion_in_caller_order(tiny_model):
    events = []

    def policy_loaded(policy, recipe):
        assert recipe == read_sidecar(tiny_model)
        events.append(("policy",))

    def reset(index, seed, env, observation):
        assert observation.obs and observation.info
        events.append(("reset", index, seed))

    def complete(index, seed, policy_seed, reward, steps):
        assert 0 < steps <= 2 and math.isfinite(reward)
        events.append(("complete", index, seed, policy_seed, reward))

    rewards, seeds = run_torch_episodes(
        tiny_model,
        variant=SHORT_VARIANT,
        episode_seeds=[18, 17],
        policy_seeds=[100, 101],
        on_policy=policy_loaded,
        on_reset=reset,
        on_complete=complete,
    )
    assert seeds == [18, 17]
    assert events == [
        ("policy",),
        ("reset", 0, 18),
        ("complete", 0, 18, 100, rewards[0]),
        ("reset", 1, 17),
        ("complete", 1, 17, 101, rewards[1]),
    ]


@pytest.mark.parametrize("stage", ["policy", "reset", "complete"])
def test_observer_failure_stops_later_episodes_and_restores_seeded_rng(tiny_model, stage):
    events = []

    def event(name):
        def observe(*values):
            index = None if name == "policy" else values[0]
            events.append((name, index))
            if name == stage:
                if name != "policy":
                    torch.rand(4)
                raise RuntimeError("observer rejected evidence")

        return observe

    torch.manual_seed(312)
    with pytest.raises(RuntimeError, match="observer rejected evidence"):
        run_torch_episodes(
            tiny_model,
            variant=SHORT_VARIANT,
            episode_seeds=[17, 18],
            policy_seeds=[100, 101],
            on_policy=event("policy"),
            on_reset=event("reset"),
            on_complete=event("complete"),
        )
    expected = [("policy", None), ("reset", 0), ("complete", 0)]
    assert events == expected[: ["policy", "reset", "complete"].index(stage) + 1]
    assert torch.equal(torch.rand(8), torch.rand(8, generator=torch.Generator().manual_seed(312)))


@pytest.mark.parametrize("observer", ["on_policy", "on_reset", "on_complete"])
def test_parallel_observers_are_rejected_before_loading_or_emitting_events(tmp_path, observer):
    events = []
    with pytest.raises(ValueError, match="serial"):
        run_torch_episodes(
            tmp_path / "missing.pt",
            variant=SHORT_VARIANT,
            episode_seeds=[17],
            workers=2,
            **{observer: lambda *values: events.append(values)},
        )
    assert events == []


@pytest.mark.parametrize("episode_seeds,policy_seeds", [([], None), ([17, 18], [100]), ([17], [True])])
def test_invalid_execution_schedule_is_rejected_before_model_loading(tmp_path, episode_seeds, policy_seeds):
    with pytest.raises(ValueError):
        run_torch_episodes(
            tmp_path / "missing.pt", variant=SHORT_VARIANT, episode_seeds=episode_seeds, policy_seeds=policy_seeds
        )


def test_existing_recipe_cli_writes_one_native_deterministic_result_without_tracking(tmp_path, tiny_model):
    output = tmp_path / "evaluation.json"
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join([str(PROJECT / "src"), str(PROJECT)]),
        "JAXBORG_EXP_DIR": str(tmp_path / "artifacts"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts/eval/eval_recipe.py"),
            "--model",
            str(tiny_model),
            "--seeds",
            "17",
            "--episodes",
            "1",
            "--deterministic",
            "--workers",
            "1",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text())
    assert result["trained_backend"] == result["eval_env"] == "cyborg"
    assert result["variant"] == "cc4_stock" and result["stochastic"] is False
    assert result["seeds"] == result["per_episode_seeds"] == [17]
    assert result["n_episodes"] == result["episodes_per_seed"] == 1
    assert result["train_run_id"] is None
    assert len(result["per_episode"]) == 1 and math.isfinite(result["per_episode"][0])
    assert result["mean_reward"] == result["per_episode"][0]
    assert "attached eval metrics" not in completed.stdout
