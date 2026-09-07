from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jaxborg.evaluation.play_priors import (
    PlayPriorsSettings,
    find_periodic_checkpoints,
    run_play_priors,
)
from jaxborg.recipe import load


def _recipe(*, teams: str = "both", play_priors=True) -> dict:
    return {
        "meta": {"name": "prior-test"},
        "algorithm": "ippo",
        "core": {"lr": 3e-4},
        "arch": {"name": "shared"},
        "train": {
            "teams": teams,
            "episode_length": 10,
            "buffer_size": 20,
            "total_timesteps": 100,
            "variant": "cc4_stock",
        },
        "eval": {"variant": "cc4_stock", "play_priors": play_priors},
        "jax": {"num_envs": 2, "checkpoint_every_updates": 2},
        "cleanrl": {
            "num_envs": 2,
            "rollout_length": 10,
            "num_rollouts_per_update": 1,
            "checkpoint_every_updates": 2,
        },
    }


def _run_files(tmp_path: Path, suffix: str = ".safetensors") -> Path:
    run_dir = tmp_path / "exp" / "ippo_jax" / "run"
    run_dir.mkdir(parents=True)
    model = run_dir / f"model_run{suffix}"
    model.touch()
    for steps in (40, 60, 80):
        (run_dir / f"checkpoint_{steps}{suffix}").touch()
    return model


def test_settings_accept_boolean_and_expanded_mapping():
    assert PlayPriorsSettings.from_recipe(_recipe()).enabled
    settings = PlayPriorsSettings.from_recipe(
        _recipe(
            play_priors={
                "seeds": "9,7-8",
                "episodes_per_seed": 3,
                "deterministic": True,
                "required": False,
            }
        )
    )

    assert settings.seeds == (7, 8, 9)
    assert settings.episodes_per_seed == 3
    assert settings.deterministic
    assert not settings.required


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"seeds": []}, "at least one"),
        ({"episodes_per_seed": 0}, "must be positive"),
        ({"unexpected": 1}, "unknown settings"),
        ("yes", "boolean or mapping"),
    ],
)
def test_settings_reject_invalid_values(value, message):
    with pytest.raises(ValueError, match=message):
        PlayPriorsSettings.from_recipe(_recipe(play_priors=value))


def test_recipe_rejects_play_priors_for_single_team_training(tmp_path):
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(_recipe(teams="blue"), sort_keys=False))

    with pytest.raises(ValueError, match="only supported when train.teams is 'both'"):
        load(str(recipe_path))


def test_recipe_rejects_play_priors_without_durable_checkpoints(tmp_path):
    recipe = _recipe()
    recipe["jax"]["checkpoint_every_updates"] = 0
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))

    with pytest.raises(ValueError, match="jax.checkpoint_every_updates to be a positive integer"):
        load(str(recipe_path))


def test_finds_only_checkpoints_on_the_periodic_training_cadence(tmp_path):
    model = _run_files(tmp_path)

    checkpoints = find_periodic_checkpoints(model, _recipe())

    assert [checkpoint.steps for checkpoint in checkpoints] == [40, 80]


def test_uses_cleanrl_cadence_for_torch_checkpoints(tmp_path):
    recipe = _recipe()
    recipe["cleanrl"]["num_envs"] = 3
    model = _run_files(tmp_path, suffix=".pt")
    model.with_name("checkpoint_120.pt").touch()

    checkpoints = find_periodic_checkpoints(model, recipe)

    assert [checkpoint.steps for checkpoint in checkpoints] == [60, 120]


def test_evaluates_each_adjacent_pair_in_both_directions_and_logs_curves(tmp_path, monkeypatch):
    from jaxborg import recipe as recipe_module

    model = _run_files(tmp_path)
    recipe = _recipe(play_priors={"seeds": [7, 9], "episodes_per_seed": 1})
    recipe["run"] = {"train_run_id": "train-123"}
    # Older/custom projection doubles do not expose the additive CIA keys.
    monkeypatch.setattr(recipe_module, "project_eval", lambda *_args, **_kwargs: {"TOPOLOGY_BANK": ()})
    calls = []
    attached = []

    def fake_evaluate(blue_path, red_path, **kwargs):
        calls.append((Path(blue_path), Path(red_path), kwargs))
        values = [2.0, 4.0] if len(calls) == 1 else [5.0, 7.0]
        return SimpleNamespace(
            blue_returns=values,
            red_returns=[-value for value in values],
            episode_seeds=[7, 9],
            policies={"blue": {"path": str(blue_path)}, "red": {"path": str(red_path)}},
            topology_paths=[],
            topology_sampling="generative",
            episode_topology_paths=[None, None],
        )

    def fake_attach(run_id, metrics, *, step=None):
        attached.append((run_id, metrics, step))

    output = tmp_path / "results.jsonl"
    result = run_play_priors(
        model,
        recipe,
        output=output,
        evaluate_fn=fake_evaluate,
        attach_metrics_fn=fake_attach,
    )

    checkpoint_40 = model.with_name("checkpoint_40.safetensors").resolve()
    checkpoint_80 = model.with_name("checkpoint_80.safetensors").resolve()
    assert result == output.resolve()
    assert [(call[0], call[1]) for call in calls] == [
        (checkpoint_80, checkpoint_40),
        (checkpoint_40, checkpoint_80),
    ]
    assert all(call[2]["seeds"] == [7, 9] for call in calls)
    assert all(call[2]["topology_sampling"] == "exhaustive" for call in calls)
    assert all("cia" not in call[2] for call in calls)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["focal_team"] for row in rows] == ["blue", "red"]
    assert [row["mean_reward"] for row in rows] == [3.0, -6.0]
    assert all(row["current_step"] == 80 and row["prior_step"] == 40 for row in rows)
    assert attached == [
        (
            "train-123",
            {
                "eval.play_priors.blue_vs_prior_red.mean_reward": 3.0,
                "eval.play_priors.red_vs_prior_blue.mean_reward": -6.0,
                "eval.play_priors.prior_step": 40.0,
            },
            80,
        )
    ]


def test_cotraining_recipe_retains_ten_periodic_checkpoints():
    recipe = load("cotraining")
    steps_per_update = recipe["jax"]["num_envs"] * recipe["train"]["episode_length"]
    num_updates = recipe["train"]["total_timesteps"] // steps_per_update
    checkpoint_every = recipe["jax"]["checkpoint_every_updates"]

    assert len(range(checkpoint_every, num_updates + 1, checkpoint_every)) == 10
    assert PlayPriorsSettings.from_recipe(recipe).enabled


def test_play_priors_logs_comparison_qualified_cia_and_writes_audit_fields(tmp_path, monkeypatch):
    from jaxborg import recipe as recipe_module

    model = _run_files(tmp_path)
    recipe = _recipe(play_priors={"seeds": [7], "episodes_per_seed": 1})
    recipe["eval"].update(
        {
            "variant": "cia_resilience",
            "cia": {
                "enabled": True,
                "metric": "resilience",
                "role_assignment": "fixed_per_topology",
            },
        }
    )
    recipe["run"] = {"train_run_id": "train-cia"}
    topology = tmp_path / "eval.snapshot.npz"
    cia_config = recipe["eval"]["cia"]
    monkeypatch.setattr(
        recipe_module,
        "project_eval",
        lambda *_args, **_kwargs: {
            "TOPOLOGY_BANK": (topology,),
            "TOPOLOGY_SAMPLING": "exhaustive",
            "CIA": cia_config,
        },
    )
    calls = []

    def fake_evaluate(blue_path, red_path, **kwargs):
        calls.append((blue_path, red_path, kwargs))
        offset = float(len(calls))
        summary = {
            "n": 1,
            "c": {"mean": -offset, "std": 0.0},
            "i": {"mean": -2.0 * offset, "std": 0.0},
            "a": {"mean": -3.0 * offset, "std": 0.0},
        }
        return SimpleNamespace(
            blue_returns=[offset],
            red_returns=[-offset],
            episode_seeds=[7],
            policies={"blue": {"path": str(blue_path)}, "red": {"path": str(red_path)}},
            topology_paths=[str(topology)],
            topology_sampling="exhaustive",
            episode_topology_paths=[str(topology)],
            cia_metric="resilience",
            cia_config=cia_config,
            cia_summary=summary,
            per_episode_cia=[{"c": -offset, "i": -2.0 * offset, "a": -3.0 * offset}],
            episode_role_map_ids=["shared-map"],
            episode_topology_fingerprints=["shared-fingerprint"],
            topology_role_maps=[{"topology_path": str(topology), "role_map_id": "shared-map"}],
        )

    attached = []
    output = tmp_path / "cia-results.jsonl"
    run_play_priors(
        model,
        recipe,
        output=output,
        evaluate_fn=fake_evaluate,
        attach_metrics_fn=lambda run_id, metrics, *, step=None: attached.append((run_id, metrics, step)),
    )

    assert all(call[2]["cia"] == cia_config for call in calls)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(row["episode_role_map_ids"] == ["shared-map"] for row in rows)
    metrics = attached[0][1]
    assert metrics["eval.play_priors.blue_vs_prior_red.cia.c.mean"] == -1.0
    assert metrics["eval.play_priors.red_vs_prior_blue.cia.c.mean"] == -2.0
    assert metrics["eval.play_priors.blue_vs_prior_red.cia.a.std"] == 0.0
