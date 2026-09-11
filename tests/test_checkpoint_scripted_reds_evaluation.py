from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaxborg.evaluation.checkpoint_scripted_reds import (
    CheckpointScriptedRedsSettings,
    run_checkpoint_scripted_reds,
)
from jaxborg.evaluation.play_priors import PeriodicCheckpoint
from jaxborg.recipe import load


def _recipe(**overrides):
    recipe = {
        "meta": {"name": "checkpoint-scripted-test"},
        "train": {"teams": "both", "episode_length": 10, "variant": "cc4_stock"},
        "jax": {"num_envs": 2, "checkpoint_every_updates": 2},
        "eval": {"variant": "cia_resilience", "checkpoint_scripted_reds": True},
        "run": {"train_run_id": "run-1", "seed": 42},
    }
    recipe["eval"].update(overrides)
    return recipe


def _checkpoints(count: int) -> list[PeriodicCheckpoint]:
    return [PeriodicCheckpoint(Path(f"/tmp/checkpoint_{i * 40}.safetensors"), i * 40) for i in range(1, count + 1)]


def test_settings_reject_unknown_keys_and_empty_reds():
    with pytest.raises(ValueError, match="unknown settings"):
        CheckpointScriptedRedsSettings.from_recipe(_recipe(checkpoint_scripted_reds={"nope": 1}))
    with pytest.raises(ValueError, match="at least one Red"):
        CheckpointScriptedRedsSettings.from_recipe(_recipe(checkpoint_scripted_reds={"reds": []}))


def test_settings_default_to_the_full_scripted_suite():
    settings = CheckpointScriptedRedsSettings.from_recipe(_recipe(checkpoint_scripted_reds=True))

    assert settings.enabled
    assert settings.reds == ("fsm", "cia_c", "cia_i", "cia_a")


def test_disabled_suite_is_a_no_op():
    assert run_checkpoint_scripted_reds("m.safetensors", _recipe(checkpoint_scripted_reds=False)) is None


def _fake_rows(reward_by_red):
    def evaluate(model_path, **kwargs):
        step = int(Path(model_path).stem.split("_")[1])
        return [
            {
                "eval_red": red,
                "mean_reward": float(reward(step)),
                "std_reward": 1.0,
                "n_episodes": 4,
                "cia_summary": {
                    "n": 4,
                    "c": {"mean": -1.0, "std": 0.5},
                    "i": {"mean": -2.0, "std": 0.5},
                    "a": {"mean": -3.0, "std": 0.5},
                },
                "train_run_id": "run-1",
            }
            for red, reward in reward_by_red.items()
        ]

    return evaluate


def test_each_checkpoint_gets_its_own_step_stamped_metrics(tmp_path, monkeypatch):
    from jaxborg.evaluation import checkpoint_scripted_reds as module

    checkpoints = _checkpoints(3)
    monkeypatch.setattr(module, "find_periodic_checkpoints", lambda *_a, **_k: checkpoints)
    monkeypatch.setattr(module, "select_checkpoints", lambda found, limit: found)

    attached: list[tuple[int, dict]] = []
    recipe = load("cotraining")
    recipe["eval"]["checkpoint_scripted_reds"]["enabled"] = True
    recipe["run"] = {"train_run_id": "run-1", "seed": 42}
    output = tmp_path / "rows.jsonl"

    run_checkpoint_scripted_reds(
        tmp_path / "model_x.safetensors",
        recipe,
        output=output,
        evaluate_fn=_fake_rows({"fsm": lambda s: -s, "cia_c": lambda s: -2 * s}),
        attach_metrics_fn=lambda run_id, metrics, step: attached.append((step, metrics)),
    )

    # One attach per checkpoint, each stamped at that checkpoint's step.
    assert [step for step, _ in attached] == [c.steps for c in checkpoints]
    first = attached[0][1]
    assert first["eval.checkpoint_scripted_reds.fsm.blue.mean_reward"] == -40.0
    assert first["eval.checkpoint_scripted_reds.cia_c.blue.mean_reward"] == -80.0
    # Worst case is the harsher Red, not the average of the two.
    assert first["eval.checkpoint_scripted_reds.blue.worst_reward"] == -80.0
    assert first["eval.checkpoint_scripted_reds.blue.mean_reward"] == -60.0
    # CIA travels with every checkpoint, not just the final model.
    assert first["eval.checkpoint_scripted_reds.fsm.blue.cia.c.mean"] == -1.0
    assert first["eval.checkpoint_scripted_reds.fsm.blue.cia.a.mean"] == -3.0

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 6  # 3 checkpoints x 2 Reds
    assert {row["checkpoint_step"] for row in rows} == {40, 80, 120}
    assert all(row["suite"] == "checkpoint_scripted_reds" for row in rows)


def test_cotraining_recipes_disable_checkpoint_curves_but_keep_final_scripted_reds():
    for path in sorted((Path(__file__).resolve().parents[1] / "recipes" / "cotraining").glob("cotraining*.yaml")):
        recipe = load(str(path))
        settings = CheckpointScriptedRedsSettings.from_recipe(recipe)
        assert not settings.enabled
        assert settings.required is False
        assert settings.max_checkpoints == 20
        # Seeds match eval.after_training so the last checkpoint is directly
        # comparable with the final-model scripted-Red numbers.
        scripted = next(e for e in recipe["eval"]["after_training"] if e["name"] == "scripted-reds")
        assert scripted["args"][scripted["args"].index("--seeds") + 1] == "1000-1009"
        assert settings.seeds == tuple(range(1000, 1010))


def test_malformed_cia_summary_does_not_lose_the_reward_metrics(tmp_path, monkeypatch, capsys):
    """Rewards are the primary signal and must survive a bad CIA summary."""
    from jaxborg.evaluation import checkpoint_scripted_reds as module

    monkeypatch.setattr(module, "find_periodic_checkpoints", lambda *_a, **_k: _checkpoints(1))
    monkeypatch.setattr(module, "select_checkpoints", lambda found, limit: found)

    def evaluate(model_path, **kwargs):
        return [
            {
                "eval_red": "fsm",
                "mean_reward": -5.0,
                "std_reward": 1.0,
                "n_episodes": 4,
                "cia_summary": {"c": {"mean": -1.0}},  # missing "n", "i", "a"
                "train_run_id": "run-1",
            }
        ]

    attached: list[dict] = []
    recipe = load("cotraining")
    recipe["eval"]["checkpoint_scripted_reds"]["enabled"] = True
    recipe["run"] = {"train_run_id": "run-1", "seed": 42}
    run_checkpoint_scripted_reds(
        tmp_path / "model_x.safetensors",
        recipe,
        output=tmp_path / "rows.jsonl",
        evaluate_fn=evaluate,
        attach_metrics_fn=lambda run_id, metrics, step: attached.append(metrics),
    )

    assert attached, "a malformed CIA summary must not suppress the whole attach"
    assert attached[0]["eval.checkpoint_scripted_reds.fsm.blue.mean_reward"] == -5.0
    assert not any("cia" in key for key in attached[0])
    assert "Skipping CIA metrics" in capsys.readouterr().out
