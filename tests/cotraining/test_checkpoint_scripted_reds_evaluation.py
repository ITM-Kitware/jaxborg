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


@pytest.mark.parametrize("reds", [["typo"], ["fsm", "fsm"], [1]])
def test_invalid_red_names_are_rejected_before_running_checkpoints(reds):
    with pytest.raises(ValueError, match="eval.checkpoint_scripted_reds.reds"):
        CheckpointScriptedRedsSettings.from_recipe(_recipe(checkpoint_scripted_reds={"reds": reds}))


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
    recipe["eval"]["checkpoint_scripted_reds"] = {"reds": ["fsm", "cia_c"], "max_checkpoints": 20}
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


HMARL_COMPARISON = {
    base + suffix
    for base in ("cotraining", "cotraining_lstm", "cotraining_mappo", "cotraining_mappo_lstm")
    for suffix in ("", "_env_diversity")
}


def test_hmarl_comparison_recipes_curve_fsm_and_cia_every_48m_steps_plus_final():
    for path in sorted((Path(__file__).resolve().parents[2] / "recipes" / "cotraining").glob("cotraining*.yaml")):
        recipe = load(str(path))
        settings = CheckpointScriptedRedsSettings.from_recipe(recipe)
        # The final-model sweep is the headline number in every recipe.
        scripted = next(e for e in recipe["eval"]["after_training"] if e["name"] == "scripted-reds")
        assert scripted["args"][scripted["args"].index("--seeds") + 1] == "1000-1009"
        if path.stem not in HMARL_COMPARISON:
            assert not settings.enabled
            continue
        assert settings.enabled and settings.required
        assert settings.reds == ("fsm", "cia_c", "cia_i", "cia_a")
        # Five 960k checkpoints per point; cross_play's per-cell budget.
        assert settings.every_steps == 4_800_000
        assert settings.every_steps % 960_000 == 0
        assert settings.include_final is True
        assert settings.seeds == (1000, 1001, 1002)
        assert settings.episodes_per_seed == 6


def test_every_steps_selects_fixed_interval_and_appends_exact_final(tmp_path, monkeypatch):
    from jaxborg.evaluation import checkpoint_scripted_reds as module

    # Stride is 2 updates x 2 envs x 10 steps = 40; saves at 40..240.
    monkeypatch.setattr(module, "find_periodic_checkpoints", lambda *_a, **_k: _checkpoints(6))
    final = tmp_path / "model_x.safetensors"
    final.touch()
    recipe = _recipe(checkpoint_scripted_reds={"reds": ["fsm"], "every_steps": 80, "include_final": True})
    recipe["run"]["total_steps"] = 250
    evaluated, attached = [], []

    def evaluate(model_path, **kwargs):
        evaluated.append(Path(model_path))
        return [{"eval_red": "fsm", "mean_reward": -1.0, "std_reward": 0.0, "n_episodes": 1}]

    output = run_checkpoint_scripted_reds(
        final,
        recipe,
        output=tmp_path / "rows.jsonl",
        evaluate_fn=evaluate,
        attach_metrics_fn=lambda run_id, metrics, step: attached.append(step),
    )

    assert attached == [80, 160, 240, 250]
    assert evaluated[-1] == final.resolve()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["checkpoint_step"] for row in rows] == [80, 160, 240, 250]


def test_every_steps_must_land_on_saved_checkpoints(tmp_path, monkeypatch):
    from jaxborg.evaluation import checkpoint_scripted_reds as module

    monkeypatch.setattr(module, "find_periodic_checkpoints", lambda *_a, **_k: _checkpoints(6))
    recipe = _recipe(checkpoint_scripted_reds={"every_steps": 60})
    with pytest.raises(ValueError, match="multiple of the 40-step checkpoint interval"):
        run_checkpoint_scripted_reds(tmp_path / "model_x.safetensors", recipe, evaluate_fn=pytest.fail)


@pytest.mark.parametrize(
    "config,message",
    [
        ({"every_steps": 80, "max_checkpoints": 4}, "every_steps or max_checkpoints, not both"),
        ({"every_steps": 0}, "every_steps must be a positive integer"),
        ({"every_steps": True}, "every_steps must be a positive integer"),
        ({"include_final": "yes"}, "include_final must be a boolean"),
    ],
)
def test_interval_settings_are_validated(config, message):
    with pytest.raises(ValueError, match=message):
        CheckpointScriptedRedsSettings.from_recipe(_recipe(checkpoint_scripted_reds=config))


def test_include_final_requires_step_provenance(tmp_path, monkeypatch):
    from jaxborg.evaluation import checkpoint_scripted_reds as module

    monkeypatch.setattr(module, "find_periodic_checkpoints", lambda *_a, **_k: _checkpoints(2))
    recipe = _recipe(checkpoint_scripted_reds={"include_final": True})
    with pytest.raises(ValueError, match="checkpoint_scripted_reds.include_final requires run.total_steps"):
        run_checkpoint_scripted_reds(tmp_path / "model_x.safetensors", recipe, evaluate_fn=pytest.fail)


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
    recipe["eval"]["checkpoint_scripted_reds"] = {"reds": ["fsm", "cia_c"], "max_checkpoints": 20}
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
