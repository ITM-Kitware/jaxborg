from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaxborg.evaluation.cross_play import (
    CrossPlaySettings,
    run_cross_play,
    summarize_matrix,
)
from jaxborg.evaluation.play_priors import PeriodicCheckpoint
from jaxborg.recipe import load


def _recipe(*, teams: str = "both", cross_play=True) -> dict:
    return {
        "meta": {"name": "cross-play-test"},
        "train": {"teams": teams, "episode_length": 10, "variant": "cc4_stock"},
        "jax": {"num_envs": 2, "checkpoint_every_updates": 2},
        "eval": {"variant": "cc4_stock", "cross_play": cross_play},
        "run": {"train_run_id": "run-1", "seed": 42},
    }


def _checkpoints(count: int) -> list[PeriodicCheckpoint]:
    return [PeriodicCheckpoint(Path(f"/tmp/checkpoint_{i * 40}.safetensors"), i * 40) for i in range(1, count + 1)]


def test_settings_reject_unknown_keys_and_degenerate_matrix():
    with pytest.raises(ValueError, match="unknown settings"):
        CrossPlaySettings.from_recipe(_recipe(cross_play={"nope": 1}))
    with pytest.raises(ValueError, match="max_checkpoints must be at least 2"):
        CrossPlaySettings.from_recipe(_recipe(cross_play={"max_checkpoints": 1}))


def test_settings_default_to_disabled_and_enable_from_bare_true():
    assert not CrossPlaySettings.from_recipe(_recipe(cross_play=False)).enabled
    assert CrossPlaySettings.from_recipe(_recipe(cross_play=True)).enabled


def test_summary_reports_progress_when_later_blue_dominates_all_history():
    # Blue improves against every Red it has seen: rows increase down the column.
    matrix = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 2.0, 2.0],
    ]

    summary = summarize_matrix(matrix, _checkpoints(3))

    assert summary["blue_worst_vs_history"] == [0.0, 1.0, 2.0]
    assert summary["blue_worst_vs_history_gain"] == 2.0
    assert summary["blue_forgetting_rate"] == 0.0


def test_summary_flags_forgetting_that_adjacent_pairs_would_miss():
    """A cycle: each Blue beats the Red before it, but the last loses to the first.

    Every adjacent comparison (matrix[i][i - 1]) improves, which is all
    play_priors ever looks at, while Blue 2 collapses against Red 0.
    """
    matrix = [
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [-9.0, 5.0, 0.0],
    ]

    summary = summarize_matrix(matrix, _checkpoints(3))

    assert [matrix[i][i - 1] for i in (1, 2)] == [5.0, 5.0]  # adjacent pairs look healthy
    assert summary["blue_worst_vs_history"] == [0.0, 0.0, -9.0]
    assert summary["blue_worst_vs_history_gain"] == -9.0  # the cycle shows here
    assert summary["blue_forgetting_rate"] > 0.0


def test_run_cross_play_evaluates_every_ordered_pair_and_writes_a_summary(tmp_path, monkeypatch):
    from jaxborg.evaluation import cross_play as module

    monkeypatch.setattr(module, "_git_commit", lambda: "test-commit")
    checkpoints = _checkpoints(3)
    monkeypatch.setattr(module, "find_periodic_checkpoints", lambda *_a, **_k: checkpoints)
    monkeypatch.setattr(
        module,
        "select_checkpoints",
        lambda found, limit: found,
    )

    seen: list[tuple[int, int]] = []

    def fake_evaluate(blue_path, red_path, **kwargs):
        blue_step = int(Path(blue_path).stem.split("_")[1])
        red_step = int(Path(red_path).stem.split("_")[1])
        seen.append((blue_step, red_step))
        payoff = float(blue_step - red_step)
        return SimpleNamespace(
            blue_returns=[payoff, payoff],
            red_returns=[-payoff, -payoff],
            episode_seeds=[1000, 1001],
            policies={"blue": {}, "red": {}},
            topology_paths=[],
            episode_topology_paths=[],
            topology_sampling="exhaustive",
            cia_summary=None,
        )

    attached: list[dict] = []
    recipe = load("cotraining")
    recipe["run"] = {"train_run_id": "run-1", "seed": 42}
    recipe["eval"]["cross_play"]["include_final"] = False
    output = tmp_path / "cross_play.jsonl"
    run_cross_play(
        tmp_path / "model_x.safetensors",
        recipe,
        output=output,
        evaluate_fn=fake_evaluate,
        attach_metrics_fn=lambda run_id, metrics, step: attached.append({**metrics, "step": step}),
    )

    assert len(seen) == 9  # 3x3, diagonal included
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    cells = [row for row in rows if row["eval_name"] == "cross_play"]
    summaries = [row for row in rows if row["eval_name"] == "cross_play_summary"]
    assert len(cells) == 9
    assert len(summaries) == 1
    assert sum(row["is_diagonal"] for row in cells) == 3
    # Payoff is blue_step - red_step, so the diagonal is 0 and Blue's worst
    # case against its own history is its oldest opponent.
    assert summaries[0]["blue_self_play"] == [0.0, 0.0, 0.0]
    assert summaries[0]["blue_worst_vs_history"] == [0.0, 0.0, 0.0]
    assert any("eval.cross_play.blue.forgetting_rate" in entry for entry in attached)


def test_run_cross_play_requires_simultaneous_training():
    with pytest.raises(ValueError, match="train.teams is 'both'"):
        run_cross_play("model.safetensors", _recipe(teams="blue"))


def test_disabled_cross_play_is_a_no_op():
    assert run_cross_play("model.safetensors", _recipe(cross_play=False)) is None


def test_comparison_recipes_require_cross_play_including_final_model():
    for name in (
        base + suffix
        for base in ("cotraining", "cotraining_lstm", "cotraining_mappo", "cotraining_mappo_lstm")
        for suffix in ("", "_env_diversity")
    ):
        settings = CrossPlaySettings.from_recipe(load(name))
        assert settings.enabled
        assert settings.max_checkpoints >= 2
        assert settings.required is True
        assert settings.include_final is True


def test_default_cross_play_shares_one_context_across_all_cells(tmp_path, monkeypatch):
    from jaxborg.evaluation import cross_play, matchup_runner

    monkeypatch.setattr(cross_play, "find_periodic_checkpoints", lambda *_a, **_k: _checkpoints(2))
    monkeypatch.setattr(cross_play, "_git_commit", lambda: "test")
    contexts = []

    def evaluate(*args, **kwargs):
        contexts.append(kwargs["context"])
        return SimpleNamespace(
            blue_returns=[0.0],
            red_returns=[0.0],
            episode_seeds=[7],
            policies={},
            topology_paths=[],
            episode_topology_paths=[],
            topology_sampling="generative",
            cia_summary=None,
        )

    monkeypatch.setattr(matchup_runner, "evaluate_matchup", evaluate)
    run_cross_play(
        tmp_path / "model.safetensors",
        _recipe(),
        output=tmp_path / "out.jsonl",
        attach_metrics_fn=lambda *_a, **_k: None,
    )
    assert len(contexts) == 4
    assert all(context is contexts[0] for context in contexts)


@pytest.mark.parametrize("final_steps", [120, 140])
def test_include_final_keeps_exact_final_blue_against_prior_reds(tmp_path, monkeypatch, final_steps):
    from jaxborg.evaluation import cross_play

    monkeypatch.setattr(cross_play, "find_periodic_checkpoints", lambda *_a: _checkpoints(3))
    monkeypatch.setattr(cross_play, "_git_commit", lambda: "test")
    final = tmp_path / "model_final.safetensors"
    final.touch()
    recipe = _recipe(cross_play={"include_final": True, "max_checkpoints": 3})
    recipe["run"]["total_steps"] = final_steps
    calls = []

    def evaluate(blue, red, **kwargs):
        calls.append((blue, red))
        return SimpleNamespace(
            blue_returns=[0.0],
            red_returns=[0.0],
            episode_seeds=[1000],
            policies={},
            topology_paths=[],
            topology_sampling="exhaustive",
            cia_summary=None,
        )

    output = run_cross_play(
        final, recipe, output=tmp_path / "out.jsonl", evaluate_fn=evaluate, attach_metrics_fn=lambda *a, **kw: None
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(calls) == 9
    final_rows = [row for row in rows[:-1] if row["blue_checkpoint"] == str(final)]
    assert len(final_rows) == 3
    assert sum(row["red_step"] < final_steps for row in final_rows) == 2
    assert rows[-1]["checkpoint_steps"][-1] == final_steps
    if final_steps == 120:
        assert all(red.name != "checkpoint_120.safetensors" for _, red in calls)


def test_include_final_rejects_missing_step_provenance(tmp_path, monkeypatch):
    from jaxborg.evaluation import cross_play

    monkeypatch.setattr(cross_play, "find_periodic_checkpoints", lambda *_a: _checkpoints(3))
    with pytest.raises(ValueError, match="run.total_steps"):
        run_cross_play(tmp_path / "model_final.safetensors", _recipe(cross_play={"include_final": True}))
    with pytest.raises(ValueError, match="include_final must be a boolean"):
        CrossPlaySettings.from_recipe(_recipe(cross_play={"include_final": "true"}))
