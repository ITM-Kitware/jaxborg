import contextlib
import io

import mlflow
import numpy as np
import pytest

from plots.plot_mlflow import (
    BandSettings,
    RunInfo,
    aggregate_curves,
    aggregate_scalars,
    figures_from_mapping,
    grid_columns,
    main,
    match_panel_keys,
    pretty_series,
    select_runs,
    series_colors,
    series_team,
    std_companion,
)


def test_std_companion_pairs_both_naming_conventions():
    assert std_companion("eval.x.cia.c.mean") == "eval.x.cia.c.std"
    assert std_companion("eval.x.blue.mean_reward") == "eval.x.blue.std_reward"
    assert std_companion("mean_reward") == "std_reward"
    assert std_companion("team.blue.return") is None
    assert std_companion("eval.x.cia.c.std") is None


def test_match_panel_keys_folds_std_and_labels_varying_segments():
    keys = {
        "eval.pp.blue_vs_prior_red.cia.c.mean",
        "eval.pp.blue_vs_prior_red.cia.c.std",
        "eval.pp.blue_vs_prior_red.cia.i.mean",
        "eval.pp.blue_vs_prior_red.cia.i.std",
        "eval.pp.blue_vs_prior_red.cia.a.mean",
        "eval.pp.blue_vs_prior_red.cia.a.std",
        "eval.pp.blue_vs_prior_red.mean_reward",
    }
    series = match_panel_keys("eval.pp.*.cia.*.mean", keys)
    assert [s.label for s in series] == ["c", "i", "a"]  # CIA order, not alphabetical
    assert all(s.std_key == s.key[:-4] + "std" for s in series)

    # A glob that also matches the std keys still yields one series per mean.
    series = match_panel_keys("eval.pp.blue_vs_prior_red.cia.c.*", keys)
    assert [(s.key, s.label) for s in series] == [("eval.pp.blue_vs_prior_red.cia.c.mean", None)]

    # Explicitly asking for a std key keeps it as an ordinary metric.
    assert [s.key for s in match_panel_keys("*.cia.i.std", keys)] == ["eval.pp.blue_vs_prior_red.cia.i.std"]
    assert match_panel_keys("missing.*", keys) == []


def test_series_team_detection_and_override():
    assert series_team("team.blue.return") == "blue"
    assert series_team("eval.play_priors.red_vs_prior_blue.cia.c.mean") == "red"
    assert series_team("eval.after_training.scripted-reds.scripted_red.fsm.blue.mean_reward") == "blue"
    assert series_team("eval.after_training.learned-red-ppo.jax_matchup.red_mean") == "red"
    assert series_team("loss_entropy") is None
    keys = {"train_episode_reward_mean", "eval.cross_play.red.forgetting_rate"}
    assert match_panel_keys("train_episode_reward_mean", keys, team="blue")[0].team == "blue"
    assert match_panel_keys("eval.cross_play.*.forgetting_rate", keys)[0].team == "red"

    figures = figures_from_mapping(
        {"figures": [{"name": "f", "metrics": [{"key": "loss_value", "panel": "Loss", "team": "blue"}]}]},
        default_name="x",
    )
    assert figures[0].panels[0].team == "blue"
    with pytest.raises(ValueError, match="team must be"):
        figures_from_mapping({"plots": {"metrics": [{"key": "a", "team": "green"}]}}, default_name="x")


def test_series_colors_by_team_with_shading_only_for_curves():
    keys = {"eval.cp.blue.mean_vs_history", "eval.cp.blue.worst_vs_history", "eval.cp.red.mean_vs_history"}
    series = match_panel_keys("eval.cp.*_vs_history", keys)
    shaded = series_colors(series)
    flat = series_colors(series, shade=False)
    blue = [s for s in series if s.team == "blue"]
    assert len({tuple(shaded[s.key]) for s in blue}) == 2  # two blue series get two shades
    assert len({flat[s.key] for s in blue}) == 1  # bars share the base team color
    assert flat[[s for s in series if s.team == "red"][0].key] != flat[blue[0].key]


def test_pretty_series_labels():
    assert pretty_series("c") == "Confidentiality"
    assert pretty_series("blue_vs_prior_red") == "Blue vs prior Red"
    assert pretty_series("cia_c") == "CIA-C"


def _run(recipe, seed, start, status="FINISHED", keys=("m",)):
    run_id, name = f"{recipe}-{seed}-{start}", f"{recipe}-seed{seed}"
    return RunInfo(run_id, name, recipe, "jax", str(seed), status, start, frozenset(keys))


def test_select_runs_keeps_newest_duplicate_and_filters():
    runs = [_run("a", 42, 1), _run("a", 42, 5), _run("a", 100, 2), _run("b", 42, 3), _run("b", 7, 4, status="FAILED")]
    kept, notes = select_runs(runs)
    assert [(r.recipe, r.seed, r.start_time) for r in kept] == [("a", "42", 5), ("a", "100", 2), ("b", "42", 3)]
    assert len(notes) == 2  # one duplicate dropped, one failed run skipped

    kept, _ = select_runs(runs, keep_duplicates=True, include_failed=True)
    assert len(kept) == 5
    kept, _ = select_runs(runs, recipes=["b"], include_failed=True)
    assert {r.recipe for r in kept} == {"b"}
    kept, _ = select_runs(runs, seeds=["100"])
    assert [r.seed for r in kept] == ["100"]


def test_aggregate_curves_uses_pooled_episode_std_when_paired():
    steps = np.array([0.0, 10.0, 20.0])
    runs = [
        (steps, np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])),
        (steps, np.array([3.0, 4.0, 5.0]), np.array([2.0, 2.0, 2.0])),
    ]
    curve = aggregate_curves(runs, BandSettings())
    np.testing.assert_allclose(curve.mean, [2.0, 3.0, 4.0])
    # sqrt(mean(std^2) + population variance of the means) = sqrt(2.5 + 1)
    np.testing.assert_allclose(curve.half_width, np.sqrt(3.5))
    assert curve.n_runs == 2


def test_aggregate_curves_band_needs_two_runs_and_handles_ragged_steps():
    single = aggregate_curves([(np.array([0.0, 1.0]), np.array([1.0, 2.0]), None)], BandSettings(kind="std"))
    assert np.all(np.isnan(single.half_width))

    runs = [
        (np.array([0.0, 10.0, 20.0]), np.array([0.0, 10.0, 20.0]), None),
        (np.array([0.0, 20.0, 30.0]), np.array([2.0, 22.0, 32.0]), None),
    ]
    curve = aggregate_curves(runs, BandSettings(kind="std"))
    np.testing.assert_allclose(curve.steps, [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(curve.mean, [1.0, 11.0, 21.0, 32.0])  # 30 only covered by run 2
    assert np.isnan(curve.half_width[-1])
    np.testing.assert_allclose(curve.half_width[:3], np.sqrt(2.0))


def test_aggregate_scalars_ci_and_paired_std():
    bar = aggregate_scalars([1.0, 2.0, 3.0], None, BandSettings(kind="ci", confidence=95))
    assert bar.mean == pytest.approx(2.0)
    assert bar.error == pytest.approx(4.302652729911275 * 1.0 / np.sqrt(3))
    paired = aggregate_scalars([1.0, 3.0], [1.0, 1.0], BandSettings())
    assert paired.error == pytest.approx(np.sqrt(1.0 + 1.0))
    assert np.isnan(aggregate_scalars([5.0], None, BandSettings()).error)


def test_grid_columns_prefers_balanced_layouts():
    assert grid_columns(4, 3) == 2
    assert grid_columns(6, 3) == 3
    assert grid_columns(5, 3) == 3
    assert grid_columns(1, 3) == 1


def _seed_tracking_db(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("ippo-cc4")
    for recipe in ("alpha", "beta"):
        for seed in (1, 2):
            with mlflow.start_run(run_name=f"ippo-jax-{recipe}-seed{seed}"):
                mlflow.set_tags({"recipe.name": recipe, "backend": "jax", "seed": str(seed)})
                for step in range(0, 500, 100):
                    mlflow.log_metric("train_episode_reward_mean", -step + seed, step=step)
                    mlflow.log_metric("team.red.return", step, step=step)
                    mlflow.log_metrics(
                        {
                            "eval.play_priors.blue_vs_prior_red.mean_reward": -1000.0 - step,
                            "eval.play_priors.blue_vs_prior_red.cia.c.mean": -1.0 - seed,
                            "eval.play_priors.blue_vs_prior_red.cia.c.std": 0.5,
                        },
                        step=step,
                    )
                mlflow.log_metrics(
                    {
                        "eval.after_training.scripted-reds.scripted_red.fsm.blue.mean_reward": -100.0 * seed,
                        "eval.after_training.scripted-reds.scripted_red.fsm.blue.std_reward": 10.0,
                        "eval.after_training.scripted-reds.scripted_red.cia_c.blue.mean_reward": -50.0,
                        "eval.after_training.scripted-reds.scripted_red.cia_c.blue.std_reward": 5.0,
                        "eval.after_training.scripted-reds.scripted_red.fsm.blue.cia.c.mean": -2.0,
                        "eval.after_training.scripted-reds.scripted_red.fsm.blue.cia.c.std": 1.0,
                    }
                )
    mlflow.set_tracking_uri("")
    return uri


def test_main_renders_default_and_custom_figures(tmp_path):
    uri = _seed_tracking_db(tmp_path)
    out = tmp_path / "plots"
    assert main(["ippo-cc4", "--tracking-uri", uri, "--out", str(out), "--formats", "png"]) == 0
    assert (out / "training.png").is_file()
    assert (out / "play_priors.png").is_file()  # reward only; CIA lives in its own figures
    assert (out / "cia.png").is_file()
    assert (out / "after_training.png").is_file()
    assert (out / "cia_after_training.png").is_file()
    assert not (out / "cross_play.png").exists()

    assert (
        main(
            [
                "ippo-cc4",
                "--tracking-uri",
                uri,
                "--out",
                str(out),
                "--formats",
                "pdf",
                "--metric",
                "team.*.return",
                "--name",
                "custom",
                "--recipe",
                "alpha",
                "--label",
                "alpha=Alpha run",
                "--smoothing",
                "3",
                "--band",
                "std",
            ]
        )
        == 0
    )
    assert (out / "custom.pdf").is_file()

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert main(["ippo-cc4", "--tracking-uri", uri, "--list-metrics"]) == 0
    listing = buffer.getvalue()
    assert "eval.play_priors.blue_vs_prior_red.cia.c.mean" in listing
    assert "(std paired)" in listing

    with pytest.raises(ValueError, match="not found"):
        main(["nope", "--tracking-uri", uri, "--out", str(out)])
