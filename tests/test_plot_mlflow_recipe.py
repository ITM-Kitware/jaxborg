from __future__ import annotations

from types import SimpleNamespace

import pytest

from plots.plot_mlflow import (
    MetricRequest,
    _series_color,
    collect_history_rows,
    default_experiment_dir,
    default_tracking_uri,
    find_runs,
    render,
    resolve_metric_requests,
    settings_from_recipe,
)


@pytest.fixture
def experiment_root(tmp_path, monkeypatch):
    monkeypatch.setattr("plots.plot_mlflow.REPO_ROOT", tmp_path)
    monkeypatch.delenv("JAXBORG_EXP_DIR", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return tmp_path


def test_default_store_falls_back_to_synced_runs_from_any_working_directory(experiment_root, monkeypatch):
    remote = experiment_root / "remote" / "jaxborg-exp"
    remote.mkdir(parents=True)
    (remote / "mlflow.db").touch()
    (experiment_root / "jaxborg-exp").mkdir()
    elsewhere = experiment_root / "plots"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert default_experiment_dir() == remote
    assert default_tracking_uri() == f"sqlite:///{remote / 'mlflow.db'}"


def test_default_store_prefers_local_runs(experiment_root):
    for root in (experiment_root / "jaxborg-exp", experiment_root / "remote" / "jaxborg-exp"):
        root.mkdir(parents=True)
        (root / "mlflow.db").touch()

    assert default_experiment_dir() == experiment_root / "jaxborg-exp"


def test_explicit_store_is_not_silently_replaced(experiment_root, monkeypatch):
    remote = experiment_root / "remote" / "jaxborg-exp"
    remote.mkdir(parents=True)
    (remote / "mlflow.db").touch()
    configured = experiment_root / "custom"
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(configured))

    assert default_experiment_dir() == configured
    with pytest.raises(FileNotFoundError, match="custom/mlflow.db"):
        default_tracking_uri()
    assert default_tracking_uri(remote) == f"sqlite:///{remote / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.example.test")
    assert default_tracking_uri() == "https://tracking.example.test"


def test_missing_default_store_reports_both_locations_without_creating_database(experiment_root):
    with pytest.raises(FileNotFoundError, match="remote/jaxborg-exp/mlflow.db"):
        default_tracking_uri()
    assert not (experiment_root / "jaxborg-exp").exists()


def _run(run_id, *, recipe="demo", backend="jax", seed=0, status="FINISHED", metrics=(), params=None):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, status=status),
        data=SimpleNamespace(
            tags={"recipe.name": recipe, "backend": backend, "seed": str(seed)},
            params=params or {},
            metrics={key: 0.0 for key in metrics},
        ),
    )


def _point(step, value, timestamp):
    return SimpleNamespace(step=step, value=value, timestamp=timestamp)


class _Client:
    def __init__(self, runs, histories=None):
        self.runs = list(runs)
        self.histories = histories or {}

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="experiment-1") if name == "ippo-cc4" else None

    def search_runs(self, **kwargs):
        assert kwargs["experiment_ids"] == ["experiment-1"]
        return self.runs

    def get_run(self, run_id):
        return next(run for run in self.runs if run.info.run_id == run_id)

    def get_metric_history(self, run_id, key):
        return self.histories.get((run_id, key), [])


def test_recipe_settings_and_metric_globs_are_resolved():
    settings = settings_from_recipe(
        {
            "plots": {
                "metrics": [
                    {"key": "team.*.return", "panel": "Returns"},
                    {"key": "loss_value", "label": "Critic", "panel": "Loss"},
                ],
                "smoothing": 3,
                "confidence": None,
                "formats": ["svg"],
                "group_by": ["backend", "param:recipe.arch.name"],
            }
        }
    )
    specs = resolve_metric_requests(
        settings.metrics,
        ["loss_value", "team.blue.return", "team.red.return"],
    )

    assert settings.smoothing == 3
    assert settings.confidence is None
    assert settings.formats == ("svg",)
    assert settings.group_by == ("tag:backend", "param:recipe.arch.name")
    assert [(spec.key, spec.label, spec.panel) for spec in specs] == [
        ("team.blue.return", "Team · Blue · Return", "Returns"),
        ("team.red.return", "Team · Red · Return", "Returns"),
        ("loss_value", "Critic", "Loss"),
    ]


def test_default_dashboard_includes_all_metrics_and_separates_evaluation():
    keys = [
        "team.blue.return",
        "team.red.return",
        "train_episode_reward_mean",
        "eval.checkpoint.blue.mean_reward",
        "eval.checkpoint.red.mean_reward",
        "eval.checkpoint_scripted_reds.blue.worst_reward",
        "eval.checkpoint_scripted_reds.SleepAgent.blue.mean_reward",
        "eval.cross_play.blue.forgetting_rate",
        "eval.checkpoint.cia.resilience",
        "team.blue.ppo_kl_divergence",
        "team.red.ppo_kl_divergence",
        "team.blue.reward_asf",
        "lr",
        "env_steps",
        "wall_time_sec",
        "steps_per_second",
        "eval.episodes",
        "eval.std_reward",
        "custom_metric",
    ]
    specs = resolve_metric_requests([], keys)
    by_key = {spec.key: spec for spec in specs}

    assert set(by_key) == set(keys)
    assert len(specs) == len(keys)
    assert by_key["team.blue.return"].panel == by_key["team.red.return"].panel == "Training return"
    assert by_key["eval.checkpoint.blue.mean_reward"].panel == "Evaluation return"
    assert by_key["eval.checkpoint.red.mean_reward"].panel == "Evaluation return"
    assert by_key["eval.checkpoint_scripted_reds.blue.worst_reward"].panel != "Evaluation return"
    assert by_key["team.blue.ppo_kl_divergence"].panel == by_key["team.red.ppo_kl_divergence"].panel
    assert by_key["team.red.return"].label == "Red"


def test_automatic_unknown_metrics_have_no_six_metric_limit():
    keys = [f"custom_{index}" for index in range(10)]
    assert {spec.key for spec in resolve_metric_requests([], keys)} == set(keys)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("team.red.return", "#c62828"),
        ("eval.checkpoint.red.mean_reward", "#c62828"),
        ("jax.game.red_return", "#c62828"),
        ("team.blue.return", "#2563a6"),
        ("eval.scripted_red.aggressive.blue.mean_reward", "#2563a6"),
    ],
)
def test_team_colors_follow_metric_even_with_custom_labels(metric, expected):
    assert _series_color(metric, "Custom label", 0) == expected
    assert _series_color(metric, "Custom label", 3) == expected


def test_find_runs_filters_recipe_backend_seed_and_status():
    runs = [
        _run("keep", backend="jax", seed=42),
        _run("wrong-recipe", recipe="other", backend="jax", seed=42),
        _run("wrong-backend", backend="cyborg", seed=42),
        _run("wrong-seed", backend="jax", seed=43),
        _run("failed", backend="jax", seed=42, status="FAILED"),
    ]

    selected = find_runs(
        _Client(runs),
        experiment_name="ippo-cc4",
        recipe_name="demo",
        backends={"jax"},
        seeds={42},
    )

    assert [run.info.run_id for run in selected] == ["keep"]


def test_collect_history_rows_deduplicates_steps_smooths_per_run_and_groups():
    runs = [
        _run("jax-1", backend="jax", seed=1, metrics=["reward"]),
        _run("torch-2", backend="cyborg", seed=2, metrics=["reward"]),
    ]
    histories = {
        ("jax-1", "reward"): [_point(10, 1.0, 1), _point(20, 3.0, 2), _point(20, 5.0, 3)],
        ("torch-2", "reward"): [_point(10, 2.0, 1), _point(20, 6.0, 2)],
    }

    rows = collect_history_rows(
        _Client(runs, histories),
        runs,
        resolve_metric_requests([MetricRequest("reward", label="Reward", panel="Return")], ["reward"]),
        smoothing=2,
        group_by=("tag:backend",),
    )

    assert [(row["run_id"], row["step"], row["value"], row["series"]) for row in rows] == [
        ("jax-1", 10, 1.0, "Reward · jax"),
        ("jax-1", 20, 3.0, "Reward · jax"),
        ("torch-2", 10, 2.0, "Reward · cyborg"),
        ("torch-2", 20, 4.0, "Reward · cyborg"),
    ]


def test_render_writes_paper_figure(tmp_path, monkeypatch):
    from matplotlib.colors import to_hex
    from matplotlib.figure import Figure

    saved_figures = []
    savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        saved_figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    rows = [
        {
            "panel": "Return",
            "metric": f"team.{team}.return",
            "series": f"{team.title()} · {backend}",
            "run_id": run_id,
            "run": f"jax, seed {seed}",
            "step": step,
            "value": value,
        }
        for team in ("red", "blue")
        for backend in ("jax", "cyborg")
        for run_id, seed, values in (("one", 1, (1.0, 2.0)), ("two", 2, (2.0, 3.0)))
        for step, value in zip((10, 20), values, strict=True)
    ]

    outputs = render(
        rows,
        tmp_path / "figure",
        title="Demo",
        columns=2,
        confidence=95,
        formats=("png", "pdf"),
        show_runs=True,
    )

    assert [path.suffix for path in outputs] == [".png", ".pdf"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    figure = saved_figures[0]
    assert figure._suptitle.get_position()[0] == 0.5
    assert figure._suptitle.get_horizontalalignment() == "center"
    axis = figure.axes[0]
    assert axis.get_title(loc="center") == "Return"
    assert axis.get_title(loc="left") == ""
    assert all(spine.get_visible() for spine in axis.spines.values())
    legend_lines = {line.get_label(): line for line in axis.lines if " · " in line.get_label()}
    assert to_hex(legend_lines["Red · jax"].get_color()) == "#c62828"
    assert to_hex(legend_lines["Blue · jax"].get_color()) == "#2563a6"
    assert legend_lines["Red · jax"].get_linestyle() != legend_lines["Red · cyborg"].get_linestyle()


def test_all_metrics_overrides_recipe_selection(tmp_path, monkeypatch):
    import mlflow

    from plots import plot_mlflow

    keys = ["team.blue.return", "eval.checkpoint.blue.mean_reward", "steps_per_second"]
    client = _Client(
        [_run("one", metrics=keys)],
        {("one", key): [_point(10, 1.0, 1)] for key in keys},
    )
    monkeypatch.setattr(mlflow, "MlflowClient", lambda **kwargs: client)
    monkeypatch.setattr(
        plot_mlflow,
        "load_recipe",
        lambda _: {"meta": {"name": "demo"}, "plots": {"metrics": ["team.blue.return"]}},
    )
    rendered_rows = []

    def capture_render(rows, *args, **kwargs):
        rendered_rows.extend(rows)
        return []

    monkeypatch.setattr(plot_mlflow, "render", capture_render)
    plot_mlflow.main(["demo", "--all-metrics", "--tracking-uri", "unused", "--output-dir", str(tmp_path)])

    assert {row["metric"] for row in rendered_rows} == set(keys)


@pytest.mark.parametrize(
    ("plots", "message"),
    [
        ({"smoothing": 0}, "smoothing"),
        ({"confidence": 100}, "confidence"),
        ({"formats": ["jpg"]}, "formats"),
        ({"group_by": ["recipe.arch.name"]}, "group_by"),
    ],
)
def test_invalid_plot_settings_are_rejected(plots, message):
    with pytest.raises(ValueError, match=message):
        settings_from_recipe({"plots": plots})
