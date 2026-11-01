from __future__ import annotations

import mlflow
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from mlflow.entities import Metric

from plots.plot_mlflow import load_figure_file, main, resolve_tracking_uri


@pytest.fixture
def experiment_root(tmp_path, monkeypatch):
    monkeypatch.setattr("plots.plot_mlflow.REPO_ROOT", tmp_path)
    monkeypatch.delenv("JAXBORG_EXP_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_default_store_uses_current_directory_or_explicit_remote_mirror(experiment_root):
    local = experiment_root / "jaxborg-exp"
    remote = experiment_root / "remote" / "jaxborg-exp"
    for root in (local, remote):
        root.mkdir(parents=True)
        (root / "mlflow.db").touch()

    assert resolve_tracking_uri(None, remote=False) == f"sqlite:///{local / 'mlflow.db'}"
    assert resolve_tracking_uri(None, remote=True) == f"sqlite:///{remote / 'mlflow.db'}"


def test_explicit_store_and_tracking_uri_take_precedence(experiment_root, monkeypatch):
    configured = experiment_root / "custom"
    configured.mkdir()
    monkeypatch.setenv("JAXBORG_EXP_DIR", str(configured))
    with pytest.raises(FileNotFoundError, match="custom/mlflow.db"):
        resolve_tracking_uri(None, remote=False)
    (configured / "mlflow.db").touch()

    assert resolve_tracking_uri(None, remote=False) == f"sqlite:///{configured / 'mlflow.db'}"
    assert resolve_tracking_uri("https://tracking.example.test", remote=True) == "https://tracking.example.test"


@pytest.mark.parametrize("remote", [False, True])
def test_missing_store_does_not_create_database(experiment_root, remote):
    with pytest.raises(FileNotFoundError, match="MLflow database not found"):
        resolve_tracking_uri(None, remote=remote)
    assert not list(experiment_root.rglob("mlflow.db"))


@pytest.fixture
def tracking_uri(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("ippo-cc4")
    for recipe, backend, seed, status, offset in (
        ("alpha", "jax", 1, "FINISHED", 0),
        ("beta", "jax", 1, "FINISHED", 10),
        ("alpha", "cyborg", 1, "FINISHED", 100),
        ("alpha", "jax", 2, "FINISHED", 200),
        ("failed", "jax", 1, "FAILED", 300),
    ):
        run = client.create_run(
            experiment_id,
            tags={"recipe.name": recipe, "backend": backend, "seed": str(seed)},
        )
        metrics = []
        for team, sign in (("blue", 1), ("red", -1)):
            metrics.extend(
                Metric(f"team.{team}.return", sign * (value + offset), timestamp, step)
                for step, value, timestamp in ((10, 1.0, 1), (20, 3.0, 2), (20, 5.0, 3), (30, 9.0, 4))
            )
        metrics.append(Metric("custom_metric", offset + 1.0, 1, 0))
        client.log_batch(run.info.run_id, metrics=metrics)
        client.set_terminated(run.info.run_id, status=status)
    return uri


@pytest.fixture
def saved_figures(monkeypatch):
    figures = []
    savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return figures


def test_recipe_file_renders_globs_filters_runs_and_smooths_histories(tmp_path, tracking_uri, saved_figures):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """plots:
  title: Return comparison
  metrics:
    - key: team.*.return
      panel: Training return
  smoothing: 2
  band: none
  confidence: 90
  formats: [svg]
"""
    )
    out = tmp_path / "plots"
    assert (
        main(
            [
                "ippo-cc4",
                "--tracking-uri",
                tracking_uri,
                "--figures",
                str(recipe),
                "--out",
                str(out),
                "--backend",
                "jax",
                "--seed",
                "1",
                "--label",
                "alpha=Alpha run",
            ]
        )
        == 0
    )

    output = out / "recipe.svg"
    assert output.is_file() and output.stat().st_size > 0
    assert sorted(path.name for path in out.iterdir()) == ["recipe.svg"]
    figure = saved_figures[0]
    assert figure.get_suptitle() == "Return comparison"
    axis = figure.axes[0]
    assert axis.get_title(loc="left") == "Training return"
    assert len(axis.lines) == 4
    for line, values, color, style in zip(
        axis.lines,
        ([1, 3, 7], [11, 13, 17], [-1, -3, -7], [-11, -13, -17]),
        ("#0077bb", "#0077bb", "#cc3311", "#cc3311"),
        ("-", "--", "-", "--"),
        strict=True,
    ):
        np.testing.assert_allclose(line.get_xdata(), [10, 20, 30])
        np.testing.assert_allclose(line.get_ydata(), values)
        assert to_hex(line.get_color()) == color
        assert line.get_linestyle() == style
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "Alpha run",
        "beta",
        "Blue agent",
        "Red agent",
    ]


def test_metric_cli_selection_overrides_recipe_file(tmp_path, tracking_uri, saved_figures):
    out = tmp_path / "plots"
    assert (
        main(
            [
                "--tracking-uri",
                tracking_uri,
                "--figures",
                str(tmp_path / "unused.yaml"),
                "--metric",
                "custom_*",
                "--name",
                "custom",
                "--recipe",
                "alpha",
                "--backend",
                "jax",
                "--seed",
                "1",
                "--formats",
                "png",
                "pdf",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    assert sorted(path.name for path in out.iterdir()) == ["custom.pdf", "custom.png"]
    assert all(path.stat().st_size > 0 for path in out.iterdir())
    axis = saved_figures[0].axes[0]
    assert axis.get_title(loc="left") == "custom_*"
    assert len(axis.patches) == 1
    assert axis.patches[0].get_height() == 1.0


def test_figure_file_supports_multiple_named_figures_and_options(tmp_path):
    path = tmp_path / "figures.yaml"
    path.write_text(
        """figures:
  - name: returns
    title: Team returns
    metrics:
      - key: team.*.return
        panel: Return
  - name: loss
    panels: [loss_value]
smoothing: 3
band: std
formats: [pdf]
"""
    )
    figures, options = load_figure_file(path)

    assert [(figure.name, figure.title) for figure in figures] == [("returns", "Team returns"), ("loss", None)]
    assert [(panel.pattern, panel.title) for panel in figures[0].panels] == [("team.*.return", "Return")]
    assert figures[1].panels[0].pattern == "loss_value"
    assert options == {"smoothing": 3, "band": "std", "formats": ["pdf"]}


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("- not-a-mapping", "must contain a YAML mapping"),
        ("metrics: []", "top-level 'figures:' list"),
        ("figures: [invalid]", "figures entries must be mappings"),
        ("plots: {metrics: [{panel: Missing key}]}", "figure metric entries"),
        ("plots: {metrics: [{key: loss_value, team: green}]}", "team must be"),
    ],
)
def test_invalid_figure_files_are_rejected(tmp_path, document, message):
    path = tmp_path / "invalid.yaml"
    path.write_text(document)
    with pytest.raises(ValueError, match=message):
        load_figure_file(path)
