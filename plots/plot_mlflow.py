#!/usr/bin/env python3
"""Export paper-style comparison figures from an MLflow experiment.

Every run in the requested MLflow experiment (``ippo-cc4`` by default) is
grouped by its ``recipe.name`` tag, seeds are aggregated into a mean with a
band or error bar, and one figure is written per figure group under
``$JAXBORG_EXP_DIR/plots/<experiment>/``.

Viewability corrections applied automatically:

* ``<key>.mean`` / ``<key>.std`` and ``mean_<x>`` / ``std_<x>`` pairs are drawn
  as a single mean series with the recorded std as the band or error bar.
* Metrics logged once per run become bar panels; step histories become curves.
* Color encodes the team (Blue agent, Red agent, gray for team-less
  diagnostics); line style and bar hatching encode the recipe.
* When the same recipe/backend/seed was trained more than once only the most
  recent run is kept (``--keep-duplicates`` disables this).

Examples::

    JAXBORG_EXP_DIR=./jaxborg-exp uv run python plots/plot_mlflow.py
    uv run python plots/plot_mlflow.py ippo-cc4 --remote --list-metrics
    uv run python plots/plot_mlflow.py ippo-cc4 --remote \\
        --metric 'team.*.return' --metric 'eval.play_priors.*.cia.*.mean'
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = "ippo-cc4"
LINESTYLES = ("-", "--", ":", "-.")
HATCHES = ("", "///", "...", "xx")
MAX_CURVE_POINTS = 400
TEAMS = ("blue", "red")
# Paul Tol's "vibrant" blue/red pair: distinguishable under common color-vision deficiencies.
TEAM_COLORS = {"blue": "#0077bb", "red": "#cc3311"}
NEUTRAL_COLOR = "#555555"

# Tokens that appear as wildcard matches in metric keys, in the order they
# should be listed and with the label they should carry.
TOKEN_ORDER = ("blue", "red", "c", "i", "a", "fsm", "cia_c", "cia_i", "cia_a")
TOKEN_LABELS = {
    "c": "Confidentiality",
    "i": "Integrity",
    "a": "Availability",
    "blue": "Blue",
    "red": "Red",
    "fsm": "FSM",
    "cia_c": "CIA-C",
    "cia_i": "CIA-I",
    "cia_a": "CIA-A",
    "vs": "vs",
    "prior": "prior",
    "mean": "mean",
    "worst": "worst",
    "history": "history",
    "self": "self",
    "play": "play",
}


# --------------------------------------------------------------------------- #
# Figure specifications
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PanelSpec:
    """One axes: every metric key matching ``pattern`` becomes a series.

    ``team`` forces the color of series whose key does not name a team (for
    example ``train_episode_reward_mean``, which is the Blue return).
    """

    pattern: str
    title: str
    team: str | None = None


@dataclass(frozen=True)
class FigureSpec:
    name: str
    title: str | None
    panels: tuple[PanelSpec, ...]


DEFAULT_FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        "training",
        None,
        (
            PanelSpec("train_episode_reward_mean", "Blue training return", team="blue"),
            PanelSpec("team.red.return", "Red training return"),
            PanelSpec("loss_entropy", "Policy entropy"),
            PanelSpec("ppo_kl_divergence", "Approximate KL"),
            PanelSpec("loss_value", "Value loss"),
            PanelSpec("ppo_explained_variance", "Explained variance"),
        ),
    ),
    FigureSpec(
        "checkpoint_eval",
        None,
        (
            PanelSpec("eval.checkpoint.*.mean_reward", "Checkpoint reward"),
            PanelSpec("eval.checkpoint_scripted_reds.blue.*reward", "Blue reward vs scripted Reds"),
        ),
    ),
    FigureSpec(
        "play_priors",
        None,
        (PanelSpec("eval.play_priors.*.mean_reward", "Reward vs prior opponent"),),
    ),
    FigureSpec(
        "cross_play",
        None,
        (
            PanelSpec("eval.cross_play.*.mean_vs_history", "Mean reward vs checkpoint history"),
            PanelSpec("eval.cross_play.*.worst_vs_history", "Worst reward vs checkpoint history"),
            PanelSpec("eval.cross_play.blue.self_play", "Blue self-play"),
            PanelSpec("eval.cross_play.*.forgetting_rate", "Forgetting rate"),
            PanelSpec("eval.cross_play.*.worst_vs_history_gain", "Worst-vs-history gain"),
        ),
    ),
    FigureSpec(
        "after_training",
        None,
        (
            PanelSpec("eval.after_training.*.scripted_red.*.blue.mean_reward", "Blue reward vs scripted Red"),
            PanelSpec("eval.after_training.*.jax_matchup.*_mean", "Learned matchup return"),
        ),
    ),
    FigureSpec(
        "cia",
        None,
        (
            PanelSpec("eval.play_priors.*.cia.c.mean", "Confidentiality vs prior opponent"),
            PanelSpec("eval.play_priors.*.cia.i.mean", "Integrity vs prior opponent"),
            PanelSpec("eval.play_priors.*.cia.a.mean", "Availability vs prior opponent"),
            PanelSpec("eval.checkpoint.cia.*.mean", "Checkpoint CIA", team="blue"),
        ),
    ),
    FigureSpec(
        "cia_after_training",
        None,
        (
            PanelSpec("eval.after_training.*.scripted_red.*.blue.cia.c.mean", "Confidentiality vs scripted Red"),
            PanelSpec("eval.after_training.*.scripted_red.*.blue.cia.i.mean", "Integrity vs scripted Red"),
            PanelSpec("eval.after_training.*.scripted_red.*.blue.cia.a.mean", "Availability vs scripted Red"),
            PanelSpec("eval.after_training.*.jax_matchup.cia.*.mean", "Learned matchup CIA", team="blue"),
        ),
    ),
)


def figures_from_mapping(document: Mapping[str, Any], *, default_name: str) -> list[FigureSpec]:
    """Parse ``{figures: [...]}`` or a recipe-style ``{plots: {...}}`` block."""

    def _panels(entries: Iterable[Any]) -> tuple[PanelSpec, ...]:
        panels = []
        for entry in entries:
            if isinstance(entry, str):
                panels.append(PanelSpec(entry, entry))
            elif isinstance(entry, Mapping) and "key" in entry:
                team = entry.get("team")
                if team is not None and team not in TEAMS:
                    raise ValueError(f"team must be one of {TEAMS}, got {team!r}")
                panels.append(PanelSpec(str(entry["key"]), str(entry.get("panel", entry["key"])), team))
            else:
                raise ValueError(f"figure metric entries must be a glob or {{key, panel}} mapping, got {entry!r}")
        return tuple(panels)

    if "figures" in document:
        figures = []
        for index, entry in enumerate(document["figures"]):
            if not isinstance(entry, Mapping):
                raise ValueError("figures entries must be mappings")
            metrics = entry.get("metrics", entry.get("panels", []))
            figures.append(FigureSpec(str(entry.get("name", f"figure{index}")), entry.get("title"), _panels(metrics)))
        return figures
    if "plots" in document and isinstance(document["plots"], Mapping):
        block = document["plots"]
        return [FigureSpec(default_name, block.get("title"), _panels(block.get("metrics", [])))]
    raise ValueError("figure file needs a top-level 'figures:' list or a 'plots:' block")


def load_figure_file(path: Path) -> tuple[list[FigureSpec], dict[str, Any]]:
    document = yaml.safe_load(path.read_text()) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    figures = figures_from_mapping(document, default_name=path.stem)
    settings = document.get("plots", document) if "plots" in document else document
    options = {k: settings[k] for k in ("smoothing", "confidence", "formats", "band") if k in settings}
    return figures, options


# --------------------------------------------------------------------------- #
# Metric key helpers
# --------------------------------------------------------------------------- #


def std_companion(key: str) -> str | None:
    """Return the std key paired with a mean key, or ``None``."""

    head, _, last = key.rpartition(".")
    prefix = f"{head}." if head else ""
    if last == "mean":
        return f"{prefix}std"
    if last.startswith("mean_"):
        return f"{prefix}std_{last[5:]}"
    return None


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    parts = []
    for char in pattern:
        if char == "*":
            parts.append("(.*?)")
        elif char == "?":
            parts.append("(.)")
        else:
            parts.append(re.escape(char))
    return re.compile("".join(parts))


def _token_rank(label: str) -> tuple[int, str]:
    if label in TOKEN_ORDER:
        return TOKEN_ORDER.index(label), label
    return len(TOKEN_ORDER), label


def pretty_series(label: str) -> str:
    if label in TOKEN_LABELS:
        return TOKEN_LABELS[label]
    return " ".join(TOKEN_LABELS.get(token, token) for token in label.split("_"))


def series_team(key: str) -> str | None:
    """Team a metric key belongs to: a whole ``blue``/``red`` segment wins, else a leading token."""

    segments = key.split(".")
    for segment in segments:
        if segment in TEAMS:
            return segment
    for segment in segments:
        head = segment.split("_", 1)[0]
        if head in TEAMS:
            return head
    return None


@dataclass(frozen=True)
class Series:
    key: str
    std_key: str | None
    label: str | None  # wildcard text, None when the pattern has no wildcard
    team: str | None = None


def match_panel_keys(pattern: str, keys: Iterable[str], *, team: str | None = None) -> list[Series]:
    """Expand a glob into series, folding ``std`` companions into their means."""

    regex = _pattern_regex(pattern)
    key_set = set(keys)
    matched: dict[str, tuple[str | None, tuple[str, ...]]] = {}
    for key in sorted(key_set):
        found = regex.fullmatch(key)
        if not found:
            continue
        std_key = std_companion(key)
        matched[key] = (std_key if std_key in key_set else None, found.groups())
    consumed = {std_key for std_key, _ in matched.values() if std_key}
    kept = {key: value for key, value in matched.items() if key not in consumed}
    if not kept:
        return []
    # Wildcard segments shared by every matched key carry no information, so
    # only the varying segments become the series label.
    groups = [value[1] for value in kept.values()]
    varying = [i for i in range(len(groups[0])) if len({g[i] for g in groups}) > 1]
    result = []
    for key, (std_key, captured) in kept.items():
        label = " / ".join(captured[i] for i in varying) if varying else None
        result.append(Series(key, std_key, label, team or series_team(key)))
    return sorted(result, key=lambda s: _token_rank(s.label or ""))


# --------------------------------------------------------------------------- #
# MLflow access
# --------------------------------------------------------------------------- #


@dataclass
class RunInfo:
    run_id: str
    name: str
    recipe: str
    backend: str
    seed: str
    status: str
    start_time: int
    metric_keys: frozenset[str]

    @property
    def group(self) -> tuple[str, str, str]:
        return (self.recipe, self.backend, self.seed)


def resolve_tracking_uri(tracking_uri: str | None, *, remote: bool) -> str:
    if tracking_uri:
        return tracking_uri
    if remote:
        db_path = REPO_ROOT / "remote" / "jaxborg-exp" / "mlflow.db"
    else:
        db_path = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve() / "mlflow.db"
    if not db_path.is_file():
        raise FileNotFoundError(
            f"MLflow database not found: {db_path}. Set JAXBORG_EXP_DIR, pass --remote, or pass --tracking-uri."
        )
    return f"sqlite:///{db_path}"


def load_runs(client: Any, experiment_name: str) -> list[RunInfo]:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        names = sorted(e.name for e in client.search_experiments())
        raise ValueError(f"experiment {experiment_name!r} not found; available: {names}")
    runs: list[RunInfo] = []
    token = None
    while True:
        page = client.search_runs([experiment.experiment_id], max_results=1000, page_token=token)
        for run in page:
            tags = run.data.tags
            runs.append(
                RunInfo(
                    run_id=run.info.run_id,
                    name=run.info.run_name or tags.get("mlflow.runName", run.info.run_id),
                    recipe=tags.get("recipe.name", "unnamed"),
                    backend=tags.get("backend", "unknown"),
                    seed=tags.get("seed", "?"),
                    status=run.info.status,
                    start_time=run.info.start_time or 0,
                    metric_keys=frozenset(run.data.metrics),
                )
            )
        token = getattr(page, "token", None)
        if not token:
            break
    return runs


def select_runs(
    runs: Sequence[RunInfo],
    *,
    recipes: Sequence[str] = (),
    backends: Sequence[str] = (),
    seeds: Sequence[str] = (),
    run_ids: Sequence[str] = (),
    include_failed: bool = False,
    keep_duplicates: bool = False,
) -> tuple[list[RunInfo], list[str]]:
    """Filter runs and, by default, keep only the newest run per recipe/backend/seed."""

    notes: list[str] = []
    kept = []
    for run in runs:
        if run_ids and not any(run.run_id.startswith(prefix) for prefix in run_ids):
            continue
        if recipes and run.recipe not in recipes:
            continue
        if backends and run.backend not in backends:
            continue
        if seeds and run.seed not in seeds:
            continue
        if not include_failed and run.status not in ("FINISHED", "RUNNING"):
            notes.append(f"skipping {run.name} ({run.run_id[:8]}): status {run.status}")
            continue
        kept.append(run)
    if keep_duplicates:
        return sorted(kept, key=lambda r: (r.recipe, r.backend, _seed_sort(r.seed), r.start_time)), notes
    newest: dict[tuple[str, str, str], RunInfo] = {}
    for run in kept:
        current = newest.get(run.group)
        if current is None or run.start_time > current.start_time:
            if current is not None:
                notes.append(f"dropping older duplicate {current.name} ({current.run_id[:8]})")
            newest[run.group] = run
        else:
            notes.append(f"dropping older duplicate {run.name} ({run.run_id[:8]})")
    return sorted(newest.values(), key=lambda r: (r.recipe, r.backend, _seed_sort(r.seed))), notes


def _seed_sort(seed: str) -> tuple[int, str]:
    return (int(seed), "") if seed.isdigit() else (sys.maxsize, seed)


class HistoryCache:
    """Lazily fetch metric histories as (steps, values) arrays."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def get(self, run: RunInfo, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        if key not in run.metric_keys:
            return None
        cache_key = (run.run_id, key)
        if cache_key not in self._cache:
            points = self._client.get_metric_history(run.run_id, key)
            frame = pd.DataFrame({"step": [p.step for p in points], "value": [p.value for p in points]})
            frame = frame.groupby("step", sort=True)["value"].last().reset_index()
            self._cache[cache_key] = (frame["step"].to_numpy(dtype=float), frame["value"].to_numpy(dtype=float))
        return self._cache[cache_key]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass
class BandSettings:
    kind: str = "ci"  # ci | std | sem | none
    confidence: float = 95.0
    smoothing: int = 1


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def spread(samples: np.ndarray, settings: BandSettings) -> np.ndarray:
    """Half-width of the across-run band for an (n_runs, n_points) array."""

    count = np.sum(~np.isnan(samples), axis=0)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # single-seed slices have no ddof=1 std
        std = np.nanstd(samples, axis=0, ddof=1)
    std = np.where(count >= 2, std, np.nan)
    if settings.kind == "none":
        return np.full(std.shape, np.nan)
    if settings.kind == "std":
        return std
    sem = std / np.sqrt(np.maximum(count, 1))
    if settings.kind == "sem":
        return sem
    if settings.kind == "ci":
        quantile = stats.t.ppf(0.5 + settings.confidence / 200.0, np.maximum(count - 1, 1))
        quantile = np.where(count >= 2, quantile, np.nan)
        return quantile * sem
    raise ValueError(f"unknown band kind {settings.kind!r}")


def pooled_std(means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Std over all episodes across runs: pooled within-run variance plus between-run variance."""

    with np.errstate(invalid="ignore"):
        within = np.nanmean(np.square(stds), axis=0)
        between = np.nanvar(means, axis=0, ddof=0)
    return np.sqrt(within + between)


@dataclass
class Curve:
    steps: np.ndarray
    mean: np.ndarray
    half_width: np.ndarray
    traces: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    n_runs: int = 0


def _interp(grid: np.ndarray, steps: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.interp(grid, steps, values)
    result[(grid < steps[0]) | (grid > steps[-1])] = np.nan
    return result


def aggregate_curves(
    histories: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    settings: BandSettings,
    *,
    max_points: int = MAX_CURVE_POINTS,
) -> Curve | None:
    """Aggregate ``(steps, values, stds_or_None)`` histories from several runs."""

    histories = [h for h in histories if h[0].size > 0]
    if not histories:
        return None
    grid = np.unique(np.concatenate([h[0] for h in histories]))
    if grid.size > max_points:
        grid = np.linspace(grid[0], grid[-1], max_points)
    smoothed = [(steps, smooth(values, settings.smoothing)) for steps, values, _ in histories]
    means = np.vstack([_interp(grid, steps, values) for steps, values in smoothed])
    paired = all(h[2] is not None for h in histories)
    if paired:
        stds = np.vstack([_interp(grid, steps, smooth(std, settings.smoothing)) for steps, _, std in histories])
        half_width = pooled_std(means, stds)
    else:
        half_width = spread(means, settings)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(means, axis=0)
    return Curve(grid, mean, half_width, traces=smoothed, n_runs=len(histories))


@dataclass
class Bar:
    mean: float
    error: float
    samples: list[float]


def aggregate_scalars(values: Sequence[float], stds: Sequence[float] | None, settings: BandSettings) -> Bar:
    sample = np.asarray(values, dtype=float)
    if stds is not None:
        error = float(pooled_std(sample[:, None], np.asarray(stds, dtype=float)[:, None])[0])
    else:
        error = float(spread(sample[:, None], settings)[0])
    return Bar(float(np.nanmean(sample)), error, [float(v) for v in sample])


# --------------------------------------------------------------------------- #
# Panel data assembly
# --------------------------------------------------------------------------- #


@dataclass
class PanelData:
    spec: PanelSpec
    series: list[Series]
    kind: str  # "curve" | "bar"
    curves: dict[tuple[str, str], Curve] = field(default_factory=dict)  # (recipe, key)
    bars: dict[tuple[str, str], Bar] = field(default_factory=dict)
    recipes: list[str] = field(default_factory=list)


def build_panel(
    spec: PanelSpec,
    runs: Sequence[RunInfo],
    histories: HistoryCache,
    settings: BandSettings,
) -> PanelData | None:
    all_keys = set().union(*(run.metric_keys for run in runs)) if runs else set()
    series = match_panel_keys(spec.pattern, all_keys, team=spec.team)
    if not series:
        return None
    by_recipe: dict[str, list[RunInfo]] = {}
    for run in runs:
        by_recipe.setdefault(run.recipe, []).append(run)

    per_run: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]] = {}
    for recipe, group in by_recipe.items():
        for item in series:
            rows = []
            for run in group:
                history = histories.get(run, item.key)
                if history is None:
                    continue
                std_history = histories.get(run, item.std_key) if item.std_key else None
                std_values = None
                if std_history is not None:
                    std_values = _interp(history[0], std_history[0], std_history[1])
                rows.append((history[0], history[1], std_values))
            if rows:
                per_run[(recipe, item.key)] = rows
    if not per_run:
        return None

    is_curve = any(steps.size > 1 for rows in per_run.values() for steps, _, _ in rows)
    panel = PanelData(spec, series, "curve" if is_curve else "bar")
    panel.recipes = [recipe for recipe in by_recipe if any(k[0] == recipe for k in per_run)]
    for (recipe, key), rows in per_run.items():
        if is_curve:
            curve = aggregate_curves(rows, settings)
            if curve is not None:
                panel.curves[(recipe, key)] = curve
        else:
            values = [float(values[-1]) for _, values, _ in rows]
            stds = [float(std[-1]) for _, _, std in rows] if all(std is not None for _, _, std in rows) else None
            panel.bars[(recipe, key)] = aggregate_scalars(values, stds, settings)
    return panel


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def apply_style() -> None:
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "sans-serif",
            "axes.titlesize": 8.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 8,
            "axes.labelweight": "normal",
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "0.92",
            "grid.linewidth": 0.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 7.5,
            "legend.title_fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "hatch.linewidth": 0.6,
            "figure.titlesize": 9,
            "figure.titleweight": "normal",
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _step_formatter(max_step: float) -> tuple[FuncFormatter, str]:
    if max_step >= 1e6:
        return FuncFormatter(lambda x, _: f"{x / 1e6:g}"), "Environment steps (M)"
    if max_step >= 1e3:
        return FuncFormatter(lambda x, _: f"{x / 1e3:g}"), "Environment steps (k)"
    return FuncFormatter(lambda x, _: f"{x:g}"), "Environment steps"


def series_colors(series: Sequence[Series], *, shade: bool = True) -> dict[str, Any]:
    """Color per series key: the team hue, shaded lighter when several curves share a team.

    Bar panels pass ``shade=False`` because their series are already told apart
    by the x-axis category.
    """

    by_team: dict[str | None, list[Series]] = {}
    for item in series:
        by_team.setdefault(item.team, []).append(item)
    colors: dict[str, Any] = {}
    for team, items in by_team.items():
        base = TEAM_COLORS.get(team, NEUTRAL_COLOR)
        if len(items) == 1 or not shade:
            for item in items:
                colors[item.key] = base
            continue
        ramp = sns.light_palette(base, n_colors=len(items) + 2, reverse=True)[: len(items)]
        for item, color in zip(items, ramp, strict=True):
            colors[item.key] = color
    return colors


def _draw_curve_panel(
    ax: plt.Axes,
    panel: PanelData,
    colors: Mapping[str, Any],
    styles: Mapping[str, str],
    *,
    traces: bool,
) -> None:
    max_step = 0.0
    band_alpha = 0.18 if len(panel.series) * len(panel.recipes) <= 3 else 0.11
    for item in panel.series:
        color = colors[item.key]
        for recipe in panel.recipes:
            curve = panel.curves.get((recipe, item.key))
            if curve is None:
                continue
            style = styles[recipe]
            max_step = max(max_step, float(curve.steps[-1]))
            if traces and curve.n_runs > 1:
                for steps, values in curve.traces:
                    ax.plot(steps, values, color=color, linestyle=style, linewidth=0.5, alpha=0.25, zorder=1)
            if not np.all(np.isnan(curve.half_width)):
                ax.fill_between(
                    curve.steps,
                    curve.mean - curve.half_width,
                    curve.mean + curve.half_width,
                    color=color,
                    alpha=band_alpha,
                    linewidth=0,
                    zorder=2,
                )
            ax.plot(curve.steps, curve.mean, color=color, linestyle=style, linewidth=1.3, zorder=3)
    formatter, label = _step_formatter(max_step)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(label)
    ax.set_xlim(left=0)
    if len(panel.series) > 1:
        handles = [Line2D([], [], color=colors[item.key], linewidth=1.8) for item in panel.series]
        labels = [pretty_series(item.label or item.key) for item in panel.series]
        ax.legend(handles, labels, loc="best", fontsize=6.5, handlelength=1.6)


def _draw_bar_panel(
    ax: plt.Axes,
    panel: PanelData,
    colors: Mapping[str, Any],
    hatches: Mapping[str, str],
    *,
    samples: bool,
) -> None:
    categories = panel.series
    recipes = panel.recipes
    n_recipes = max(len(recipes), 1)
    group_width = 0.8
    width = group_width / n_recipes
    rng = np.random.default_rng(0)
    for r_index, recipe in enumerate(recipes):
        for c_index, item in enumerate(categories):
            bar = panel.bars.get((recipe, item.key))
            if bar is None:
                continue
            x = c_index - group_width / 2 + width * (r_index + 0.5)
            ax.bar(
                x,
                bar.mean,
                width=width * 0.92,
                color=colors[item.key],
                hatch=hatches[recipe],
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )
            if not math.isnan(bar.error):
                ax.errorbar(x, bar.mean, yerr=bar.error, color="0.2", linewidth=0.7, capsize=2, capthick=0.7, zorder=4)
            if samples and len(bar.samples) > 1:
                jitter = rng.uniform(-width * 0.2, width * 0.2, size=len(bar.samples))
                ax.scatter(x + jitter, bar.samples, s=6, color="0.15", alpha=0.7, linewidths=0, zorder=5)
    ax.axhline(0, color="0.3", linewidth=0.6, zorder=1)
    ax.set_xticks(range(len(categories)))
    if len(categories) == 1 and categories[0].label is None:
        ax.set_xticklabels([""])
        ax.tick_params(axis="x", length=0)
    else:
        tick_labels = [pretty_series(item.label or item.key) for item in categories]
        if sum(len(label) for label in tick_labels) > 28:
            ax.set_xticklabels(tick_labels, rotation=30, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.6, len(categories) - 0.4)
    ax.grid(False, axis="x")


def grid_columns(n_panels: int, max_columns: int) -> int:
    """Fewest rows within ``max_columns``; among those, the layout with the fewest empty slots."""

    candidates = range(1, max(1, min(max_columns, n_panels)) + 1)
    return min(candidates, key=lambda ncols: (math.ceil(n_panels / ncols), (-n_panels) % ncols, -ncols))


def recipe_encodings(recipes: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Line style and bar hatch per recipe, assigned in sorted recipe order."""

    styles = {recipe: LINESTYLES[i % len(LINESTYLES)] for i, recipe in enumerate(recipes)}
    hatches = {recipe: HATCHES[i % len(HATCHES)] for i, recipe in enumerate(recipes)}
    return styles, hatches


def render_figure(
    figure: FigureSpec,
    panels: Sequence[PanelData],
    recipes: Sequence[str],
    labels: Mapping[str, str],
    *,
    columns: int,
    traces: bool,
    samples: bool,
) -> plt.Figure:
    styles, hatches = recipe_encodings(recipes)
    ncols = grid_columns(len(panels), columns)
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.35 * ncols, 1.95 * nrows + 0.35),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    for ax, panel in zip(axes.flat, panels, strict=False):
        colors = series_colors(panel.series, shade=panel.kind == "curve")
        if panel.kind == "curve":
            _draw_curve_panel(ax, panel, colors, styles, traces=traces)
        else:
            _draw_bar_panel(ax, panel, colors, hatches, samples=samples)
        ax.set_title(panel.spec.title, loc="left")
        sns.despine(ax=ax)

    has_curves = any(panel.kind == "curve" for panel in panels)
    has_bars = any(panel.kind == "bar" for panel in panels)
    handles: list[Any] = []
    names: list[str] = []
    for recipe in recipes:
        if not any(recipe in panel.recipes for panel in panels):
            continue
        parts: list[Any] = []
        if has_bars:
            parts.append(Patch(facecolor="0.55", edgecolor="white", hatch=hatches[recipe], linewidth=0.5))
        if has_curves:
            parts.append(Line2D([], [], color="0.25", linestyle=styles[recipe], linewidth=1.3))
        handles.append(tuple(parts) if len(parts) > 1 else parts[0])
        names.append(labels.get(recipe, recipe))
    for team in TEAMS:
        if any(item.team == team for panel in panels for item in panel.series):
            handles.append(Patch(facecolor=TEAM_COLORS[team], linewidth=0))
            names.append(f"{TOKEN_LABELS[team]} agent")
    fig.legend(
        handles,
        names,
        loc="outside lower center",
        ncol=min(len(handles), 5),
        handlelength=2.6 if (has_bars and has_curves) else 1.8,
        columnspacing=1.4,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
    )
    if figure.title:
        fig.suptitle(figure.title)
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI (default: sqlite db under $JAXBORG_EXP_DIR)")
    parser.add_argument("--remote", action="store_true", help="use the mirror pulled by scripts/sync/pull_runs.sh")
    parser.add_argument("--out", help="output directory (default: <exp dir>/plots/<experiment>)")
    parser.add_argument("--formats", nargs="+", default=None, help="image formats (default: png pdf)")
    parser.add_argument("--list-metrics", action="store_true", help="print available metric keys and exit")
    parser.add_argument("--metric", action="append", default=[], help="metric glob; each becomes one panel")
    parser.add_argument("--name", default="custom", help="figure name used with --metric")
    parser.add_argument("--figures", type=Path, help="YAML with a 'figures:' list or a recipe 'plots:' block")
    parser.add_argument("--recipe", action="append", default=[], help="only runs with this recipe.name tag")
    parser.add_argument("--backend", action="append", default=[], help="only runs with this backend tag")
    parser.add_argument("--seed", action="append", default=[], help="only runs with this seed tag")
    parser.add_argument("--run-id", action="append", default=[], help="only runs whose id starts with this")
    parser.add_argument("--include-failed", action="store_true", help="include FAILED/KILLED runs")
    parser.add_argument("--keep-duplicates", action="store_true", help="keep every run per recipe/backend/seed")
    parser.add_argument("--label", action="append", default=[], metavar="RECIPE=LABEL", help="legend label override")
    parser.add_argument("--band", choices=("ci", "std", "sem", "none"), default=None, help="seed band (default ci)")
    parser.add_argument("--confidence", type=float, default=None, help="confidence level for --band ci (default 95)")
    parser.add_argument("--smoothing", type=int, default=None, help="rolling-mean window in logged points (default 1)")
    parser.add_argument("--columns", type=int, default=3, help="panels per row")
    parser.add_argument("--no-traces", action="store_true", help="hide faint per-seed curves")
    parser.add_argument("--no-samples", action="store_true", help="hide per-seed points on bar panels")
    return parser.parse_args(argv)


def _parse_labels(entries: Sequence[str]) -> dict[str, str]:
    labels = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--label expects RECIPE=LABEL, got {entry!r}")
        recipe, label = entry.split("=", 1)
        labels[recipe.strip()] = label.strip()
    return labels


def _print_metrics(runs: Sequence[RunInfo]) -> None:
    counts: dict[str, int] = {}
    for run in runs:
        for key in run.metric_keys:
            counts[key] = counts.get(key, 0) + 1
    all_keys = set(counts)
    print(f"{'metric key':<80} runs")
    for key in sorted(counts):
        paired = "  (std paired)" if std_companion(key) in all_keys else ""
        print(f"{key:<80} {counts[key]:>4}{paired}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from mlflow.tracking import MlflowClient

    tracking_uri = resolve_tracking_uri(args.tracking_uri, remote=args.remote)
    client = MlflowClient(tracking_uri=tracking_uri)
    runs, notes = select_runs(
        load_runs(client, args.experiment),
        recipes=args.recipe,
        backends=args.backend,
        seeds=args.seed,
        run_ids=args.run_id,
        include_failed=args.include_failed,
        keep_duplicates=args.keep_duplicates,
    )
    for note in notes:
        print(f"note: {note}")
    if not runs:
        print(f"no runs selected in experiment {args.experiment!r} at {tracking_uri}", file=sys.stderr)
        return 1
    if args.list_metrics:
        _print_metrics(runs)
        return 0

    file_options: dict[str, Any] = {}
    if args.metric:
        figures = [FigureSpec(args.name, None, tuple(PanelSpec(m, m) for m in args.metric))]
    elif args.figures:
        figures, file_options = load_figure_file(args.figures)
    else:
        figures = list(DEFAULT_FIGURES)

    settings = BandSettings(
        kind=args.band or file_options.get("band", "ci"),
        confidence=args.confidence if args.confidence is not None else float(file_options.get("confidence", 95)),
        smoothing=args.smoothing if args.smoothing is not None else int(file_options.get("smoothing", 1)),
    )
    formats = args.formats or list(file_options.get("formats", ["png", "pdf"]))
    labels = _parse_labels(args.label)
    recipes = sorted({run.recipe for run in runs})

    if args.out:
        out_dir = Path(args.out)
    elif tracking_uri.startswith("sqlite:///"):
        out_dir = Path(tracking_uri[len("sqlite:///") :]).parent / "plots" / args.experiment
    else:
        out_dir = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve() / "plots" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"experiment {args.experiment!r} at {tracking_uri}")
    print(f"{'recipe':<32} {'backend':<8} {'seed':<6} {'status':<9} run")
    for run in runs:
        print(f"{run.recipe:<32} {run.backend:<8} {run.seed:<6} {run.status:<9} {run.run_id[:8]}")

    apply_style()
    histories = HistoryCache(client)
    written: list[Path] = []
    for figure in figures:
        panels = [p for p in (build_panel(spec, runs, histories, settings) for spec in figure.panels) if p is not None]
        if not panels:
            print(f"figure {figure.name}: no matching metrics, skipped")
            continue
        fig = render_figure(
            figure,
            panels,
            recipes,
            labels,
            columns=args.columns,
            traces=not args.no_traces,
            samples=not args.no_samples,
        )
        for fmt in formats:
            path = out_dir / f"{figure.name}.{fmt}"
            fig.savefig(path, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
        print(f"figure {figure.name}: {len(panels)} panels -> {out_dir / figure.name}.{{{','.join(formats)}}}")
    if not written:
        print("nothing written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
