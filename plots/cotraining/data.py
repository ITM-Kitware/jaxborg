"""Load completed runs and paired evaluations from MLflow SQLite and JSONL.

Runs must finish at least 95% of their recorded step budget; repeated evaluations use
the latest eval_id. Training curves use a 30-update mean, with sample SD across
seeds; end-of-training components average the last 30 updates."""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ("single", "diverse")


TEAMS = ("blue", "red")


RECIPE_FAMILIES = {
    "cotraining": "ippo",
    "cotraining_lstm": "lstm",
    "cotraining_rnn": "gru",
    "cotraining_mappo": "mappo",
    "cotraining_mappo_joint_obs": "mappo_joint_obs",
}


FAMILY_ORDER = ("ippo", "lstm", "gru", "mappo", "mappo_joint_obs")


MAX_STEPS = 70e6


MIN_STEP_FRACTION = 0.95


TRAINING_KEYS = {
    "return": "team.{team}.return",
    "entropy": "team.{team}.loss_entropy",
    "explained_variance": "team.{team}.ppo_explained_variance",
}


COMPONENT_KEYS = (
    "team.blue.return",
    "team.blue.reward_ria",
    "team.blue.reward_lwf",
    "team.blue.reward_asf",
    "team.red.reward_ria",
    "team.red.reward_lwf",
    "team.red.reward_asf",
    "team.red.action_cost",
    "team.blue.action_cost",
    "team.blue.actor_fraction",
    "team.red.actor_fraction",
    "backend.jax.game.impact_count",
    "backend.jax.game.green_lwf_count",
    "backend.jax.game.green_asf_count",
)


def family(recipe: str) -> str | None:
    """Learner family for a co-training recipe name, or None for anything else."""
    return RECIPE_FAMILIES.get(recipe.removesuffix("_env_diversity"))


def condition(recipe: str) -> str:
    return "diverse" if recipe.endswith("_env_diversity") else "single"


def _recipe_from_run_name(name: str) -> str | None:
    """``mappo-jax-cotraining_mappo-seed42`` -> ``cotraining_mappo``."""
    if "-jax-" not in name or "-seed" not in name:
        return None
    return name.split("-jax-", 1)[1].rsplit("-seed", 1)[0]


def _cia(record: dict) -> dict[str, float]:
    summary = record.get("cia_summary") or {}
    out = {}
    for key in ("c", "i", "a"):
        value = summary.get(key)
        out[key] = value.get("mean") if isinstance(value, dict) else value
    return out


@dataclass(frozen=True)
class CompletedRuns:
    by_name: dict[str, str]  # MLflow run name -> run_uuid
    ids: frozenset[str]


def completed_runs(db_path: Path, min_fraction: float = MIN_STEP_FRACTION, *, families=None) -> CompletedRuns:
    """Finished, non-deleted co-training runs that reached ``min_fraction`` of the step budget."""
    conn = sqlite3.connect(db_path)
    best: dict[str, tuple[float, int, str]] = {}
    excluded = []
    query = "select run_uuid, name, status, lifecycle_stage, coalesce(start_time, 0) from runs"
    for uuid, name, status, stage, start in conn.execute(query).fetchall():
        recipe = _recipe_from_run_name(name or "")
        if recipe is None or family(recipe) is None or (families is not None and family(recipe) not in families):
            continue
        count, last = conn.execute(
            "select count(*), max(step) from metrics where run_uuid=? and key='team.blue.return'", (uuid,)
        ).fetchone()
        last = float(last or 0)
        budget_row = conn.execute(
            "select value from params where run_uuid=? and key='recipe.train.total_timesteps'", (uuid,)
        ).fetchone()
        budget = float(budget_row[0]) if budget_row else MAX_STEPS
        if stage != "active":
            reason = f"lifecycle {stage}"
        elif status != "FINISHED":
            reason = f"status {status}"
        elif not count or last < min_fraction * budget:
            reason = f"stopped at {last / 1e6:.1f}M steps"
        else:
            reason = None
        if reason:
            excluded.append((name, uuid, reason))
            continue
        if name not in best or (start, last) > best[name][:2]:
            best[name] = (start, last, uuid)
    for name, uuid, reason in sorted(excluded):
        print(f"exclude run {name} ({uuid[:8]}): {reason}")
    by_name = {name: uuid for name, (_, _, uuid) in best.items()}
    return CompletedRuns(by_name=by_name, ids=frozenset(by_name.values()))


def _latest(frame: pd.DataFrame, keys: list[str], kind: str) -> pd.DataFrame:
    """Keep the most recent evaluation when the same model was evaluated more than once."""
    if frame.empty:
        return frame
    before = len(frame)
    frame = frame.sort_values("eval_id").drop_duplicates(keys, keep="last").reset_index(drop=True)
    if len(frame) < before:
        print(f"keep latest of repeated {kind} evaluations: dropped {before - len(frame)} older record(s)")
    return frame


def _report_dropped(kind: str, dropped: int) -> None:
    if dropped:
        print(f"drop {dropped} {kind} record(s) from runs that did not complete")


def _blue_run_id(record: dict) -> str | None:
    return record.get("train_run_id") or (record.get("blue_policy") or {}).get("train_run_id")


def load_scripted(eval_dir: Path, completed_ids: frozenset[str]) -> pd.DataFrame:
    return _load_scripted(eval_dir, completed_ids, checkpoint=False)


def load_checkpoint_scripted(eval_dir: Path, completed_ids: frozenset[str]) -> pd.DataFrame:
    return _load_scripted(eval_dir, completed_ids, checkpoint=True)


def _load_scripted(eval_dir: Path, completed_ids: frozenset[str], *, checkpoint: bool) -> pd.DataFrame:
    pattern = "*_checkpoint_scripted_reds_*.jsonl" if checkpoint else "*scripted-reds*.jsonl"
    kind = "checkpoint scripted-Red" if checkpoint else "scripted-Red"
    keys = ["family", "condition", "seed", "red"] + (["step"] if checkpoint else [])
    rows, dropped = [], 0
    for path in eval_dir.glob(pattern):
        with path.open() as fh:
            for line in fh:
                record = json.loads(line)
                if not checkpoint and "checkpoint_step" in record:
                    continue
                recipe = record["recipe_name"]
                if family(recipe) is None:
                    continue
                if _blue_run_id(record) not in completed_ids:
                    dropped += 1
                    continue
                rows.append(
                    {
                        "family": family(recipe),
                        "condition": condition(recipe),
                        "red": record["eval_red"],
                        "eval_id": record.get("eval_id", ""),
                        "seed": record["blue_policy"]["train_seed"],
                        **({"step": float(record["checkpoint_step"])} if checkpoint else {}),
                        "reward": record["mean_reward"],
                        **({} if checkpoint else {"reward_std": record["std_reward"]}),
                        **_cia(record),
                    }
                )
    _report_dropped(kind, dropped)
    return paired_rows(_latest(pd.DataFrame(rows), keys, kind), [k for k in keys if k != "condition"])


def load_matchups(eval_dir: Path, completed_ids: frozenset[str]) -> pd.DataFrame:
    files = (
        glob.glob(str(eval_dir / "*matchup_learned-red-ppo*.jsonl"))
        + glob.glob(str(eval_dir / "*matchup_cross-seed-play*.jsonl"))
        + glob.glob(str(eval_dir / "*cross_seed*" / "blue_*.json"))
    )
    rows, dropped = [], 0
    for path in files:
        with open(path) as fh:
            d = json.load(fh)
        recipe = d["recipe_name"]
        if family(recipe) is None:
            continue
        blue, red = d["policies"]["blue"], d["policies"]["red"]
        if blue.get("train_run_id") not in completed_ids or red.get("train_run_id") not in completed_ids:
            dropped += 1
            continue
        rows.append(
            {
                "family": family(recipe),
                "condition": condition(recipe),
                "kind": "self" if "learned-red" in os.path.basename(path) else "cross_seed",
                "eval_id": d.get("eval_id", ""),
                "blue_seed": blue["train_seed"],
                "red_seed": red["train_seed"],
                "blue_return": d["blue_mean_return"],
                "red_return": d["red_mean_return"],
                **_cia(d),
            }
        )
    _report_dropped("learned-Red matchup", dropped)
    return paired_rows(
        _latest(pd.DataFrame(rows), ["family", "condition", "kind", "blue_seed", "red_seed"], "learned-Red matchup"),
        ["family", "kind", "blue_seed", "red_seed"],
    )


def load_cross_play(eval_dir: Path, completed_ids: frozenset[str]) -> dict[tuple[str, str, int], dict]:
    out, dropped, repeated = {}, 0, 0
    for path in glob.glob(str(eval_dir / "*cross_play*.jsonl")):
        with open(path) as fh:
            records = [json.loads(line) for line in fh]
        summary = records[-1]
        if summary.get("eval_name") != "cross_play_summary" or family(summary["recipe_name"]) is None:
            continue
        if summary.get("train_run_id") not in completed_ids:
            dropped += 1
            continue
        recipe = summary["recipe_name"]
        key = (family(recipe), condition(recipe), int(summary["train_seed"]))
        eval_id = summary.get("eval_id", "")
        if key in out:
            repeated += 1
            if eval_id <= out[key]["eval_id"]:
                continue
        out[key] = {
            "eval_id": eval_id,
            "steps": np.asarray(summary["steps"], dtype=float),
            "matrix": np.asarray(summary["blue_payoff_matrix"], dtype=float),
        }
    _report_dropped("cross-play", dropped)
    if repeated:
        print(f"keep latest of repeated cross-play evaluations: dropped {repeated} older record(s)")
    return out


def cross_play_by_step(cross_play: dict) -> pd.DataFrame:
    """Per-checkpoint panel scores.

    ``matrix[i, j]`` is Blue checkpoint ``i`` against Red checkpoint ``j``. A
    Blue checkpoint's score is its row mean (every Red checkpoint of the run);
    a Red checkpoint's score is the negated column mean (every Blue checkpoint).
    """
    rows = []
    for (fam, cond, seed), data in cross_play.items():
        matrix, steps = data["matrix"], data["steps"]
        for idx, step in enumerate(steps):
            base = {"family": fam, "condition": cond, "seed": seed, "step": step}
            rows.append({**base, "team": "blue", "return": matrix[idx, :].mean()})
            rows.append({**base, "team": "red", "return": -matrix[:, idx].mean()})
    return pd.DataFrame(rows)


def zero_sum_decomposition(cross_play: dict) -> pd.DataFrame:
    """Separate the two teams' movement inside the zero-sum self-play change.

    With ``m[i, j]`` = Blue return for Blue checkpoint ``i`` against Red checkpoint ``j``:
    self-play change ``m[-1, -1] - m[0, 0]``; Blue's change with Red frozen at the first
    checkpoint ``m[-1, 0] - m[0, 0]``; Red's return gain with Blue frozen at the first
    checkpoint ``-(m[0, -1] - m[0, 0])``.
    """
    rows = []
    for (fam, cond, seed), data in cross_play.items():
        m = data["matrix"]
        rows.append(
            {
                "family": fam,
                "condition": cond,
                "seed": seed,
                "first_step": data["steps"][0],
                "last_step": data["steps"][-1],
                "selfplay_change": m[-1, -1] - m[0, 0],
                "blue_vs_frozen_red": m[-1, 0] - m[0, 0],
                "red_vs_frozen_blue": -(m[0, -1] - m[0, 0]),
            }
        )
    return pd.DataFrame(rows)


def load_training(db_path: Path, runs: CompletedRuns, window: int = 30) -> pd.DataFrame:
    """Smoothed per-seed training metrics for both teams, one row per logged update."""
    conn = sqlite3.connect(db_path)
    frames = []
    for name, uuid in runs.by_name.items():
        recipe = _recipe_from_run_name(name)
        seed = int(name.rsplit("seed", 1)[1])
        for metric, template in TRAINING_KEYS.items():
            for team in TEAMS:
                rows = conn.execute(
                    "select step, value from metrics where run_uuid=? and key=? order by step",
                    (uuid, template.format(team=team)),
                ).fetchall()
                if not rows:
                    continue
                frame = pd.DataFrame(rows, columns=["step", "value"])
                frame["value"] = frame["value"].rolling(window, min_periods=1, center=True).mean()
                frame["family"] = family(recipe)
                frame["condition"] = condition(recipe)
                frame["seed"] = seed
                frame["team"] = team
                frame["metric"] = metric
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_team_components(db_path: Path, runs: CompletedRuns, last_updates: int = 30) -> pd.DataFrame:
    """End-of-training means of reward components and action statistics, one row per run."""
    conn = sqlite3.connect(db_path)
    rows = []
    for name, uuid in runs.by_name.items():
        recipe = _recipe_from_run_name(name)
        row = {"family": family(recipe), "condition": condition(recipe), "seed": int(name.rsplit("seed", 1)[1])}
        for key in COMPONENT_KEYS:
            values = [
                value
                for (value,) in conn.execute(
                    "select value from metrics where run_uuid=? and key=? order by step desc limit ?",
                    (uuid, key, last_updates),
                )
            ]
            row[key] = float(np.mean(values)) if values else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rows(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep identical training seeds in both conditions for each reported comparison."""
    if frame.empty:
        return frame
    counts = frame.groupby(keys, dropna=False)["condition"].transform("nunique")
    dropped = int((counts != len(CONDITIONS)).sum())
    if dropped:
        print(f"exclude {dropped} unpaired evaluation row(s): waiting for matching training seeds")
    return frame[counts == len(CONDITIONS)].copy()


def paired_families(frame: pd.DataFrame) -> list[str]:
    """Families that have results in both conditions; others would mislead side by side."""
    if frame.empty:
        return []
    present = frame.groupby("family")["condition"].nunique()
    return [f for f in FAMILY_ORDER if present.get(f, 0) == 2]


def _seed_band(runs: list[pd.DataFrame], grid: np.ndarray, x: str, y: str):
    """Interpolate each seed onto ``grid`` (NaN outside its range); return mean, mean - SD, mean + SD.

    The SD is the sample standard deviation across seeds (ddof=1); it is NaN where only one seed covers
    the grid point, so the band is simply not drawn there.
    """
    stacked = []
    for run in runs:
        run = run.sort_values(x)
        values = np.interp(grid, run[x], run[y])
        values[(grid < run[x].min()) | (grid > run[x].max())] = np.nan
        stacked.append(values)
    arr = np.vstack(stacked)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # grid points past every seed's last step
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.full_like(mean, np.nan)
        return mean, mean - sd, mean + sd
