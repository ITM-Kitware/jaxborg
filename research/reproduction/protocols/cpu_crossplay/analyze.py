"""Verify complete cross-play and test fixed-opponent and panel comparisons."""

import csv
import hashlib
import json
import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from statistics import NormalDist, fmean, stdev


@lru_cache(None)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)


def find_cycle(edges):
    graph = {}
    for a, b in edges:
        graph.setdefault(a, []).append(b)
    done = set()
    active = []

    def visit(node):
        if node in active:
            return active[active.index(node) :] + [node]
        if node in done:
            return None
        active.append(node)
        for other in graph.get(node, []):
            found = visit(other)
            if found:
                return found
        active.pop()
        done.add(node)
        return None

    for node in sorted(graph):
        found = visit(node)
        if found:
            return found
    return None


def analyze(root):
    m = json.loads((root / "manifest.json").read_text())
    steps = m["steps"]
    seeds = list(range(51000000, 51004096))
    cells = {}
    cell_rows = []
    for spec in m["cells"]:
        path = root / "eval" / (spec["name"] + ".json")
        d = json.loads(path.read_text())
        assert d["seeds"] == d["per_episode_seeds"] == seeds and d["n_episodes"] == d["total_episodes"] == 4096
        assert (
            d["eval_env"] == "jax_joint"
            and d["policy_backend"] == "jax"
            and d["variant"] == "cc4_stock"
            and d["stochastic"]
        )
        assert d["git_commit"] == m["source_revision"] and d["episodes_per_seed"] == 1
        assert d["topology_sampling"] == "exhaustive" and len(d["topology_paths"]) == 1
        assert d["per_episode_topology_paths"] == d["topology_paths"] * 4096
        assert sha(d["topology_paths"][0]) == m["topology"]["sha256"]
        for team in ["blue", "red"]:
            p = d["policies"][team]
            age = spec[team + "_steps"]
            provenance = p["bundle_provenance"]
            assert p["team"] == team and p["backend"] == "jax" and p["arch"]["name"] == "mappo"
            assert provenance["seed"] == 43 and provenance["total_steps"] == age
            assert provenance["train_run_id"] == p["train_run_id"] == m["source_training_run_id"]
            assert sha(p["path"]) == m["inputs"][str(age)]["sha256"]
            values = d["per_episode_" + team + "_returns"]
            assert len(values) == 4096
            close(fmean(values), d[team + "_mean_return"])
            close(stdev(values), d[team + "_std_return"])
        assert all(
            math.isfinite(b) and b == -r for b, r in zip(d["per_episode_blue_returns"], d["per_episode_red_returns"])
        )
        if spec["reused"]:
            assert sha(path) == m["reused_cells"][spec["name"]]["sha256"]
        cells[(spec["blue_steps"], spec["red_steps"])] = d
        cell_rows.append(
            {
                **spec,
                "blue_mean": d["blue_mean_return"],
                "red_mean": d["red_mean_return"],
                "result_sha256": sha(path),
                "canonical_reference": f"runs:/{d['eval_id']}/evaluations/result.json",
            }
        )
    assert len(cells) == 9
    critical = NormalDist().inv_cdf(1 - 0.05 / (2 * 24))
    contrasts = []
    pairs = list(combinations(steps, 2))
    edges = []
    for team in ["blue", "red"]:
        for early, late in pairs:
            arrays = []
            for opponent in steps:
                early_pair = (early, opponent) if team == "blue" else (opponent, early)
                late_pair = (late, opponent) if team == "blue" else (opponent, late)
                a = cells[early_pair]["per_episode_" + team + "_returns"]
                b = cells[late_pair]["per_episode_" + team + "_returns"]
                diffs = [x - y for x, y in zip(a, b)]
                arrays.append(diffs)
                mean, sd = fmean(diffs), stdev(diffs)
                se = sd / 64
                low, high = mean - critical * se, mean + critical * se
                row = {
                    "team": team,
                    "earlier_steps": early,
                    "later_steps": late,
                    "scope": "fixed_opponent",
                    "opponent_steps": opponent,
                    "mean_earlier_minus_later_own_return": mean,
                    "paired_sd": sd,
                    "standard_error": se,
                    "simultaneous_95_low": low,
                    "simultaneous_95_high": high,
                    "earlier_better": low > 0,
                    "later_better": high < 0,
                    "equivalent_within_provisional_50": low > -50 and high < 50,
                    "n_paired_roots": 4096,
                }
                contrasts.append(row)
                if low > 0:
                    edges.append((late_pair, early_pair))
                elif high < 0:
                    edges.append((early_pair, late_pair))
            values = [fmean(row) for row in zip(*arrays)]
            mean, sd = fmean(values), stdev(values)
            se = sd / 64
            low, high = mean - critical * se, mean + critical * se
            contrasts.append(
                {
                    "team": team,
                    "earlier_steps": early,
                    "later_steps": late,
                    "scope": "equal_three_opponent_panel",
                    "opponent_steps": "equal_panel",
                    "mean_earlier_minus_later_own_return": mean,
                    "paired_sd": sd,
                    "standard_error": se,
                    "simultaneous_95_low": low,
                    "simultaneous_95_high": high,
                    "earlier_better": low > 0,
                    "later_better": high < 0,
                    "equivalent_within_provisional_50": low > -50 and high < 50,
                    "n_paired_roots": 4096,
                }
            )
    assert len(contrasts) == 24
    rankings = {
        team: [
            {
                "opponent_steps": o,
                "best_to_worst_point_means": sorted(
                    steps, key=lambda s: (-cells[(s, o) if team == "blue" else (o, s)][team + "_mean_return"], s)
                ),
            }
            for o in steps
        ]
        for team in ["blue", "red"]
    }
    comparisons = []
    for team in ["blue", "red"]:
        for early, late in pairs:
            rows = [
                c
                for c in contrasts
                if c["team"] == team
                and c["earlier_steps"] == early
                and c["later_steps"] == late
                and c["scope"] == "fixed_opponent"
            ]
            comparisons.append(
                {
                    "team": team,
                    "earlier_steps": early,
                    "later_steps": late,
                    "earlier_better_against_all_three": all(c["earlier_better"] for c in rows),
                    "later_better_against_all_three": all(c["later_better"] for c in rows),
                    "supported_opponent_ranking_reversal": any(c["earlier_better"] for c in rows)
                    and any(c["later_better"] for c in rows),
                }
            )
    cycle = find_cycle(edges)
    report = {
        "status": "complete_nine_cell_exploratory_cpu_crossplay",
        "execution_platform": "cpu",
        "all_nine_cells_cpu": True,
        "study_manifest_sha256": sha(root / "manifest.json"),
        "steps": steps,
        "protocol": m["protocol"],
        "normal_critical_value": critical,
        "cells": cell_rows,
        "contrasts": contrasts,
        "blue_mean_returns_rows_blue_columns_red": [[cells[(b, r)]["blue_mean_return"] for r in steps] for b in steps],
        "descriptive_rankings": rankings,
        "checkpoint_pair_findings": comparisons,
        "supported_better_response_cycle": cycle,
        "supported_better_response_edges": edges,
        "training_cycle_established": False,
        "nash_equilibrium_certified": False,
        "limits": m["limits"],
    }
    folder = root / "analysis"
    with (folder / "crossplay-report.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    for name, rows in [("crossplay-contrasts.csv", contrasts), ("crossplay-cells.csv", cell_rows)]:
        with (folder / name).open("x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return report
