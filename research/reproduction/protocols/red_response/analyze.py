"""Verify selected-attacker confirmation and its four registered contrasts."""

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, fmean, stdev


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root):
    m = json.loads((root / "manifest.json").read_text())
    state = json.loads((root / "state.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    pool = json.loads((root / "candidate-pool.json").read_text())["candidates"]
    assert state["manifest_sha256"] == sha(root / "manifest.json") and state["selection_sha256"] == sha(
        root / "selection.json"
    )
    assert len(state["training"]) == 6 and len(state["validation"]) == 18
    assert len(state["smoke"]) == 2 and selection["candidate_pool_sha256"] == sha(root / "candidate-pool.json")
    for target in m["targets"]:
        winner = min(pool, key=lambda c: (state["validation"][f"b{target}-{c['name']}"]["blue_mean"], pool.index(c)))
        assert winner == selection["selected"][str(target)]
    ranges = m["evaluation_seed_ranges"]
    cells = {}
    for split in ["validation", "confirmation"]:
        seeds = list(range(ranges[split][0], ranges[split][1] + 1))
        for key, item in state[split].items():
            assert sha(item["path"]) == item["sha256"]
            d = json.loads(Path(item["path"]).read_text())
            assert d["per_episode_seeds"] == d["seeds"] == seeds and d["n_episodes"] == d["total_episodes"] == len(
                seeds
            )
            assert d["stochastic"] and d["variant"] == "cc4_stock" and d["git_commit"] == m["source_revision"]
            assert d["policy_backend"] == "jax" and d["eval_env"] == "jax_joint" and d["episodes_per_seed"] == 1
            assert d["topology_sampling"] == "exhaustive" and len(d["topology_paths"]) == 1
            assert d["per_episode_topology_paths"] == d["topology_paths"] * len(seeds)
            assert sha(d["topology_paths"][0]) == m["topology"]["sha256"]
            target = int(key.split("-")[0][1:])
            candidate = next(c for c in pool if key == f"b{target}-{c['name']}")
            for team, expected in [("blue", m["inputs"][str(target)]), ("red", candidate)]:
                p = d["policies"][team]
                assert sha(p["path"]) == expected["sha256"]
                provenance = p["bundle_provenance"]
                assert p["team"] == team and p["backend"] == "jax"
                assert provenance["total_steps"] == (target if team == "blue" else candidate["source_steps"])
                assert provenance["seed"] == (43 if team == "blue" else candidate["seed"])
                assert provenance["train_run_id"] == (
                    m["source_training_run_id"] if team == "blue" else candidate["run_id"]
                )
                values = d["per_episode_" + team + "_returns"]
                assert len(values) == len(seeds)
                assert math.isclose(fmean(values), d[team + "_mean_return"], abs_tol=1e-8)
                assert math.isclose(stdev(values), d[team + "_std_return"], abs_tol=1e-8)
            assert all(
                math.isfinite(b) and b == -r
                for b, r in zip(d["per_episode_blue_returns"], d["per_episode_red_returns"])
            )
            if split == "confirmation":
                cells[key] = d
    baseline = pool[0]
    critical = NormalDist().inv_cdf(1 - 0.05 / (2 * 4))
    contrasts = []

    def add(name, a, b, direction):
        values = [direction * (x - y) for x, y in zip(a["per_episode_blue_returns"], b["per_episode_blue_returns"])]
        mean, sd = fmean(values), stdev(values)
        se = sd / math.sqrt(len(values))
        low, high = mean - critical * se, mean + critical * se
        contrasts.append(
            {
                "contrast": name,
                "mean_gain": mean,
                "paired_sd": sd,
                "standard_error": se,
                "n_paired_roots": len(values),
                "simultaneous_95_low": low,
                "simultaneous_95_high": high,
                "positive_gain_detected": low > 0,
                "gain_over_provisional_50_detected": low > 50,
                "equivalent_within_provisional_50": low > -50 and high < 50,
                "candidate_eval_id": a["eval_id"],
                "baseline_eval_id": b["eval_id"],
            }
        )

    for target in m["targets"]:
        c = selection["selected"][str(target)]
        add(
            f"selected_red_gain_at_blue{target}",
            cells[f"b{target}-{c['name']}"],
            cells[f"b{target}-{baseline['name']}"],
            -1,
        )
    for target in m["targets"]:
        c = selection["selected"][str(target)]
        add(
            f"blue150_minus_blue125_vs_selected_for{target}",
            cells[f"b150000000-{c['name']}"],
            cells[f"b124800000-{c['name']}"],
            1,
        )
    report = {
        "status": "complete_matched_response_comparison",
        "manifest_sha256": sha(root / "manifest.json"),
        "selection_sha256": sha(root / "selection.json"),
        "selection": selection["selected"],
        "protocol": m["confirmation"],
        "normal_critical_value": critical,
        "contrasts": contrasts,
        "nash_equilibrium_certified": False,
        "cells": [{"name": k, "eval_id": v["eval_id"], "blue_mean": v["blue_mean_return"]} for k, v in cells.items()],
        "limits": m["limits"],
    }
    with (root / "analysis/response-report.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    with (root / "analysis/response-contrasts.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0]))
        w.writeheader()
        w.writerows(contrasts)
    return report
