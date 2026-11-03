"""Verify and summarize the four predeclared unilateral checkpoint gains."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, fmean, stdev


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root):
    m = json.loads((root / "manifest.json").read_text())
    seeds = list(range(m["episode_seed_range"][0], m["episode_seed_range"][1] + 1))
    cells, identities = {}, {}
    for spec in m["cells"]:
        path = root / "eval" / (spec["name"] + ".json")
        d = json.loads(path.read_text())
        assert d["git_commit"] == m["source_revision"] and d["eval_env"] == "jax_joint"
        assert d["variant"] == "cc4_stock" and d["policy_backend"] == "jax" and d["stochastic"]
        assert d["n_episodes"] == d["total_episodes"] == len(seeds) == 4096
        assert d["seeds"] == d["per_episode_seeds"] == seeds and d["episodes_per_seed"] == 1
        assert d["topology_sampling"] == "exhaustive" and len(d["topology_paths"]) == 1
        assert d["per_episode_topology_paths"] == d["topology_paths"] * len(seeds)
        assert sha(d["topology_paths"][0]) == m["topology"]["sha256"]
        b, r = d["per_episode_blue_returns"], d["per_episode_red_returns"]
        assert len(b) == len(r) == len(seeds)
        assert all(math.isfinite(x) and x == -y for x, y in zip(b, r))
        for team in ("blue", "red"):
            steps = spec[team + "_steps"]
            p = d["policies"][team]
            provenance = p["bundle_provenance"]
            assert p["team"] == team and p["backend"] == "jax" and provenance["seed"] == 43
            assert (
                provenance["total_steps"] == steps
                and provenance["train_run_id"] == p["train_run_id"] == m["source_training_run_id"]
            )
            assert sha(p["path"]) == m["inputs"][str(steps)]["sha256"]
            assert p["observation_dim"] == (450 if team == "blue" else 706)
            assert p["action_dim"] == (242 if team == "blue" else 1106)
            assert p["arch"]["name"] == "mappo"
            values = d["per_episode_" + team + "_returns"]
            assert math.isclose(d[team + "_mean_return"], fmean(values), abs_tol=1e-8)
            assert math.isclose(d[team + "_std_return"], stdev(values), abs_tol=1e-8)
            identity = {k: v for k, v in p.items() if k != "path"}
            key = (team, steps)
            assert key not in identities or identities[key] == identity
            identities[key] = identity
        cells[spec["name"]] = {"record": d, "sha256": sha(path)}
    critical = NormalDist().inv_cdf(1 - 0.05 / (2 * 4))
    baseline = cells["b150000000-r150000000"]["record"]
    contrasts = []
    for team in ("blue", "red"):
        for steps in (99_840_000, 124_800_000):
            name = f"b{steps}-r150000000" if team == "blue" else f"b150000000-r{steps}"
            challenger = cells[name]["record"]
            diffs = [
                a - b
                for a, b in zip(
                    challenger["per_episode_" + team + "_returns"], baseline["per_episode_" + team + "_returns"]
                )
            ]
            mean, sd = fmean(diffs), stdev(diffs)
            se = sd / math.sqrt(len(seeds))
            low, high = mean - critical * se, mean + critical * se
            contrasts.append(
                {
                    "team": team,
                    "challenger_steps": steps,
                    "incumbent_steps": 150_000_000,
                    "frozen_opponent_steps": 150_000_000,
                    "paired_episode_roots": len(seeds),
                    "mean_own_return_gain": mean,
                    "paired_sd": sd,
                    "standard_error": se,
                    "simultaneous_95_low": low,
                    "simultaneous_95_high": high,
                    "profitable_deviation_detected": low > 0,
                    "gain_over_provisional_50_detected": low > 50,
                    "equivalent_within_provisional_50": low > -50 and high < 50,
                    "challenger_reference": f"runs:/{challenger['eval_id']}/evaluations/result.json",
                    "incumbent_reference": f"runs:/{baseline['eval_id']}/evaluations/result.json",
                }
            )
    report = {
        "status": "complete_four_contrast_checkpoint_audit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "study_manifest_sha256": sha(root / "manifest.json"),
        "source_revision": m["source_revision"],
        "source_training_run_id": m["source_training_run_id"],
        "protocol": m["protocol"],
        "normal_critical_value": critical,
        "contrasts": contrasts,
        "evidence_against_final_pair_equilibrium": any(c["profitable_deviation_detected"] for c in contrasts),
        "nash_equilibrium_certified": False,
        "descriptive_final_pair": {
            "blue_mean_return": baseline["blue_mean_return"],
            "red_mean_return": baseline["red_mean_return"],
            "n_episodes": len(seeds),
        },
        "cells": [
            {"name": k, "result_sha256": v["sha256"], "eval_id": v["record"]["eval_id"]} for k, v in cells.items()
        ],
        "limits": m["limits"],
    }
    path = root / "analysis/audit-report.json"
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    with (root / "analysis/audit-contrasts.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    print(json.dumps(analyze(parser.parse_args().directory), indent=2))
