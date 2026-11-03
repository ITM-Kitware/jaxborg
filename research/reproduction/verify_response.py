"""Verify retained training provenance and independently recompute paired contrasts."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, fmean, stdev

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--study-root", type=Path, required=True)
args = parser.parse_args()
ROOT = args.study_root.resolve()
E = ROOT / "evidence"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def stamp(value):
    return datetime.fromisoformat(value)


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9), (a, b)


m, state = read(E / "manifest.json"), read(E / "state.json")
a = read(E / "independent-remote-audit.json")
selection = read(E / "selection.json")
pool = read(E / "candidate-pool.json")["candidates"]
reported = read(E / "analysis/response-report.json")
inventory = read(E / "run-inventory.json")
assert len(inventory) == 33
assert sum(x["kind"] == "training" for x in inventory) == 8
assert sum(x["kind"] == "evaluation" for x in inventory) == 24
assert state["status"] == "complete" and a["completion"]["complete"]
assert sha(E / "manifest.json") == a["manifest_sha256"] == state["manifest_sha256"] == reported["manifest_sha256"]
assert sha(E / "manifest.json") == sha(ROOT.parent / "launch-records/response-125m-150m-20261101/manifest.json")
assert (
    sha(E / "selection.json")
    == a["selection_stat"]["sha256"]
    == state["selection_sha256"]
    == reported["selection_sha256"]
)
assert sha(E / "candidate-pool.json") == selection["candidate_pool_sha256"]
assert selection["validation"] == state["validation"]
assert a["topology_sha256"] == m["topology"]["sha256"] == sha(E / "inputs/topology-seed0.npz")
assert a["evaluation_recipe_sha256"] == m["evaluation_recipe"]["sha256"] == sha(E / "recipe-evaluation.yaml")
for name, digest in m["launcher_hashes"].items():
    assert sha(E / name) == digest, name
for target, records in m["recipes"].items():
    for seed, record in records.items():
        assert sha(E / "configs" / Path(record["path"]).name) == record["sha256"]
for target, item in m["inputs"].items():
    observed = a["inputs"][target]
    assert observed["file_sha256"] == observed["canonical_file_sha256"] == item["sha256"]
    assert observed["parameter_hashes"] == item["parameter_hashes"]
    assert observed["sidecar_sha256"] == item["sidecar_sha256"]

expected_names = {f"b{target}-seed{seed}" for target in m["targets"] for seed in m["training_seeds"]}
assert set(state["training"]) == set(a["training"]) == expected_names
assert len(pool) == 9 and len({x["name"] for x in pool}) == 9
assert set(x["name"] for x in pool) == expected_names | {"saved-r99840000", "saved-r124800000", "saved-r150000000"}
assert selection["tie_order"] == [x["name"] for x in pool]
full_runs = [x for x in inventory if x["actual_steps"] == 9984000]
assert len(full_runs) == 6 and {x["run_id"] for x in full_runs} == {x["run_id"] for x in state["training"].values()}
assert all(x["database_status"] == "FINISHED" for x in inventory)
assert sum(x["actual_steps"] == 4000 for x in inventory) == 2
training = []
initial_red_hashes = {}
for name, record in state["training"].items():
    observed = a["training"][name]
    owner = observed["manifest"]
    target, seed = str(record["target_steps"]), record["training_seed"]
    assert owner["status"] == "FINISHED" and owner["error"] is None
    assert owner["source"]["git_commit"] == m["source_revision"] and not owner["source"]["dirty"]
    assert owner["trainable_teams"] == ["red"] and owner["frozen_teams"] == ["blue"]
    assert owner["seed"] == seed and owner["policy_sources"]["red"] == {"kind": "fresh", "seed": seed}
    assert owner["policy_sources"]["blue"]["kind"] == "checkpoint"
    assert observed["frozen_input_sha256"] == m["inputs"][target]["sha256"]
    assert observed["file_sha256"] == record["model_sha256"]
    assert owner["actual_steps"] == observed["bundle_provenance"]["total_steps"] == 9984000
    assert owner["requested_steps"] == 10000000 and owner["completed_updates"] == 208
    assert owner["seed_contract"]["environment_reset_root"] == seed
    assert owner["seed_contract"]["policy_and_rollout_root"] == seed + 1
    checks = owner["parameter_checks"]
    assert not checks["blue"]["changed"] and checks["red"]["changed"]
    assert (
        checks["blue"]["initial_sha256"]
        == checks["blue"]["current_sha256"]
        == observed["parameter_hashes"]["blue"]
        == m["inputs"][target]["parameter_hashes"]["blue"]
    )
    assert checks["red"]["current_sha256"] == observed["parameter_hashes"]["red"]
    assert checks["red"]["initial_sha256"] != observed["parameter_hashes"]["red"]
    assert owner["initial_parameter_hashes"]["red"] == checks["red"]["initial_sha256"]
    initial_red_hashes.setdefault(seed, set()).add(checks["red"]["initial_sha256"])
    metrics = [json.loads(line) for line in observed["metrics_jsonl"].splitlines()]
    assert [x["update_idx"] for x in metrics] == list(range(1, 209))
    assert [x["env_steps"] for x in metrics] == [48000 * x for x in range(1, 209)]
    assert all(
        x["team.blue.loss_policy"] == x["team.blue.loss_value"] == x["team.blue.ppo_grad_norm"] == 0 for x in metrics
    )
    training.append(
        {
            "name": name,
            "run_id": record["run_id"],
            "seed": seed,
            "target_steps": int(target),
            "actual_steps": 9984000,
            "wall_seconds": record["wall_seconds"],
            "initial_red_sha256": checks["red"]["initial_sha256"],
            "final_red_sha256": observed["parameter_hashes"]["red"],
            "frozen_blue_sha256": observed["parameter_hashes"]["blue"],
        }
    )
assert all(len(x) == 1 for x in initial_red_hashes.values())
assert len(set(next(iter(x)) for x in initial_red_hashes.values())) == 3

cells = {"validation": {}, "confirmation": {}}
all_validation_ends, all_confirmation_starts = [], []
for split, count in [("validation", 18), ("confirmation", 6)]:
    assert len(state[split]) == len(list((E / split).glob("*.json"))) == count
    bounds = m["evaluation_seed_ranges"][split]
    roots = list(range(bounds[0], bounds[1] + 1))
    for name, record in state[split].items():
        path = E / split / (name + ".json")
        d, observed = read(path), a["evaluations"][split + "/" + name]
        owner = observed["manifest"]
        assert sha(path) == observed["file_sha256"] == observed["canonical_sha256"] == record["sha256"]
        assert owner["status"] == observed["database_status_start_end"][0] == "FINISHED"
        assert owner["source"]["git_commit"] == d["git_commit"] == m["source_revision"] and not owner["source"]["dirty"]
        assert owner["outputs"]["evaluations/result.json"]["sha256"] == sha(path)
        assert d["seeds"] == d["per_episode_seeds"] == roots
        assert d["n_episodes"] == d["total_episodes"] == len(roots)
        assert d["episodes_per_seed"] == 1 and d["policy_backend"] == "jax" and d["stochastic"]
        assert len(d["per_episode_topology_paths"]) == len(roots)
        assert set(observed["topology_sha256"].values()) == {m["topology"]["sha256"]}
        target, candidate_name = name[1:].split("-", 1)
        candidate = next(x for x in pool if x["name"] == candidate_name)
        assert observed["policy_file_sha256"]["blue"] == m["inputs"][target]["sha256"]
        assert observed["policy_file_sha256"]["red"] == candidate["sha256"]
        assert d["policies"]["red"]["bundle_provenance"]["seed"] == candidate["seed"]
        assert d["policies"]["red"]["bundle_provenance"]["total_steps"] == candidate["source_steps"]
        for team in ["blue", "red"]:
            values = d[f"per_episode_{team}_returns"]
            assert len(values) == len(roots) and all(math.isfinite(x) for x in values)
            near(fmean(values), d[team + "_mean_return"])
            near(stdev(values), d[team + "_std_return"])
        assert all(b == -r for b, r in zip(d["per_episode_blue_returns"], d["per_episode_red_returns"]))
        near(record["blue_mean"], d["blue_mean_return"])
        cells[split][name] = d
        if split == "validation":
            all_validation_ends.append(stamp(owner["ended_utc"]))
        else:
            all_confirmation_starts.append(stamp(owner["started_utc"]))
assert max(all_validation_ends) <= stamp(selection["frozen_utc"]) < min(all_confirmation_starts)
assert a["selection_stat"]["mtime_ns"] / 1e9 < min(x.timestamp() for x in all_confirmation_starts)
for target in m["targets"]:
    best = min(pool, key=lambda x: cells["validation"][f"b{target}-{x['name']}"]["blue_mean_return"])
    assert best == selection["selected"][str(target)] == reported["selection"][str(target)]
assert set(m["evaluation_seed_ranges"]["validation"]).isdisjoint(m["evaluation_seed_ranges"]["confirmation"])

critical = NormalDist().inv_cdf(1 - 0.05 / (2 * 4))
contrasts = []
for target in m["targets"]:
    chosen = selection["selected"][str(target)]["name"]
    contrasts.append((f"selected_red_gain_at_blue{target}", f"b{target}-{chosen}", f"b{target}-saved-r150000000", -1))
for target in m["targets"]:
    chosen = selection["selected"][str(target)]["name"]
    contrasts.append(
        (f"blue150_minus_blue125_vs_selected_for{target}", f"b150000000-{chosen}", f"b124800000-{chosen}", 1)
    )
statistics = []
for name, candidate, baseline, direction in contrasts:
    x, y = cells["confirmation"][candidate], cells["confirmation"][baseline]
    differences = [direction * (u - v) for u, v in zip(x["per_episode_blue_returns"], y["per_episode_blue_returns"])]
    mean, sd = fmean(differences), stdev(differences)
    se = sd / math.sqrt(len(differences))
    row = {
        "contrast": name,
        "mean_gain": mean,
        "paired_sd": sd,
        "standard_error": se,
        "n_paired_roots": len(differences),
        "simultaneous_95_low": mean - critical * se,
        "simultaneous_95_high": mean + critical * se,
    }
    original = next(x for x in reported["contrasts"] if x["contrast"] == name)
    for key, value in row.items():
        if key != "contrast":
            near(value, original[key])
    statistics.append(row)

verification = {
    "passed": True,
    "verified_utc": datetime.now(timezone.utc).isoformat(),
    "manifest_sha256": sha(E / "manifest.json"),
    "selection_sha256": sha(E / "selection.json"),
    "full_training_runs": 6,
    "paired_training_seeds": [71001, 72001, 73001],
    "fresh_initial_red_hashes_match_across_paired_targets": True,
    "frozen_blue_bytes_and_parameter_hashes_verified": True,
    "complete_updates_per_attempt": 208,
    "actual_steps_per_attempt": 9984000,
    "validation_cells": 18,
    "confirmation_cells": 6,
    "validation_max_end_utc": max(all_validation_ends).isoformat(),
    "selection_frozen_utc": selection["frozen_utc"],
    "confirmation_min_start_utc": min(all_confirmation_starts).isoformat(),
    "selection_reproduced_from_all_nine_candidates": True,
    "raw_roots_returns_means_and_standard_deviations_verified": True,
    "canonical_artifact_hashes_verified": True,
    "four_statistics_reproduced": True,
    "normal_critical_value": critical,
    "training": training,
    "contrasts": statistics,
}
save(ROOT / "verification.json", verification)
with (ROOT / "verified-contrasts.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(statistics[0]))
    writer.writeheader()
    writer.writerows(statistics)
print(
    json.dumps(
        {
            "passed": True,
            "training_runs": len(training),
            "validation_cells": len(cells["validation"]),
            "confirmation_cells": len(cells["confirmation"]),
            "contrasts": statistics,
        },
        indent=2,
    )
)
