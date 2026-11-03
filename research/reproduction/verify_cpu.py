"""Independently verify the fixed CPU matrix from retained episode outcomes."""

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from statistics import NormalDist

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--study-root", type=Path, required=True)
args = parser.parse_args()
ROOT = args.study_root.resolve()
E = ROOT / "evidence"
STEPS = [99840000, 124800000, 150000000]
ROOTS = list(range(51000000, 51004096))
EXPECTED_INPUTS = {
    "99840000": "f0eff76a30ad9e5640ed9e6d19b39d49b343dd399d1a6ab280eb02bdc955b1c3",
    "124800000": "f8e850688d52d1398a93dbb4f80945982b313a04271d417f615cfac547bc67fe",
    "150000000": "8e237355d06f50f8fcaafcc4f6575ce2abaff0941b3f49972bb4fef9caf05f56",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-8), (a, b)


def moments(values):
    mean = math.fsum(values) / len(values)
    variance = math.fsum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


m, remote, state = read(E / "manifest.json"), read(E / "independent-remote-audit.json"), read(E / "state.json")
registered = read(ROOT / "launch-records/manifest.json")
initial = read(ROOT / "registered-manifest-before-affinity.json")
digest = sha(E / "manifest.json")
assert digest == sha(ROOT / "launch-records/manifest.json") == remote["manifest_sha256"]
assert digest == "d35b3b4c789201bf85cab04b32b17b78608085139734fb883f47a95c16fa4aef"
for key in [
    "inputs",
    "topology",
    "recipe",
    "steps",
    "source_training_run_id",
    "source_revision",
    "cells",
    "episode_seed_range",
    "analysis_rules",
    "protocol",
    "timing_preflight",
]:
    assert m[key] == initial[key], key
assert m == registered
assert m["steps"] == STEPS and m["episode_seed_range"] == [ROOTS[0], ROOTS[-1]]
assert m["protocol"]["episodes_per_cell"] == 4096 and m["protocol"]["family_size"] == 24
assert m["protocol"]["alpha"] == 0.05 and m["protocol"]["episode_length"] == 500
assert m["resources"]["gpus"] == 0 and m["resources"]["cpus"] == 24
assert m["resources"]["memory_gib"] == 32 and m["resources"]["parallel_evaluators"] == 3
assert remote["tracking_uri"] == "sqlite:///" + str(Path(m["store"]) / "mlflow.db")
assert not remote["checkout_status"] and remote["checkout_head"] == m["source_revision"]
assert not remote["our_squeue"].strip()
assert remote["recipe_sha256"] == m["recipe"]["sha256"]
assert (
    remote["topology_sha256"]
    == m["topology"]["sha256"]
    == "a750655717241a8cb8aefecefa01d1173d610e8b783cf4dc8dab8fab2158d210"
)
assert remote["parent_manifest_sha256"] == m["parent_manifest"]["sha256"]
assert remote["launcher_hashes"] == m["launcher_hashes"]
for filename, expected in m["launcher_hashes"].items():
    assert sha(E / filename) == sha(ROOT / "launch-records" / filename) == expected
for age, record in remote["registered_input_hashes"].items():
    assert (
        record["policy_sha256"] == record["source_policy_sha256"] == EXPECTED_INPUTS[age] == m["inputs"][age]["sha256"]
    )
    assert record["sidecar_sha256"] == m["inputs"][age]["sidecar_sha256"]

jobs = {row["JobID"]: row for row in csv.DictReader(io.StringIO(remote["sacct"]), delimiter="|")}
assert jobs["4073"]["State"] == "COMPLETED" and jobs["4073"]["ExitCode"] == "0:0"
assert "cpu=24" in jobs["4073"]["AllocTRES"] and "mem=32G" in jobs["4073"]["AllocTRES"]
assert "gpu" not in jobs["4073"]["AllocTRES"]
assert state["status"] == state["phase"] == "complete" and len(state["cells"]) == 9
completion = read(E / "provenance/completion.json")
assert completion["complete"] and completion["new_cpu_cells"] == 9 and completion["gpu_cells_reused_in_matrix"] == 0
assert completion["manifest_sha256"] == state["manifest_sha256"] == digest
assert len(list((E / "eval").glob("*.json"))) == 9

allocation = read(E / "provenance/attempt-4073/cpu-allocation.json")
linux_topology = [
    tuple(map(int, line.split(",")))
    for line in allocation["linux_topology"].splitlines()
    if line and not line.startswith("#")
]
linux_order = [row[0] for row in sorted(linux_topology, key=lambda row: (row[2], row[1], row[0]))]
assert allocation["linux_cpu_ids"] == [linux_order[n] for n in allocation["slurm_abstract_cpu_ids"]]
groups = [set(g) for g in allocation["worker_groups"]]
assert len(groups) == 3 and all(len(g) == 8 for g in groups)
assert len(set.union(*groups)) == 24 and set.union(*groups) == set(allocation["linux_cpu_ids"])
assert all(not (a & b) for a, b in combinations(groups, 2))
cells, cell_records, affinities, source_records = {}, [], {}, []
for spec in m["cells"]:
    name = spec["name"]
    d, observed = read(E / "eval" / (name + ".json")), remote["cells"][name]
    owner = observed["manifest"]
    assert not spec["reused"] and spec["execution_platform"] == "cpu"
    assert (
        sha(E / "eval" / (name + ".json"))
        == observed["sha256"]
        == observed["canonical_sha256"]
        == state["cells"][name]["sha256"]
    )
    output = owner["outputs"]["evaluations/result.json"]
    assert output["complete"] and output["sha256"] == observed["sha256"]
    assert owner["status"] == "FINISHED" and owner["kind"] == "evaluation"
    assert owner["run_id"] == d["eval_id"] == state["cells"][name]["eval_id"]
    assert owner["source"]["git_commit"] == d["git_commit"] == m["source_revision"]
    assert not owner["source"]["dirty"] and owner["source"]["canonical"]
    source_records.append(owner["source"]["files"])
    runtime = owner["environment"]["runtime"]
    assert runtime["SLURM_JOB_ID"] == "4073" and runtime["JAX_PLATFORMS"] == "cpu"
    assert runtime["JAXBORG_ALLOW_CPU"] == "1" and runtime["OMP_NUM_THREADS"] == "8"
    assert owner["effective_config"]["accelerator"] == [{"platform": "cpu", "kind": "cpu", "id": 0}]
    config = owner["effective_config"]
    assert config["seeds"] == config["episode_seeds"] == ROOTS
    assert not config["deterministic"] and config["episodes_per_seed"] == 1
    assert config["variant"]["name"] == "cc4_stock" and config["variant"]["num_steps"] == 500
    assert config["variant"]["red_reward"] == "zero_sum"
    assert d["seeds"] == d["per_episode_seeds"] == ROOTS
    assert d["n_episodes"] == d["total_episodes"] == 4096 and d["episodes_per_seed"] == 1
    assert d["policy_backend"] == "jax" and d["eval_env"] == "jax_joint" and d["stochastic"]
    assert d["variant"] == "cc4_stock" and d["topology_sampling"] == "exhaustive"
    assert len(d["topology_paths"]) == 1
    assert d["per_episode_topology_paths"] == d["topology_paths"] * 4096
    assert observed["retained_topology_sha256"] == m["topology"]["sha256"]
    assert set(observed["topology_hashes"].values()) == {m["topology"]["sha256"]}
    inputs = {x["role"]: x for x in owner["inputs"]}
    for team in ["blue", "red"]:
        age, policy = spec[team + "_steps"], d["policies"][team]
        expected = m["inputs"][str(age)]
        assert observed["policy_file_hashes"][team] == inputs[team + " policy"]["sha256"] == EXPECTED_INPUTS[str(age)]
        assert inputs[team + " policy"]["canonical_reference"] == expected["canonical_reference"]
        assert (
            observed["sidecar_file_hashes"][team]
            == inputs[team + " policy"]["sidecar_sha256"]
            == expected["sidecar_sha256"]
        )
        assert policy["team"] == team and policy["arch"]["name"] == "mappo"
        assert policy["bundle_provenance"]["seed"] == policy["train_seed"] == 43
        assert policy["bundle_provenance"]["total_steps"] == age
        assert policy["train_run_id"] == m["source_training_run_id"]
        values = d["per_episode_" + team + "_returns"]
        assert len(values) == 4096 and all(math.isfinite(x) for x in values)
        mean, sd = moments(values)
        close(mean, d[team + "_mean_return"])
        close(sd, d[team + "_std_return"])
    assert all(b == -r for b, r in zip(d["per_episode_blue_returns"], d["per_episode_red_returns"]))
    affinity = read(E / "provenance/attempt-4073" / (name + "-affinity.json"))
    assert affinity["job_id"] == "4073" and set(affinity["linux_cpu_ids"]) in groups
    step = jobs["4073." + affinity["step_id"]]
    assert step["State"] == "COMPLETED" and step["ExitCode"] == "0:0"
    assert "cpu=8" in step["AllocTRES"] and "mem=10G" in step["AllocTRES"] and "gpu" not in step["AllocTRES"]
    command = read(E / "provenance/attempt-4073" / (name + "-command.json"))["command"]
    assert "--cpus-per-task=8" in command and "--mem=10G" in command and "--cpu-bind=none" in command
    assert command[command.index("--seeds") + 1] == "51000000-51004095"
    affinities[name] = {"cpus": set(affinity["linux_cpu_ids"]), "start": step["Start"], "end": step["End"]}
    cells[(spec["blue_steps"], spec["red_steps"])] = d
    cell_records.append(
        {
            **spec,
            "eval_id": d["eval_id"],
            "sha256": observed["sha256"],
            "blue_mean": moments(d["per_episode_blue_returns"])[0],
            "red_mean": moments(d["per_episode_red_returns"])[0],
            "runtime_seconds": d["wall_time_s"],
        }
    )
assert all(record == source_records[0] for record in source_records)
assert len(cells) == len({d["eval_id"] for d in cells.values()}) == 9
for a, b in combinations(affinities.values(), 2):
    if max(a["start"], b["start"]) < min(a["end"], b["end"]):
        assert not a["cpus"] & b["cpus"]

critical = NormalDist().inv_cdf(1 - 0.05 / (2 * 24))
contrasts, edges, panel_covariance = [], set(), []
for team, (early, late) in product(["blue", "red"], combinations(STEPS, 2)):
    paired = []
    for opponent in STEPS:
        before = (early, opponent) if team == "blue" else (opponent, early)
        after = (late, opponent) if team == "blue" else (opponent, late)
        values = [
            a - b
            for a, b in zip(
                cells[before]["per_episode_" + team + "_returns"], cells[after]["per_episode_" + team + "_returns"]
            )
        ]
        paired.append(values)
        mean, sd = moments(values)
        se = sd / math.sqrt(len(values))
        low, high = mean - critical * se, mean + critical * se
        contrasts.append(
            {
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
                "equivalent_within_provisional_50": -50 < low and high < 50,
                "n_paired_roots": 4096,
            }
        )
        if low > 0:
            edges.add((after, before))
        elif high < 0:
            edges.add((before, after))
    panel = [math.fsum(differences) / 3 for differences in zip(*paired)]
    mean, sd = moments(panel)
    means = [moments(a)[0] for a in paired]
    covariances = [
        [math.fsum((x - means[i]) * (y - means[j]) for x, y in zip(paired[i], paired[j])) / 4095 for j in range(3)]
        for i in range(3)
    ]
    close(sd**2, math.fsum(math.fsum(row) for row in covariances) / 9)
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
            "equivalent_within_provisional_50": -50 < low and high < 50,
            "n_paired_roots": 4096,
        }
    )
    panel_covariance.append(
        {
            "team": team,
            "earlier_steps": early,
            "later_steps": late,
            "opponent_difference_covariance": covariances,
            "panel_variance": sd**2,
        }
    )
assert len(contrasts) == 24
production = read(E / "analysis/crossplay-report.json")


def contrast_key(row):
    return tuple(row[k] for k in ["team", "earlier_steps", "later_steps", "scope", "opponent_steps"])


reported = {contrast_key(row): row for row in production["contrasts"]}
assert len(reported) == 24
for row in contrasts:
    reference = reported[contrast_key(row)]
    for key, value in row.items():
        if isinstance(value, float):
            close(value, reference[key])
        else:
            assert value == reference[key], (contrast_key(row), key)
close(critical, production["normal_critical_value"])
assert edges == {(tuple(a), tuple(b)) for a, b in production["supported_better_response_edges"]}

nodes = list(product(STEPS, STEPS))
reach = [[(a, b) in edges for b in nodes] for a in nodes]
for k in range(9):
    for i in range(9):
        for j in range(9):
            reach[i][j] |= reach[i][k] and reach[k][j]
has_cycle = any(reach[i][i] for i in range(9))
assert has_cycle == (production["supported_better_response_cycle"] is not None)
assert not has_cycle
findings = []
for team, (early, late) in product(["blue", "red"], combinations(STEPS, 2)):
    rows = [
        c
        for c in contrasts
        if c["team"] == team
        and c["earlier_steps"] == early
        and c["later_steps"] == late
        and c["scope"] == "fixed_opponent"
    ]
    findings.append(
        {
            "team": team,
            "earlier_steps": early,
            "later_steps": late,
            "earlier_better_against_all_three": all(c["simultaneous_95_low"] > 0 for c in rows),
            "later_better_against_all_three": all(c["simultaneous_95_high"] < 0 for c in rows),
            "supported_opponent_ranking_reversal": min(c["simultaneous_95_high"] for c in rows)
            < 0
            < max(c["simultaneous_95_low"] for c in rows),
        }
    )
assert findings == production["checkpoint_pair_findings"]
rankings = {
    team: [
        {
            "opponent_steps": opponent,
            "best_to_worst_point_means": sorted(
                STEPS,
                key=lambda age: -cells[(age, opponent) if team == "blue" else (opponent, age)][team + "_mean_return"],
            ),
        }
        for opponent in STEPS
    ]
    for team in ["blue", "red"]
}
assert rankings == production["descriptive_rankings"]
matrix = [[moments(cells[(b, r)]["per_episode_blue_returns"])[0] for r in STEPS] for b in STEPS]
assert matrix == production["blue_mean_returns_rows_blue_columns_red"]
for row in production["cells"]:
    own = next(x for x in cell_records if x["name"] == row["name"])
    close(row["blue_mean"], own["blue_mean"])
    close(row["red_mean"], own["red_mean"])
    assert row["result_sha256"] == own["sha256"]

bridge = []
for name, spec in m["gpu_reference_cells"].items():
    path = ROOT.parent / "crossplay-100m-125m-150m/evidence/eval" / (name + ".json")
    assert (
        sha(path)
        == spec["sha256"]
        == remote["gpu_reference_hashes"][name]["study_sha256"]
        == remote["gpu_reference_hashes"][name]["source_sha256"]
    )
    gpu, cpu = read(path), read(E / "eval" / (name + ".json"))
    assert gpu["per_episode_seeds"] == cpu["per_episode_seeds"] == ROOTS
    differences = [c - g for c, g in zip(cpu["per_episode_blue_returns"], gpu["per_episode_blue_returns"])]
    bridge.append(
        {
            "cell": name,
            "roots_with_different_returns": sum(d != 0 for d in differences),
            "mean_cpu_minus_gpu_blue_return": math.fsum(differences) / 4096,
            "descriptive_only": True,
        }
    )
assert bridge == read(E / "analysis/cpu-provenance.json")["gpu_reference_descriptive_comparison"]
benchmark = read(E / "benchmark/result.json")
assert (
    sha(E / "benchmark/result.json")
    == remote["benchmark"]["sha256"]
    == remote["benchmark"]["canonical_sha256"]
    == "0d74e1ecbb76d0f5cf50842c8628f4801effe4b573823e6340aa867e466033b9"
)
assert benchmark["eval_id"] == "885ef74a988a44ff80f0124c1013827f"
assert benchmark["per_episode_seeds"] == list(range(55000000, 55000128))
assert remote["benchmark"]["manifest"]["status"] == "FINISHED"
assert remote["analysis"]["manifest"]["status"] == "FINISHED"
assert remote["analysis"]["manifest"]["run_id"] == completion["analysis_reference"].split("/")[1]
for filename, archived in remote["analysis"]["archived_sha256"].items():
    assert sha(E / "analysis" / filename) == archived
runs = {run["run_id"]: run for run in remote["study_runs"]}
for cell in cell_records:
    assert runs[cell["eval_id"]]["status"] == "FINISHED"
for run_id in ["232b308742d640c9a35fde8c0e162710", "39b426dd07a74548af8476173bff8db5"]:
    assert runs[run_id]["status"] == "KILLED"
assert not any(run["status"] == "RUNNING" for run in runs.values())

summary = {
    "status": "independently_verified_complete",
    "verified_utc": datetime.now(timezone.utc).isoformat(),
    "job_id": 4073,
    "slurm_elapsed": jobs["4073"]["Elapsed"],
    "completed_utc": completion["completed_utc"],
    "analysis_reference": completion["analysis_reference"],
    "study_manifest_sha256": digest,
    "steps": STEPS,
    "blue_mean_returns_rows_blue_columns_red": matrix,
    "cells": cell_records,
    "contrasts": contrasts,
    "normal_critical_value": critical,
    "family_size": 24,
    "descriptive_rankings": rankings,
    "checkpoint_pair_findings": findings,
    "supported_better_response_cycle": has_cycle,
    "supported_better_response_edges": sorted(edges),
    "panel_covariance_checks": panel_covariance,
    "gpu_bridge_descriptive_only": bridge,
    "training_cycle_established": False,
    "nash_equilibrium_certified": False,
    "limits": m["limits"],
}
save(ROOT / "summary.json", summary)
save(
    ROOT / "verification.json",
    {
        "passed": True,
        "verified_utc": summary["verified_utc"],
        "raw_cells_verified": 9,
        "new_cpu_scientific_episodes": 9 * 4096,
        "aligned_roots_per_cell": 4096,
        "all_policy_topology_recipe_and_source_hashes_match": True,
        "canonical_raw_and_analysis_artifacts_match": True,
        "cpu_runtime_and_opt_in_verified": True,
        "eight_thread_disjoint_affinities_verified": True,
        "overlapping_step_memory_and_cpu_reservations_verified": True,
        "all_attempts_preserved": True,
        "scientific_protocol_unchanged_across_orchestration_repairs": True,
        "benchmark_preserved_and_separate": True,
        "independently_recomputed_contrasts": 24,
        "family_size": 24,
        "panels_preserve_within_root_covariance": True,
        "dominance_rank_reversal_and_graph_findings_match": True,
        "supported_response_cycle": has_cycle,
        "gpu_bridge_checks_descriptive_only": 5,
        "our_jobs_remaining": 0,
    },
)
with (ROOT / "verified-contrasts.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(contrasts[0]))
    writer.writeheader()
    writer.writerows(contrasts)
print(
    json.dumps(
        {
            "passed": True,
            "cells": 9,
            "contrasts": 24,
            "red_rankings": rankings["red"],
            "pair_findings": findings,
            "supported_cycle": has_cycle,
            "resources_released": True,
        },
        indent=2,
    )
)
