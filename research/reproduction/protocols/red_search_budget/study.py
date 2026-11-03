"""Register, execute and verify a fixed10M-versus20M attacker-budget study."""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist, fmean, stdev

ROOT = Path(__file__).resolve().parent
PARENT = Path.home() / "jaxborg-exp/shared/studies/response-125m-150m-20261101"
SEED = 74001
BOUNDS = [54000000, 54002047]


@lru_cache(None)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def state_save(state):
    path = ROOT / "state.tmp"
    path.write_text(json.dumps(state, indent=2) + "\n")
    path.replace(ROOT / "state.json")


def require_source(m):
    assert not subprocess.check_output(
        ["git", "-C", m["checkout"], "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    assert (
        subprocess.check_output(["git", "-C", m["checkout"], "rev-parse", "HEAD"], text=True).strip()
        == m["source_revision"]
    )
    for item in m["inputs"].values():
        assert sha(item["path"]) == item["sha256"] and sha(item["sidecar_path"]) == item["sidecar_sha256"]
    assert sha(m["topology"]["path"]) == m["topology"]["sha256"]


def audit_seeds(store):
    collection = Path.home() / "src/cyber/plans/jax/cc4/equilibrium-experiments"
    paths = set(collection.rglob("seed-ledger*.json")) | set(collection.rglob("manifest.json"))
    paths |= set(store.glob("studies/*/seed-ledger.json")) | set(store.glob("studies/*/manifest.json"))
    paths |= set(store.glob("campaigns/*/manifest.json")) | set(store.glob("artifacts/*/artifacts/evaluations/*.json"))
    audit = {"files_checked": 0, "collisions": [], "unreadable": []}
    ranges = [BOUNDS, [SEED, SEED + 1]]

    def inspect(value, key, path):
        relevant = any(w in key.lower() for w in ["seed", "validation", "confirmation", "smoke", "future"])
        if isinstance(value, dict):
            for k, v in value.items():
                inspect(v, key + "/" + k if relevant else k, path)
        elif isinstance(value, list):
            if relevant and len(value) == 2 and all(type(x) is int for x in value) and value[0] <= value[1]:
                if any(max(a, value[0]) <= min(b, value[1]) for a, b in ranges):
                    audit["collisions"].append([str(path), key, value])
            for v in value:
                inspect(v, key, path)
        elif relevant and type(value) is int and any(a <= value <= b for a, b in ranges):
            audit["collisions"].append([str(path), key, value])

    for path in sorted(paths):
        if path.is_relative_to(ROOT):
            continue
        try:
            inspect(json.loads(path.read_text()), "", path)
            audit["files_checked"] += 1
        except (UnicodeError, json.JSONDecodeError) as e:
            audit["unreadable"].append([str(path), str(e)])
    assert not audit["collisions"] and not audit["unreadable"], audit
    return audit


def prepare():
    assert os.environ.get("JAX_PLATFORMS") == "cpu" and not (ROOT / "manifest.json").exists()
    parent = json.loads((PARENT / "manifest.json").read_text())
    require_source(parent)
    for name in ["configs", "training", "eval", "analysis", "provenance", "logs"]:
        (ROOT / name).mkdir(exist_ok=True)
    audit = audit_seeds(Path(parent["store"]))
    sys.path[:0] = [parent["checkout"] + "/src", parent["checkout"]]
    import yaml

    from jaxborg.recipe import load, project_jax
    from jaxborg.response_oracle import assert_recipe_contract, budget

    recipes = {}
    for target in parent["targets"]:
        source = parent["recipes"][str(target)]["71001"]
        assert sha(source["path"]) == source["sha256"]
        recipe = yaml.safe_load(Path(source["path"]).read_text())
        original = copy.deepcopy(recipe)
        recipe["meta"] = {
            "name": f"{ROOT.name}-b{target}",
            "source": "Continuous fixed-opponent Red search budget comparison",
        }
        recipe["train"]["total_timesteps"] = 20000000
        recipe["jax"]["checkpoint_every_updates"] = 104
        comparison = copy.deepcopy(recipe)
        comparison["meta"] = original["meta"]
        comparison["train"]["total_timesteps"] = original["train"]["total_timesteps"]
        comparison["jax"]["checkpoint_every_updates"] = original["jax"]["checkpoint_every_updates"]
        assert comparison == original, "Unplanned recipe changes"
        assert_recipe_contract(recipe)
        assert recipe["train"]["teams"] == "red" and set(recipe["train"]["opponents"]) == {"blue"}
        assert sha(recipe["train"]["opponents"]["blue"]["path"]) == parent["inputs"][str(target)]["sha256"]
        assert budget(recipe) == {
            "requested_steps": 20000000,
            "completed_steps": 19968000,
            "updates": 416,
            "steps_per_update": 48000,
        }
        config = project_jax(recipe, team="red")
        assert config["LR"] == 0.0003 and not config["ANNEAL_LR"] and config["GAE_LAMBDA"] == 0.95
        path = ROOT / "configs" / f"b{target}-seed{SEED}.yaml"
        with path.open("x") as f:
            yaml.safe_dump(recipe, f, sort_keys=False)
        load(path)
        recipes[str(target)] = {
            "path": str(path),
            "sha256": sha(path),
            "source_recipe_path": source["path"],
            "source_recipe_sha256": source["sha256"],
        }
    now = datetime.now(timezone.utc).isoformat()
    save(
        ROOT / "seed-ledger.json",
        {
            "registered_utc": now,
            "training_seed": SEED,
            "training_policy_root": SEED + 1,
            "intentional_pairing": "Same fresh training seed across two frozenBlue targets",
            "evaluation_seed_range": BOUNDS,
            "audit": audit,
        },
    )
    m = {
        "study": ROOT.name,
        "registered_utc": now,
        "checkout": parent["checkout"],
        "python": parent["python"],
        "store": parent["store"],
        "source_revision": parent["source_revision"],
        "source_training_run_id": parent["source_training_run_id"],
        "inputs": parent["inputs"],
        "topology": parent["topology"],
        "evaluation_recipe": parent["evaluation_recipe"],
        "source_protocol_path": str(PARENT / "manifest.json"),
        "source_protocol_sha256": sha(PARENT / "manifest.json"),
        "targets": parent["targets"],
        "training_seed": SEED,
        "recipes": recipes,
        "evaluation_seed_range": BOUNDS,
        "training": {
            "requested_steps": 20000000,
            "actual_steps": 19968000,
            "updates": 416,
            "steps_per_update": 48000,
            "midpoint_steps": 9984000,
            "midpoint_update": 208,
            "checkpoint_every_updates": 104,
            "initialization": (
                "Fresh Red parameters and optimizer; no continuation or warm start from an earlier10M run"
            ),
            "comparison": "10M and20M checkpoints on the same continuous trajectory per frozenBlue",
        },
        "protocol": {
            "episodes_per_cell": 2048,
            "batch_size": 64,
            "episode_length": 500,
            "variant": "cc4_stock",
            "stochastic": True,
            "topology_sampling": "exhaustive",
            "max_cells": 8,
            "family_size": 8,
            "alpha": 0.05,
            "practical_margin": 50,
            "contrasts": (
                "At each frozenBlue: Red20M minus Red10M, Red10M minus incumbentRed150M, Red20M "
                "minus incumbentRed150M, and Red20M minus the independently "
                "parent-validation-selected Red own return"
            ),
            "intervals": (
                "Two-sided Bonferroni paired normal approximation across all8 gains, retaining "
                "correction for duplicate labels"
            ),
            "selection": "Fixed10M and20M checkpoints; no validation-based checkpoint selection",
            "parent_selected_baseline": (
                "Use the completed parent study selection.json without reselecting on this study cohort"
            ),
            "deduplication": (
                "Evaluate an identical retained Red model once perBlue and preserve logical "
                "labels without independent-count claims"
            ),
            "stopping": (
                "Fixed2 training trajectories and2048 roots percell, without seed replacement, "
                "sample extension or extra checkpoints"
            ),
        },
        "resources": {"partition": "community", "gpus": 1, "cpus": 8, "memory_gib": 48, "walltime": "07:30:00"},
        "runtime_budget": {
            "conservative_seconds_per_training": 9000,
            "reserved_evaluation_seconds": 5000,
            "margin_seconds": 1800,
        },
        "internal_cutoff": "2026-11-02T07:55:00-05:00",
        "user_cutoff": "2026-11-02T08:00:00-05:00",
        "limits": [
            "One paired fresh response seed and one topology; not a response-seed population result",
            "A positive20M-minus10M gain shows more search helped this target and seed",
            "A null or negative gain cannot establish search saturation or an exploitability upper bound",
            "IncumbentRed150 is the named baseline at bothBlue125 andBlue150",
            "If the full plan cannot fit, do not shrink the experiment or present partial outcomes as complete",
        ],
        "launcher_hashes": {p.name: sha(p) for p in ROOT.iterdir() if p.suffix in [".py", ".sh"]},
    }
    save(ROOT / "manifest.json", m)
    print(
        json.dumps(
            {
                "status": "registered",
                "manifest_sha256": sha(ROOT / "manifest.json"),
                "seed_records_audited": audit["files_checked"],
                "training_seed": SEED,
                "actual_steps": 19968000,
                "midpoint_steps": 9984000,
            }
        )
    )


def analyze():
    m = json.loads((ROOT / "manifest.json").read_text())
    state = json.loads((ROOT / "state.json").read_text())
    assets = json.loads((ROOT / "checkpoint-inputs.json").read_text())
    assert state["manifest_sha256"] == sha(ROOT / "manifest.json")
    assert len(state["training"]) == 2 and len(state["evaluation"]) == 8
    seeds = list(range(BOUNDS[0], BOUNDS[1] + 1))
    cells = {}
    cell_rows = []
    for target in m["targets"]:
        for label in ["mid10m", "final20m", "incumbent150m", "parent_selected"]:
            name = f"b{target}-{label}"
            path = Path(state["evaluation"][name]["path"])
            d = json.loads(path.read_text())
            assert sha(path) == state["evaluation"][name]["sha256"]
            assert d["seeds"] == d["per_episode_seeds"] == seeds and d["n_episodes"] == d["total_episodes"] == 2048
            assert (
                d["git_commit"] == m["source_revision"]
                and d["eval_env"] == "jax_joint"
                and d["policy_backend"] == "jax"
            )
            assert d["variant"] == "cc4_stock" and d["stochastic"] and d["episodes_per_seed"] == 1
            assert d["topology_sampling"] == "exhaustive" and len(d["topology_paths"]) == 1
            assert (
                d["per_episode_topology_paths"] == d["topology_paths"] * 2048
                and sha(d["topology_paths"][0]) == m["topology"]["sha256"]
            )
            red = assets[str(target)][label]
            for team, expected_seed, expected_steps, expected_run, expected_sha in [
                ("blue", 43, target, m["source_training_run_id"], m["inputs"][str(target)]["sha256"]),
                ("red", red["seed"], red["steps"], red["run_id"], red["sha256"]),
            ]:
                p = d["policies"][team]
                provenance = p["bundle_provenance"]
                assert p["team"] == team and p["backend"] == "jax" and sha(p["path"]) == expected_sha
                assert provenance["seed"] == expected_seed and provenance["total_steps"] == expected_steps
                assert provenance["train_run_id"] == p["train_run_id"] == expected_run
                values = d["per_episode_" + team + "_returns"]
                assert len(values) == 2048
                assert math.isclose(fmean(values), d[team + "_mean_return"], abs_tol=1e-8)
                assert math.isclose(stdev(values), d[team + "_std_return"], abs_tol=1e-8)
            assert all(
                math.isfinite(b) and b == -r
                for b, r in zip(d["per_episode_blue_returns"], d["per_episode_red_returns"])
            )
            cells[name] = d
            cell_rows.append(
                {
                    "cell": name,
                    "blue_steps": target,
                    "red_label": label,
                    "blue_mean": d["blue_mean_return"],
                    "red_mean": d["red_mean_return"],
                    "result_sha256": sha(path),
                    "canonical_reference": f"runs:/{d['eval_id']}/evaluations/result.json",
                }
            )
    critical = NormalDist().inv_cdf(1 - 0.05 / 16)
    contrasts = []
    for target in m["targets"]:
        for a, b in [
            ("final20m", "mid10m"),
            ("mid10m", "incumbent150m"),
            ("final20m", "incumbent150m"),
            ("final20m", "parent_selected"),
        ]:
            left, right = cells[f"b{target}-{a}"], cells[f"b{target}-{b}"]
            values = [x - y for x, y in zip(left["per_episode_red_returns"], right["per_episode_red_returns"])]
            mean, sd = fmean(values), stdev(values)
            se = sd / math.sqrt(2048)
            low, high = mean - critical * se, mean + critical * se
            contrasts.append(
                {
                    "blue_steps": target,
                    "candidate": a,
                    "baseline": b,
                    "mean_red_gain": mean,
                    "paired_sd": sd,
                    "standard_error": se,
                    "simultaneous_95_low": low,
                    "simultaneous_95_high": high,
                    "positive_gain_detected": low > 0,
                    "gain_over_provisional_50_detected": low > 50,
                    "equivalent_within_provisional_50": low > -50 and high < 50,
                    "n_paired_roots": 2048,
                }
            )
    report = {
        "status": "complete_paired_attacker_budget_study",
        "manifest_sha256": sha(ROOT / "manifest.json"),
        "protocol": m["protocol"],
        "training": state["training"],
        "normal_critical_value": critical,
        "cells": cell_rows,
        "contrasts": contrasts,
        "limits": m["limits"],
    }
    save(ROOT / "analysis/budget-report.json", report)
    with (ROOT / "analysis/budget-contrasts.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0]))
        w.writeheader()
        w.writerows(contrasts)
    return report


def run():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "community"
    assert os.environ.get("JAX_PLATFORMS") == "cuda"
    child_env = {**os.environ, "JAX_PLATFORMS": "cuda"}
    child_env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    m = json.loads((ROOT / "manifest.json").read_text())
    require_source(m)
    assert sha(m["evaluation_recipe"]["path"]) == m["evaluation_recipe"]["sha256"]
    for name, digest in m["launcher_hashes"].items():
        assert sha(ROOT / name) == digest
    for item in m["recipes"].values():
        assert sha(item["path"]) == item["sha256"]
    cutoff = min(datetime.fromisoformat(m["internal_cutoff"]).timestamp(), time.time() + 7.4 * 3600)
    required = (
        2 * m["runtime_budget"]["conservative_seconds_per_training"]
        + m["runtime_budget"]["reserved_evaluation_seconds"]
        + m["runtime_budget"]["margin_seconds"]
    )
    state = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "manifest_sha256": sha(ROOT / "manifest.json"),
        "training": {},
        "evaluation": {},
    }
    assert not (ROOT / "state.json").exists()
    state_save(state)
    if cutoff - time.time() < required:
        state.update(
            status="not_started_insufficient_time",
            reason="Complete registered plan does not conservatively fit before cutoff",
        )
        state_save(state)
        raise SystemExit("Insufficient time for the complete fixed study")
    os.chdir(m["checkout"])
    sys.path[:0] = [m["checkout"] + "/src", m["checkout"]]
    from jaxborg.checkpoint import load_jax_bundle
    from jaxborg.research_tracking import parameter_hash
    from jaxborg.tracking import read_manifest, resolve_artifact

    parent_state = json.loads((PARENT / "state.json").read_text())
    parent_complete = json.loads((PARENT / "provenance/completion.json").read_text())
    parent_selection = json.loads((PARENT / "selection.json").read_text())
    assert parent_state["status"] == "complete" and parent_complete["complete"]
    assert parent_state["manifest_sha256"] == m["source_protocol_sha256"]
    assert sha(PARENT / "manifest.json") == m["source_protocol_sha256"]
    assert parent_state["selection_sha256"] == sha(PARENT / "selection.json")
    save(
        ROOT / "parent-selection-used.json",
        {
            "parent_selection_sha256": sha(PARENT / "selection.json"),
            "parent_manifest_sha256": sha(PARENT / "manifest.json"),
            "selection": parent_selection,
        },
    )
    durations = []
    assets = {}

    def execute(command):
        subprocess.run(command, env=child_env, check=True, timeout=max(1, cutoff - time.time() - 30))

    for index, target in enumerate(m["targets"]):
        upper = max(m["runtime_budget"]["conservative_seconds_per_training"], max(durations, default=0) * 1.2)
        required = (
            (2 - index) * upper
            + m["runtime_budget"]["reserved_evaluation_seconds"]
            + m["runtime_budget"]["margin_seconds"]
        )
        if cutoff - time.time() < required:
            state.update(
                status="incomplete_time_budget", reason="Remaining registered work does not conservatively fit"
            )
            state_save(state)
            raise SystemExit(state["reason"])
        dest = ROOT / "training" / f"b{target}.json"
        assert not dest.exists()
        start = time.monotonic()
        print("START_TRAINING", target, SEED, flush=True)
        execute(
            [
                m["python"],
                "scripts/train/algorithms/ippo_jax.py",
                "--recipe",
                m["recipes"][str(target)]["path"],
                "--seed",
                str(SEED),
                "--tag",
                f"{m['study']}-b{target}-seed{SEED}",
                "--run-result",
                str(dest),
            ]
        )
        result = json.loads(dest.read_text())
        assert result["actual_steps"] == 19968000
        owner = read_manifest(result["run_id"])[0]
        assert (
            owner["status"] == "FINISHED" and owner["trainable_teams"] == ["red"] and owner["frozen_teams"] == ["blue"]
        )
        assert owner["policy_sources"]["red"] == {"kind": "fresh", "seed": SEED}
        assert sha(owner["policy_sources"]["blue"]["path"]) == m["inputs"][str(target)]["sha256"]
        expected = m["inputs"][str(target)]["parameter_hashes"]["blue"]
        checks = owner["parameter_checks"]
        assert not checks["blue"]["changed"] and checks["red"]["changed"]
        assets[str(target)] = {}
        for label, steps, reference in [
            ("mid10m", 9984000, f"runs:/{result['run_id']}/checkpoints/checkpoint_9984000.safetensors"),
            ("final20m", 19968000, result["final_checkpoint"]),
        ]:
            path = resolve_artifact(reference)
            bundle = load_jax_bundle(path)
            assert bundle.provenance["seed"] == SEED and bundle.provenance["total_steps"] == steps
            assert parameter_hash(bundle.policies["blue"].weights) == expected
            if label == "final20m":
                assert parameter_hash(bundle.policies["red"].weights) == checks["red"]["current_sha256"]
            assets[str(target)][label] = {
                "checkpoint": reference,
                "sha256": sha(path),
                "seed": SEED,
                "steps": steps,
                "run_id": result["run_id"],
            }
        incumbent = m["inputs"]["150000000"]
        assets[str(target)]["incumbent150m"] = {
            "checkpoint": incumbent["canonical_reference"],
            "sha256": incumbent["sha256"],
            "seed": 43,
            "steps": 150000000,
            "run_id": m["source_training_run_id"],
        }
        selected = parent_selection["selected"][str(target)]
        assert sha(resolve_artifact(selected["checkpoint"])) == selected["sha256"]
        assets[str(target)]["parent_selected"] = {
            "checkpoint": selected["checkpoint"],
            "sha256": selected["sha256"],
            "seed": selected["seed"],
            "steps": selected["source_steps"],
            "run_id": selected["run_id"],
        }
        duration = time.monotonic() - start
        durations.append(duration)
        state["training"][str(target)] = {**result, "wall_seconds": duration, "frozen_blue_verified": True}
        state_save(state)
        print("TRAINING_COMPLETE", target, duration, flush=True)
    save(ROOT / "checkpoint-inputs.json", assets)
    for target in m["targets"]:
        seen = {}
        for label in ["mid10m", "final20m", "incumbent150m", "parent_selected"]:
            name = f"b{target}-{label}"
            path = ROOT / "eval" / (name + ".json")
            assert not path.exists()
            identity = assets[str(target)][label]["sha256"]
            if identity in seen:
                state["evaluation"][name] = {**seen[identity], "reused_label": True}
                state_save(state)
                continue
            print("START_EVALUATION", name, flush=True)
            execute(
                [
                    m["python"],
                    "scripts/eval/eval_matchup.py",
                    "--recipe",
                    m["evaluation_recipe"]["path"],
                    "--policy-backend",
                    "jax",
                    "--blue-path",
                    m["inputs"][str(target)]["canonical_reference"],
                    "--red-path",
                    assets[str(target)][label]["checkpoint"],
                    "--episodes-per-seed",
                    "1",
                    "--seeds",
                    "54000000-54002047",
                    "--topology-path",
                    m["topology"]["path"],
                    "--topology-sampling",
                    "exhaustive",
                    "--name",
                    m["study"] + "-" + name,
                    "--output",
                    str(path),
                ]
            )
            state["evaluation"][name] = {"path": str(path), "sha256": sha(path)}
            seen[identity] = state["evaluation"][name]
            state_save(state)
    report = analyze()
    from jaxborg.recipe import eval_variant, load
    from jaxborg.tracking import Run, input_artifact

    recipe = load(m["evaluation_recipe"]["path"])
    variant = eval_variant(recipe)
    records = [
        input_artifact(ROOT / "manifest.json", role="registered continuous attacker budget study"),
        input_artifact(ROOT / "checkpoint-inputs.json", role="fixed checkpoint identities"),
    ]
    records += [input_artifact(v["path"], role=k) for k, v in state["evaluation"].items()]
    with Run(
        recipe,
        backend="cpu",
        kind="analysis",
        name=m["study"] + "-analysis",
        config={"variant": variant, "budget_protocol": m["protocol"]},
        inputs=records,
    ) as owner:
        for name in ["budget-report.json", "budget-contrasts.csv"]:
            owner.publish(ROOT / "analysis" / name, "analysis/" + name)
        for name in m["launcher_hashes"]:
            owner.publish(ROOT / name, "protocol/" + name)
        reference = f"runs:/{owner.run_id}/analysis/budget-report.json"
    state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    state_save(state)
    save(
        ROOT / "provenance/completion.json",
        {
            "complete": True,
            "completed_utc": state["completed_utc"],
            "job_id": os.environ["SLURM_JOB_ID"],
            "analysis_reference": reference,
        },
    )
    print(
        json.dumps({"status": report["status"], "contrasts": report["contrasts"], "analysis_reference": reference}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "run", "analyze"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.phase]()
