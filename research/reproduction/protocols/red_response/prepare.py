"""Seal a matched fresh-Red search against two retained MAPPO defenders."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORE = Path(os.environ["JAXBORG_EXP_DIR"]).resolve()
SOURCE = STORE / "studies/training-replication-20261029"
CHECKOUT = Path(os.environ["JAXBORG_EVALUATION_CHECKOUT"]).resolve()
REVISION = "6f17b5546bc7fd2773d30d33444f44f1bcbb1b7b"
TRAIN_SEEDS = [71001, 72001, 73001]
SMOKE_SEEDS = [710001, 710011]
RANGES = {"validation": [52000000, 52000511], "confirmation": [53000000, 53004095]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def copy_checked(source, dest, expected):
    assert sha(source) == expected, str(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(source, dest)
    assert sha(dest) == expected
    return {"path": str(dest), "sha256": expected, "source_path": str(source)}


def audit_seeds():
    collection = Path(os.environ["JAXBORG_SEED_AUDIT_ROOT"]).resolve()
    paths = set(collection.rglob("seed-ledger*.json")) | set(collection.rglob("manifest.json"))
    paths |= set(STORE.glob("studies/*/seed-ledger.json")) | set(STORE.glob("studies/*/manifest.json"))
    paths |= set(STORE.glob("campaigns/*/manifest.json")) | set(STORE.glob("artifacts/*/artifacts/evaluations/*.json"))
    bounds = list(RANGES.values()) + [[s, s + 1] for s in TRAIN_SEEDS + SMOKE_SEEDS]
    audit = {"files_checked": 0, "collisions": [], "unreadable": []}

    def inspect(value, key, path):
        relevant = any(w in key.lower() for w in ("seed", "confirmation", "validation", "smoke", "future"))
        if isinstance(value, dict):
            for k, v in value.items():
                inspect(v, key + "/" + k if relevant else k, path)
        elif isinstance(value, list):
            if relevant and len(value) == 2 and all(type(x) is int for x in value) and value[0] <= value[1]:
                if any(max(a, value[0]) <= min(b, value[1]) for a, b in bounds):
                    audit["collisions"].append([str(path), key, value])
            for v in value:
                inspect(v, key, path)
        elif relevant and type(value) is int and any(a <= value <= b for a, b in bounds):
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


def main():
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert not (ROOT / "manifest.json").exists()
    assert not subprocess.check_output(
        ["git", "-C", str(CHECKOUT), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    assert subprocess.check_output(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True).strip() == REVISION
    sys.path[:0] = [str(CHECKOUT / "src"), str(CHECKOUT)]
    import jax
    import yaml

    from jaxborg.blue_observation_contract import recipe_blue_obs_size
    from jaxborg.checkpoint import load_jax_bundle, read_sidecar
    from jaxborg.recipe import load, project_jax, team_recipe, train_variant
    from jaxborg.research_tracking import parameter_hash
    from jaxborg.response_oracle import assert_recipe_contract, budget

    assert all(d.platform == "cpu" for d in jax.devices())
    for folder in [
        "inputs",
        "configs",
        "training",
        "smoke",
        "validation",
        "confirmation",
        "analysis",
        "provenance",
        "logs",
    ]:
        (ROOT / folder).mkdir(exist_ok=True)
    audit = audit_seeds()
    source = json.loads((SOURCE / "manifest.json").read_text())
    dynamic = json.loads((SOURCE / "evaluation/input-manifest.json").read_text())
    assert source["source_revision"] == REVISION and dynamic["study_manifest_sha256"] == sha(SOURCE / "manifest.json")
    inputs = {}
    for steps in [99840000, 124800000, 150000000]:
        original = dynamic["new_inputs"][str(steps)]
        folder = ROOT / "inputs" / str(steps)
        model = copy_checked(original["path"], folder / Path(original["path"]).name, original["sha256"])
        sidecar = copy_checked(
            original["sidecar_path"], folder / Path(original["sidecar_path"]).name, original["sidecar_sha256"]
        )
        bundle = load_jax_bundle(model["path"])
        assert bundle.provenance["seed"] == 43 and bundle.provenance["total_steps"] == steps
        inputs[str(steps)] = {
            **original,
            **model,
            "sidecar_path": sidecar["path"],
            "parameter_hashes": {team: parameter_hash(p.weights) for team, p in bundle.policies.items()},
        }
    topology = copy_checked(
        source["topology"]["path"], ROOT / "inputs/topology-seed0.npz", source["topology"]["sha256"]
    )
    template_path = Path(
        "/data/shared/jaxborg/jaxborg-harml-comparison/ippo_jax/cotraining_seed42_hmarl_comparison_50m/checkpoint_9600000.safetensors"
    )
    expected = "71554f36f3ab7e9300355df4d109f09c400dca1df82c385a205710bb863c05ca"
    template = copy_checked(template_path, ROOT / "inputs/template" / template_path.name, expected)
    sidecar_path = template_path.with_name("recipe_checkpoint_9600000.yaml")
    template["sidecar"] = copy_checked(sidecar_path, ROOT / "inputs/template" / sidecar_path.name, sha(sidecar_path))
    template_recipe = read_sidecar(template["path"])
    recipes = {}
    smokes = {}
    budgets = {}
    for target, smoke_seed in zip([124800000, 150000000], SMOKE_SEEDS):
        item = inputs[str(target)]
        source_recipe = read_sidecar(item["path"])
        assert source_recipe["run"]["seed"] == 43 and source_recipe["run"]["total_steps"] == target
        assert recipe_blue_obs_size(source_recipe) == recipe_blue_obs_size(template_recipe) == 450
        assert source_recipe["run"]["blue_observation_version"] == 2
        assert train_variant(source_recipe) == train_variant(template_recipe)
        assert (
            template_recipe["algorithm"] == "ippo" and team_recipe(template_recipe, "red")["arch"]["name"] == "shared"
        )
        base = copy.deepcopy(template_recipe)
        base.pop("run", None)
        base.pop("__source_path__", None)
        base["meta"] = {"name": ROOT.name, "source": "Fresh Red against a frozen MAPPO Blue"}
        base["train"].update(
            teams="red",
            opponents={"blue": {"path": item["path"]}},
            total_timesteps=10000000,
            topology_bank=[topology["path"]],
        )
        base["train"].pop("topology_generation", None)
        base["eval"] = {
            "variant": "cc4_stock",
            "cia": {"enabled": False},
            "topology_bank": [topology["path"]],
            "topology_sampling": "exhaustive",
            "allow_training_topologies": True,
            "policy_backend": "jax",
            "after_training": [],
            "scripted_red": {"after_training": False},
            **{
                suite: {"enabled": False}
                for suite in ["play_priors", "cross_play", "cross_seed_play", "checkpoint_scripted_reds"]
            },
        }
        base["mlflow"] = {"checkpoint_eval": {"every_steps": 0}}
        assert_recipe_contract(base)
        budgets[str(target)] = budget(base)
        assert budgets[str(target)] == {
            "requested_steps": 10000000,
            "completed_steps": 9984000,
            "updates": 208,
            "steps_per_update": 48000,
        }
        projected = project_jax(base, team="red")
        assert projected["LR"] == 0.0003 and projected["GAE_LAMBDA"] == 0.95 and not projected["ANNEAL_LR"]
        recipes[str(target)] = {}
        for seed in TRAIN_SEEDS:
            recipe = copy.deepcopy(base)
            recipe["meta"]["name"] = f"{ROOT.name}-b{target}-seed{seed}"
            path = ROOT / "configs" / f"b{target}-seed{seed}.yaml"
            with path.open("x") as f:
                yaml.safe_dump(recipe, f, sort_keys=False)
            load(path)
            recipes[str(target)][str(seed)] = {"path": str(path), "sha256": sha(path)}
        smoke = copy.deepcopy(base)
        smoke["train"]["total_timesteps"] = 4000
        smoke["jax"].update(num_envs=4, num_minibatches=4, update_epochs=1, checkpoint_every_updates=1)
        smoke["meta"]["name"] = f"{ROOT.name}-smoke-b{target}"
        path = ROOT / "configs" / f"smoke-b{target}.yaml"
        with path.open("x") as f:
            yaml.safe_dump(smoke, f, sort_keys=False)
        load(path)
        smokes[str(target)] = {
            "path": str(path),
            "sha256": sha(path),
            "training_seed": smoke_seed,
            "actual_steps": budget(smoke)["completed_steps"],
        }
    evaluation = copy_checked(source["recipe"]["path"], ROOT / "recipe-evaluation.yaml", source["recipe"]["sha256"])
    now = datetime.now(timezone.utc).isoformat()
    save(
        ROOT / "seed-ledger.json",
        {
            "registered_utc": now,
            "training_seeds": TRAIN_SEEDS,
            "training_policy_roots": [s + 1 for s in TRAIN_SEEDS],
            "intentional_pairing": "Same3 training roots across2 frozen-Blue targets",
            "smoke_training_seeds": SMOKE_SEEDS,
            "smoke_policy_roots": [s + 1 for s in SMOKE_SEEDS],
            "evaluation_seed_ranges": RANGES,
            "audit": audit,
        },
    )
    m = {
        "study": ROOT.name,
        "registered_utc": now,
        "checkout": str(CHECKOUT),
        "python": str(CHECKOUT / ".venv/bin/python"),
        "store": str(STORE),
        "source_revision": REVISION,
        "source_training_run_id": dynamic["training_run_id"],
        "inputs": inputs,
        "template": template,
        "topology": topology,
        "evaluation_recipe": evaluation,
        "targets": [124800000, 150000000],
        "training_seeds": TRAIN_SEEDS,
        "recipes": recipes,
        "smoke_recipes": smokes,
        "budgets": budgets,
        "launcher_hashes": {p.name: sha(p) for p in ROOT.iterdir() if p.suffix in [".py", ".sh"]},
        "evaluation_seed_ranges": RANGES,
        "candidate_rule": "Final only; six fresh Reds plus saved Red99.84M,124.8M,150M; no outcome-dependent additions",
        "selection_rule": (
            "Lowest validation Blue mean per defender; ties favor savedRed150, savedRed99.84, "
            "savedRed124.8, then target order within seed order"
        ),
        "confirmation": {
            "episodes_per_cell": 4096,
            "max_cells": 6,
            "family_size": 4,
            "alpha": 0.05,
            "practical_margin": 50,
            "contrasts": [
                "selected-for125 Red own gain vs Red150 atBlue125",
                "selected-for150 Red own gain vs Red150 atBlue150",
                "Blue150 minus Blue125 vs selected-for125 Red",
                "Blue150 minus Blue125 vs selected-for150 Red",
            ],
            "intervals": (
                "Paired Bonferroni normal approximation; keep4-family correction if selected policies coincide"
            ),
            "stopping": (
                "Fixed samples; selection persisted before confirmation; no sample extension or "
                "training-seed replacement"
            ),
        },
        "resources": {"partition": "community", "gpus": 1, "cpus": 8, "memory_gib": 48, "walltime_hours": 14},
        "internal_cutoff": "2026-11-02T07:55:00-05:00",
        "user_cutoff": "2026-11-02T08:00:00-05:00",
        "runtime_budget": {
            "conservative_training_seconds_per_attempt": 5400,
            "reserved_evaluation_seconds": 10800,
            "margin_seconds": 1800,
        },
        "limits": [
            "One trained defender trajectory and one topology",
            "Three response seeds per defender are a bounded search, not a saturated best response",
            (
                "Positive gains are conditional profitable deviations; no global exploitability "
                "upper bound or Nash certificate"
            ),
            "Incomplete planned training or evaluation is reported incomplete, without shrinking the candidate pool",
        ],
    }
    save(ROOT / "manifest.json", m)
    print(
        json.dumps(
            {
                "status": "registered",
                "manifest_sha256": sha(ROOT / "manifest.json"),
                "training_budgets": budgets,
                "seed_records_audited": audit["files_checked"],
            }
        )
    )


if __name__ == "__main__":
    main()
