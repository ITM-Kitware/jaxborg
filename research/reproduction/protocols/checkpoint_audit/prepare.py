"""Register fixed checkpoint deviations from the 150M MAPPO pair."""

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORE = Path(os.environ["JAXBORG_EXP_DIR"]).resolve()
SOURCE = STORE / "studies/training-replication-20261029"
CHECKOUT = Path(os.environ["JAXBORG_EVALUATION_CHECKOUT"]).resolve()
REVISION = "6f17b5546bc7fd2773d30d33444f44f1bcbb1b7b"
STEPS = (99_840_000, 124_800_000, 150_000_000)
SEEDS = (51_000_000, 51_004_095)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def copy_checked(source, destination, expected):
    assert sha(source) == expected, str(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists(), str(destination)
    shutil.copy2(source, destination)
    assert sha(destination) == expected
    return {"path": str(destination), "sha256": expected, "source_path": str(source)}


def seed_audit():
    collection = Path(os.environ["JAXBORG_SEED_AUDIT_ROOT"]).resolve()
    paths = set(collection.rglob("seed-ledger*.json")) | set(collection.rglob("manifest.json"))
    paths |= set(STORE.glob("studies/*/seed-ledger.json")) | set(STORE.glob("studies/*/manifest.json"))
    paths |= set(STORE.glob("campaigns/*/manifest.json")) | set(STORE.glob("artifacts/*/artifacts/evaluations/*.json"))
    audit = {"files_checked": 0, "collisions": [], "unreadable": []}

    def inspect(value, key, path):
        relevant = any(word in key.lower() for word in ("seed", "confirmation", "validation", "smoke", "future"))
        if isinstance(value, dict):
            for child, item in value.items():
                inspect(item, key + "/" + child if relevant else child, path)
        elif isinstance(value, list):
            if relevant and len(value) == 2 and all(type(x) is int for x in value) and value[0] <= value[1]:
                if max(SEEDS[0], value[0]) <= min(SEEDS[1], value[1]):
                    audit["collisions"].append([str(path), key, value])
            for item in value:
                inspect(item, key, path)
        elif relevant and type(value) is int and SEEDS[0] <= value <= SEEDS[1]:
            audit["collisions"].append([str(path), key, value])

    for path in sorted(paths):
        if path.is_relative_to(ROOT):
            continue
        try:
            inspect(json.loads(path.read_text()), "", path)
            audit["files_checked"] += 1
        except (UnicodeError, json.JSONDecodeError) as error:
            audit["unreadable"].append([str(path), str(error)])
    assert not audit["collisions"] and not audit["unreadable"], audit
    return audit


def main():
    assert not (ROOT / "manifest.json").exists(), "Study is already registered"
    assert not subprocess.check_output(
        ["git", "-C", str(CHECKOUT), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    assert subprocess.check_output(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True).strip() == REVISION
    for name in ("inputs", "eval", "analysis", "logs", "provenance"):
        (ROOT / name).mkdir(exist_ok=True)
    source = json.loads((SOURCE / "manifest.json").read_text())
    dynamic = json.loads((SOURCE / "evaluation/input-manifest.json").read_text())
    complete = json.loads((SOURCE / "provenance/completion.json").read_text())
    assert complete["complete"] and complete["training_complete"]
    assert source["source_revision"] == REVISION
    assert dynamic["study_manifest_sha256"] == sha(SOURCE / "manifest.json")
    inputs = {}
    for step in STEPS:
        item = dynamic["new_inputs"][str(step)]
        assert item["source_step"] == step and item["source_run_id"] == complete["training_result"]["run_id"]
        folder = ROOT / "inputs" / str(step)
        model = copy_checked(item["path"], folder / Path(item["path"]).name, item["sha256"])
        sidecar = copy_checked(item["sidecar_path"], folder / Path(item["sidecar_path"]).name, item["sidecar_sha256"])
        inputs[str(step)] = {**item, **model, "sidecar_path": sidecar["path"]}
    topology = copy_checked(
        source["topology"]["path"], ROOT / "inputs/topology-seed0.npz", source["topology"]["sha256"]
    )
    recipe = copy_checked(source["recipe"]["path"], ROOT / "recipe-evaluation.yaml", source["recipe"]["sha256"])
    source_manifest = copy_checked(
        SOURCE / "manifest.json", ROOT / "inputs/source-study-manifest.json", sha(SOURCE / "manifest.json")
    )
    now = datetime.now(timezone.utc).isoformat()
    save(
        ROOT / "seed-ledger.json",
        {
            "registered_utc": now,
            "status": "reserved_before_execution",
            "evaluation_seeds": list(SEEDS),
            "audit": seed_audit(),
        },
    )
    cells = [
        {"name": f"b{b}-r{r}", "blue_steps": b, "red_steps": r}
        for b, r in [
            (150_000_000, 150_000_000),
            (99_840_000, 150_000_000),
            (124_800_000, 150_000_000),
            (150_000_000, 99_840_000),
            (150_000_000, 124_800_000),
        ]
    ]
    manifest = {
        "study": ROOT.name,
        "registered_utc": now,
        "checkout": str(CHECKOUT),
        "source_revision": REVISION,
        "python": str(CHECKOUT / ".venv/bin/python"),
        "store": str(STORE),
        "source_training_run_id": complete["training_result"]["run_id"],
        "source_training_seed": 43,
        "inputs": inputs,
        "topology": topology,
        "recipe": recipe,
        "source_manifest": source_manifest,
        "launcher_hashes": {p.name: sha(p) for p in ROOT.iterdir() if p.suffix in (".py", ".sh")},
        "cells": cells,
        "episode_seed_range": list(SEEDS),
        "protocol": {
            "episodes_per_cell": 4096,
            "episode_length": 500,
            "batch_size": 64,
            "stochastic": True,
            "variant": "cc4_stock",
            "family_size": 4,
            "alpha": 0.05,
            "practical_margin": 50,
            "primary": (
                "Four unilateral own-return gains: 99.84M and124.8M replace150M for Blue or Red, "
                "holding the opposing team at150M"
            ),
            "intervals": "Two-sided Bonferroni paired normal intervals over four gains",
            "pairing": "Same4096 fresh ordered roots in every cell",
            "stopping": "Fixed sample, no outcome-dependent extension or selection",
            "practical_margin_status": "Provisional design tolerance, not a validated mission threshold",
        },
        "resources": {"partition": "community", "gpus": 1, "cpus": 8, "memory_gib": 32, "walltime_hours": 3},
        "hard_stop": "2026-11-02T07:55:00-05:00",
        "limits": [
            "One trained policy pair on one training topology",
            "Only four named checkpoint deviations; no learned best-response search",
            (
                "Positive gains supply conditional evidence against equilibrium; absence of "
                "detected gain cannot certify equilibrium"
            ),
            "Paired normal interval coverage is approximate",
        ],
    }
    save(ROOT / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "study": ROOT.name,
                "manifest_sha256": sha(ROOT / "manifest.json"),
                "cells": len(cells),
                "seed_records_audited": json.loads((ROOT / "seed-ledger.json").read_text())["audit"]["files_checked"],
            }
        )
    )


if __name__ == "__main__":
    main()
