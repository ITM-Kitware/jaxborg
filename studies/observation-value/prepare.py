"""Freeze existing policies and disjoint calibration and evaluation cases."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retain(item, directory):
    source = Path(item["path"])
    if digest(source) != item["sha256"]:
        raise ValueError(f"Frozen source input differs: {source}")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | 0o060)
    if digest(destination) != item["sha256"]:
        raise ValueError("Retained input copy differs")
    return {**item, "path": str(destination), "source": str(source.resolve())}


def prepare(args):
    previous = read(args.previous / "inputs/manifest.json")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "results", "evidence", "baselines", "sensitivity", "logs", "submissions", "checks", "tmp"):
        (root / name).mkdir()
    blues = []
    for identity in ("lstm42-final", "lstm100-final", "mappo43-final"):
        source = next(p for p in previous["blues"] if p["id"] == identity)
        blues.append(
            {
                **source,
                **{role: retain(source[role], root / "inputs/policies" / identity) for role in ("model", "sidecar")},
            }
        )
    reds = [
        {
            **p,
            **{role: retain(p[role], root / "inputs/policies" / p["id"]) for role in ("model", "sidecar")},
        }
        for p in previous["reds"]
    ]
    topologies = [
        {**p, **retain(p, root / "inputs/topologies" / str(index))} for index, p in enumerate(previous["topologies"])
    ]
    replay_red = "br-blue42-final-red17"
    replay = {}
    for blue in blues:
        path = args.previous / "results" / f"{blue['id']}__{replay_red}__sampled.json"
        reference = read(path)
        episodes = [
            p
            for p in reference["episodes"]
            if p["topology_sha256"] == topologies[0]["sha256"] and p["seed"] in (21000, 21001)
        ]
        if len(episodes) != 2:
            raise ValueError("Historical replay selection is incomplete")
        replay[blue["id"]] = {
            "red": replay_red,
            "episodes": episodes,
            "reference": retain({"path": str(path), "sha256": digest(path)}, root / "inputs/replay"),
        }
    manifest = {
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": root.name,
        "previous_manifest": retain(
            {
                "path": str(args.previous / "inputs/manifest.json"),
                "sha256": digest(args.previous / "inputs/manifest.json"),
            },
            root / "inputs/previous",
        ),
        "blues": blues,
        "reds": reds,
        "topologies": topologies,
        "replay": replay,
        "calibration_roots": [31000, 31001],
        "test_roots": list(range(32000, 32032)),
        "horizon": 500,
        "collection_stride": 5,
        "practical_margin": 100.0,
        "primary_modes": ["sampled", "blind"],
        "selection": "Same three final Blue policies and eight Reds as the frozen learned-Red panel; no substitutions.",
        "baseline": {
            "information": "Per-Blue, per-agent constant canonical logits; current legal mask only at execution.",
            "calibration": "Masked cross entropy to teacher probabilities at every tick; genuine decisions only.",
            "pooling": "One vector per agent across eight Reds and four layouts; no opponent or phase input.",
            "freeze": "Fit and hash once after all calibration cells and before inspecting fresh test outcomes.",
        },
        "inference": {
            "estimand": "Full sampled policy minus fitted blind baseline; equal weight for each fixed Red/layout/root.",
            "unit": "Episode root, clustering all fixed opponent/layout cases within root.",
            "interval": "Two-sided 95% Student t interval over 32 root means, separately for each Blue checkpoint.",
            "decision": "Useful if interval lower >100; comparable if within [-100,100]; otherwise inconclusive.",
            "limit": "Conditional on frozen policies, four familiar layouts, calibration fit and episode distribution.",
        },
        "missing_cases": "No partial-panel primary conclusion. Report gaps; retries receive new attempt identities.",
        "budget": {"calibration_episodes": 192, "test_episodes": 6144, "replay_episodes": 6},
        "estimated_peak_bytes": 32_000_000_000,
        "storage_includes": (
            "Frozen inputs, calibration decisions, sensitivity records, evidence copies, "
            "MLflow artifacts and peak temporary staging; no training."
        ),
    }
    trials = []
    for blue in blues:
        trials.append(
            {
                "id": f"{blue['id']}__replay",
                "condition_id": "replay",
                "replicate_id": blue["id"],
                "matched_unit_id": None,
            }
        )
        for red in reds:
            pair = f"{blue['id']}__{red['id']}"
            for mode in ("calibration", "sampled", "blind"):
                trials.append(
                    {
                        "id": pair + "__" + mode,
                        "condition_id": mode,
                        "replicate_id": blue["id"],
                        "matched_unit_id": pair,
                    }
                )
    protocol = {
        "schema_version": 1,
        "kind": "protocol",
        "id": root.name + "-v1",
        "study_id": root.name,
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {"attempt": [], "evaluation": []},
        "metric_definitions": [
            {
                "name": "blue_return",
                "unit": "score",
                "direction": "higher",
                "aggregation": "arithmetic mean",
                "sampling_unit": "episode root within fixed policy panel",
                "uncertainty": manifest["inference"]["interval"],
            }
        ],
        "trials": trials,
    }
    write(root / "inputs/manifest.json", manifest)
    write(root / "inputs/protocol.json", protocol)
    print(json.dumps({"root": str(root), "budget": manifest["budget"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    prepare(parser.parse_args())
