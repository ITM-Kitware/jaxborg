"""Retain exact baseline inputs and recipes for paired adaptive responses."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
BASELINE_SOURCE = PROJECT / "studies/lstm-fsm-competence"
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE_PREPARE = load_helper("adaptive_baseline_prepare", BASELINE_SOURCE / "prepare.py")
BASELINE_EVALUATE = load_helper("adaptive_baseline_evaluate", BASELINE_SOURCE / "evaluate.py")
sha256 = BASELINE_PREPARE.sha256
source_identity = BASELINE_PREPARE.source_identity
BASELINE_REVISION = "0aef148bbebb45818e71b20e9b5a52be031f93f1"
BLUE_STEPS = {"primary": 49968000, "final": 124992000}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def trial_settings(trial):
    for phase in ("smoke-resume", "smoke", "timing"):
        if trial == f"{phase}-blue42-primary-red17":
            return {
                "phase": phase,
                "blue_seed": 42,
                "age": "primary",
                "red_seed": 17,
                "updates": 10 if phase == "timing" else 4,
            }
    for blue in (42, 100):
        for age in BLUE_STEPS:
            for red in (17, 83):
                if trial == f"br-blue{blue}-{age}-red{red}":
                    return {"phase": "full", "blue_seed": blue, "age": age, "red_seed": red, "updates": 208}
    raise ValueError(f"Undeclared training trial: {trial}")


def recipe_for_trial(base, trial, checkpoints, training_bank, evaluation_bank):
    settings = trial_settings(trial)
    recipe = copy.deepcopy(base)
    target = f"blue{settings['blue_seed']}-{settings['age']}"
    recipe["train"]["opponents"] = {"blue": {"path": checkpoints[target]["model"]}}
    recipe["train"]["topology_bank"] = [str(Path(path).resolve()) for path in training_bank]
    recipe["eval"]["topology_bank"] = [str(Path(path).resolve()) for path in evaluation_bank]
    recipe["train"]["total_timesteps"] = 10000000 if settings["phase"] == "full" else settings["updates"] * 48000
    return recipe


def verified_owned_checkpoint(records, evidence, attempt_id, trial, age):
    indexed = {record["id"]: record for record in records}
    owner = indexed.get(attempt_id, {})
    if owner.get("facts", {}).get("source_revision", {}).get("value") != BASELINE_REVISION:
        raise ValueError("Baseline checkpoint must come from the declared producing revision")
    return BASELINE_EVALUATE.checkpoint_selection(records, evidence, trial, attempt_id, target_steps=BLUE_STEPS[age])[
        -1
    ]


def copy_checkpoints(baseline, root, records):
    checkpoints = {}
    for blue in (42, 100):
        trial = f"fsm-seed{blue}"
        receipt = read_json(baseline / "runs" / trial / "receipt.json")
        if Path(receipt["evidence_root"]).resolve() != baseline / "evidence":
            raise ValueError("Baseline receipt references another evidence store")
        for age, steps in BLUE_STEPS.items():
            selected = verified_owned_checkpoint(records, baseline / "evidence", receipt["attempt_id"], trial, age)
            contract = BASELINE_EVALUATE.validate_bundle(selected["model"], selected["sidecar"], steps)
            identity = f"blue{blue}-{age}"
            directory = root / "inputs/blue" / identity
            directory.mkdir(parents=True)
            copied = {}
            for role, filename in (("model", f"model_{identity}.safetensors"), ("sidecar", f"recipe_{identity}.yaml")):
                destination = directory / filename
                shutil.copyfile(selected[role], destination)
                if sha256(destination) != selected[f"{role}_sha256"]:
                    raise ValueError("Baseline checkpoint changed while copying")
                copied[role] = str(destination)
            checkpoints[identity] = {
                **selected,
                **copied,
                "id": identity,
                "blue_seed": blue,
                "age": age,
                "source_revision": BASELINE_REVISION,
                "source_model": selected["model"],
                "source_sidecar": selected["sidecar"],
                "policy": contract,
            }
    return checkpoints


def prepare(root, baseline, expected_revision):
    from jaxborg.evaluation.capture_store import load_records
    from jaxborg.experiment_records.core import validate_records
    from jaxborg.recipe import load

    root, baseline = Path(root).resolve(), Path(baseline).resolve()
    if root.is_relative_to(PROJECT) or root.is_relative_to(baseline):
        raise ValueError("Use a separate external output directory")
    if root.exists():
        raise FileExistsError("Use a fresh output directory; retained inputs are immutable")
    source_identity(expected_revision)
    baseline_manifest = read_json(baseline / "inputs/manifest.json")
    if baseline_manifest["source_revision"] != BASELINE_REVISION:
        raise ValueError("Expected the declared baseline training source")
    BASELINE_PREPARE.verify_input_files(baseline, baseline_manifest)
    protocol = read_json(STUDY / "protocol.json")
    validate_records([protocol])
    root.mkdir(parents=True)
    banks, reused = BASELINE_PREPARE.copy_verified_banks(baseline, root)
    checkpoints = copy_checkpoints(baseline, root, load_records(baseline / "evidence"))
    base = load(str(STUDY / "recipe.yaml"))
    base.pop("__source_path__", None)
    recipes = {}
    for trial in protocol["trials"]:
        identity = trial["id"]
        if identity.startswith("smoke-resume-"):
            recipes[identity] = recipes[identity.replace("smoke-resume-", "smoke-", 1)]
            continue
        recipe = recipe_for_trial(base, identity, checkpoints, banks["train"], banks["eval"])
        destination = root / "inputs/recipes" / f"{identity}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(recipe, sort_keys=False))
        load(str(destination))
        recipes[identity] = str(destination)
    write_json(root / "inputs/protocol.json", protocol)
    source_files = sorted(path for path in STUDY.iterdir() if path.is_file()) + [
        BASELINE_SOURCE / "prepare.py",
        BASELINE_SOURCE / "evaluate.py",
    ]
    retained_source = {}
    for path in source_files:
        relative = path.relative_to(PROJECT)
        destination = root / "inputs/source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        retained_source[relative.as_posix()] = sha256(destination)
    artifact_paths = sorted(path for path in (root / "inputs").rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "source_revision": expected_revision,
        "baseline_study_root": str(baseline),
        "baseline_manifest_sha256": sha256(baseline / "inputs/manifest.json"),
        "blue_checkpoints": checkpoints,
        "recipes": recipes,
        "source_files": retained_source,
        "training_topologies": [
            {"path": str(path), "sha256": sha256(path), "source_seed": index}
            for index, path in enumerate(banks["train"])
        ],
        "evaluation_topologies": [
            {"path": str(path), "sha256": sha256(path), "source_seed": 200 + index}
            for index, path in enumerate(banks["eval"])
        ],
        "reused_inputs": reused,
        "deadline_utc": "2026-11-09T13:00:00+00:00",
        "input_sha256": {path.relative_to(root).as_posix(): sha256(path) for path in artifact_paths},
    }
    write_json(root / "inputs/manifest.json", manifest)
    print(json.dumps({"study_root": str(root), "manifest": str(root / "inputs/manifest.json")}))
    return manifest


def verify(root, expected_revision):
    root = Path(root).resolve()
    source_identity(expected_revision)
    manifest = read_json(root / "inputs/manifest.json")
    if manifest["source_revision"] != expected_revision:
        raise ValueError("Prepared inputs belong to another source revision")
    BASELINE_PREPARE.verify_input_files(root, manifest)
    for relative, expected in manifest["source_files"].items():
        if sha256(PROJECT / relative) != expected:
            raise ValueError(f"Study helper source differs: {relative}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--baseline-study", type=Path)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.baseline_study is None:
            parser.error("prepare requires --baseline-study")
        prepare(args.study_root, args.baseline_study, args.expected_revision)
    else:
        verify(args.study_root, args.expected_revision)
        print("Frozen study inputs verified")


if __name__ == "__main__":
    main()
