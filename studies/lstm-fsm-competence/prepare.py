"""Freeze baseline recipes and topology bytes before launching either seed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_identity(expected_revision):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=PROJECT, text=True)
    if len(expected_revision) != 40 or revision != expected_revision or status:
        raise ValueError("Use clean, committed source at the full expected revision")
    return revision


def recipes_with_banks(recipe, training_bank, evaluation_bank):
    """Freeze a shared topology bank, changing only budget for the smoke run."""
    full = copy.deepcopy(recipe)
    for section, bank in (("train", training_bank), ("eval", evaluation_bank)):
        full[section].pop("topology_generation", None)
        full[section]["topology_bank"] = [str(Path(path).resolve()) for path in bank]
    smoke = copy.deepcopy(full)
    smoke["train"]["total_timesteps"] = 3 * full["jax"]["num_envs"] * full["train"]["episode_length"]
    return full, smoke


def prepare(root, expected_revision, reuse_inputs=None):
    from jaxborg.experiment_records.core import validate_records
    from jaxborg.recipe import load, materialize_topology_bank

    root = Path(root).resolve()
    if root == PROJECT or PROJECT in root.parents:
        raise ValueError("Study artifacts must be outside the source checkout")
    if root.exists():
        raise FileExistsError("Use a new study root; existing inputs are immutable")
    revision = source_identity(expected_revision)
    protocol = json.loads((STUDY / "protocol.json").read_text())
    validate_records([protocol])
    recipe = load(str(STUDY / "recipe.yaml"))
    recipe.pop("__source_path__", None)
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    if reuse_inputs is not None:
        banks, reuse = copy_verified_banks(reuse_inputs, root)
    else:
        banks, reuse = {}, None
        for section in ("train", "eval"):
            settings = copy.deepcopy(recipe[section])
            settings["topology_generation"]["cache_dir"] = str(inputs / "topologies" / section)
            banks[section] = materialize_topology_bank(settings, scope=section, repo_root=PROJECT)
    full, smoke = recipes_with_banks(recipe, banks["train"], banks["eval"])
    for phase, effective in (("full", full), ("smoke", smoke)):
        path = inputs / f"recipe-{phase}.yaml"
        path.write_text(yaml.safe_dump(effective, sort_keys=False))
        load(str(path))
    write_json(inputs / "protocol.json", protocol)
    artifact_paths = sorted(path for path in inputs.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "source_revision": revision,
        "python": platform.python_version(),
        "source_files": {
            str(path.relative_to(PROJECT)): sha256(path)
            for path in (STUDY / "recipe.yaml", STUDY / "protocol.json", Path(__file__))
        },
        "input_sha256": {str(path.relative_to(root)): sha256(path) for path in artifact_paths},
        "training_topology_count": len(banks["train"]),
        "evaluation_topology_count": len(banks["eval"]),
        "topology_generation": {section: recipe[section]["topology_generation"] for section in ("train", "eval")},
        "deadline_utc": "2026-11-09T13:00:00+00:00",
        "reused_inputs": reuse,
    }
    write_json(inputs / "manifest.json", manifest)
    print(json.dumps({"study_root": str(root), "manifest": str(inputs / "manifest.json"), "source_revision": revision}))


def verify(root, expected_revision):
    root = Path(root).resolve()
    source_identity(expected_revision)
    manifest = json.loads((root / "inputs/manifest.json").read_text())
    if manifest["source_revision"] != expected_revision:
        raise ValueError("Inputs were prepared at a different source revision")
    verify_input_files(root, manifest)
    return manifest


def verify_input_files(root, manifest):
    root = Path(root).resolve()
    for relative, digest in manifest["input_sha256"].items():
        path = (root / relative).resolve()
        if root not in path.parents or sha256(path) != digest:
            raise ValueError(f"Changed or invalid input: {relative}")


def copy_verified_banks(previous_root, root):
    """Copy exact old topology bytes without requiring its old checkout revision."""
    previous_root, root = Path(previous_root).resolve(), Path(root).resolve()
    if root == previous_root or previous_root in root.parents:
        raise ValueError("The reused study must remain unchanged; use a separate new study root")
    manifest_path = previous_root / "inputs/manifest.json"
    previous = json.loads(manifest_path.read_text())
    verify_input_files(previous_root, previous)
    recipe_path = previous_root / "inputs/recipe-full.yaml"
    if "inputs/recipe-full.yaml" not in previous["input_sha256"]:
        raise ValueError("Reused recipe was not retained in the previous input manifest")
    recipe = yaml.safe_load(recipe_path.read_text())
    banks, copies = {}, []
    for section, count in (("train", 100), ("eval", 4)):
        declared = recipe[section]["topology_bank"]
        if len(declared) != count or len(set(declared)) != count:
            raise ValueError(f"Expected {count} distinct reused {section} topologies")
        banks[section] = []
        for index, name in enumerate(declared):
            source = Path(name).resolve()
            if previous_root not in source.parents:
                raise ValueError("Reused topology lies outside the previous study inputs")
            relative = source.relative_to(previous_root).as_posix()
            digest = previous["input_sha256"].get(relative)
            if digest is None or sha256(source) != digest:
                raise ValueError("Reused topology is absent from the manifest or changed")
            destination = root / "inputs/topologies" / section / f"snapshot-{index:03d}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError("Refusing to overwrite reused topology")
            shutil.copyfile(source, destination)
            if sha256(destination) != digest:
                raise ValueError("Reused topology changed while copying")
            banks[section].append(destination)
            copies.append({"source": str(source), "destination": str(destination.relative_to(root)), "sha256": digest})
    reuse = {
        "previous_study_root": str(previous_root),
        "previous_source_revision": previous["source_revision"],
        "previous_manifest_sha256": sha256(manifest_path),
        "topologies": copies,
    }
    return banks, reuse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--reuse-inputs", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.study_root, args.expected_revision, args.reuse_inputs)
    else:
        if args.reuse_inputs is not None:
            parser.error("--reuse-inputs is only valid with prepare")
        verify(args.study_root, args.expected_revision)
        print("Frozen inputs verified")


if __name__ == "__main__":
    main()
