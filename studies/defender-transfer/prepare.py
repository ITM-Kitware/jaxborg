"""Prepare the frozen defender screening protocol without executing policies.

Topology generation uses JAX and must run in the allocated execution environment.
The original training topology bytes were not retained in the supplied archive.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import struct
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "defender-transfer"
TRAINING_SOURCE = "45957e2e18fea380bb81d2a62cb2c7ddb5c2034e"
SEEDS = list(range(21000, 21016))
BANK_SETTINGS = {
    "anchor": ([0], None),
    "familiar": ([1, 2, 3], None),
    "unseen": ([200, 201, 202, 203], None),
    "cia": ([100, 101, 102, 103], 3),
}
CONDITIONS = {
    "single": "cotraining_lstm_separate",
    "diverse": "cotraining_lstm_env_diversity_separate",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_identity(expected_revision):
    if not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
        raise ValueError("--source-revision must be a full lowercase Git commit SHA")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT, text=True
    )
    if revision != expected_revision or status:
        raise ValueError("Preparation requires a clean checkout at --source-revision")
    return {
        "revision": revision,
        "clean": True,
        "python": platform.python_version(),
        "source_files": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in [
                Path(__file__).resolve(),
                REPO_ROOT / "pyproject.toml",
                REPO_ROOT / "uv.lock",
                REPO_ROOT / "src/jaxborg/scenarios/cc4/topology.py",
                REPO_ROOT / "src/jaxborg/scenarios/cc4/topology_cli.py",
                REPO_ROOT / "src/jaxborg/evaluation/cia/fixed_topology.py",
            ]
        },
        "installed_versions": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy")},
    }


def bundle_metadata(path):
    with Path(path).open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > 1024 * 1024:
            raise ValueError(f"Unexpectedly large safetensors header: {path}")
        header = json.loads(stream.read(length))
    bundle = json.loads(header["__metadata__"]["jaxborg_bundle"])
    if bundle["schema_version"] != 1 or bundle["backend"] != "jax":
        raise ValueError(f"Expected a version 1 JAX bundle: {path}")
    for team, obs, actions, trunk in (("blue", 450, 242, "separate"), ("red", 706, 1106, "shared")):
        entry = bundle["policies"][team]
        expected_arch = {
            "name": "recurrent",
            "cell": "lstm",
            "trunk": trunk,
            "hidden_dim": 256,
            "hidden_layers": 1,
            "activation": "tanh",
        }
        if entry["obs_dim"] != obs or entry["action_dim"] != actions or entry["arch"] != expected_arch:
            raise ValueError(f"Unexpected {team} policy contract: {path}")
        if not entry["trainable"] or not any(key.startswith(f"{team}/") for key in header):
            raise ValueError(f"Missing trained {team} weights: {path}")
    return bundle


def policy_catalog(archive_root):
    catalog = {"blue": [], "red": []}
    for condition, recipe_name in CONDITIONS.items():
        for seed in (42, 100):
            tag = f"{recipe_name}_seed{seed}"
            directory = archive_root / "ippo_jax" / tag
            for stage, steps in (("early", 960000), ("final", 49968000)):
                model = directory / (
                    f"checkpoint_{steps}.safetensors" if stage == "early" else f"model_{tag}.safetensors"
                )
                sidecar = directory / (f"recipe_checkpoint_{steps}.yaml" if stage == "early" else f"recipe_{tag}.yaml")
                recipe = yaml.safe_load(sidecar.read_text())
                bundle = bundle_metadata(model)
                run = recipe["run"]
                expected_count = 1 if condition == "single" else 100
                generation = recipe["train"]["topology_generation"]
                if (
                    run["seed"] != seed
                    or run["total_steps"] != steps
                    or run["git_commit"] != TRAINING_SOURCE
                    or bundle["provenance"]["seed"] != seed
                    or bundle["provenance"]["total_steps"] != steps
                    or recipe["cage4_enhanced_obs"] is not True
                    or recipe["train"]["variant"] != "cc4_stock"
                    or generation["generator"] != "jax"
                    or generation["seed_start"] != 0
                    or generation["count"] != expected_count
                    or generation.get("op_zone_servers") is not None
                ):
                    raise ValueError(f"Policy does not match the planned cohort: {model}")
                if recipe["core"]["lr"] != 0.0003 or recipe["train"].get("adaptive_updates"):
                    raise ValueError(f"Expected standard updates and learning rate: {sidecar}")
                shared = {
                    "condition": condition,
                    "training_seed": seed,
                    "stage": stage,
                    "steps": steps,
                    "model": str(model),
                    "sidecar": str(sidecar),
                    "sha256": sha256(model),
                    "sidecar_sha256": sha256(sidecar),
                    "size_bytes": model.stat().st_size,
                    "origin": "external archived policy; no retained training producer is asserted",
                    "training_source_revision": run["git_commit"],
                    "historical_train_run_id": run.get("train_run_id"),
                    "training_topology_declaration": generation,
                    "training_layout_identity": "original snapshot bytes unavailable; generation seeds only",
                }
                for team in ("blue", "red") if stage == "final" else ("blue",):
                    catalog[team].append(
                        {
                            **shared,
                            "id": f"{team}-{condition}-{seed}-{stage}",
                            "team": team,
                            "contract": bundle["policies"][team],
                        }
                    )
    return catalog


def archived_benchmark(archive_root):
    references = {}
    sources = []
    selected_recipes = set(CONDITIONS.values())
    for path in sorted((archive_root / "eval").glob("cotraining_lstm*_matchup_learned-red-ppo_*.jsonl")):
        row = json.loads(path.read_text())
        if row.get("recipe_name") not in selected_recipes:
            continue
        if row.get("variant") != "cia_resilience" or row.get("git_commit") != TRAINING_SOURCE:
            continue
        used = False
        for topology in row.get("topology_role_maps", []):
            match = re.search(r"/jax_ops3_seed_(\d+)\.snapshot\.npz$", topology["topology_path"])
            if match is None or int(match[1]) not in BANK_SETTINGS["cia"][0]:
                continue
            seed = int(match[1])
            identity = {
                key: topology[key] for key in ("topology_fingerprint", "role_map_id", "assignment_version", "roles")
            }
            if seed in references and identity != references[seed]:
                raise ValueError(f"Conflicting archived CIA topology identities for seed {seed}")
            references[seed] = identity
            used = True
        if used:
            sources.append({"path": str(path), "sha256": sha256(path), "eval_id": row["eval_id"]})
    if set(references) != set(BANK_SETTINGS["cia"][0]):
        raise ValueError("Archive lacks all four required CIA benchmark fingerprints")
    return {"sources": sources, "by_seed": references}


def generate_banks(output, reference):
    import numpy as np

    from jaxborg.evaluation.cia.fixed_topology import (
        canonical_topology_fingerprint,
        eligible_operational_server_indices,
        fixed_role_assignment,
    )
    from jaxborg.scenarios.cc4.topology import load_topology, load_topology_metadata
    from jaxborg.scenarios.cc4.topology_cli import export_generated

    banks = {}
    discrepancies = []
    for bank_id, (seeds, servers) in BANK_SETTINGS.items():
        directory = output / "topologies" / bank_id
        directory.mkdir(parents=True)
        snapshots = []
        for seed in seeds:
            path = directory / f"jax_seed_{seed:010d}.snapshot.npz"
            export_generated(seed, path, op_zone_servers=servers)
            const = load_topology(path)
            fingerprint = canonical_topology_fingerprint(const)
            eligible = list(eligible_operational_server_indices(const))
            roles = fixed_role_assignment(const).audit_role_map() if len(eligible) >= 3 else None
            snapshot = {
                "path": str(path),
                "sha256": sha256(path),
                "fingerprint": fingerprint,
                "source_seed": seed,
                "op_zone_servers": servers,
                "role_eligible": len(eligible) >= 3,
                "eligible_host_indices": eligible,
                "eligible_host_count": len(eligible),
                "role_map": roles,
                "host_count": int(np.asarray(const.host_active).sum()),
                "server_count": int((np.asarray(const.host_active) & np.asarray(const.host_is_server)).sum()),
                "metadata": load_topology_metadata(path),
                "provenance": "reconstructed seed-matched layout; not verified observed training topology",
            }
            if bank_id == "cia":
                expected = reference["by_seed"][seed]
                matches = roles == expected and fingerprint == expected["topology_fingerprint"]
                snapshot["archived_benchmark_match"] = matches
                snapshot["archived_benchmark_identity"] = expected
                if matches:
                    snapshot["provenance"] = (
                        "canonical topology arrays and role map verified against archived CIA evaluation"
                    )
                else:
                    discrepancies.append(
                        {"seed": seed, "expected": expected, "actual": roles, "fingerprint": fingerprint}
                    )
            snapshots.append(snapshot)
        banks[bank_id] = {
            "id": bank_id,
            "paths": [item["path"] for item in snapshots],
            "snapshots": snapshots,
            "source_seeds": seeds,
            "op_zone_servers": servers,
            "scripted_eligible": all(item["role_eligible"] for item in snapshots),
            "sampling": "exhaustive",
            "interpretation": {
                "anchor": "seed0 reconstructed anchor shared by both declared training banks",
                "familiar": "seeds1..3 declared in diverse training only; reconstructed, not verified training bytes",
                "unseen": "seeds200..203 excluded from both declared training seed banks",
                "cia": "separate operational-server-count and resilience-role benchmark; not pure topology transfer",
            }[bank_id],
        }
    return banks, discrepancies


def game_variant(kind, bank_id):
    return {
        "name": "cia_resilience" if bank_id == "cia" else "cc4_stock",
        "red_agent": "resilience" if bank_id == "cia" else "finite_state",
        "target_weight": 5.0,
        "op_zone_servers": 3 if bank_id == "cia" else None,
        "resilience_roles": kind == "scripted",
        "num_steps": 500,
        "red_reward": "zero_sum",
        "blue_block_policy": "cc4",
        "cage4_enhanced_obs": True,
    }


def effective_recipe(kind, bank_id, bank):
    return {
        "meta": {
            "name": f"defender_transfer_{kind}_{bank_id}",
            "notes": "Frozen-policy screening. The manifest retains original sidecars and the exact runtime variant.",
        },
        "algorithm": "ippo",
        "cage4_enhanced_obs": True,
        "train": {"variant": "cc4_stock"},
        "eval": {
            "variant": "cia_resilience" if bank_id == "cia" else "cc4_stock",
            "policy_backend": "jax",
            "topology_bank": bank["paths"],
            "topology_sampling": "exhaustive",
            "cia": {"enabled": kind == "scripted", "metric": "resilience", "role_assignment": "fixed_per_topology"},
        },
    }


def study_cells(policies, banks, output):
    recipes = output / "recipes"
    recipes.mkdir()
    recipe_paths = {}
    for kind, bank_ids in (("scripted", tuple(BANK_SETTINGS)), ("learned", ("anchor", "unseen"))):
        for bank_id in bank_ids:
            path = recipes / f"{kind}-{bank_id}.yaml"
            path.write_text(yaml.safe_dump(effective_recipe(kind, bank_id, banks[bank_id]), sort_keys=False))
            recipe_paths[kind, bank_id] = path
    cells = []
    for stage in ("final", "early"):
        blues = [policy for policy in policies["blue"] if policy["stage"] == stage]
        for kind, bank_ids in (("scripted", tuple(BANK_SETTINGS)), ("learned", ("anchor", "unseen"))):
            for blue in blues:
                for bank_id in bank_ids:
                    reds = [None] if kind == "scripted" else policies["red"]
                    for red in reds:
                        opponent = red["id"] if red else "scripts"
                        identity = f"{kind}:{blue['id']}:{opponent}:{bank_id}"
                        eligible = kind == "learned" or banks[bank_id]["scripted_eligible"]
                        recipe_path = recipe_paths[kind, bank_id]
                        cells.append(
                            {
                                "id": identity,
                                "trial_id": identity,
                                "kind": kind,
                                "stage": stage,
                                "blue_id": blue["id"],
                                "red_id": red["id"] if red else None,
                                "reds": (["fsm", "cia_c", "cia_i", "cia_a"] if bank_id == "cia" else ["fsm"])
                                if red is None
                                else [],
                                "bank_id": bank_id,
                                "recipe_path": str(recipe_path),
                                "recipe_sha256": sha256(recipe_path),
                                "variant": game_variant(kind, bank_id),
                                "condition_id": f"{kind}:{blue['condition']}:{stage}:{opponent}:{bank_id}",
                                "replicate_id": f"training-{blue['condition']}-{blue['training_seed']}",
                                "matched_unit_id": f"screening:{kind}:{bank_id}:21000-21015",
                                "executable": eligible,
                                "blocked_reason": None
                                if eligible
                                else (
                                    "At least one declared snapshot has fewer than three role-eligible servers; "
                                    "no seed was replaced or excluded"
                                ),
                            }
                        )
    return cells


def protocol_for(manifest):
    encoded_manifest = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    identity = hashlib.sha256(encoded_manifest.encode()).hexdigest()
    return {
        "schema_version": 1,
        "kind": "protocol",
        "id": f"{STUDY_ID}:protocol:{identity}",
        "study_id": STUDY_ID,
        "revision": "1",
        "selection_rule": "explicit",
        "required_facts": {
            "attempt": ["resolved_config", "source_identity"],
            "evaluation": ["resolved_config", "source_identity"],
        },
        "trials": [
            {key: cell[key] for key in ("id", "condition_id", "replicate_id", "matched_unit_id")}
            for cell in manifest["cells"]
        ],
        "metric_definitions": [
            {
                "name": "blue_return",
                "unit": "score",
                "direction": "higher",
                "aggregation": "arithmetic mean",
                "sampling_unit": (
                    "episode conditional on fixed policies and layout; scripted opponents reported separately"
                ),
                "uncertainty": (
                    "exploratory episode variation only; checkpoint ages and opponent cells "
                    "are not independent training replicates"
                ),
            }
        ],
        "extensions": {
            "defender_transfer": {
                "schema_version": 1,
                "source": "manifest.json",
                "values": {"manifest_sha256": identity, "phase": "screening", "confirmatory_selection": False},
            }
        },
    }


def prepare(archive_root, output, source_revision):
    archive_root = Path(archive_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.is_relative_to(REPO_ROOT) or output.is_relative_to(archive_root):
        raise ValueError("Preparation output must be outside both source checkout and archive inputs")
    if output.exists():
        raise FileExistsError(f"Preparation requires a new output directory: {output}")
    source = source_identity(source_revision)
    policies = policy_catalog(archive_root)
    benchmark = archived_benchmark(archive_root)
    output.mkdir(parents=True)
    report = {"status": "running", "source": source, "archive_root": str(archive_root), "output": str(output)}
    write_json(output / "preparation.json", report)
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        banks, discrepancies = generate_banks(output, benchmark)
        write_json(output / "topology-inventory.json", banks)
        if discrepancies:
            report["benchmark_discrepancies"] = discrepancies
            raise ValueError("Reconstructed CIA benchmark differs from archived canonical topology or role identity")
        manifest = {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "source_revision": source_revision,
            "preparation_source": source,
            "archive_root": str(archive_root),
            "policies": policies,
            "topology_banks": banks,
            "archived_cia_benchmark": benchmark,
            "screening": {
                "seeds": SEEDS,
                "episodes_per_seed": 1,
                "horizon": 500,
                "stochastic": True,
                "deterministic": False,
                "default_stage": "final",
                "confirmatory_selection": False,
            },
            "cells": study_cells(policies, banks, output),
            "limitations": [
                "Only two training runs per condition; checkpoint ages and episodes do not add training replication.",
                "Matching training seed labels across conditions does not establish paired training randomness.",
                "Stock layouts use declared seeds and current source; original training snapshots were unavailable.",
                "The CIA bank changes operational-server configuration and roles and is a separate benchmark.",
                (
                    "Scripted and learned executors have different episode RNG contracts; compare policies "
                    "within each executor, not paired scripted-versus-learned differences."
                ),
                "Older checkpoints are planned but deferred; exploratory scores do not select a confirmatory winner.",
            ],
        }
        protocol = protocol_for(manifest)
        from jaxborg.experiment_records.core import validate_records

        validate_records([protocol])
        if source_identity(source_revision) != source:
            raise ValueError("Preparation source or installed dependencies changed during generation")
        write_json(output / "manifest.json", manifest)
        write_json(output / "protocol.json", protocol)
        report.update(
            status="succeeded",
            protocol_id=protocol["id"],
            cells=len(manifest["cells"]),
            blocked_cells=[cell["id"] for cell in manifest["cells"] if not cell["executable"]],
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "preparation.json", report)
        raise
    write_json(output / "preparation.json", report)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    manifest = prepare(args.archive_root, args.output, args.source_revision)
    print(json.dumps({"output": str(args.output.resolve()), "expected_cells": len(manifest["cells"])}))


if __name__ == "__main__":
    main()
