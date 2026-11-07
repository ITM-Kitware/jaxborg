"""Summarize explicitly selected frozen-defender evaluations using only the standard library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import platform
import statistics
from collections import defaultdict
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite observation, got {value!r}")
    return float(value)


def table(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def artifact_path(root, artifact):
    relative = Path(artifact["uri"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact URI must remain inside the evidence store")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact resolves outside the evidence store")
    if path.stat().st_size != artifact["size_bytes"] or digest(path) != artifact["sha256"]:
        raise ValueError(f"Artifact byte identity differs: {artifact['id']}")
    return path


def inventory(root):
    attempts, artifacts = [], {}
    for directory in sorted((root / "attempts").glob("*")):
        terminal = directory / "terminal.json"
        path = terminal if terminal.exists() else directory / "current.json"
        records = read_json(path)
        owners = [row for row in records if row["kind"] == "attempt"]
        evaluations = [row for row in records if row["kind"] == "evaluation"]
        if len(owners) != 1 or len(evaluations) != 1:
            raise ValueError(f"Expected one attempt and evaluation in {path}")
        owner, evaluation = owners[0], evaluations[0]
        if evaluation["attempt_id"] != owner["id"] or evaluation["status"] != owner["status"]:
            raise ValueError(f"Inconsistent execution identity/status: {path}")
        attempts.append(
            {
                "attempt": owner,
                "evaluation": evaluation,
                "records": records,
                "terminal": terminal.exists(),
                "snapshot": str(path),
                "sha256": digest(path),
            }
        )
        for row in records:
            if row["kind"] == "artifact":
                if row["id"] in artifacts and artifacts[row["id"]] != row:
                    raise ValueError("Conflicting artifact identity")
                artifacts[row["id"]] = row
    return attempts, artifacts


def selected_payload(root, attempt, artifacts, cell, manifest):
    records = attempt["records"]
    verified = {}
    for record in records:
        if record["kind"] == "artifact":
            verified[record["id"]] = artifact_path(root, record)
    for identity in attempt["evaluation"]["inputs"]:
        if identity not in verified:
            verified[identity] = artifact_path(root, artifacts[identity])
    configurations = [row for row in records if row.get("role") == "resolved evaluation configuration"]
    input_maps = [row for row in records if row.get("role") == "execution input roles"]
    if len(configurations) != 1 or len(input_maps) != 1:
        raise ValueError("Selected execution lacks unique configuration/input-role artifacts")
    config = read_json(verified[configurations[0]["id"]])
    inputs = read_json(verified[input_maps[0]["id"]])
    if config["execution"]["source_revision"] != manifest["source_revision"]:
        raise ValueError("Selected evaluator source revision differs from manifest")
    if (
        cell["kind"] == "scripted"
        and "recipe_sha256" in cell
        and artifacts[inputs["effective recipe"]]["sha256"] != cell["recipe_sha256"]
    ):
        raise ValueError("Selected effective recipe differs from manifest")
    for team in ("blue", "red"):
        if not cell.get(f"{team}_id"):
            continue
        policy = next(item for item in manifest["policies"][team] if item["id"] == cell[f"{team}_id"])
        for role, expected in ((f"{team} checkpoint", policy["sha256"]), (f"{team} sidecar", policy["sidecar_sha256"])):
            if artifacts[inputs[role]]["sha256"] != expected:
                raise ValueError(f"Selected {role} differs from manifest")
    snapshots = manifest["topology_banks"][cell["bank_id"]]["snapshots"]
    hashes = [item["sha256"] for item in snapshots]
    if config["topology"]["sha256"] != hashes:
        raise ValueError("Selected topology bank differs from manifest")
    if len(set(hashes)) != len(hashes):
        raise ValueError("Topology bank contains repeated snapshot bytes")
    topology_paths = {}
    for index, expected in enumerate(hashes):
        artifact = artifacts[inputs[f"topology {index}"]]
        if artifact["sha256"] != expected:
            raise ValueError("Retained topology input differs from manifest")
        topology_paths[artifact["uri"]] = expected
    settings = manifest["screening"]
    variant = config["base_variant" if cell["kind"] == "scripted" else "variant"]
    if "variant" in cell and variant != cell["variant"]:
        raise ValueError("Selected game variant differs from declared cell")
    if variant["num_steps"] != settings["horizon"] or config["deterministic"]:
        raise ValueError("Selected horizon or policy sampling differs from manifest")
    actual_seeds = config.get("seeds", config.get("seed_roles", {}).get("episode_roots"))
    if sorted(actual_seeds) != sorted(settings["seeds"]) or config["episodes_per_seed"] != 1:
        raise ValueError("Selected episode roots differ from manifest")
    role = "scripted opponent episode results" if cell["kind"] == "scripted" else "joint matchup result"
    results = []
    for artifact in records:
        if artifact.get("role") == role:
            data = verified[artifact["id"]].read_text()
            results.extend(json.loads(line) for line in data.splitlines() if line.strip())
    return config, results, topology_paths


def extract_episodes(cell, blue, evaluation_id, results, topology_paths, manifest):
    scripted = cell["kind"] == "scripted"
    expected_reds = cell["reds"] if scripted else [cell["red_id"]]
    if len(results) != len(expected_reds):
        raise ValueError("Selected execution lacks the declared opponent result count")
    episodes = []
    seen_reds = set()
    expected_seeds = set(manifest["screening"]["seeds"])
    for result in results:
        opponent = result["eval_red"] if scripted else cell["red_id"]
        if opponent not in expected_reds or opponent in seen_reds:
            raise ValueError("Unexpected or duplicate result opponent")
        seen_reds.add(opponent)
        rewards = result["per_episode"] if scripted else result["blue_returns"]
        seeds = result["per_episode_seeds"] if scripted else result["episode_seeds"]
        paths = result["per_episode_topology_paths"] if scripted else result["episode_topology_paths"]
        roles = result.get("episode_role_map_ids") or [""] * len(rewards)
        fingerprints = result.get(
            "per_episode_topology_fingerprints" if scripted else "episode_topology_fingerprints"
        ) or [""] * len(rewards)
        cia = result.get("per_episode_cia") or [{} for _ in rewards]
        if any(len(values) != len(rewards) for values in (seeds, paths, roles, fingerprints, cia)):
            raise ValueError("Episode columns have different lengths")
        cases = set()
        for reward, seed, path, role, fingerprint, scores in zip(rewards, seeds, paths, roles, fingerprints, cia):
            matches = [value for uri, value in topology_paths.items() if str(path).endswith("/" + uri)]
            if len(matches) != 1:
                raise ValueError("Episode topology does not match a retained input")
            topology = matches[0]
            key = topology, seed
            if key in cases or seed not in expected_seeds:
                raise ValueError("Duplicate or undeclared episode case")
            if scripted and (not role or not fingerprint):
                raise ValueError("Scripted episode lacks fixed topology/role identity")
            cases.add(key)
            episodes.append(
                {
                    "trial_id": cell["trial_id"],
                    "evaluation_id": evaluation_id,
                    "blue_id": blue["id"],
                    "condition": blue["condition"],
                    "training_seed": blue["training_seed"],
                    "stage": blue["stage"],
                    "kind": cell["kind"],
                    "bank_id": cell["bank_id"],
                    "opponent": opponent,
                    "topology_sha256": topology,
                    "topology_fingerprint": fingerprint,
                    "role_map_id": role,
                    "episode_seed": seed,
                    "reward": finite(reward),
                    **{name: finite(scores[name]) if name in scores else None for name in ("c", "i", "a")},
                }
            )
        if cases != {(topology, seed) for topology in topology_paths.values() for seed in expected_seeds}:
            raise ValueError("Selected result is missing declared topology/seed cases")
    return episodes


IDENTITY = (
    "trial_id",
    "evaluation_id",
    "blue_id",
    "condition",
    "training_seed",
    "stage",
    "kind",
    "bank_id",
    "opponent",
)
METRICS = ("reward", "c", "i", "a")


def summaries(episodes):
    groups = defaultdict(list)
    for row in episodes:
        groups[tuple(row[key] for key in IDENTITY) + (row["topology_sha256"],)].append(row)
    output = []
    for keys, values in sorted(groups.items()):
        output.append(
            {
                **dict(zip((*IDENTITY, "topology_sha256"), keys)),
                "n_episodes": len(values),
                **{
                    name: statistics.mean(row[name] for row in values) if values[0][name] is not None else None
                    for name in METRICS
                },
            }
        )
    return output


def contrasts(episodes):
    policies = defaultdict(dict)
    metadata = {}
    for row in episodes:
        group = row["kind"], row["bank_id"], row["opponent"], row["blue_id"]
        case = row["topology_sha256"], row["role_map_id"], row["episode_seed"]
        if case in policies[group]:
            raise ValueError("Multiple selected trials contain the same Blue/opponent/bank case")
        policies[group][case] = row
        metadata[group] = row
    output = []
    for group, candidate in policies.items():
        meta = metadata[group]
        for other, reference in policies.items():
            ref = metadata[other]
            if group[:3] != other[:3] or meta["training_seed"] != ref["training_seed"]:
                continue
            name = None
            if meta["condition"] == ref["condition"] and meta["stage"] == "final" and ref["stage"] == "early":
                name = "final_minus_early"
            if meta["stage"] == ref["stage"] and meta["condition"] == "diverse" and ref["condition"] == "single":
                name = "diverse_minus_single"
            if name is None:
                continue
            if candidate.keys() != reference.keys():
                raise ValueError("Contrast requires exactly matched topology, role and episode identities")
            topology_ids = sorted({key[0] for key in candidate})
            for topology in [*topology_ids, "ALL_EQUAL_TOPOLOGY_WEIGHT"]:
                keys = [key for key in candidate if topology == "ALL_EQUAL_TOPOLOGY_WEIGHT" or key[0] == topology]
                output.append(
                    {
                        "contrast": name,
                        "kind": group[0],
                        "bank_id": group[1],
                        "opponent": group[2],
                        "training_seed": meta["training_seed"],
                        "candidate_blue_id": group[3],
                        "reference_blue_id": other[3],
                        "candidate_evaluation_id": meta["evaluation_id"],
                        "reference_evaluation_id": ref["evaluation_id"],
                        "topology_sha256": topology,
                        "n_paired_episodes": len(keys),
                        **{
                            f"{metric}_mean_difference": statistics.mean(
                                candidate[key][metric] - reference[key][metric] for key in keys
                            )
                            if candidate[keys[0]][metric] is not None and reference[keys[0]][metric] is not None
                            else None
                            for metric in METRICS
                        },
                    }
                )
    return output


def analyze(manifest_path, root, selection_path, output):
    manifest_path, root, selection_path, output = (
        Path(path).resolve() for path in (manifest_path, root, selection_path, output)
    )
    if output.exists():
        raise FileExistsError("Analysis output must be a new directory")
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Generated analysis belongs outside the source checkout")
    manifest, selection = read_json(manifest_path), read_json(selection_path)
    cells = {cell["trial_id"]: cell for cell in manifest["cells"]}
    if len(cells) != len(manifest["cells"]) or not isinstance(selection, dict) or set(selection) - set(cells):
        raise ValueError("Selection must map declared trial IDs to exact evaluation IDs")
    if any(not isinstance(value, str) for value in selection.values()):
        raise ValueError("Every selected evaluation identity must be a string")
    blues = {blue["id"]: blue for blue in manifest["policies"]["blue"]}
    attempts, artifacts = inventory(root)
    by_trial = defaultdict(list)
    for attempt in attempts:
        by_trial[attempt["attempt"]["trial_id"]].append(attempt)
    episodes, coverage, retained = [], [], []
    for trial_id, cell in cells.items():
        available = by_trial[trial_id]
        identity = selection.get(trial_id)
        selected = [attempt for attempt in available if attempt["evaluation"]["id"] == identity]
        status = "no_attempt" if not available else "unselected"
        count = 0
        if identity is not None:
            if len(selected) != 1:
                raise ValueError(f"Selection does not identify one evaluation for {trial_id}")
            attempt = selected[0]
            status = attempt["evaluation"]["status"] if attempt["terminal"] else "running"
            if status == "succeeded":
                config, results, topology_paths = selected_payload(root, attempt, artifacts, cell, manifest)
                values = extract_episodes(cell, blues[cell["blue_id"]], identity, results, topology_paths, manifest)
                episodes.extend(values)
                count = len(values)
                retained.append(
                    {
                        "trial_id": trial_id,
                        "snapshot": attempt,
                        "configuration": config,
                        "results": results,
                        "topology_paths": topology_paths,
                    }
                )
        coverage.append(
            {
                "trial_id": trial_id,
                "blue_id": cell["blue_id"],
                "stage": blues[cell["blue_id"]]["stage"],
                "kind": cell["kind"],
                "bank_id": cell["bank_id"],
                "planned_executable": cell.get("executable", True),
                "blocked_reason": cell.get("blocked_reason") or "",
                "selected_evaluation_id": identity or "",
                "status": status,
                "attempt_count": len(available),
                "succeeded_attempt_count": sum(
                    a["terminal"] and a["evaluation"]["status"] == "succeeded" for a in available
                ),
                "selected_episode_count": count,
            }
        )
    summary, paired = summaries(episodes), contrasts(episodes)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest-input.json", manifest)
    write_json(output / "selection.json", selection)
    write_json(output / "analysis-inputs.json", {"selected_evidence": retained, "attempt_inventory": attempts})
    table(
        output / "coverage.csv",
        coverage,
        (
            "trial_id",
            "blue_id",
            "stage",
            "kind",
            "bank_id",
            "planned_executable",
            "blocked_reason",
            "selected_evaluation_id",
            "status",
            "attempt_count",
            "succeeded_attempt_count",
            "selected_episode_count",
        ),
    )
    table(
        output / "episodes.csv",
        episodes,
        (*IDENTITY, "topology_sha256", "topology_fingerprint", "role_map_id", "episode_seed", *METRICS),
    )
    table(output / "summary.csv", summary, (*IDENTITY, "topology_sha256", "n_episodes", *METRICS))
    table(
        output / "paired-contrasts.csv",
        paired,
        (
            "contrast",
            "kind",
            "bank_id",
            "opponent",
            "training_seed",
            "candidate_blue_id",
            "reference_blue_id",
            "candidate_evaluation_id",
            "reference_evaluation_id",
            "topology_sha256",
            "n_paired_episodes",
            *(f"{metric}_mean_difference" for metric in METRICS),
        ),
    )
    gaps = [row for row in coverage if row["status"] != "succeeded"]
    write_json(output / "gaps.json", gaps)
    (output / "analyze.py").write_bytes(Path(__file__).read_bytes())
    text = (
        f"Screening analysis: {len(coverage) - len(gaps)}/{len(coverage)} declared cells selected and complete; "
        f"{len(episodes)} episodes; {len(gaps)} gaps. No retry was selected automatically. "
        "summary.csv preserves every policy, opponent and topology. paired-contrasts.csv reports final-minus-early "
        "and diverse-minus-single differences against identical fixed opponents and episode cases. "
        "Matching training seed labels do not establish paired training runs. These are conditional descriptive "
        "comparisons of saved policies, with no population confidence intervals or superiority claim. Two training "
        "seeds do not establish a general method effect. Equal numbers of episodes per topology produce equal "
        "topology weights. Scripted and learned-Red executors are separate; numeric seeds do not pair them. "
        "A lower score on reserved networks may reflect network difficulty, not causal transfer failure. "
        "Reconstructed training-seed networks are not verified familiar layouts. Role-based CIA scores are "
        "temporal means of service-impact proxies, not exfiltration/corruption counts. Green failures, defensive "
        "disruption and action success are not measured separately here. No attack-free or simple-defense baseline "
        "is inferred. Missing or unselected cells cannot support a contrast. See coverage.csv and gaps.json."
    )
    (output / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Defender screening</title><p>" + html.escape(text) + "</p>"
    )
    write_json(
        output / "provenance.json",
        {
            "python": platform.python_version(),
            "dependencies": "standard library",
            "manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)},
            "selection": {"path": str(selection_path), "sha256": digest(selection_path)},
            "evidence_root": str(root),
            "verification": (
                "Selected retained artifact bytes checked against capture hashes; not full record-schema validation"
            ),
        },
    )
    write_json(output / "hashes.json", {path.name: digest(path) for path in sorted(output.iterdir())})
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.manifest, args.root, args.selection, args.output)
