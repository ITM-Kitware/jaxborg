"""Normalize local evidence without inferring scientific membership from names."""

import hashlib
import json
import math
import sqlite3
import statistics
from pathlib import Path


def known(value, source):
    return {"value": value, "source": source}


def unknown(reason, source):
    return {"value": None, "reason": reason, "source": source}


def record(kind, identity, **fields):
    return {"schema_version": 1, "kind": kind, "id": identity, **fields}


def status(value):
    statuses = {
        "FINISHED": "succeeded",
        "FAILED": "failed",
        "RUNNING": "running",
        "SCHEDULED": "planned",
        "KILLED": "cancelled",
    }
    if value not in statuses:
        raise ValueError(f"Unrecognized retained execution status: {value}")
    return statuses[value]


def file_record(path, identity, producer=None, role="retained evidence"):
    path = Path(path).resolve()
    return record(
        "artifact",
        identity,
        producer_id=producer,
        uri=path.as_uri(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
        role=role,
    )


def frozen_cells(protocol, state, pid):
    """Keep intended cells even when no corresponding execution was launched."""
    trials, cells, episode_budgets = [], {}, {}

    def assign(run_id, cell):
        if run_id in cells:
            raise ValueError("One run is mapped to multiple intended scientific trials")
        cells[run_id] = cell

    for seed in protocol["training_seeds"]:
        name = f"seed-{seed}"
        cell = dict(
            id=f"{pid}:training:{name}",
            condition_id="fresh-red-frozen-blue",
            replicate_id=str(seed),
            matched_unit_id=None,
        )
        trials.append(cell)
        attempt = state.get("training", {}).get(name)
        if attempt:
            assign(attempt["run_id"], cell)
    for cohort, candidates, requested in (
        ("validation", protocol["selection"]["candidates"], protocol["validation_episodes"]),
        ("test", ["original", "selected"], protocol["final_test_episodes"]),
    ):
        for candidate in candidates:
            cell = dict(
                id=f"{pid}:{cohort}:{candidate}",
                condition_id=f"{cohort}:{candidate}",
                replicate_id="one-source-policy",
                matched_unit_id=cohort,
            )
            trials.append(cell)
            reference = state.get(cohort, {}).get(candidate)
            if reference:
                if not reference.startswith("runs:/") or "/" not in reference[6:]:
                    raise ValueError("Invalid saved evaluation run reference")
                run_id = reference[6:].split("/", 1)[0]
                assign(run_id, cell)
                episode_budgets[run_id] = requested
    return trials, cells, episode_budgets


def frozen_bundle(root):
    """Import indexed bytes and explicit manifest relationships from one bundle."""
    root = Path(root).resolve()
    index = json.loads((root / "index.json").read_text())
    if index["format"] != "frozen-defender-response-v1":
        raise ValueError("Unsupported bundle")
    paths, artifacts = {}, []
    namespace = "frozen:" + hashlib.sha256(index["source_store"].encode()).hexdigest()
    for name, entry in sorted(index["files"].items()):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Bundle path escapes root")
        artifact = file_record(path, f"{namespace}:artifact:{name}")
        if artifact["sha256"] != entry["sha256"] or artifact["size_bytes"] != entry["size"]:
            raise ValueError(f"Bundle hash mismatch: {name}")
        artifacts.append(artifact)
        if entry.get("reference"):
            if entry["reference"] in paths:
                raise ValueError("Duplicate indexed artifact reference")
            paths[entry["reference"]] = (path, artifact)
    protocol_path = paths[index["protocol"]][0]
    protocol = json.loads(protocol_path.read_text())
    pid = f"{namespace}:protocol:{protocol['campaign']}"
    state_path = paths[index["state"]][0]
    state = json.loads(state_path.read_text())
    trials, cells, episode_budgets = frozen_cells(protocol, state, pid)
    manifests = [
        (ref, json.loads(path.read_text()))
        for ref, (path, _) in paths.items()
        if ref.endswith("/manifest.json") and "/protocol/" not in ref
    ]
    manifests = [(ref, owner) for ref, owner in manifests if "run_id" in owner and "kind" in owner]
    attempts, evaluations = [], []
    for ref, owner in manifests:
        if ref != f"runs:/{owner['run_id']}/manifest.json":
            raise ValueError("Producer manifest run identity differs from index")
        aid = f"{namespace}:attempt:{owner['run_id']}"
        cell = cells.get(owner["run_id"])
        tid = (
            cell["id"]
            if cell
            else unknown("Source/report evidence is not an independent scientific trial", str(paths[ref][0]))
        )
        inputs = [
            paths[item["canonical_reference"]][1]["id"]
            for item in owner.get("inputs", [])
            if item.get("canonical_reference") in paths
        ]
        facts = {
            key: known(owner[key], f"{paths[ref][0]}#/{key}")
            for key in ("seed", "source", "environment", "effective_config", "game_contract", "seed_contract")
            if owner.get(key) is not None
        }
        facts["unretained_inputs"] = known(
            [item for item in owner.get("inputs", []) if item.get("canonical_reference") not in paths],
            f"{paths[ref][0]}#/inputs",
        )
        facts["source_store"] = known(index["source_store"], f"{root / 'index.json'}#/source_store")
        attempts.append(
            record(
                "attempt",
                aid,
                protocol_id=pid,
                trial_id=tid,
                status=status(owner["status"]),
                execution_kind=owner["kind"],
                retry_of=unknown(
                    "Retained protocol/state select this run but do not record retry genealogy", str(paths[ref][0])
                ),
                facts=facts,
                budgets=[
                    {
                        "unit": "environment_steps",
                        "requested": owner.get("requested_steps")
                        if owner.get("requested_steps") is not None
                        else unknown("Not recorded in manifest", str(paths[ref][0])),
                        "completed": owner.get("actual_steps")
                        if owner.get("actual_steps") is not None
                        else unknown("Not recorded in manifest", str(paths[ref][0])),
                    }
                ],
                inputs=inputs,
            )
        )
        for reference, (_, artifact) in paths.items():
            if reference.startswith(f"runs:/{owner['run_id']}/"):
                relative_name = reference.split("/", 2)[2]
                if relative_name != "manifest.json":
                    output = owner["outputs"].get(relative_name)
                    if (
                        not output
                        or not output["complete"]
                        or output["sha256"] != artifact["sha256"]
                        or output["size"] != artifact["size_bytes"]
                    ):
                        raise ValueError(f"Producer output identity or completion differs: {relative_name}")
                artifact["producer_id"] = aid
        result_ref = f"runs:/{owner['run_id']}/evaluations/result.json"
        if result_ref in paths:
            path, raw = paths[result_ref]
            result = json.loads(path.read_text())
            efacts = {
                key: known(result[key], f"{path}#/{key}")
                for key in ("seeds", "policies", "eval_env", "policy_backend", "variant")
                if key in result
            }
            efacts["game_contract"] = known(owner["game_contract"], f"{paths[ref][0]}#/game_contract")
            efacts["unretained_inputs"] = facts["unretained_inputs"]
            for key in ("blue_mean_return", "blue_std_return", "n_episodes", "total_episodes"):
                efacts[key] = known(result[key], f"{path}#/{key}")
            returns = result["per_episode_blue_returns"]
            if len(returns) < 2 or not math.isclose(
                statistics.stdev(returns), result["blue_std_return"], rel_tol=1e-12
            ):
                raise ValueError("Recorded episode SD does not match sample SD from retained returns")
            if not math.isclose(statistics.mean(returns), result["blue_mean_return"], rel_tol=1e-12):
                raise ValueError("Recorded episode mean differs from retained returns")
            evaluations.append(
                record(
                    "evaluation",
                    f"{namespace}:evaluation:{owner['run_id']}",
                    attempt_id=aid,
                    inputs=inputs,
                    status=attempts[-1]["status"],
                    facts=efacts,
                    budgets=[
                        {
                            "unit": "episodes",
                            "requested": episode_budgets.get(
                                owner["run_id"],
                                unknown("No authoritative requested episode budget", str(protocol_path)),
                            ),
                            "completed": len(returns),
                        }
                    ],
                    metrics=[
                        {"name": key, "value": result[key], "unit": "score"}
                        for key in ("blue_mean_return", "blue_std_return")
                    ],
                    raw_artifacts=[raw["id"]],
                )
            )
    return [
        record(
            "protocol",
            pid,
            study_id=protocol["campaign"],
            revision=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            selection_rule="explicit",
            required_facts={
                "attempt": ["source_store", "source", "game_contract"],
                "evaluation": ["seeds", "policies", "game_contract"],
            },
            trials=trials,
            metric_definitions=[
                {
                    "name": "blue_mean_return",
                    "unit": "score",
                    "direction": "higher",
                    "aggregation": "arithmetic mean",
                    "sampling_unit": "evaluation episode conditional on fixed policies",
                    "uncertainty": "none in imported mean; paired effect interval computed by saved analysis",
                },
                {
                    "name": "blue_std_return",
                    "unit": "score",
                    "direction": "neutral",
                    "aggregation": "sample standard deviation (ddof=1)",
                    "sampling_unit": "evaluation episode conditional on fixed policies",
                    "uncertainty": "descriptive episode variability, not effect interval or training-seed uncertainty",
                },
            ],
            extensions={
                "frozen_bundle": {
                    "schema_version": 1,
                    "source": str(root / "index.json"),
                    "values": {
                        "bundle_root": str(root),
                        "bundle_index_sha256": hashlib.sha256((root / "index.json").read_bytes()).hexdigest(),
                        "protocol_source": str(protocol_path),
                        "state_source": str(state_path),
                    },
                }
            },
        ),
        *attempts,
        *evaluations,
        *artifacts,
        file_record(root / "index.json", f"{namespace}:bundle-index"),
    ]


def smoke_store(root):
    """Keep every SQLite run distinct and leave unlinked adjacent files unowned."""
    root = Path(root).resolve()
    database = root / "mlflow.db"
    namespace = "sqlite:" + hashlib.sha256(str(database).encode()).hexdigest()
    attempts = []
    if Path(str(database) + "-wal").exists():
        raise ValueError("Retained database must be checkpointed without a WAL before import")
    database_sha256 = hashlib.sha256(database.read_bytes()).hexdigest()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("BEGIN")
        connection.row_factory = sqlite3.Row
        for run in connection.execute("SELECT * FROM runs ORDER BY run_uuid"):
            rid = run["run_uuid"]
            source = f"{database}#runs[run_uuid={rid}]"
            params = dict(connection.execute("SELECT key,value FROM params WHERE run_uuid=? ORDER BY key", (rid,)))
            tags = dict(connection.execute("SELECT key,value FROM tags WHERE run_uuid=? ORDER BY key", (rid,)))

            metrics = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM metrics WHERE run_uuid=? ORDER BY key,step,timestamp", (rid,)
                )
            ]
            observed_steps = [row["value"] for row in metrics if row["key"] == "env_steps"]
            completed = unknown(
                "Recorded env_steps axis lacks verified cumulative-work semantics for these executions",
                source + "/metrics",
            )
            attempts.append(
                record(
                    "attempt",
                    namespace + ":attempt:" + rid,
                    protocol_id=unknown("No retained authoritative protocol association", source),
                    trial_id=unknown("No retained authoritative intended trial association", source),
                    status=status(run["status"]),
                    execution_kind="training",
                    retry_of=unknown("No retained retry relationship", source),
                    budgets=[
                        {
                            "unit": "environment_steps",
                            "requested": int(params["recipe.train.total_timesteps"])
                            if "recipe.train.total_timesteps" in params
                            else unknown("No requested budget recorded", source + "/params"),
                            "completed": completed,
                        }
                    ],
                    inputs=[],
                    facts={
                        "run": known(dict(run), source),
                        "params": known(params, source + "/params"),
                        "tags": known(tags, source + "/tags"),
                        "metrics": known(metrics, source + "/metrics"),
                        "observed_env_steps_max": known(max(observed_steps), source + "/metrics")
                        if observed_steps
                        else unknown("No env_steps metric retained", source + "/metrics"),
                        "scientific_protocol": unknown("No retained authoritative protocol association", source),
                        "retry_relationship": unknown(
                            "Same name and seed do not establish retry or replication", source
                        ),
                        "checkpoint_link": unknown(
                            "No trustworthy link from either run to adjacent recipe/model files", source
                        ),
                    },
                )
            )
    artifacts = [
        file_record(path, namespace + ":artifact:" + path.relative_to(root).as_posix())
        for path in sorted(root.rglob("*"))
        if path.is_file() and (path == database or path.suffix in {".yaml", ".jsonl", ".safetensors"})
    ]
    if hashlib.sha256(database.read_bytes()).hexdigest() != database_sha256:
        raise ValueError("Retained database changed during import")
    return [*attempts, *artifacts]
