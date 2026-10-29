"""Durable local evidence for explicitly declared training and evaluation trials."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from pathlib import Path

from jaxborg.experiment_records import validate_facts, validate_records

_EPISODE_ROLE = "evaluation episode observation"
_TERMINAL = {"succeeded", "failed", "cancelled"}


@dataclass(frozen=True)
class Capture:
    root: Path
    directory: Path
    attempt_id: str
    evaluation_id: str | None
    inputs: dict[str, Path]
    input_ids: dict[str, str] = field(default_factory=dict)


def json_configuration(value):
    """Normalize resolved configuration while retaining supplied input bytes separately."""

    def default(item):
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        raise TypeError(f"Unsupported resolved setting: {type(item).__name__}")

    return json.loads(json.dumps(value, default=default, allow_nan=False))


def _path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or any(part in {"..", "."} for part in relative.parts):
        raise ValueError("Evidence paths must be relative and remain inside the store")
    path = root / relative
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Evidence paths must not follow symbolic links")
    return path


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(root, relative, payload, *, exclusive=True):
    path = _path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _path(root, path.relative_to(root).with_name(f".{path.name}.{uuid.uuid4().hex}.tmp"))
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_json(root, relative, value, *, exclusive=True):
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    return _write_bytes(root, relative, payload, exclusive=exclusive)


def _read_json(root, relative):
    return json.loads(_path(root, relative).read_text())


def _store(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = _path(root, "store.json")
    if not identity.exists():
        try:
            _write_json(root, "store.json", {"schema_version": 1, "id": uuid.uuid4().hex})
        except FileExistsError:
            pass
    return root, _store_identity(root)


def _store_identity(root):
    store = _read_json(root, "store.json")
    if set(store) != {"schema_version", "id"} or store["schema_version"] != 1:
        raise ValueError("Invalid evaluation store identity")
    if not isinstance(store["id"], str) or uuid.UUID(hex=store["id"]).hex != store["id"]:
        raise ValueError("Invalid evaluation store identity")
    return store["id"]


def declare_protocol(root, protocol):
    """Retain the supplied complete protocol once, rejecting revisions in place."""
    validate_records([protocol])
    if protocol["kind"] != "protocol":
        raise ValueError("Expected a protocol record")
    root, _ = _store(root)
    path = _path(root, "protocol.json")
    if path.exists():
        if _read_json(root, "protocol.json") != protocol:
            raise ValueError("Store already contains a different protocol")
    else:
        try:
            _write_json(root, "protocol.json", protocol)
        except FileExistsError:
            if _read_json(root, "protocol.json") != protocol:
                raise ValueError("Store already contains a different protocol") from None
    return path


def _snapshot(capture):
    relative = capture.directory.relative_to(capture.root)
    if _path(capture.root, relative / "terminal.json").exists():
        raise ValueError("Evaluation is already terminal")
    records = _read_json(capture.root, relative / "current.json")
    if records[0]["id"] != capture.attempt_id or (
        capture.evaluation_id is not None and records[1]["id"] != capture.evaluation_id
    ):
        raise ValueError("Capture identity differs from its retained snapshot")
    return records


def _save(capture, records, *, terminal=False):
    relative = capture.directory.relative_to(capture.root)
    if _path(capture.root, relative / "terminal.json").exists():
        raise ValueError("Evaluation is already terminal")
    if terminal:
        previous_ids = {record["id"] for record in _read_json(capture.root, relative / "current.json")}
        existing = load_records(capture.root, verify=False)
        validate_records([record for record in existing if record["id"] not in previous_ids] + records)
        _check_snapshot(
            capture.root,
            _store_identity(capture.root),
            _read_json(capture.root, "protocol.json"),
            capture.directory,
            records,
            True,
        )
    _write_json(capture.root, relative / ("terminal.json" if terminal else "current.json"), records, exclusive=terminal)


def _record(kind, identity, **fields):
    return {"schema_version": 1, "kind": kind, "id": identity, **fields}


def _artifact(capture, path, role, producer):
    payload = path.read_bytes()
    return _record(
        "artifact",
        f"{capture.attempt_id}:artifact:{uuid.uuid4().hex}",
        producer_id=producer,
        uri=path.relative_to(capture.root).as_posix(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        role=role,
    )


def _copy(capture, source, relative, role, *, producer=None):
    source = Path(source)
    if not source.is_file():
        raise ValueError(f"Retained input is not a file: {source}")
    path = _write_bytes(capture.root, relative, source.read_bytes())
    return path, _artifact(capture, path, role, producer)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _owners(records):
    return [record for record in records if record["kind"] in {"attempt", "evaluation"}]


def start_evaluation(
    root,
    *,
    protocol,
    trial_id,
    retry_of=None,
    requested_episodes,
    resolved_config,
    facts=None,
    inputs: dict[str, Path],
    source_files: dict[str, Path],
    retained_inputs=None,
    correction=None,
):
    """Persist an evaluation identity before copying or loading its inputs."""
    _positive_integer(requested_episodes, "Requested episodes")
    return _start(
        root,
        protocol=protocol,
        trial_id=trial_id,
        retry_of=retry_of,
        execution_kind="evaluation",
        budgets=[{"unit": "episodes", "requested": requested_episodes, "completed": 0}],
        resolved_config=resolved_config,
        facts=facts,
        inputs=inputs,
        source_files=source_files,
        retained_inputs=retained_inputs,
        correction=correction,
    )


def start_training(
    root,
    *,
    protocol,
    trial_id,
    requested_steps,
    requested_updates,
    resolved_config,
    inputs: dict[str, Path],
    source_files: dict[str, Path],
    facts=None,
    retry_of=None,
):
    """Persist a training attempt before initialization or checkpoint creation."""
    _positive_integer(requested_steps, "Requested steps")
    _positive_integer(requested_updates, "Requested updates")
    return _start(
        root,
        protocol=protocol,
        trial_id=trial_id,
        retry_of=retry_of,
        execution_kind="training",
        budgets=[
            {"unit": "environment_steps", "requested": requested_steps, "completed": 0},
            {"unit": "joint_rollout_updates", "requested": requested_updates, "completed": 0},
        ],
        resolved_config=resolved_config,
        facts=facts,
        inputs=inputs,
        source_files=source_files,
    )


def _start(
    root,
    *,
    protocol,
    trial_id,
    retry_of,
    execution_kind,
    budgets,
    resolved_config,
    facts,
    inputs,
    source_files,
    retained_inputs=None,
    correction=None,
):
    resolved_config = json_configuration(resolved_config)
    declare_protocol(root, protocol)
    root, store_id = _store(root)
    existing = load_records(root)
    if trial_id not in {cell["id"] for cell in protocol["trials"]}:
        raise ValueError("Unknown trial in supplied protocol")
    if retry_of is not None:
        previous = next((record for record in existing if record["id"] == retry_of), None)
        if (
            previous is None
            or previous["kind"] != "attempt"
            or previous["protocol_id"] != protocol["id"]
            or previous["trial_id"] != trial_id
            or previous["status"] not in _TERMINAL
        ):
            raise ValueError("Retry must reference a terminal attempt of this protocol and trial")
    execution_id = uuid.uuid4().hex
    directory = _path(root, Path("attempts") / execution_id)
    capture = Capture(
        root,
        directory,
        f"evaluation-store:{store_id}:attempt:{execution_id}",
        f"evaluation-store:{store_id}:evaluation:{execution_id}" if execution_kind == "evaluation" else None,
        {},
    )
    records = [
        _record(
            "attempt",
            capture.attempt_id,
            protocol_id=protocol["id"],
            trial_id=trial_id,
            status="running",
            execution_kind=execution_kind,
            retry_of=retry_of,
            facts=dict(facts or {}),
            budgets=budgets,
            inputs=[],
        ),
    ]
    if capture.evaluation_id is not None:
        records.append(
            _record(
                "evaluation",
                capture.evaluation_id,
                attempt_id=capture.attempt_id,
                inputs=[],
                status="running",
                facts=dict(facts or {}),
                budgets=budgets,
                metrics=[],
                raw_artifacts=[],
            )
        )
        if correction is not None:
            records[1]["extensions"] = {
                "evaluation_correction": {
                    "schema_version": 1,
                    "source": "configuration.json#/correction",
                    "values": correction,
                }
            }
            resolved_config = {**resolved_config, "correction": correction}
    # Retry references belong to the complete store graph, not an isolated attempt.
    validate_records([*existing, *records])
    pending = _path(root, Path(".pending") / execution_id)
    pending.mkdir(parents=True, exist_ok=False)
    _write_json(root, pending.relative_to(root) / "current.json", records)
    directory.parent.mkdir(parents=True, exist_ok=True)
    os.rename(pending, directory)
    _sync_directory(directory.parent)
    try:
        relative = directory.relative_to(root)
        config_path = _write_json(root, relative / "configuration.json", resolved_config)
        config = _artifact(capture, config_path, f"resolved {execution_kind} configuration", capture.attempt_id)
        records.append(config)
        for owner in _owners(records):
            owner["inputs"].append(config["id"])
            owner["facts"]["resolved_config"] = {"value": resolved_config, "source": config["uri"]}
        _save(capture, records)
        copied_inputs = {}
        for role, source in inputs.items():
            if not isinstance(role, str) or not role.strip():
                raise ValueError("Input roles must be nonempty strings")
            original = Path(source).resolve()
            if original in copied_inputs:
                path, artifact = copied_inputs[original]
                capture.inputs[role] = path
                capture.input_ids[role] = artifact["id"]
                continue
            path, artifact = _copy(
                capture,
                source,
                relative
                / "inputs"
                / hashlib.sha256(str(Path(source).resolve().parent).encode()).hexdigest()[:16]
                / Path(source).name,
                role,
            )
            capture.inputs[role] = path
            capture.input_ids[role] = artifact["id"]
            copied_inputs[original] = (path, artifact)
            records.append(artifact)
            for owner in _owners(records):
                owner["inputs"].append(artifact["id"])
        indexed = {record["id"]: record for record in existing}
        for role, identity in (retained_inputs or {}).items():
            artifact = indexed.get(identity)
            if not isinstance(role, str) or not role.strip() or role in capture.inputs:
                raise ValueError("Retained input roles must be distinct nonempty strings")
            if artifact is None or artifact["kind"] != "artifact":
                raise ValueError("Retained input must identify an artifact in this store")
            capture.inputs[role] = _path(root, artifact["uri"])
            capture.input_ids[role] = identity
            for owner in _owners(records):
                if identity not in owner["inputs"]:
                    owner["inputs"].append(identity)
        source_hashes = {}
        for name, source in source_files.items():
            # Validate the key separately so an absolute key cannot discard the source directory.
            _path(root, name)
            _, artifact = _copy(capture, source, relative / "source" / name, f"{execution_kind} source")
            records.append(artifact)
            source_hashes[name] = artifact["sha256"]
            for owner in _owners(records):
                owner["inputs"].append(artifact["id"])
        manifest_path = _write_json(root, relative / "source-files.json", source_hashes)
        manifest = _artifact(capture, manifest_path, f"{execution_kind} source manifest", capture.attempt_id)
        records.append(manifest)
        for owner in _owners(records):
            owner["inputs"].append(manifest["id"])
            owner["facts"]["source_files"] = {"value": source_hashes, "source": manifest["uri"]}
        input_path = _write_json(root, relative / "input-artifacts.json", capture.input_ids)
        input_map = _artifact(capture, input_path, "execution input roles", capture.attempt_id)
        records.append(input_map)
        for owner in _owners(records):
            owner["inputs"].append(input_map["id"])
            owner["facts"]["input_artifacts"] = {"value": dict(capture.input_ids), "source": input_map["uri"]}
        _save(capture, records)
        return capture
    except BaseException as error:
        error._evaluation_capture = capture
        _save(capture, records)
        finish = finish_evaluation if capture.evaluation_id is not None else finish_training
        finish(capture, status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", error=error)
        raise


def _verify(root, records):
    local_ids = {record["id"] for record in records}
    external_inputs = {identity for owner in _owners(records) for identity in owner["inputs"]} - local_ids
    if external_inputs:
        records = [
            *records,
            *(record for record in load_records(root, verify=False) if record["id"] in external_inputs),
        ]
    for record in records:
        if record["kind"] == "artifact":
            try:
                payload = _path(root, record["uri"]).read_bytes()
            except (OSError, ValueError) as error:
                raise ValueError(f"Cannot verify retained artifact {record['uri']}: {error}") from error
            if len(payload) != record["size_bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"Retained artifact differs from its recorded bytes: {record['uri']}")


def add_artifact(capture, path, *, role, name=None):
    """Retain an immutable output copy owned by this execution."""
    records = _snapshot(capture)
    if not isinstance(role, str) or not role.strip():
        raise ValueError("Artifact role must be a nonempty string")
    name = name or Path(path).name
    _path(capture.root, name)
    _, artifact = _copy(
        capture,
        path,
        capture.directory.relative_to(capture.root) / "artifacts" / name,
        role,
        producer=capture.evaluation_id or capture.attempt_id,
    )
    records.append(artifact)
    if capture.evaluation_id is not None:
        records[1]["raw_artifacts"].append(artifact["id"])
    _save(capture, records)
    return artifact["id"]


def _episodes(capture, records):
    observations = [
        _read_json(capture.root, record["uri"])
        for record in records
        if record["kind"] == "artifact"
        and record["role"] == _EPISODE_ROLE
        and Path(record["uri"]).parent == capture.directory.relative_to(capture.root) / "episodes"
    ]
    requested = records[0]["budgets"][0]["requested"]
    seen = set()
    for observation in observations:
        required = {"episode_index", "seed", "reward"}
        if not isinstance(observation, dict) or not required <= observation.keys():
            raise ValueError("Malformed retained episode observation")
        if observation.keys() - required - {"policy_seed", "steps"}:
            raise ValueError("Unknown retained episode fields")
        index = observation["episode_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < requested or index in seen:
            raise ValueError("Duplicate or out-of-budget retained episode index")
        seen.add(index)
        reward = observation["reward"]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("Retained episode reward must be finite")
        for key in ("seed", "policy_seed", "steps"):
            if key in observation:
                value = observation[key]
                if isinstance(value, bool) or not isinstance(value, int) or (key == "steps" and value < 0):
                    raise ValueError(f"Invalid retained episode {key}")
    return observations


def _progress(records, observations):
    for owner in _owners(records):
        owner["budgets"] = _episode_budgets(owner["budgets"][0]["requested"], observations)


def _episode_budgets(requested, observations):
    budgets = [{"unit": "episodes", "requested": requested, "completed": len(observations)}]
    if observations and all("steps" in observation for observation in observations):
        budgets.append(
            {
                "unit": "completed_episode_environment_steps",
                "requested": {
                    "value": None,
                    "reason": "Only completed episode steps are observed",
                    "source": "episode observations",
                },
                "completed": sum(observation["steps"] for observation in observations),
            }
        )
    return budgets


def _evaluation_metrics(protocol, status, observations):
    if status == "succeeded" and any(
        metric["name"] == "blue_return"
        and metric["unit"] == "score"
        and metric["aggregation"] in {"mean", "arithmetic mean"}
        for metric in protocol["metric_definitions"]
    ):
        return [
            {"name": "blue_return", "value": statistics.mean(row["reward"] for row in observations), "unit": "score"}
        ]
    return []


def record_episode(capture, *, episode_index, seed, reward, policy_seed=None, steps=None):
    """Retain one completed episode before reporting its budget as completed."""
    records = _snapshot(capture)
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise ValueError("Episode index must be an integer")
    if not 0 <= episode_index < records[0]["budgets"][0]["requested"]:
        raise ValueError("Episode index is outside the declared budget")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Episode seed must be an integer")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError("Episode reward must be finite")
    observation = {"episode_index": episode_index, "seed": seed, "reward": reward}
    for key, value in (("policy_seed", policy_seed), ("steps", steps)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or (key == "steps" and value < 0):
                raise ValueError(f"Invalid episode {key}")
            observation[key] = value
    relative = capture.directory.relative_to(capture.root) / "episodes" / f"{episode_index}.json"
    if any(record["kind"] == "artifact" and record["uri"] == relative.as_posix() for record in records):
        raise FileExistsError(f"Episode index is already retained: {episode_index}")
    path = _write_json(capture.root, relative, observation)
    artifact = _artifact(capture, path, _EPISODE_ROLE, capture.evaluation_id)
    records.append(artifact)
    records[1]["raw_artifacts"].append(artifact["id"])
    for owner in _owners(records):
        budgets = owner["budgets"]
        completed = budgets[0]["completed"]
        owner["budgets"] = [{**budgets[0], "completed": completed + 1}]
        if steps is not None and (completed == 0 or len(budgets) == 2):
            step_budget = _episode_budgets(budgets[0]["requested"], [observation])[1]
            step_budget["completed"] += budgets[1]["completed"] if completed else 0
            owner["budgets"].append(step_budget)
    _save(capture, records)


def finish_evaluation(capture, *, status, error=None):
    """Seal terminal evidence; successful execution requires every declared episode."""
    if status not in _TERMINAL:
        raise ValueError("Completion status must be succeeded, failed, or cancelled")
    if status == "succeeded" and error is not None:
        raise ValueError("Successful evaluation cannot include an execution error")
    records = _snapshot(capture)
    integrity_error = None
    try:
        _verify(capture.root, records)
        observations = _episodes(capture, records)
        _verify_retained_facts(capture, records)
    except (OSError, ValueError) as failure:
        if status == "succeeded":
            raise
        integrity_error = str(failure)
        observations = None
    if observations is not None:
        _progress(records, observations)
    if integrity_error is not None:
        integrity_path = _write_json(
            capture.root,
            capture.directory.relative_to(capture.root) / "integrity-error.json",
            {"message": integrity_error},
        )
        artifact = _artifact(capture, integrity_path, "evaluation evidence integrity failure", capture.evaluation_id)
        records.append(artifact)
        records[1]["raw_artifacts"].append(artifact["id"])
        for owner in _owners(records):
            owner["facts"]["integrity_error"] = {"value": integrity_error, "source": artifact["uri"]}
    if status == "succeeded":
        requested = records[0]["budgets"][0]["requested"]
        if sorted(observation["episode_index"] for observation in observations) != list(range(requested)):
            raise ValueError("Successful evaluation requires all declared episodes")
    protocol = _read_json(capture.root, "protocol.json")
    records[1]["metrics"] = _evaluation_metrics(protocol, status, observations)
    if error is not None:
        error_path = _write_json(
            capture.root,
            capture.directory.relative_to(capture.root) / "error.json",
            {
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        artifact = _artifact(capture, error_path, "evaluation failure", capture.evaluation_id)
        records.append(artifact)
        records[1]["raw_artifacts"].append(artifact["id"])
        for owner in _owners(records):
            owner["facts"]["error"] = {
                "value": {"type": type(error).__name__, "message": str(error)},
                "source": artifact["uri"],
            }
    for owner in _owners(records):
        owner["status"] = status
    _save(capture, records, terminal=True)


def load_records(root, verify=True):
    """Read one snapshot per execution, retaining every declared scientific cell."""
    root = Path(root).resolve()
    store_id = _store_identity(root)
    protocol = _read_json(root, "protocol.json")
    records = [protocol]
    snapshots = []
    attempts = _path(root, "attempts")
    if attempts.exists():
        for directory in sorted(attempts.iterdir()):
            _path(root, directory.relative_to(root))
            if not directory.is_dir():
                raise ValueError("Unexpected entry in evaluation attempts")
            terminal = directory.relative_to(root) / "terminal.json"
            current = directory.relative_to(root) / "current.json"
            is_terminal = _path(root, terminal).exists()
            snapshot = _read_json(root, terminal if is_terminal else current)
            records.extend(snapshot)
            snapshots.append((directory, snapshot, is_terminal))
    indexed = validate_records(records)
    for directory, snapshot, is_terminal in snapshots:
        _check_snapshot(root, store_id, protocol, directory, snapshot, is_terminal)
    if verify:
        _verify(root, records)
        for directory, snapshot, _ in snapshots:
            evaluation_id = snapshot[1]["id"] if len(snapshot) > 1 and snapshot[1]["kind"] == "evaluation" else None
            capture = Capture(root, directory, snapshot[0]["id"], evaluation_id, {})
            if evaluation_id is None:
                updates = _training_updates(capture, snapshot)
                if snapshot[0]["budgets"] != _training_budgets(
                    snapshot[0]["budgets"], updates[-1] if updates else None
                ):
                    raise ValueError("Training budgets differ from retained updates")
                if snapshot[0]["status"] == "succeeded" and len(updates) != snapshot[0]["budgets"][1]["requested"]:
                    raise ValueError("Successful training lacks its declared updates")
                _verify_retained_facts(capture, snapshot)
                continue
            observations = _episodes(capture, snapshot)
            requested = snapshot[0]["budgets"][0]["requested"]
            expected_budgets = _episode_budgets(requested, observations)
            if any(owner["budgets"] != expected_budgets for owner in _owners(snapshot)):
                raise ValueError("Budgets differ from retained episode observations")
            if snapshot[1]["metrics"] != _evaluation_metrics(protocol, snapshot[1]["status"], observations):
                raise ValueError("Metrics differ from retained episode observations and status")
            if snapshot[0]["status"] == "succeeded" and len(observations) != requested:
                raise ValueError("Successful snapshot lacks its declared episodes")
            _verify_retained_facts(capture, snapshot)
    return [indexed[identity] for identity in sorted(indexed)]


def _verify_retained_facts(capture, records):
    relative = capture.directory.relative_to(capture.root)
    execution_kind = records[0]["execution_kind"]
    for fact_name, filename, role in (
        ("resolved_config", "configuration.json", f"resolved {execution_kind} configuration"),
        ("source_files", "source-files.json", f"{execution_kind} source manifest"),
        ("input_artifacts", "input-artifacts.json", "execution input roles"),
    ):
        uri = (relative / filename).as_posix()
        artifacts = [
            artifact
            for artifact in records
            if artifact["kind"] == "artifact"
            if artifact["uri"] == uri or artifact["role"] == role
        ]
        if not artifacts:
            if any(fact_name in owner["facts"] for owner in _owners(records)):
                raise ValueError(f"Fact {fact_name} has no retained supporting artifact")
            continue
        if len(artifacts) != 1 or artifacts[0]["uri"] != uri or artifacts[0]["role"] != role:
            raise ValueError(f"Ambiguous retained artifact for {fact_name}")
        artifact = artifacts[0]
        retained = _read_json(capture.root, uri)
        for owner in _owners(records):
            fact = owner["facts"].get(fact_name)
            if (
                fact is None
                or fact["source"] != uri
                or json.dumps(fact["value"], sort_keys=True, allow_nan=False)
                != json.dumps(retained, sort_keys=True, allow_nan=False)
                or artifact["id"] not in owner["inputs"]
            ):
                raise ValueError(f"Fact {fact_name} differs from its retained supporting artifact")
        if fact_name == "resolved_config":
            correction = next(
                (
                    owner.get("extensions", {}).get("evaluation_correction")
                    for owner in _owners(records)
                    if owner["kind"] == "evaluation"
                ),
                None,
            )
            if (correction["values"] if correction else None) != retained.get("correction"):
                raise ValueError("Correction relation differs from retained execution configuration")
        if fact_name == "input_artifacts":
            if not isinstance(retained, dict) or any(
                not isinstance(role_name, str)
                or not role_name.strip()
                or any(identity not in owner["inputs"] for owner in _owners(records))
                for role_name, identity in retained.items()
            ):
                raise ValueError("Retained input roles differ from execution inputs")
        if fact_name == "source_files":
            source_root = relative / "source"
            source_artifacts = [
                artifact
                for artifact in records
                if artifact["kind"] == "artifact"
                if artifact["role"] == f"{execution_kind} source"
            ]
            expected = {}
            for source in source_artifacts:
                path = Path(source["uri"])
                if not path.is_relative_to(source_root):
                    raise ValueError("Retained source artifact lies outside the source snapshot")
                name = path.relative_to(source_root).as_posix()
                if name in expected:
                    raise ValueError("Duplicate retained source artifact path")
                expected[name] = source["sha256"]
                if any(source["id"] not in owner["inputs"] for owner in _owners(records)):
                    raise ValueError("Retained source artifact is missing from execution inputs")
            if retained != expected:
                raise ValueError("Source manifest differs from retained source artifact paths and hashes")


def _check_snapshot(root, store_id, protocol, directory, records, terminal):
    if not records or records[0]["kind"] != "attempt":
        raise ValueError("Snapshot must begin with its attempt")
    attempt = records[0]
    owners = _owners(records)
    expected_kinds = ["attempt", "evaluation"] if attempt["execution_kind"] == "evaluation" else ["attempt"]
    if [owner["kind"] for owner in owners] != expected_kinds:
        raise ValueError("Snapshot contains unexpected execution records")
    evaluation = owners[1] if len(owners) == 2 else None
    if attempt["execution_kind"] not in {"training", "evaluation"}:
        raise ValueError("Unsupported retained execution kind")
    if (attempt["execution_kind"] == "evaluation") != (evaluation is not None):
        raise ValueError("Snapshot must begin with its attempt and evaluation")
    if records[: len(owners)] != owners or any(record["kind"] != "artifact" for record in records[len(owners) :]):
        raise ValueError("Snapshot contains unexpected execution records")
    execution_id = directory.name
    if uuid.UUID(hex=execution_id).hex != execution_id:
        raise ValueError("Invalid attempt directory identity")
    if (
        attempt["id"] != f"evaluation-store:{store_id}:attempt:{execution_id}"
        or attempt["protocol_id"] != protocol["id"]
    ):
        raise ValueError("Snapshot identity differs from store, directory, or protocol")
    if evaluation is not None:
        if (
            evaluation["id"] != f"evaluation-store:{store_id}:evaluation:{execution_id}"
            or evaluation["attempt_id"] != attempt["id"]
        ):
            raise ValueError("Snapshot identity differs from store, directory, or protocol")
        if attempt["status"] != evaluation["status"] or attempt["budgets"] != evaluation["budgets"]:
            raise ValueError("Attempt and evaluation status and budgets must match")
    if (terminal and attempt["status"] not in _TERMINAL) or (not terminal and attempt["status"] != "running"):
        raise ValueError("Snapshot status differs from its lifecycle location")
    units = ["episodes"] if evaluation is not None else ["environment_steps", "joint_rollout_updates"]
    for owner in owners:
        budgets = owner["budgets"]
        if len(budgets) < len(units):
            raise ValueError("Snapshot lacks its declared work budget")
        for budget, unit in zip(budgets, units):
            if (
                budget["unit"] != unit
                or type(budget["requested"]) is not int
                or budget["requested"] <= 0
                or type(budget["completed"]) is not int
                or not 0 <= budget["completed"] <= budget["requested"]
            ):
                raise ValueError("Snapshot lacks a positive integer work budget")
    relative = directory.relative_to(root)
    for artifact in records[len(owners) :]:
        uri = Path(artifact["uri"])
        if uri.is_absolute() or ".." in uri.parts or not uri.is_relative_to(relative):
            raise ValueError("Artifact lies outside its execution directory")


def _training_updates(capture, records):
    updates = [
        _read_json(capture.root, record["uri"])
        for record in records
        if record["kind"] == "artifact" and record["role"] == "training update observation"
    ]
    updates.sort(key=lambda row: row["update_idx"])
    budgets = _training_budgets(records[0]["budgets"])
    for row in updates:
        _validate_training_update(row, budgets)
        budgets = _training_budgets(budgets, row)
    return updates


def _validate_training_update(row, budgets):
    index = budgets[1]["completed"] + 1
    if type(row.get("update_idx")) is not int or row["update_idx"] != index:
        raise ValueError("Training updates must be consecutive and unique")
    if index > budgets[1]["requested"]:
        raise ValueError("Training updates exceed the requested budget")
    steps = row.get("env_steps")
    if type(steps) is not int or not budgets[0]["completed"] < steps <= budgets[0]["requested"]:
        raise ValueError("Training environment steps must increase within the requested budget")
    for offset, team in enumerate(("blue", "red"), 2):
        count = row.get(f"team.{team}.update_count")
        previous = budgets[offset]["completed"] if budgets[1]["completed"] else 0
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or not math.isfinite(count)
            or int(count) != count
            or not previous <= count <= index
        ):
            raise ValueError("Training must retain cumulative per-team update counts")


def _training_budgets(budgets, last=None):
    last = last or {}
    result = [
        {**budgets[0], "completed": last.get("env_steps", 0)},
        {**budgets[1], "completed": last.get("update_idx", 0)},
    ]
    if last:
        result.extend(
            {
                "unit": f"{team}_policy_updates",
                "requested": {
                    "value": None,
                    "reason": "Per-team update schedule is retained in resolved configuration",
                    "source": "configuration.json",
                },
                "completed": int(last[f"team.{team}.update_count"]),
            }
            for team in ("blue", "red")
        )
    return result


def record_training_update(capture, row):
    """Persist one completed outer-loop update and its actual per-team work."""
    records = _snapshot(capture)
    if capture.evaluation_id is not None:
        raise ValueError("Training updates require a training attempt")
    _validate_training_update(row, records[0]["budgets"])
    relative = capture.directory.relative_to(capture.root) / "updates" / f"{row['update_idx']}.json"
    path = _write_json(capture.root, relative, row)
    records.append(_artifact(capture, path, "training update observation", capture.attempt_id))
    records[0]["budgets"] = _training_budgets(records[0]["budgets"], row)
    _save(capture, records)


def finish_training(capture, *, status, error=None):
    """Seal completed training work and owned outputs, including partial failures."""
    if status not in _TERMINAL or capture.evaluation_id is not None:
        raise ValueError("Training completion requires a training attempt and terminal status")
    if status == "succeeded" and error is not None:
        raise ValueError("Successful training cannot include an execution error")
    records = _snapshot(capture)
    try:
        _verify(capture.root, records)
        updates = _training_updates(capture, records)
        _verify_retained_facts(capture, records)
    except (OSError, ValueError) as integrity_error:
        if status == "succeeded":
            raise
        updates = None
        path = _write_json(
            capture.root,
            capture.directory.relative_to(capture.root) / "integrity-error.json",
            {"message": str(integrity_error)},
        )
        artifact = _artifact(capture, path, "training evidence integrity failure", capture.attempt_id)
        records.append(artifact)
        records[0]["facts"]["integrity_error"] = {"value": str(integrity_error), "source": artifact["uri"]}
    if updates is not None:
        records[0]["budgets"] = _training_budgets(records[0]["budgets"], updates[-1] if updates else None)
    if status == "succeeded" and len(updates) != records[0]["budgets"][1]["requested"]:
        raise ValueError("Successful training requires all declared updates")
    if error is not None:
        value = {"type": type(error).__name__, "message": str(error)}
        path = _write_json(capture.root, capture.directory.relative_to(capture.root) / "error.json", value)
        artifact = _artifact(capture, path, "training failure", capture.attempt_id)
        records.append(artifact)
        records[0]["facts"]["error"] = {"value": value, "source": artifact["uri"]}
    records[0]["status"] = status
    _save(capture, records, terminal=True)


def record_facts(capture, facts):
    """Add sourced execution facts without replacing already retained facts."""
    records = _snapshot(capture)
    validate_facts(facts)
    for owner in _owners(records):
        for name, fact in facts.items():
            if name in owner["facts"] and owner["facts"][name] != fact:
                raise ValueError(f"Cannot replace retained execution fact: {name}")
            owner["facts"][name] = fact
    _save(capture, records)
