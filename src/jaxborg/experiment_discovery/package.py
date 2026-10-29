"""Immutable, verified packages for two retained scientific evidence formats."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from jaxborg.experiment_records import validate_collation, validate_records

_PROJECT = Path(__file__).resolve().parents[3]
_KINDS = {"frozen-defender", "evaluation-capture"}


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs)


def _relative(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Invalid package path: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in name.split("/")) or path.as_posix() != name:
        raise ValueError(f"Invalid package path: {name}")
    return path


def _file(root, name):
    relative = _relative(name)
    path = root.joinpath(*relative.parts)
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError(f"Symbolic link in evidence path: {name}")
    if not path.is_file():
        raise ValueError(f"Missing evidence file: {name}")
    return path


def _identity(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = os.fstat(stream.fileno()).st_size
    return {"sha256": digest, "size_bytes": size}


def _digest(manifest):
    content = {key: value for key, value in manifest.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _frozen(root, provenance_root):
    from jaxborg.experiment_producers.local import _frozen_bundle

    index = _read(_file(root, "index.json"))
    names = ["index.json", *index["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Bundle file closure contains duplicate paths")
    for name in names:
        _file(root, name)
    records = _frozen_bundle(root, provenance_root)
    # This is the maintained analyzer, never code contained in the input bundle.
    analyzer = runpy.run_path(str(_PROJECT / "experiments/frozen-defender-response/analyze.py"))
    verified, references = analyzer["verified_bundle"](root)
    analyzer["validate"](references, verified)
    return records, names


def _capture(root):
    from jaxborg.evaluation.capture_store import load_records

    records = load_records(root, verify=True)
    running = [r["id"] for r in records if r["kind"] == "attempt" and r["status"] == "running"]
    if running:
        raise ValueError("Running capture attempts cannot be packaged: " + ", ".join(running))
    names = ["store.json", "protocol.json"]
    for directory in sorted((root / "attempts").iterdir()) if (root / "attempts").exists() else []:
        names.append((directory / "terminal.json").relative_to(root).as_posix())
    names.extend(r["uri"] for r in records if r["kind"] == "artifact")
    if len(names) != len(set(names)):
        raise ValueError("Capture file closure contains duplicate paths")
    for name in names:
        _file(root, name)
    return records, names


def _snapshot(kind, root, provenance_root):
    if kind == "frozen-defender":
        records, names = _frozen(root, provenance_root)
    elif kind == "evaluation-capture":
        records, names = _capture(root)
    else:
        raise ValueError(f"Unsupported evidence kind: {kind}")
    validate_records(records)
    return records, sorted(names)


def _artifact_map(kind, records, origin):
    result = {}
    for record in records:
        if record["kind"] != "artifact":
            continue
        if kind == "frozen-defender":
            # Frozen locators are produced from this explicit root and indexed name.
            matches = [name for name in origin["names"] if (Path(origin["root"]) / name).as_uri() == record["uri"]]
            if len(matches) != 1:
                raise ValueError(f"Artifact locator is outside original bundle: {record['id']}")
            name = matches[0]
        else:
            name = record["uri"]
        _relative(name)
        result[record["id"]] = {
            "path": "input/" + name,
            "uri": record["uri"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
    return result


def _unretained(records):
    return [
        {"record_id": record["id"], "fact": fact}
        for record in records
        if record["kind"] in {"attempt", "evaluation"}
        for name, fact in record.get("facts", {}).items()
        if name == "unretained_inputs" and fact["value"]
    ]


def _analysis_selection(selection, root):
    from jaxborg.experiment_producers.source import analysis_source_identity

    path = _PROJECT / "experiments/frozen-defender-response/analyze.py"
    if selection["analysis"]["source"] != analysis_source_identity(path):
        raise ValueError("Analyzer source differs from saved selection")
    analyzer = runpy.run_path(str(path))
    index, references = analyzer["verified_bundle"](root)
    protocol = _read(references[index["protocol"]])
    config = analyzer["resolve_config"](protocol, selection["analysis"]["configuration"], {})
    if selection["analysis"]["configuration"] != config:
        raise ValueError("Resolved analysis configuration differs from saved selection")


def _selection_records(selection, records, kind):
    validate_collation(selection)
    indexed = validate_records(selection["records"])
    for record in records:
        if indexed.get(record["id"]) != record:
            raise ValueError(f"Saved record differs from packaged evidence: {record['id']}")
    if kind == "evaluation-capture" and indexed != validate_records(records):
        raise ValueError("Capture collation must contain exactly the authoritative store records")


def _check_selection(selection, records, kind, root):
    _selection_records(selection, records, kind)
    if kind == "frozen-defender":
        # The original provenance records already match; enforce the maintained selection rules.
        from jaxborg.experiment_producers.selection import verify_frozen_selection

        verify_frozen_selection(selection, root)
        _analysis_selection(selection, root)


def fresh_output(output, input_root=None):
    """Resolve a fresh output outside the checkout and optional source evidence."""
    output = Path(output).expanduser().absolute()
    if output.is_symlink() or output.exists():
        raise ValueError("Output already exists; choose a fresh directory")
    output = output.resolve()
    if output.is_relative_to(_PROJECT):
        raise ValueError("Generated evidence must be outside the source checkout")
    if input_root is not None and (output.is_relative_to(input_root) or input_root.is_relative_to(output)):
        raise ValueError("Package output must be separate from input evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def prepare_package(kind, input_root, output, selection=None):
    """Validate a closed input snapshot and expose a new package atomically."""
    root = Path(input_root).expanduser().absolute()
    if root.is_symlink():
        raise ValueError("Evidence root must not be a symbolic link")
    root = root.resolve()
    output = fresh_output(output, root)
    records, names = _snapshot(kind, root, root)
    selected_bytes = None
    if selection is not None:
        selected_bytes = Path(selection).read_bytes()
        _check_selection(json.loads(selected_bytes, object_pairs_hook=_pairs), records, kind, root)
    identities = {name: _identity(_file(root, name)) for name in names}
    with tempfile.TemporaryDirectory(prefix=".evidence-", dir=output.parent) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        for name in names:
            destination = stage / "input" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_file(root, name), destination)
            if _identity(destination) != identities[name]:
                raise ValueError(f"Evidence changed during preparation: {name}")
        (stage / "records.json").write_bytes(_json(records))
        if selected_bytes is not None:
            (stage / "selection.json").write_bytes(selected_bytes)
        repeated, repeated_names = _snapshot(kind, root, root)
        if (
            repeated != records
            or repeated_names != names
            or any(_identity(_file(root, name)) != identities[name] for name in names)
        ):
            raise ValueError("Evidence changed during preparation")
        files = {"input/" + name: identities[name] for name in names}
        files["records.json"] = _identity(stage / "records.json")
        if selected_bytes is not None:
            files["selection.json"] = _identity(stage / "selection.json")
        manifest = {
            "schema_version": 2,
            "kind": kind,
            "input_root": "input",
            "provenance_root": str(root),
            "records": "records.json",
            "selection": "selection.json" if selected_bytes is not None else None,
            "files": files,
            "artifacts": _artifact_map(kind, records, {"root": str(root), "names": names}),
            "unretained_references": _unretained(records),
        }
        manifest["sha256"] = _digest(manifest)
        (stage / "manifest.json").write_bytes(_json(manifest))
        verify_package(stage)
        os.rename(stage, output)
    return manifest


def verify_package(root, expected_digest=None):
    """Verify file integrity and supported producer semantics without executing inputs."""
    root = Path(root).expanduser().absolute()
    if root.is_symlink():
        raise ValueError("Package root must not be a symbolic link")
    root = root.resolve()
    manifest = _read(_file(root, "manifest.json"))
    required = {
        "schema_version",
        "kind",
        "input_root",
        "provenance_root",
        "records",
        "selection",
        "files",
        "artifacts",
        "unretained_references",
        "sha256",
    }
    if set(manifest) != required or manifest["schema_version"] != 2 or manifest["kind"] not in _KINDS:
        raise ValueError("Unsupported evidence package manifest")
    if (
        manifest["input_root"] != "input"
        or manifest["records"] != "records.json"
        or manifest["selection"] not in {None, "selection.json"}
    ):
        raise ValueError("Unexpected package layout")
    if manifest["sha256"] != _digest(manifest) or (
        expected_digest is not None and manifest["sha256"] != expected_digest
    ):
        raise ValueError("Package digest differs from expected identity")
    origin = Path(manifest["provenance_root"])
    if not origin.is_absolute() or ".." in origin.parts or origin.as_posix() != manifest["provenance_root"]:
        raise ValueError("Invalid original provenance root")
    for name, identity in manifest["files"].items():
        if name == "manifest.json" or set(identity) != {"sha256", "size_bytes"}:
            raise ValueError(f"Invalid packaged file: {name}")
        if _identity(_file(root, name)) != identity:
            raise ValueError(f"Package file hash or size differs: {name}")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symbolic link in package: {path.relative_to(root)}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError(f"Unsupported package entry: {path.relative_to(root)}")
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Package file closure differs from manifest")
    records = _read(root / "records.json")
    validate_records(records)
    regenerated, names = _snapshot(manifest["kind"], root / "input", origin)
    if regenerated != records:
        raise ValueError("Packaged records differ from authoritative evidence or original locators")
    expected_files = {"input/" + name for name in names} | {"records.json"}
    if manifest["selection"]:
        expected_files.add("selection.json")
        selected = _read(root / "selection.json")
        _selection_records(selected, records, manifest["kind"])
        if manifest["kind"] == "frozen-defender":
            from jaxborg.experiment_producers.selection import _verify_frozen_records

            _verify_frozen_records(selected, records)
            _analysis_selection(selected, root / "input")
    if set(manifest["files"]) != expected_files:
        raise ValueError("Package contains files outside the authoritative evidence closure")
    if manifest["artifacts"] != _artifact_map(manifest["kind"], records, {"root": str(origin), "names": names}):
        raise ValueError("Artifact materialization map differs from original identity")
    if manifest["unretained_references"] != _unretained(records):
        raise ValueError("Unretained evidence limitations differ from original records")
    return manifest


def load_package_records(root):
    verify_package(root)
    return _read(Path(root) / "records.json")


def package_paths(root):
    manifest = verify_package(root)
    root = Path(root).resolve()
    return {
        "input": str(root / "input"),
        "records": str(root / "records.json"),
        "selection": str(root / "selection.json") if manifest["selection"] else None,
        "materialization": str(root / "manifest.json"),
    }


def frozen_materialization(input_root, descriptor):
    descriptor = Path(descriptor).resolve()
    if descriptor.name != "manifest.json":
        raise ValueError("Materialization requires the package manifest")
    manifest = verify_package(descriptor.parent)
    if manifest["kind"] != "frozen-defender" or Path(input_root).resolve() != descriptor.parent / "input":
        raise ValueError("Materialization does not identify this frozen input root")
    return _read(descriptor.parent / "records.json")


def inspect_package(root):
    manifest = verify_package(root)
    records = _read(Path(root) / "records.json")
    attempts = [r for r in records if r["kind"] == "attempt"]
    protocols = [r for r in records if r["kind"] == "protocol"]
    attempted = {r["trial_id"] for r in attempts if isinstance(r["trial_id"], str)}
    return {
        "sha256": manifest["sha256"],
        "kind": manifest["kind"],
        "records": records,
        "protocols": protocols,
        "attempts": attempts,
        "evaluations": [r for r in records if r["kind"] == "evaluation"],
        "missing": [
            {"protocol_id": p["id"], "trial_id": t["id"], "reason": "no_attempt"}
            for p in protocols
            for t in p["trials"]
            if t["id"] not in attempted
        ],
        "unretained_references": manifest["unretained_references"],
    }


def materialize_package(package, output):
    package = Path(package).resolve()
    manifest = verify_package(package)
    output = fresh_output(output, package)
    with tempfile.TemporaryDirectory(prefix=".materialize-", dir=output.parent) as temporary:
        stage = Path(temporary) / "package"
        shutil.copytree(package, stage, symlinks=True)
        verify_package(stage, manifest["sha256"])
        os.rename(stage, output)
    return package_paths(output)
