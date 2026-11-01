"""Validate the indexed files of a frozen-defender evidence bundle."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _file(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Invalid bundle path: {name}")
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or relative.as_posix() != name
    ):
        raise ValueError(f"Invalid bundle path: {name}")
    path = root.joinpath(*relative.parts)
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError(f"Symbolic link in evidence path: {name}")
    if not path.is_file():
        raise ValueError(f"Missing bundle file: {name}")
    return path


def verified_bundle(root):
    """Return the verified index, reference paths, and indexed file identities."""
    root = Path(root).resolve()
    index = json.loads(_file(root, "index.json").read_text(), object_pairs_hook=_pairs)
    if index["format"] != "frozen-defender-response-v1":
        raise ValueError("Unsupported bundle")
    if "index.json" in index["files"]:
        raise ValueError("Bundle file closure contains duplicate paths")
    references, identities = {}, {}
    for name, item in index["files"].items():
        path = _file(root, name)
        with path.open("rb") as stream:
            identity = {
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                "size_bytes": os.fstat(stream.fileno()).st_size,
            }
        if identity != {"sha256": item["sha256"], "size_bytes": item["size"]}:
            raise ValueError(f"Hash mismatch: {name}")
        identities[name] = identity
        if item.get("reference"):
            ref = item["reference"]
            if ref in references:
                raise ValueError(f"Duplicate artifact reference: {ref}")
            references[ref] = path
    for key in ("protocol", "state", "reference_summary"):
        if index[key] not in references:
            raise ValueError(f"Missing {key} reference")
    if index["configuration"] not in index["files"]:
        raise ValueError("Analysis configuration is not indexed")
    if not index["source_store"]:
        raise ValueError("Missing original tracking-store identity")
    return index, references, identities
