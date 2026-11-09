"""Numeric full-state checkpoints restored against a trusted initialized template.

The archive contains no pickle, Python code, optimizer functions or serialized
PyTreeDef. Static behavior comes from the caller's matching source/configuration;
every dynamic leaf is checked by stable key path, shape, dtype and content hash.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_SCHEMA = 1
_MANIFEST = "manifest_json"


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _array(leaf):
    # TrainState.create starts with a Python step=0; the first JIT call makes
    # this a default-width JAX integer. Normalize Python scalars only, never
    # silently downcast an explicitly typed array from a checkpoint/template.
    if type(leaf) in (bool, int, float, complex):
        leaf = jnp.asarray(leaf)
    array = np.asarray(jax.device_get(leaf))
    if array.dtype.kind not in "biufc":
        raise ValueError(f"Training state requires numeric leaves, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("Training state contains a nonfinite numeric leaf")
    weak = bool(getattr(leaf, "weak_type", False))
    return array, weak


def _leaves(state):
    pairs, definition = jax.tree_util.tree_flatten_with_path(state)
    if not pairs:
        raise ValueError("Training state must contain numeric leaves")
    arrays, leaves = {}, []
    for index, (path, leaf) in enumerate(pairs):
        array, weak = _array(leaf)
        key = f"leaf_{index}"
        arrays[key] = array
        leaves.append(
            {
                "key": key,
                "path": jax.tree_util.keystr(path),
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "weak_type": weak,
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        )
    return arrays, leaves, definition


def _descriptor(payload, manifest):
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "state_sha256": hashlib.sha256(_json_bytes(manifest["leaves"])).hexdigest(),
        "metadata": manifest["metadata"],
        "leaf_count": len(manifest["leaves"]),
    }


def save_training_state(path, *, state, metadata):
    """Atomically create an immutable numeric archive and return its identities."""
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata must be a dictionary")
    arrays, leaves, _ = _leaves(state)
    manifest = {"schema_version": _SCHEMA, "metadata": metadata, "leaves": leaves}
    arrays[_MANIFEST] = np.frombuffer(_json_bytes(manifest), dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Link publishes only complete bytes and refuses overwriting a previous
        # checkpoint. Both paths are in one directory/filesystem.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _descriptor(payload, manifest)


def _read(path, expected_sha256=None):
    payload = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Training-state archive hash differs from its recorded identity")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if _MANIFEST not in archive.files:
            raise ValueError("Training-state archive has no manifest")
        raw = archive[_MANIFEST]
        if raw.dtype != np.dtype("uint8") or raw.ndim != 1:
            raise ValueError("Invalid training-state manifest representation")
        manifest = json.loads(raw.tobytes().decode("utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != _SCHEMA
            or not isinstance(manifest.get("metadata"), dict)
            or not isinstance(manifest.get("leaves"), list)
            or not manifest["leaves"]
        ):
            raise ValueError("Unsupported training-state schema")
        _json_bytes(manifest)
        entries = manifest["leaves"]
        keys = [entry["key"] for entry in entries]
        paths = [entry["path"] for entry in entries]
        if (
            keys != [f"leaf_{index}" for index in range(len(entries))]
            or len(paths) != len(set(paths))
            or len(archive.files) != len(entries) + 1
            or set(archive.files) != {*keys, _MANIFEST}
        ):
            raise ValueError("Training-state leaves are missing, duplicated or reordered")
        arrays = []
        for entry in entries:
            array = np.asarray(archive[entry["key"]])
            if (
                array.dtype.kind not in "biufc"
                or array.dtype.str != entry["dtype"]
                or list(array.shape) != entry["shape"]
                or type(entry["weak_type"]) is not bool
                or not np.isfinite(array).all()
                or hashlib.sha256(array.tobytes(order="C")).hexdigest() != entry["sha256"]
            ):
                raise ValueError(f"Training-state leaf differs from its manifest: {entry['path']}")
            arrays.append(array)
    return arrays, manifest, _descriptor(payload, manifest)


def inspect_training_state(path, *, expected_sha256=None):
    """Validate the archive and expose metadata and scientific-state identity."""
    return _read(path, expected_sha256)[2]


def read_training_state_metadata(path, *, expected_sha256=None):
    return inspect_training_state(path, expected_sha256=expected_sha256)["metadata"]


def load_training_state(path, *, template, expected_metadata=None, expected_sha256=None):
    """Restore dynamic leaves using a caller-initialized, matching state tree.

    ``expected_metadata`` is an exact key/value subset, typically the source,
    runtime and hyperparameter contract. Completed counts remain available to
    the caller, which must enforce the selected training budget and lineage.
    """
    arrays, manifest, _ = _read(path, expected_sha256)
    metadata = manifest["metadata"]
    if expected_metadata is not None:
        expected_metadata = json.loads(_json_bytes(expected_metadata))
        if any(key not in metadata or metadata[key] != value for key, value in expected_metadata.items()):
            raise ValueError("Training-state metadata differs from the expected continuation contract")
    _, template_leaves, definition = _leaves(template)
    entries = manifest["leaves"]
    contract_fields = ("key", "path", "shape", "dtype")
    if len(entries) != len(template_leaves) or any(
        any(saved[field] != fresh[field] for field in contract_fields)
        for saved, fresh in zip(entries, template_leaves, strict=True)
    ):
        raise ValueError("Training-state tree differs from the initialized template")
    restored = []
    for entry, array in zip(entries, arrays, strict=True):
        leaf = jnp.asarray(array)
        if entry["weak_type"]:
            # Public lax.convert_element_type deliberately forces strong typing.
            # This pinned JAX primitive preserves weak array avals as well as
            # weak scalars; CC4's vmapped constants can have either form.
            from jax._src.lax.lax import _convert_element_type

            leaf = _convert_element_type(leaf, array.dtype, weak_type=True)
        actual, weak = _array(leaf)
        if (
            actual.dtype.str != entry["dtype"]
            or list(actual.shape) != entry["shape"]
            or weak != entry["weak_type"]
            or hashlib.sha256(actual.tobytes(order="C")).hexdigest() != entry["sha256"]
        ):
            raise ValueError("Runtime cannot preserve the checkpoint's array/dtype/weak-type contract")
        restored.append(leaf)
    return jax.tree_util.tree_unflatten(definition, restored), metadata
