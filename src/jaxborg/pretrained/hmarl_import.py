"""Convert the released H-MARL actors to a JAX-readable safetensors bundle.

Only the leading NumPy weight dictionary is decoded, through a restricted
unpickler. RLlib configuration, Python functions, critics and optimizer state
are neither executed nor included in the output. No Ray or Torch is needed.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import pickletools
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

UPSTREAM_REPO = "https://github.com/adityavs14/Hierarchical-MARL"
UPSTREAM_COMMIT = "6fe960f5931f2d7dc5ef522f09ca4a179f377ee4"
MANIFEST_PATH = Path(__file__).with_name("hmarl_manifest.json")
FORMAT = "jaxborg-hmarl-actors-v1"


class _ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
            return np._core.numeric._frombuffer
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}")


def read_weight_prefix(raw: bytes) -> dict[str, np.ndarray]:
    """Read the first flat weights dictionary, never the executable RLlib tail."""
    parts = []
    previous = None
    for op, arg, pos in pickletools.genops(raw):
        if previous is not None:
            prev_op, prev_pos = previous
            if prev_op != "FRAME":
                parts.append(raw[prev_pos:pos])
        if op.name == "SETITEMS":
            # STOP returns the completed weights dictionary from the stack.
            # Leave the outer dictionary unfinished; it may use SETITEM or SETITEMS.
            parts.append(b"u.")
            break
        previous = (op.name, pos)
    else:
        raise ValueError("Checkpoint does not have the expected leading weights dictionary")
    result = _ArrayUnpickler(io.BytesIO(b"".join(parts))).load()
    if (
        not isinstance(result, dict)
        or not result
        or any(
            not isinstance(key, str) or not key.startswith("internal_model.") or not isinstance(value, np.ndarray)
            for key, value in result.items()
        )
    ):
        raise ValueError("Invalid H-MARL weight dictionary")
    return result


def actor_arrays(weights: dict[str, np.ndarray], *, agent: int, branch: str) -> dict[str, np.ndarray]:
    subnets = 3 if agent == 4 else 1
    obs_dim = 1 + subnets * {"investigate": 48, "recover": 16, "master": 75}[branch]
    action_dim = 2 if branch == "master" else 2 + 80 * subnets
    layers = ("_hidden_layers.0", "_hidden_layers.1", "_logits")
    dims = ((obs_dim, 256), (256, 256), (256, action_dim))
    arrays = {}
    for index, (layer, (inputs, outputs)) in enumerate(zip(layers, dims, strict=True)):
        prefix = f"internal_model.{layer}._model.0"
        weight, bias = weights[f"{prefix}.weight"], weights[f"{prefix}.bias"]
        if weight.shape != (outputs, inputs) or bias.shape != (outputs,):
            raise ValueError(f"Unexpected Agent{agent}_{branch} layer {index} shapes: {weight.shape}, {bias.shape}")
        for name, array in (("kernel", weight.T), ("bias", bias)):
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Non-finite or non-float32 actor tensor: {prefix}.{name}")
            arrays[f"agent{agent}.{branch}.{index}.{name}"] = np.ascontiguousarray(array)
    return arrays


def import_checkpoint(output: Path, *, upstream_dir: Path | None = None) -> Path:
    """Fetch pinned, hash-checked weights (or read a local upstream checkout)."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    tensors = {}
    for entry in manifest["checkpoints"]:
        path = entry["path"]
        if upstream_dir is None:
            url = f"https://raw.githubusercontent.com/adityavs14/Hierarchical-MARL/{UPSTREAM_COMMIT}/{path}"
            with urllib.request.urlopen(url, timeout=120) as response:
                raw = response.read()
        else:
            raw = (upstream_dir / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        tensors.update(actor_arrays(read_weight_prefix(raw), agent=entry["agent"], branch=entry["branch"]))
        print(f"Imported Agent{entry['agent']}_{entry['branch']}", flush=True)
    metadata = {
        "format": FORMAT,
        "upstream_repository": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "architecture": "256,256;tanh;linear_logits;NoFilter;raw_observations",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".safetensors", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        save_file(tensors, temporary_path, metadata=metadata)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/pretrained/hmarl/actors.safetensors"))
    parser.add_argument("--upstream-dir", type=Path, help="Optional local checkout instead of downloading weights")
    args = parser.parse_args()
    print(import_checkpoint(args.output, upstream_dir=args.upstream_dir))


if __name__ == "__main__":
    main()
