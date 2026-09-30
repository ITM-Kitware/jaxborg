"""Research caller adapters over the shared run/artifact interface."""

import hashlib

import jax
import numpy as np

from jaxborg.tracking import input_artifact


def parameter_hash(params):
    """Hash shapes, dtypes and exact parameter bytes, independent of device."""
    digest = hashlib.sha256()
    for key, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(key).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def training_inputs(recipe, config, opponent_paths):
    inputs = []
    for team, path in opponent_paths.items():
        ref = recipe["train"]["opponents"][team]
        reference = ref.get("path", path) if isinstance(ref, dict) else ref
        inputs.append(input_artifact(reference, role=f"frozen {team} policy"))
    inputs.extend(input_artifact(path, role="training topology") for path in config.get("TOPOLOGY_BANK", ()))
    return inputs


def publish_checkpoint(run, path, sidecar, step):
    sidecar_name = f"checkpoints/{sidecar.name}"
    run.publish(sidecar, sidecar_name, step=step)
    reference = run.publish(path, f"checkpoints/{path.name}", sidecar=sidecar_name, step=step)
    print(f"Checkpoint: {reference}", flush=True)
    return reference


def publish_training_result(run, tag, destination=None):
    result = {
        "run_id": run.run_id,
        "final_checkpoint": f"runs:/{run.run_id}/checkpoints/model_{tag}.safetensors",
        "actual_steps": run.manifest["actual_steps"],
        "requested_steps": run.manifest["requested_steps"],
    }
    run.write_json("training/result.json", result)
    if destination:
        run.export("training/result.json", destination)
    return result
