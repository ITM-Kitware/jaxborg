"""Publish a training-owned policy and its recipe on the producing MLflow run."""

import hashlib
from pathlib import Path


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def log_policy_checkpoint(client, run_id, *, team, step, model, sidecar):
    """Publish the descriptor after both files have been uploaded."""
    directory = f"checkpoints/{team}/{step}"
    files = {}
    for role, path in (("model", Path(model)), ("recipe", Path(sidecar))):
        files[role] = {"path": f"{directory}/{path.name}", "sha256": file_digest(path)}
        client.log_artifact(run_id, str(path), artifact_path=directory)
    descriptor = {"run_id": run_id, "team": team, "step": step, "files": files}
    client.log_dict(run_id, descriptor, f"{directory}/checkpoint.json")
    return descriptor


def verify_policy_origin(client, run_id, *, team, step, model, sidecar):
    """Match local files to the training run's published checkpoint descriptor."""
    import json
    from tempfile import TemporaryDirectory

    prefix = f"checkpoints/{team}/{step}"
    with TemporaryDirectory() as tmp:
        descriptor = Path(client.download_artifacts(run_id, f"{prefix}/checkpoint.json", tmp))
        saved = json.loads(descriptor.read_text())
    expected = {
        "run_id": run_id,
        "team": team,
        "step": step,
        "files": {
            role: {"path": f"{prefix}/{path.name}", "sha256": file_digest(path)}
            for role, path in (("model", Path(model)), ("recipe", Path(sidecar)))
        },
    }
    if saved != expected:
        raise ValueError("Policy files differ from the selected training checkpoint")
    return saved
