"""Checkpoint sidecar — recipe travels with model weights.

Each training run writes:

    $JAXBORG_EXP_DIR/<algo>_<backend>/<tag>/
        model_<tag>.pt              (Torch bundle)
        model_<tag>.safetensors     (Flax bundle)
        recipe_<tag>.yaml           ← this module writes it
        checkpoint_<tag>.pt         (full optimizer state, optional)

Versioned model bundles contain one entry per learned policy (Blue and/or
Red), including frozen learned opponents. Legacy single-policy files remain
readable and are interpreted as Blue-only.

`recipe_<tag>.yaml` is the **resolved** recipe: the recipe dict that the
trainer actually consumed (post CLI overrides), plus a `run` block with
seed, commit, timestamp, total_steps, and (when known) the MLflow run id.

The eval script reads it back to instantiate the right architecture and to
attach eval metrics to the same MLflow run.
"""

from __future__ import annotations

import copy
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import jax
import yaml
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors import safe_open
from safetensors.flax import load_file, save_file

_FLAX_KEY_SEP = "/"
_JAX_BUNDLE_METADATA_KEY = "jaxborg_bundle"
_TORCH_BUNDLE_SCHEMA_KEY = "jaxborg_bundle_schema"
BUNDLE_SCHEMA_VERSION = 1
VALID_TEAMS = ("blue", "red")


@dataclass
class PolicyBundleEntry:
    """One parameter-sharing policy and the contract needed to recreate it."""

    weights: Any
    team: str
    obs_dim: int
    action_dim: int
    arch: dict[str, Any] = field(default_factory=dict)
    trainable: bool = True
    source: Any = None

    @property
    def params(self) -> Any:
        """Flax-friendly alias for ``weights``."""
        return self.weights

    @property
    def state_dict(self) -> Any:
        """Torch-friendly alias for ``weights``."""
        return self.weights


@dataclass
class ModelBundle:
    """Loaded multi-policy model bundle."""

    backend: str
    policies: dict[str, PolicyBundleEntry]
    schema_version: int = BUNDLE_SCHEMA_VERSION
    provenance: dict[str, Any] = field(default_factory=dict)
    legacy: bool = False


def _coerce_policy_entry(team: str, value: PolicyBundleEntry | Mapping[str, Any]) -> PolicyBundleEntry:
    if team not in VALID_TEAMS:
        raise ValueError(f"policy team must be one of {VALID_TEAMS}, got {team!r}")
    if isinstance(value, PolicyBundleEntry):
        entry = value
        if entry.team != team:
            raise ValueError(f"policy key {team!r} does not match entry.team {entry.team!r}")
    elif isinstance(value, Mapping):
        weights = value.get("weights")
        if weights is None:
            weights = value.get("params")
        if weights is None:
            weights = value.get("state_dict")
        if weights is None:
            raise ValueError(f"policy {team!r} requires weights, params, or state_dict")
        missing = [key for key in ("obs_dim", "action_dim", "arch") if key not in value]
        if missing:
            raise ValueError(f"policy {team!r} is missing bundle metadata: {missing}")
        entry = PolicyBundleEntry(
            weights=weights,
            team=str(value.get("team", team)),
            obs_dim=int(value["obs_dim"]),
            action_dim=int(value["action_dim"]),
            arch=dict(value["arch"]),
            trainable=bool(value.get("trainable", True)),
            source=value.get("source"),
        )
        if entry.team != team:
            raise ValueError(f"policy key {team!r} does not match entry.team {entry.team!r}")
    else:
        raise TypeError(f"policy {team!r} must be PolicyBundleEntry or a mapping")
    if entry.obs_dim <= 0:
        raise ValueError(f"policy {team!r} obs_dim must be positive, got {entry.obs_dim}")
    if entry.action_dim <= 0:
        raise ValueError(f"policy {team!r} action_dim must be positive, got {entry.action_dim}")
    if not isinstance(entry.arch, dict) or not entry.arch.get("name"):
        raise ValueError(f"policy {team!r} arch must be a mapping containing name")
    return entry


def _coerce_policies(
    policies: Mapping[str, PolicyBundleEntry | Mapping[str, Any]],
) -> dict[str, PolicyBundleEntry]:
    if not policies:
        raise ValueError("a model bundle must contain at least one policy")
    return {team: _coerce_policy_entry(team, value) for team, value in policies.items()}


def _entry_manifest(entry: PolicyBundleEntry) -> dict[str, Any]:
    return {
        "team": entry.team,
        "obs_dim": int(entry.obs_dim),
        "action_dim": int(entry.action_dim),
        "arch": entry.arch,
        "trainable": bool(entry.trainable),
        "source": entry.source,
    }


def _json_safe(value: Any) -> Any:
    """Convert metadata to primitives accepted by both JSON and weights-only Torch."""
    return json.loads(json.dumps(value, default=str))


def _manifest(*, backend: str, policies: Mapping[str, PolicyBundleEntry], provenance: Mapping | None) -> dict:
    return _json_safe(
        {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "backend": backend,
            "policies": {team: _entry_manifest(entry) for team, entry in policies.items()},
            "provenance": dict(provenance or {}),
        }
    )


def _entry_from_manifest(meta: Mapping[str, Any], weights: Any) -> PolicyBundleEntry:
    from jaxborg.blue_observation_contract import observation_version_from_size

    arch = dict(meta["arch"])
    if arch.get("name") in ("mappo", "recurrent_mappo"):
        arch.setdefault(
            "blue_observation_version",
            observation_version_from_size(int(meta["obs_dim"])) if meta["team"] == "blue" else 1,
        )
    return PolicyBundleEntry(
        weights=weights,
        team=str(meta["team"]),
        obs_dim=int(meta["obs_dim"]),
        action_dim=int(meta["action_dim"]),
        arch=arch,
        trainable=bool(meta.get("trainable", True)),
        source=meta.get("source"),
    )


def _validate_loaded_policy(
    bundle: ModelBundle,
    team: str,
    *,
    expected_obs_dim: int | None,
    expected_action_dim: int | None,
) -> PolicyBundleEntry:
    if team not in bundle.policies:
        raise ValueError(f"bundle has no {team!r} policy; available teams: {sorted(bundle.policies)}")
    entry = bundle.policies[team]
    if expected_obs_dim is not None and entry.obs_dim not in (0, int(expected_obs_dim)):
        raise ValueError(f"{team} observation dimension mismatch: model={entry.obs_dim}, expected={expected_obs_dim}")
    if expected_action_dim is not None and entry.action_dim not in (0, int(expected_action_dim)):
        raise ValueError(f"{team} action dimension mismatch: model={entry.action_dim}, expected={expected_action_dim}")
    return entry


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return ""


def _git_branch() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return ""


def write_sidecar(
    path: Path,
    recipe: dict[str, Any],
    *,
    seed: int,
    total_steps: int,
    backend: str,
    train_run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the resolved recipe + run metadata to `path`. Returns `path`.

    `recipe` must be the recipe dict as the trainer consumed it. Internal
    keys (`__source_path__`) are preserved under `meta.source_path` and the
    underscore key is dropped from the on-disk YAML.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = copy.deepcopy({k: v for k, v in recipe.items() if not str(k).startswith("__")})
    src = recipe.get("__source_path__")
    if src:
        payload.setdefault("meta", {})["source_path"] = src

    from jaxborg.blue_observation_contract import enhanced_obs_version

    payload["run"] = {
        "blue_observation_version": enhanced_obs_version(recipe),
        "seed": int(seed),
        "total_steps": int(total_steps),
        "backend": backend,
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_run_id": train_run_id,
    }
    if extra:
        payload["run"].update(extra)

    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def save_jax_params(path: str | Path, params: Any, *, action_dim: int) -> Path:
    """Write Flax params + minimal metadata to a safetensors file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_dict(jax.device_get(params), sep=_FLAX_KEY_SEP)
    save_file(flat, str(path), metadata={"action_dim": str(int(action_dim))})
    return path


def load_jax_params(path: str | Path) -> tuple[dict, int]:
    """Inverse of `save_jax_params`. New bundles return their Blue policy.

    This compatibility API intentionally remains Blue-defaulting so existing
    evaluation and transfer scripts can read both legacy single-policy files
    and versioned bundles unchanged.
    """
    path = Path(path)
    with safe_open(str(path), framework="flax") as f:
        meta = f.metadata() or {}
    if _JAX_BUNDLE_METADATA_KEY in meta:
        entry = load_jax_policy(path, team="blue")
        return entry.weights, entry.action_dim
    flat = load_file(str(path))
    params = unflatten_dict(flat, sep=_FLAX_KEY_SEP)
    action_dim = int(meta.get("action_dim", 0))
    return params, action_dim


def save_jax_bundle(
    path: str | Path,
    policies: Mapping[str, PolicyBundleEntry | Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Save one or more Flax policies in a versioned safetensors bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = _coerce_policies(policies)
    tensors: dict[str, Any] = {}
    for team, entry in entries.items():
        flat = flatten_dict(jax.device_get(entry.weights), sep=_FLAX_KEY_SEP)
        tensors.update({f"{team}{_FLAX_KEY_SEP}{key}": value for key, value in flat.items()})
    manifest = _manifest(backend="jax", policies=entries, provenance=provenance)
    metadata = {
        _JAX_BUNDLE_METADATA_KEY: json.dumps(manifest, sort_keys=True, default=str),
    }
    if "blue" in entries:
        metadata["action_dim"] = str(entries["blue"].action_dim)
    save_file(tensors, str(path), metadata=metadata)
    return path


def load_jax_bundle(path: str | Path) -> ModelBundle:
    """Load a versioned Flax bundle, or adapt a legacy file as Blue-only."""
    path = Path(path)
    flat = load_file(str(path))
    with safe_open(str(path), framework="flax") as f:
        metadata = f.metadata() or {}
    encoded = metadata.get(_JAX_BUNDLE_METADATA_KEY)
    if encoded is None:
        params = unflatten_dict(flat, sep=_FLAX_KEY_SEP)
        return ModelBundle(
            backend="jax",
            policies={
                "blue": PolicyBundleEntry(
                    weights=params,
                    team="blue",
                    obs_dim=0,
                    action_dim=int(metadata.get("action_dim", 0)),
                    arch={},
                    trainable=True,
                    source={"path": str(path.resolve())},
                )
            },
            schema_version=0,
            provenance={"path": str(path.resolve())},
            legacy=True,
        )

    manifest = json.loads(encoded)
    version = int(manifest.get("schema_version", 0))
    if version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported JAX bundle schema version {version}; expected {BUNDLE_SCHEMA_VERSION}")
    if manifest.get("backend") != "jax":
        raise ValueError(f"Expected a JAX bundle, found backend {manifest.get('backend')!r}")
    policy_manifests = manifest.get("policies", {})
    if not isinstance(policy_manifests, Mapping):
        raise ValueError("JAX bundle policies must be a mapping")
    policies: dict[str, PolicyBundleEntry] = {}
    for team, entry_meta in policy_manifests.items():
        prefix = f"{team}{_FLAX_KEY_SEP}"
        team_flat = {key[len(prefix) :]: value for key, value in flat.items() if key.startswith(prefix)}
        if not team_flat:
            raise ValueError(f"JAX bundle manifest declares {team!r}, but it has no tensors")
        entry = _entry_from_manifest(entry_meta, unflatten_dict(team_flat, sep=_FLAX_KEY_SEP))
        policies[team] = _coerce_policy_entry(team, entry)
    if not policies:
        raise ValueError("JAX bundle contains no policies")
    return ModelBundle(
        backend="jax",
        policies=policies,
        schema_version=version,
        provenance=dict(manifest.get("provenance") or {}),
    )


def load_jax_policy(
    path: str | Path,
    team: str = "blue",
    *,
    expected_obs_dim: int | None = None,
    expected_action_dim: int | None = None,
) -> PolicyBundleEntry:
    """Load and validate one team from a Flax model bundle."""
    return _validate_loaded_policy(
        load_jax_bundle(path),
        team,
        expected_obs_dim=expected_obs_dim,
        expected_action_dim=expected_action_dim,
    )


def save_torch_bundle(
    path: str | Path,
    policies: Mapping[str, PolicyBundleEntry | Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Save one or more Torch policies in a versioned ``.pt`` bundle."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = _coerce_policies(policies)
    manifest = _manifest(backend="cyborg", policies=entries, provenance=provenance)
    payload = {
        _TORCH_BUNDLE_SCHEMA_KEY: BUNDLE_SCHEMA_VERSION,
        "backend": "cyborg",
        "policies": {
            team: {**manifest["policies"][team], "state_dict": entry.weights} for team, entry in entries.items()
        },
        "provenance": manifest["provenance"],
    }
    torch.save(payload, path)
    return path


def load_torch_bundle(path: str | Path, *, map_location: str = "cpu") -> ModelBundle:
    """Load a versioned Torch bundle, or adapt a bare state_dict as Blue-only."""
    import torch

    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Torch model at {path} is not a state mapping")
    if _TORCH_BUNDLE_SCHEMA_KEY not in payload:
        return ModelBundle(
            backend="cyborg",
            policies={
                "blue": PolicyBundleEntry(
                    weights=payload,
                    team="blue",
                    obs_dim=0,
                    action_dim=0,
                    arch={},
                    trainable=True,
                    source={"path": str(path.resolve())},
                )
            },
            schema_version=0,
            provenance={"path": str(path.resolve())},
            legacy=True,
        )

    version = int(payload[_TORCH_BUNDLE_SCHEMA_KEY])
    if version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Torch bundle schema version {version}; expected {BUNDLE_SCHEMA_VERSION}")
    if payload.get("backend") != "cyborg":
        raise ValueError(f"Expected a CybORG/Torch bundle, found backend {payload.get('backend')!r}")
    policy_payloads = payload.get("policies", {})
    if not isinstance(policy_payloads, Mapping):
        raise ValueError("Torch bundle policies must be a mapping")
    policies: dict[str, PolicyBundleEntry] = {}
    for team, entry_payload in policy_payloads.items():
        if not isinstance(entry_payload, Mapping) or "state_dict" not in entry_payload:
            raise ValueError(f"Torch bundle policy {team!r} has no state_dict")
        entry = _entry_from_manifest(entry_payload, entry_payload["state_dict"])
        policies[team] = _coerce_policy_entry(team, entry)
    if not policies:
        raise ValueError("Torch bundle contains no policies")
    return ModelBundle(
        backend="cyborg",
        policies=policies,
        schema_version=version,
        provenance=dict(payload.get("provenance") or {}),
    )


def load_torch_policy(
    path: str | Path,
    team: str = "blue",
    *,
    expected_obs_dim: int | None = None,
    expected_action_dim: int | None = None,
    map_location: str = "cpu",
) -> PolicyBundleEntry:
    """Load and validate one team from a Torch model bundle."""
    return _validate_loaded_policy(
        load_torch_bundle(path, map_location=map_location),
        team,
        expected_obs_dim=expected_obs_dim,
        expected_action_dim=expected_action_dim,
    )


def read_sidecar(model_path: str | Path) -> dict[str, Any]:
    """Load `recipe_<tag>.{yaml|yml}` adjacent to `model_path`."""
    model_path = Path(model_path)
    name = model_path.name
    if name.startswith("model_"):
        stem = name[len("model_") :]
        stem = stem.rsplit(".", 1)[0]
    else:
        stem = model_path.stem
    candidates = [
        model_path.with_name(f"recipe_{stem}.yaml"),
        model_path.with_name(f"recipe_{stem}.yml"),
    ]
    for c in candidates:
        if c.exists():
            return yaml.safe_load(c.read_text())
    raise FileNotFoundError(f"No recipe sidecar found next to {model_path} (looked for {[str(c) for c in candidates]})")
