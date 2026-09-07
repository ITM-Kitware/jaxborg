"""Deterministic resilience roles and paired evaluation cases.

Evaluation role assignment is deliberately independent of episode RNG. A
snapshot's canonical array contents determine its AUTH/DB/WEB hosts, so moving
or copying the file, changing bank order, or evaluating it in another process
does not change the assignment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from jaxborg.constants import GLOBAL_MAX_HOSTS, SUBNET_IDS
from jaxborg.scenarios.cc4.topology import TOPOLOGY_SNAPSHOT_FIELDS, load_topology
from jaxborg.scenarios.cc4.topology_roles import (
    ROLE_AUTH,
    ROLE_DB,
    ROLE_NAMES,
    ROLE_NONE,
    ROLE_WEB,
)
from jaxborg.state import SimulatorConst

TOPOLOGY_FINGERPRINT_VERSION = "jaxborg-topology-fingerprint-v1"
FIXED_ROLE_ASSIGNMENT_VERSION = "jaxborg-fixed-resilience-roles-v1"
MIN_RESILIENCE_ROLE_HOSTS = 3

_OPERATIONAL_SUBNET_IDS = frozenset(
    {
        SUBNET_IDS["OPERATIONAL_ZONE_A"],
        SUBNET_IDS["OPERATIONAL_ZONE_B"],
    }
)

TopologySource = str | Path | SimulatorConst


def _length_prefix(value: bytes) -> bytes:
    return len(value).to_bytes(8, byteorder="big", signed=False) + value


def _canonical_array(value: Any) -> np.ndarray:
    """Return a contiguous, little-endian array suitable for stable hashing."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("topology fingerprint fields must not contain Python objects")
    little_endian_dtype = array.dtype.newbyteorder("<")
    if array.dtype != little_endian_dtype:
        array = array.astype(little_endian_dtype, copy=False)
    return np.ascontiguousarray(array)


def _resolve_const(source: TopologySource) -> SimulatorConst:
    if isinstance(source, (str, Path)):
        return load_topology(Path(source).expanduser().resolve())
    if isinstance(source, SimulatorConst):
        return source
    raise TypeError(f"topology must be a snapshot path or SimulatorConst, got {type(source).__name__}")


def canonical_topology_fingerprint(source: TopologySource) -> str:
    """Hash all canonical snapshot arrays, excluding mutable path/metadata.

    Field names, dtypes, and shapes are framed along with the values, avoiding
    ambiguous concatenations. Snapshot provenance is intentionally excluded:
    byte-identical topology arrays copied to a new file must retain their
    identity and role assignment.
    """

    const = _resolve_const(source)
    digest = hashlib.sha256()
    digest.update(_length_prefix(TOPOLOGY_FINGERPRINT_VERSION.encode("utf-8")))
    for field_name in TOPOLOGY_SNAPSHOT_FIELDS:
        array = _canonical_array(getattr(const, field_name))
        digest.update(_length_prefix(field_name.encode("utf-8")))
        digest.update(_length_prefix(array.dtype.str.encode("ascii")))
        shape = ",".join(str(size) for size in array.shape).encode("ascii")
        digest.update(_length_prefix(shape))
        digest.update(_length_prefix(array.tobytes(order="C")))
    return digest.hexdigest()


def eligible_operational_server_indices(source: TopologySource) -> tuple[int, ...]:
    """Return active operational-zone server indices in canonical host order."""

    const = _resolve_const(source)
    active = np.asarray(const.host_active, dtype=bool)
    servers = np.asarray(const.host_is_server, dtype=bool)
    subnets = np.asarray(const.host_subnet)
    if active.shape != (GLOBAL_MAX_HOSTS,):
        raise ValueError(f"topology host_active has shape {active.shape}; expected {(GLOBAL_MAX_HOSTS,)}")
    mask = active & servers & np.isin(subnets, tuple(_OPERATIONAL_SUBNET_IDS))
    return tuple(int(index) for index in np.flatnonzero(mask))


def validate_fixed_role_eligibility(source: TopologySource) -> tuple[int, ...]:
    """Validate and return candidates for a complete AUTH/DB/WEB assignment."""

    candidates = eligible_operational_server_indices(source)
    if len(candidates) < MIN_RESILIENCE_ROLE_HOSTS:
        label = str(source) if isinstance(source, (str, Path)) else "topology"
        raise ValueError(
            f"topology snapshot {label} has only {len(candidates)} op-zone server candidates "
            f"(need ≥{MIN_RESILIENCE_ROLE_HOSTS} for AUTH/DB/WEB roles)"
        )
    return candidates


def _role_map_id(host_roles: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_length_prefix(FIXED_ROLE_ASSIGNMENT_VERSION.encode("utf-8")))
    roles = np.asarray(host_roles, dtype="<i4")
    digest.update(_length_prefix(roles.tobytes(order="C")))
    return digest.hexdigest()


@dataclass(frozen=True)
class FixedRoleAssignment:
    """One fixed host-index role map for one topology fingerprint."""

    topology_fingerprint: str
    host_roles: tuple[int, ...]
    role_map_id: str
    eligible_host_indices: tuple[int, ...]

    @property
    def role_array(self) -> jnp.ndarray:
        """Return the fixed map in the env's ``int32[num_hosts]`` format."""

        return jnp.asarray(self.host_roles, dtype=jnp.int32)

    @property
    def role_by_host_index(self) -> dict[int, int]:
        return {host_index: role for host_index, role in enumerate(self.host_roles) if role != ROLE_NONE}

    @property
    def host_role_map(self) -> dict[int, int]:
        """Alias using the terminology expected by CIA consumers."""

        return self.role_by_host_index

    def audit_role_map(self) -> dict[str, Any]:
        """Return a compact JSON-serializable record for result artifacts."""

        roles = [
            {
                "host_index": host_index,
                "role_id": role,
                "role": ROLE_NAMES[role],
            }
            for host_index, role in sorted(
                self.role_by_host_index.items(),
                key=lambda item: item[1],
            )
        ]
        return {
            "topology_fingerprint": self.topology_fingerprint,
            "role_map_id": self.role_map_id,
            "assignment_version": FIXED_ROLE_ASSIGNMENT_VERSION,
            "roles": roles,
        }


def fixed_role_assignment(
    source: TopologySource,
    *,
    topology_fingerprint: str | None = None,
) -> FixedRoleAssignment:
    """Assign AUTH, DB, and WEB by deterministic hash rank.

    Candidate host indices are ranked by SHA-256 of the versioned assignment
    algorithm, topology fingerprint, and host index. The first three receive
    AUTH, DB, and WEB respectively.
    """

    const = _resolve_const(source)
    fingerprint = topology_fingerprint or canonical_topology_fingerprint(const)
    candidates = validate_fixed_role_eligibility(const)

    def rank(host_index: int) -> tuple[bytes, int]:
        payload = (f"{FIXED_ROLE_ASSIGNMENT_VERSION}\0{fingerprint}\0{host_index}").encode("ascii")
        return hashlib.sha256(payload).digest(), host_index

    selected = sorted(candidates, key=rank)[:MIN_RESILIENCE_ROLE_HOSTS]
    roles = [ROLE_NONE] * GLOBAL_MAX_HOSTS
    for host_index, role in zip(
        selected,
        (ROLE_AUTH, ROLE_DB, ROLE_WEB),
        strict=True,
    ):
        roles[host_index] = role
    host_roles = tuple(roles)
    return FixedRoleAssignment(
        topology_fingerprint=fingerprint,
        host_roles=host_roles,
        role_map_id=_role_map_id(host_roles),
        eligible_host_indices=candidates,
    )


@dataclass(frozen=True)
class EvaluationCase:
    """One paired rollout case in topology -> seed -> replicate order."""

    topology_index: int
    topology_path: Path
    topology_fingerprint: str
    base_seed: int
    replicate_index: int
    episode_seed: int
    host_roles: tuple[int, ...]
    role_map_id: str

    @property
    def role_array(self) -> jnp.ndarray:
        return jnp.asarray(self.host_roles, dtype=jnp.int32)

    @property
    def role_by_host_index(self) -> dict[int, int]:
        return {host_index: role for host_index, role in enumerate(self.host_roles) if role != ROLE_NONE}

    @property
    def host_role_map(self) -> dict[int, int]:
        return self.role_by_host_index

    def audit_role_map(self) -> dict[str, Any]:
        return FixedRoleAssignment(
            topology_fingerprint=self.topology_fingerprint,
            host_roles=self.host_roles,
            role_map_id=self.role_map_id,
            eligible_host_indices=(),
        ).audit_role_map()


def build_evaluation_cases(
    topology_paths: Sequence[str | Path],
    seeds: Sequence[int],
    episodes_per_seed: int,
) -> tuple[EvaluationCase, ...]:
    """Build deterministic exhaustive cases in topology/seed/replicate order."""

    if not topology_paths:
        raise ValueError("CIA evaluation requires a non-empty topology bank")
    if isinstance(episodes_per_seed, bool) or not isinstance(episodes_per_seed, int):
        raise ValueError("episodes_per_seed must be an integer")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be positive")
    if not seeds:
        raise ValueError("CIA evaluation requires at least one episode seed")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("evaluation seeds must be non-negative integers")

    cases: list[EvaluationCase] = []
    for topology_index, raw_path in enumerate(topology_paths):
        topology_path = Path(raw_path).expanduser().resolve()
        const = load_topology(topology_path)
        fingerprint = canonical_topology_fingerprint(const)
        assignment = fixed_role_assignment(
            const,
            topology_fingerprint=fingerprint,
        )
        for base_seed in seeds:
            for replicate_index in range(episodes_per_seed):
                cases.append(
                    EvaluationCase(
                        topology_index=topology_index,
                        topology_path=topology_path,
                        topology_fingerprint=fingerprint,
                        base_seed=base_seed,
                        replicate_index=replicate_index,
                        episode_seed=base_seed + replicate_index,
                        host_roles=assignment.host_roles,
                        role_map_id=assignment.role_map_id,
                    )
                )
    return tuple(cases)


__all__ = [
    "FIXED_ROLE_ASSIGNMENT_VERSION",
    "MIN_RESILIENCE_ROLE_HOSTS",
    "TOPOLOGY_FINGERPRINT_VERSION",
    "EvaluationCase",
    "FixedRoleAssignment",
    "build_evaluation_cases",
    "canonical_topology_fingerprint",
    "eligible_operational_server_indices",
    "fixed_role_assignment",
    "validate_fixed_role_eligibility",
]
