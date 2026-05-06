"""Canonical resilience role assignment — single source of truth.

The resilience metric, the trajectory recorder, the CybORG-side biased red
agents, and the JAX topology factory all need to agree on:

  1. Which hostnames are operational-zone servers (eligible for resilience roles).
  2. Which 3 of those are tagged AUTH / DB / WEB in any given episode.

Importing from this module everywhere keeps those four call sites in sync by
construction.

Role integer convention is the same on both JAX and Python sides — JAX broadcasts
plain ints when comparing against jnp arrays, so no jnp-typed wrappers are needed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable

import jax
import jax.numpy as jnp

from jaxborg.constants import GLOBAL_MAX_HOSTS, SUBNET_IDS

if TYPE_CHECKING:
    from jaxborg.state import SimulatorConst

ROLE_NONE = 0
ROLE_AUTH = 1  # authentication server  (CIA: C + I)
ROLE_DB = 2  # database server         (CIA: C + A)
ROLE_WEB = 3  # frontend web server     (CIA: I + A)

ROLE_NAMES: dict[int, str] = {ROLE_NONE: "none", ROLE_AUTH: "auth", ROLE_DB: "db", ROLE_WEB: "web"}

# CC4 hostnames that follow this pattern are eligible for resilience roles.
OPERATIONAL_SERVER_RE = re.compile(r"^operational_zone_[ab]_subnet_server_host_\d+$")


def is_operational_server(hostname: str) -> bool:
    return bool(OPERATIONAL_SERVER_RE.match(hostname))


def assign_resilience_roles(hostnames: Iterable[str]) -> dict[str, int]:
    """Assign AUTH / DB / WEB to the 3 lowest-sorted operational-zone server hostnames.

    Deterministic given the same input host set. If fewer than 3 eligible
    hostnames exist, only the available roles are assigned and the rest stay
    implicitly ROLE_NONE (omitted from the returned dict).
    """
    candidates = sorted(h for h in hostnames if is_operational_server(h))
    return dict(zip(candidates[:3], (ROLE_AUTH, ROLE_DB, ROLE_WEB)))


def role_name(role: int) -> str:
    return ROLE_NAMES.get(int(role), "unknown")


# ---------------------------------------------------------------------------
# JAX-side role assignment — operates on a SimulatorConst rather than a Python
# hostname list. Matches the hostname-list rule (sort by host index) so the
# trajectory recorder and JAX env produce identical role maps for the same
# active-host set.

_RESILIENCE_ZONE_SUBNETS = (
    SUBNET_IDS["OPERATIONAL_ZONE_A"],
    SUBNET_IDS["OPERATIONAL_ZONE_B"],
)


def assign_resilience_roles_from_const(const: "SimulatorConst") -> jax.Array:
    """JAX-traceable role assignment from a ``SimulatorConst``.

    Picks the 3 lowest-index active operational-zone server hosts and tags
    them AUTH/DB/WEB. Returns a ``(GLOBAL_MAX_HOSTS,) int32`` array; non-tagged
    hosts are 0 (``ROLE_NONE``).

    Stable: same active-host set → same assignment, JIT-friendly.
    """
    is_resilience_zone = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
    for sid in _RESILIENCE_ZONE_SUBNETS:
        is_resilience_zone = is_resilience_zone | (const.host_subnet == sid)

    candidates = const.host_active & const.host_is_server & is_resilience_zone

    # Lowest-index-first ordering: non-candidates sort to int32-max.
    scores = jnp.where(candidates, jnp.arange(GLOBAL_MAX_HOSTS), jnp.iinfo(jnp.int32).max)
    ranks = jnp.argsort(scores)
    auth_host = ranks[0]
    db_host = ranks[1]
    web_host = ranks[2]

    n_candidates = jnp.sum(candidates.astype(jnp.int32))
    idx = jnp.arange(GLOBAL_MAX_HOSTS)
    host_resilience_role = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
    host_resilience_role = jnp.where(
        (idx == auth_host) & (n_candidates >= 1), jnp.int32(ROLE_AUTH), host_resilience_role
    )
    host_resilience_role = jnp.where((idx == db_host) & (n_candidates >= 2), jnp.int32(ROLE_DB), host_resilience_role)
    host_resilience_role = jnp.where((idx == web_host) & (n_candidates >= 3), jnp.int32(ROLE_WEB), host_resilience_role)
    return host_resilience_role
