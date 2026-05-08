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

import random
import re
from typing import Iterable

import jax
import jax.numpy as jnp

from jaxborg.constants import GLOBAL_MAX_HOSTS, SUBNET_IDS
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


def assign_resilience_roles(
    hostnames: Iterable[str],
    rng: random.Random,
) -> dict[str, int]:
    """Assign AUTH / DB / WEB to 3 randomly-chosen operational-zone server hostnames.

    ``rng`` should be seeded from the env's per-episode seed so the same
    episode reproduces the same role map. If fewer than 3 eligible hostnames
    exist, only the available roles are assigned (rest stay ROLE_NONE).
    """
    candidates = sorted(h for h in hostnames if is_operational_server(h))
    if len(candidates) < 3:
        raise ValueError(f"need ≥3 op-zone server candidates for AUTH/DB/WEB, got {len(candidates)}")
    rng.shuffle(candidates)
    return dict(zip(candidates, (ROLE_AUTH, ROLE_DB, ROLE_WEB)))


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


def assign_resilience_roles_from_const(
    const: SimulatorConst,
    key: jax.Array,
) -> jax.Array:
    """JAX-traceable role assignment from a ``SimulatorConst``.

    Tags 3 randomly-chosen active operational-zone server hosts as AUTH /
    DB / WEB. Returns a ``(GLOBAL_MAX_HOSTS,) int32`` array; non-tagged
    hosts are 0 (``ROLE_NONE``). ``key`` should be a fresh per-episode key
    so roles rotate across the op-zone server set rather than pinning to
    the same 3 hosts every episode.
    """
    is_resilience_zone = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
    for sid in _RESILIENCE_ZONE_SUBNETS:
        is_resilience_zone = is_resilience_zone | (const.host_subnet == sid)
    candidates = const.host_active & const.host_is_server & is_resilience_zone

    noise = jax.random.uniform(key, shape=(GLOBAL_MAX_HOSTS,))
    scores = jnp.where(candidates, noise, jnp.float32(jnp.inf))
    ranks = jnp.argsort(scores)
    auth_host, db_host, web_host = ranks[0], ranks[1], ranks[2]

    idx = jnp.arange(GLOBAL_MAX_HOSTS)
    host_resilience_role = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
    host_resilience_role = jnp.where(idx == auth_host, jnp.int32(ROLE_AUTH), host_resilience_role)
    host_resilience_role = jnp.where(idx == db_host, jnp.int32(ROLE_DB), host_resilience_role)
    host_resilience_role = jnp.where(idx == web_host, jnp.int32(ROLE_WEB), host_resilience_role)
    return host_resilience_role
