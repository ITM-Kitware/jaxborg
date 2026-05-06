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
from typing import Iterable

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
