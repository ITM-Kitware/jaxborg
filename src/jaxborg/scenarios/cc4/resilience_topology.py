"""Resilience-augmented CC4 topology generator.

Extends the base ``build_topology`` with three resilience-critical servers
placed randomly among the active server hosts in Operational Zone A and B:

  RESILIENCE_ROLE_AUTH (1) — authentication server (CIA: C + I)
  RESILIENCE_ROLE_DB   (2) — database server       (CIA: C + A)
  RESILIENCE_ROLE_WEB  (3) — frontend web server   (CIA: I + A)

Usage::

    const, host_resilience_role = build_resilience_topology(key, num_steps=500)
    # host_resilience_role: (GLOBAL_MAX_HOSTS,) int32 — 0 for all non-resilience hosts

The returned ``SimulatorConst`` is identical to the base topology — only the
extra ``host_resilience_role`` array carries server identity.  Pass both to
the resilience-aware red agent in ``resilience_red_fsm.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxborg.constants import GLOBAL_MAX_HOSTS, SUBNET_IDS
from jaxborg.scenarios.cc4.topology import build_topology
from jaxborg.state import SimulatorConst

RESILIENCE_ROLE_NONE = jnp.int32(0)
RESILIENCE_ROLE_AUTH = jnp.int32(1)  # authentication server
RESILIENCE_ROLE_DB = jnp.int32(2)  # database server
RESILIENCE_ROLE_WEB = jnp.int32(3)  # frontend web server

_RESILIENCE_ZONE_SUBNETS = (
    SUBNET_IDS["OPERATIONAL_ZONE_A"],
    SUBNET_IDS["OPERATIONAL_ZONE_B"],
)


def _assign_resilience_roles(const: SimulatorConst) -> jax.Array:
    """Deterministically assign auth/db/web roles to 3 Operational Zone server hosts.

    Selects the 3 eligible hosts with the lowest global indices so that any two
    episodes with the same active-host set produce identical role assignments.
    If fewer than 3 eligible hosts exist only the available roles are assigned;
    the rest stay RESILIENCE_ROLE_NONE.
    """
    is_resilience_zone = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
    for sid in _RESILIENCE_ZONE_SUBNETS:
        is_resilience_zone = is_resilience_zone | (const.host_subnet == sid)

    candidates = const.host_active & const.host_is_server & is_resilience_zone

    # Sort eligible hosts by index (ascending); non-candidates sort to the end.
    scores = jnp.where(candidates, jnp.arange(GLOBAL_MAX_HOSTS), jnp.iinfo(jnp.int32).max)
    ranks = jnp.argsort(scores)
    auth_host = ranks[0]
    db_host = ranks[1]
    web_host = ranks[2]

    n_candidates = jnp.sum(candidates.astype(jnp.int32))
    idx = jnp.arange(GLOBAL_MAX_HOSTS)
    host_resilience_role = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)

    host_resilience_role = jnp.where(
        (idx == auth_host) & (n_candidates >= 1),
        RESILIENCE_ROLE_AUTH,
        host_resilience_role,
    )
    host_resilience_role = jnp.where(
        (idx == db_host) & (n_candidates >= 2),
        RESILIENCE_ROLE_DB,
        host_resilience_role,
    )
    host_resilience_role = jnp.where(
        (idx == web_host) & (n_candidates >= 3),
        RESILIENCE_ROLE_WEB,
        host_resilience_role,
    )
    return host_resilience_role


def build_resilience_topology(
    key: jax.Array,
    num_steps: int = 500,
    *,
    training_mode: bool = False,
) -> tuple[SimulatorConst, jax.Array]:
    """Build a CC4 topology and assign resilience server roles.

    Args:
        key:           JAX PRNG key.
        num_steps:     Episode length passed to the base topology.
        training_mode: Passed through to ``build_topology``.

    Returns:
        const:                 Standard ``SimulatorConst`` (unchanged).
        host_resilience_role:  ``(GLOBAL_MAX_HOSTS,)`` int32 — RESILIENCE_ROLE_*
                               per host; 0 for normal hosts.
    """
    const = build_topology(key, num_steps, training_mode=training_mode)
    host_resilience_role = _assign_resilience_roles(const)
    return const, host_resilience_role


def resilience_role_name(role: int) -> str:
    return {0: "none", 1: "auth", 2: "db", 3: "web"}.get(int(role), "unknown")
