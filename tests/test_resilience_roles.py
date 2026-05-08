"""Pin the canonical resilience role assignment.

The JAX topology factory, the trajectory recorder, the resilience metric, and
the CybORG-side biased red agents must all agree on which hostnames are
operational-zone servers and which 3 of them are AUTH/DB/WEB. They share one
implementation in ``jaxborg.scenarios.cc4.topology_roles``; this test pins
it so the next refactor doesn't silently desync the four call sites.
"""

from __future__ import annotations

import random

import jax
import jax.numpy as jnp

from jaxborg.scenarios.cc4.topology import build_topology
from jaxborg.scenarios.cc4.topology_roles import (
    OPERATIONAL_SERVER_RE,
    ROLE_AUTH,
    ROLE_DB,
    ROLE_WEB,
    assign_resilience_roles,
    is_operational_server,
    role_name,
)


def test_operational_server_regex_matches_cc4_hostnames():
    assert is_operational_server("operational_zone_a_subnet_server_host_0")
    assert is_operational_server("operational_zone_b_subnet_server_host_12")
    assert not is_operational_server("operational_zone_a_subnet_user_host_0")
    assert not is_operational_server("restricted_zone_a_subnet_server_host_0")
    assert not is_operational_server("router_internet")
    assert not is_operational_server("")


def test_assign_picks_three_op_zone_servers_with_canonical_roles():
    hosts = [
        "router_internet",
        "operational_zone_b_subnet_server_host_2",
        "operational_zone_a_subnet_server_host_5",
        "operational_zone_a_subnet_server_host_1",
        "operational_zone_a_subnet_user_host_0",  # not a server — ignored
        "operational_zone_b_subnet_server_host_0",
    ]
    roles = assign_resilience_roles(hosts, random.Random(42))
    assert len(roles) == 3
    assert set(roles.values()) == {ROLE_AUTH, ROLE_DB, ROLE_WEB}
    for host in roles:
        assert is_operational_server(host)


def test_assign_is_reproducible_for_same_seed():
    hosts = [f"operational_zone_a_subnet_server_host_{i}" for i in range(6)]
    a = assign_resilience_roles(hosts, random.Random(123))
    b = assign_resilience_roles(hosts, random.Random(123))
    assert a == b


def test_assign_is_input_order_invariant():
    hosts = [f"operational_zone_a_subnet_server_host_{i}" for i in range(4)]
    forward = assign_resilience_roles(hosts, random.Random(7))
    reversed_in = assign_resilience_roles(list(reversed(hosts)), random.Random(7))
    assert forward == reversed_in


def test_assign_varies_across_seeds():
    hosts = [f"operational_zone_a_subnet_server_host_{i}" for i in range(6)]
    seeds_seen = {tuple(sorted(assign_resilience_roles(hosts, random.Random(s)).items())) for s in range(50)}
    # 6 hosts × 3 roles = 120 permutations; expect plenty of variety in 50 seeds.
    assert len(seeds_seen) >= 10


def test_assign_raises_on_fewer_than_three_candidates():
    import pytest

    with pytest.raises(ValueError, match="need ≥3"):
        assign_resilience_roles(
            [
                "operational_zone_a_subnet_server_host_0",
                "operational_zone_a_subnet_server_host_1",
            ],
            random.Random(0),
        )
    with pytest.raises(ValueError):
        assign_resilience_roles([], random.Random(0))


def test_role_constants_and_names_match_metric_expectations():
    # The resilience metric and the JAX topology both depend on these exact
    # int values; CIA-event ↔ role-set lookups break if they drift.
    assert (ROLE_AUTH, ROLE_DB, ROLE_WEB) == (1, 2, 3)
    assert role_name(ROLE_AUTH) == "auth"
    assert role_name(ROLE_DB) == "db"
    assert role_name(ROLE_WEB) == "web"
    assert role_name(0) == "none"
    assert role_name(99) == "unknown"


def test_count_resilience_candidates_matches_active_op_zone_servers():
    from jaxborg.constants import SUBNET_IDS
    from jaxborg.scenarios.cc4.topology_roles import count_resilience_candidates

    const = build_topology(jax.random.PRNGKey(0))
    is_op = (const.host_subnet == SUBNET_IDS["OPERATIONAL_ZONE_A"]) | (
        const.host_subnet == SUBNET_IDS["OPERATIONAL_ZONE_B"]
    )
    expected = int(jnp.sum(const.host_active & const.host_is_server & is_op))
    assert count_resilience_candidates(const) == expected


def test_make_jax_env_rejects_resilience_variant_with_too_few_op_zone_servers():
    import pytest

    from jaxborg.evaluation.jax_env_factory import make_jax_env
    from jaxborg.scenarios.cc4.game_variant import GameVariant

    bad = GameVariant(name="bad", resilience_roles=True, op_zone_servers=0)
    with pytest.raises(ValueError, match="op_zone_servers=0"):
        make_jax_env(bad)

    # Sanity: a sensible variant constructs without error. The CIA metric
    # needs at least 3 op-zone server candidates per episode.
    make_jax_env(GameVariant(name="ok", resilience_roles=True, op_zone_servers=1))


def test_make_jax_env_rejects_snapshot_without_enough_op_zone_servers(tmp_path):
    import json

    import numpy as np
    import pytest

    from jaxborg.constants import CC4_CONFIG
    from jaxborg.evaluation.jax_env_factory import make_jax_env
    from jaxborg.scenarios.cc4.game_variant import GameVariant
    from jaxborg.scenarios.cc4.topology import (
        TOPOLOGY_SNAPSHOT_FIELDS,
        TOPOLOGY_SNAPSHOT_METADATA_KEY,
        _scenario_config_digest,
        build_topology,
    )

    const = build_topology(jax.random.PRNGKey(0))
    const = const.replace(host_is_server=jnp.zeros_like(const.host_is_server))
    snapshot_path = tmp_path / "no_servers.npz"
    metadata = json.dumps({"scenario_config": _scenario_config_digest(CC4_CONFIG)})
    np.savez(
        snapshot_path,
        **{name: np.asarray(getattr(const, name)) for name in TOPOLOGY_SNAPSHOT_FIELDS},
        **{TOPOLOGY_SNAPSHOT_METADATA_KEY: np.asarray(metadata)},
    )

    bad = GameVariant(name="snap", resilience_roles=True)
    with pytest.raises(ValueError, match="op-zone server candidates"):
        make_jax_env(bad, topology_path=snapshot_path)


def test_regex_is_anchored():
    # Substring matches must not slip through.
    assert OPERATIONAL_SERVER_RE.match("operational_zone_a_subnet_server_host_0")
    assert not OPERATIONAL_SERVER_RE.match("prefix_operational_zone_a_subnet_server_host_0")
    assert not OPERATIONAL_SERVER_RE.match("operational_zone_a_subnet_server_host_0_suffix")
