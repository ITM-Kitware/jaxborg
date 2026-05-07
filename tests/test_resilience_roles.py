"""Pin the canonical resilience role assignment.

The JAX topology factory, the trajectory recorder, the resilience metric, and
the CybORG-side biased red agents must all agree on which hostnames are
operational-zone servers and which 3 of them are AUTH/DB/WEB. They share one
implementation in ``jaxborg.scenarios.cc4.topology_roles``; this test pins
it so the next refactor doesn't silently desync the four call sites.
"""

from __future__ import annotations

import random

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


def test_assign_handles_fewer_than_three_candidates():
    rng = random.Random(0)
    assert assign_resilience_roles([], random.Random(0)) == {}
    assert assign_resilience_roles(["operational_zone_a_subnet_server_host_0"], rng) == {
        "operational_zone_a_subnet_server_host_0": ROLE_AUTH,
    }
    assert assign_resilience_roles(["router_internet", "user_host_0"], random.Random(0)) == {}


def test_role_constants_and_names_match_metric_expectations():
    # The resilience metric and the JAX topology both depend on these exact
    # int values; CIA-event ↔ role-set lookups break if they drift.
    assert (ROLE_AUTH, ROLE_DB, ROLE_WEB) == (1, 2, 3)
    assert role_name(ROLE_AUTH) == "auth"
    assert role_name(ROLE_DB) == "db"
    assert role_name(ROLE_WEB) == "web"
    assert role_name(0) == "none"
    assert role_name(99) == "unknown"


def test_regex_is_anchored():
    # Substring matches must not slip through.
    assert OPERATIONAL_SERVER_RE.match("operational_zone_a_subnet_server_host_0")
    assert not OPERATIONAL_SERVER_RE.match("prefix_operational_zone_a_subnet_server_host_0")
    assert not OPERATIONAL_SERVER_RE.match("operational_zone_a_subnet_server_host_0_suffix")
