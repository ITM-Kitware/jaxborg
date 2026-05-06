"""Pin the canonical resilience role assignment.

The JAX topology factory, the trajectory recorder, the resilience metric, and
the CybORG-side biased red agents must all agree on which hostnames are
operational-zone servers and which 3 of them are AUTH/DB/WEB. They share one
implementation in ``jaxborg.scenarios.cc4.topology_roles``; this test pins
it so the next refactor doesn't silently desync the four call sites.
"""

from __future__ import annotations

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


def test_assign_picks_lowest_three_sorted_op_zone_servers():
    hosts = [
        "router_internet",
        "operational_zone_b_subnet_server_host_2",
        "operational_zone_a_subnet_server_host_5",
        "operational_zone_a_subnet_server_host_1",
        "operational_zone_a_subnet_user_host_0",  # not a server — ignored
        "operational_zone_b_subnet_server_host_0",
    ]
    roles = assign_resilience_roles(hosts)
    # Sorted op-zone servers: a_..._1, a_..._5, b_..._0, b_..._2  (alpha sort)
    assert roles == {
        "operational_zone_a_subnet_server_host_1": ROLE_AUTH,
        "operational_zone_a_subnet_server_host_5": ROLE_DB,
        "operational_zone_b_subnet_server_host_0": ROLE_WEB,
    }


def test_assign_is_deterministic_across_input_order():
    hosts = [
        "operational_zone_b_subnet_server_host_0",
        "operational_zone_a_subnet_server_host_1",
        "operational_zone_a_subnet_server_host_5",
    ]
    assert assign_resilience_roles(hosts) == assign_resilience_roles(reversed(hosts))


def test_assign_handles_fewer_than_three_candidates():
    assert assign_resilience_roles([]) == {}
    assert assign_resilience_roles(["operational_zone_a_subnet_server_host_0"]) == {
        "operational_zone_a_subnet_server_host_0": ROLE_AUTH,
    }
    assert assign_resilience_roles(["router_internet", "user_host_0"]) == {}


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
