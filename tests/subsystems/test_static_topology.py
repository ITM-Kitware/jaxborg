import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    MISSION_PHASES,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
    NUM_SERVICES,
    NUM_SUBNETS,
    SUBNET_IDS,
    SUBNET_NAMES,
)
from jaxborg.topology import (
    BLUE_AGENT_SUBNETS,
    CYBORG_SUFFIX_TO_ID,
    _compute_mission_phases,
    _subnet_nacl_adjacency,
)


@pytest.fixture(scope="module")
def cyborg_env():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)


class TestPureToplogy:
    def test_shapes(self, jax_const):
        c = jax_const
        assert c.host_active.shape == (GLOBAL_MAX_HOSTS,)
        assert c.host_subnet.shape == (GLOBAL_MAX_HOSTS,)
        assert c.host_is_router.shape == (GLOBAL_MAX_HOSTS,)
        assert c.host_is_server.shape == (GLOBAL_MAX_HOSTS,)
        assert c.host_is_user.shape == (GLOBAL_MAX_HOSTS,)
        assert c.subnet_adjacency.shape == (NUM_SUBNETS, NUM_SUBNETS)
        assert c.data_links.shape == (GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS)
        assert c.initial_services.shape == (GLOBAL_MAX_HOSTS, NUM_SERVICES)
        assert c.blue_agent_subnets.shape == (NUM_BLUE_AGENTS, NUM_SUBNETS)
        assert c.blue_agent_hosts.shape == (NUM_BLUE_AGENTS, GLOBAL_MAX_HOSTS)
        assert c.red_start_hosts.shape == (NUM_RED_AGENTS,)
        assert c.phase_boundaries.shape == (MISSION_PHASES,)
        assert c.allowed_subnet_pairs.shape == (MISSION_PHASES, NUM_SUBNETS, NUM_SUBNETS)

    def test_host_counts_in_range(self, jax_const):
        c = jax_const
        num_active = int(jnp.sum(c.host_active))
        assert 9 * 4 + 1 <= num_active <= 8 * 17 + 1
        assert int(c.num_hosts) == num_active

    def test_every_subnet_has_hosts(self, jax_const):
        c = jax_const
        for sid in range(NUM_SUBNETS):
            count = int(jnp.sum(c.host_active & (c.host_subnet == sid)))
            assert count >= 1, f"Subnet {SUBNET_NAMES[sid]} has no hosts"

    def test_non_internet_subnets_have_router(self, jax_const):
        c = jax_const
        for sid in range(NUM_SUBNETS):
            sname = SUBNET_NAMES[sid]
            subnet_mask = c.host_active & (c.host_subnet == sid)
            if sname == "INTERNET":
                assert int(jnp.sum(subnet_mask & c.host_is_router)) == 0
            else:
                assert int(jnp.sum(subnet_mask & c.host_is_router)) == 1

    def test_internet_has_one_host(self, jax_const):
        c = jax_const
        sid = SUBNET_IDS["INTERNET"]
        count = int(jnp.sum(c.host_active & (c.host_subnet == sid)))
        assert count == 1

    def test_host_type_mutually_exclusive(self, jax_const):
        c = jax_const
        for h in range(int(c.num_hosts)):
            if not c.host_active[h]:
                continue
            types = int(c.host_is_router[h]) + int(c.host_is_server[h]) + int(c.host_is_user[h])
            sid = int(c.host_subnet[h])
            if SUBNET_NAMES[sid] == "INTERNET":
                assert types == 0
            else:
                assert types == 1

    def test_routers_dont_respond_to_ping(self, jax_const):
        c = jax_const
        for h in range(int(c.num_hosts)):
            if c.host_is_router[h]:
                assert not c.host_respond_to_ping[h]

    def test_user_and_server_respond_to_ping(self, jax_const):
        c = jax_const
        for h in range(int(c.num_hosts)):
            if c.host_is_user[h] or c.host_is_server[h]:
                assert c.host_respond_to_ping[h]

    def test_all_non_router_hosts_have_ssh(self, jax_const):
        from jaxborg.constants import SERVICE_IDS

        c = jax_const
        ssh_idx = SERVICE_IDS["SSHD"]
        for h in range(int(c.num_hosts)):
            if c.host_active[h] and not c.host_is_router[h] and SUBNET_NAMES[int(c.host_subnet[h])] != "INTERNET":
                assert c.initial_services[h, ssh_idx]

    def test_operational_hosts_have_otservice(self, jax_const):
        from jaxborg.constants import SERVICE_IDS

        c = jax_const
        ot_idx = SERVICE_IDS["OTSERVICE"]
        for h in range(int(c.num_hosts)):
            if not c.host_active[h]:
                continue
            sid = int(c.host_subnet[h])
            sname = SUBNET_NAMES[sid]
            if sname in ("OPERATIONAL_ZONE_A", "OPERATIONAL_ZONE_B") and (c.host_is_user[h] or c.host_is_server[h]):
                assert c.initial_services[h, ot_idx]

    def test_data_links_symmetric(self, jax_const):
        dl = np.array(jax_const.data_links)
        np.testing.assert_array_equal(dl, dl.T)

    def test_data_links_no_self_loops(self, jax_const):
        dl = np.array(jax_const.data_links)
        assert not np.any(np.diag(dl))

    def test_non_router_hosts_linked_to_their_router(self, jax_const):
        c = jax_const
        dl = np.array(c.data_links)
        for h in range(int(c.num_hosts)):
            if not c.host_active[h]:
                continue
            sid = int(c.host_subnet[h])
            sname = SUBNET_NAMES[sid]
            if c.host_is_router[h] or sname == "INTERNET":
                continue
            router_hosts = [
                r
                for r in range(int(c.num_hosts))
                if c.host_active[r] and c.host_is_router[r] and int(c.host_subnet[r]) == sid
            ]
            assert len(router_hosts) == 1
            assert dl[h, router_hosts[0]]

    def test_subnet_adjacency_matches_nacls(self):
        adj = _subnet_nacl_adjacency()
        S = SUBNET_IDS
        assert adj[S["RESTRICTED_ZONE_A"], S["OPERATIONAL_ZONE_A"]]
        assert adj[S["OPERATIONAL_ZONE_A"], S["RESTRICTED_ZONE_A"]]
        assert not adj[S["OPERATIONAL_ZONE_A"], S["CONTRACTOR_NETWORK"]]
        assert adj[S["INTERNET"], S["CONTRACTOR_NETWORK"]]
        assert not adj[S["CONTRACTOR_NETWORK"], S["INTERNET"]]

    def test_blue_agent_subnets(self, jax_const):
        c = jax_const
        for i, snames in enumerate(BLUE_AGENT_SUBNETS):
            for sname in snames:
                sid = SUBNET_IDS[sname]
                assert c.blue_agent_subnets[i, sid]
            unassigned = [SUBNET_IDS[sn] for sn in SUBNET_NAMES if sn not in snames]
            for sid in unassigned:
                assert not c.blue_agent_subnets[i, sid]

    def test_blue_agent_hosts_match_subnets(self, jax_const):
        c = jax_const
        for i in range(NUM_BLUE_AGENTS):
            for h in range(int(c.num_hosts)):
                if not c.host_active[h]:
                    continue
                sid = int(c.host_subnet[h])
                if c.blue_agent_subnets[i, sid]:
                    assert c.blue_agent_hosts[i, h]
                else:
                    assert not c.blue_agent_hosts[i, h]

    def test_red_agent_0_starts_in_contractor(self, jax_const):
        c = jax_const
        start = int(c.red_start_hosts[0])
        assert SUBNET_NAMES[int(c.host_subnet[start])] == "CONTRACTOR_NETWORK"

    def test_all_red_agents_have_valid_start_hosts(self, jax_const):
        c = jax_const
        for i in range(NUM_RED_AGENTS):
            start = int(c.red_start_hosts[i])
            assert c.host_active[start], f"red_agent_{i} start host {start} is not active"

    def test_green_agents_on_user_hosts(self, jax_const):
        c = jax_const
        num_user = int(jnp.sum(c.host_is_user))
        assert int(c.num_green_agents) == num_user
        for h in range(int(c.num_hosts)):
            if c.host_is_user[h]:
                assert c.green_agent_active[h]
                assert c.green_agent_host[h] >= 0
            else:
                assert not c.green_agent_active[h]

    def test_mission_phases_sum(self):
        assert sum(_compute_mission_phases(500)) == 500
        assert sum(_compute_mission_phases(100)) == 100
        assert sum(_compute_mission_phases(99)) == 99
        assert sum(_compute_mission_phases(101)) == 101

    def test_phase_boundaries(self, jax_const):
        c = jax_const
        assert int(c.phase_boundaries[0]) == 0
        for i in range(1, MISSION_PHASES):
            assert int(c.phase_boundaries[i]) > int(c.phase_boundaries[i - 1])

    def test_allowed_subnet_pairs_symmetric(self, jax_const):
        pairs = np.array(jax_const.allowed_subnet_pairs)
        for p in range(MISSION_PHASES):
            np.testing.assert_array_equal(pairs[p], pairs[p].T)


class TestDifferentialWithCybORG:
    def test_host_count_matches(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        state = cyborg_env.environment_controller.state
        assert int(c.num_hosts) == len(state.hosts)

    def test_subnet_assignment_matches(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(state.hosts.keys())
        for idx, hostname in enumerate(sorted_hosts):
            cyborg_subnet = state.hostname_subnet_map[hostname]
            expected_sid = CYBORG_SUFFIX_TO_ID[cyborg_subnet]
            assert int(c.host_subnet[idx]) == expected_sid, f"Mismatch for {hostname}"

    def test_host_types_match(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(state.hosts.keys())
        for idx, hostname in enumerate(sorted_hosts):
            if hostname == "root_internet_host_0":
                assert not c.host_is_router[idx]
                assert not c.host_is_server[idx]
                assert not c.host_is_user[idx]
            elif "_router" in hostname:
                assert c.host_is_router[idx]
            elif "_server_host_" in hostname:
                assert c.host_is_server[idx]
            elif "_user_host_" in hostname:
                assert c.host_is_user[idx]

    def test_respond_to_ping_matches(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(state.hosts.keys())
        for idx, hostname in enumerate(sorted_hosts):
            expected = state.hosts[hostname].respond_to_ping
            assert bool(c.host_respond_to_ping[idx]) == expected, f"Mismatch for {hostname}"

    def test_data_links_match_cyborg(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(state.hosts.keys())
        hostname_to_idx = {h: i for i, h in enumerate(sorted_hosts)}
        dl = np.array(c.data_links)

        for hostname, host in state.hosts.items():
            h = hostname_to_idx[hostname]
            for iface in host.interfaces:
                if iface.interface_type == "wired":
                    for dl_name in iface.data_links:
                        if dl_name in hostname_to_idx:
                            j = hostname_to_idx[dl_name]
                            assert dl[h, j], f"Missing link {hostname} -> {dl_name}"

    def test_services_match_cyborg(self, cyborg_env):
        from jaxborg.constants import SERVICE_IDS
        from jaxborg.topology import build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(state.hosts.keys())
        for idx, hostname in enumerate(sorted_hosts):
            host = state.hosts[hostname]
            if host.services:
                for svc_name in host.services:
                    svc_str = str(svc_name).split(".")[-1] if "." in str(svc_name) else str(svc_name)
                    if svc_str in SERVICE_IDS:
                        assert c.initial_services[idx, SERVICE_IDS[svc_str]], f"Missing service {svc_str} on {hostname}"

    def test_red_agent_0_only_active_at_reset(self, cyborg_env):
        from jaxborg.env import _init_red_state
        from jaxborg.state import create_initial_state
        from jaxborg.topology import build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        state = create_initial_state()
        state = state.replace(host_services=c.initial_services)
        state = _init_red_state(c, state)
        # Only red_agent_0 should be active at reset
        assert bool(state.red_agent_active[0])
        for r in range(1, NUM_RED_AGENTS):
            assert not bool(state.red_agent_active[r]), f"red_agent_{r} should not be active at reset"

    def test_green_count_matches(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        scenario = cyborg_env.environment_controller.state.scenario
        cyborg_green_count = sum(1 for name in scenario.agents if name.startswith("green_agent_"))
        assert int(c.num_green_agents) == cyborg_green_count

    def test_phase_boundaries_match(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        scenario = cyborg_env.environment_controller.state.scenario
        cumulative = 0
        for i, phase_len in enumerate(scenario.mission_phases):
            assert int(c.phase_boundaries[i]) == cumulative
            cumulative += phase_len

    def test_allowed_subnet_pairs_match(self, cyborg_env):
        from jaxborg.topology import _compute_allowed_subnet_pairs, build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        scenario = cyborg_env.environment_controller.state.scenario
        expected = _compute_allowed_subnet_pairs(scenario.allowed_subnets_per_mphase)
        np.testing.assert_array_equal(np.array(c.allowed_subnet_pairs), expected)

    def test_host_order_matches_cyborg(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg
        from jaxborg.translate import build_mappings_from_cyborg

        build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)
        hostnames = [mappings.idx_to_hostname[i] for i in range(mappings.num_hosts)]
        assert hostnames == sorted(hostnames), f"Host ordering is not alphabetical: {hostnames[:10]}..."

    def test_initial_foothold_privilege_matches_cyborg(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg
        from jaxborg.translate import build_mappings_from_cyborg

        mappings = build_mappings_from_cyborg(cyborg_env)
        const = build_const_from_cyborg(cyborg_env)
        cyborg_state = cyborg_env.environment_controller.state

        for agent_name, sessions in cyborg_state.sessions.items():
            if not agent_name.startswith("red_agent_"):
                continue
            red_idx = int(agent_name.split("_")[-1])
            if red_idx >= NUM_RED_AGENTS:
                continue
            for sess in sessions.values():
                if sess.hostname not in mappings.hostname_to_idx:
                    continue
                host_idx = mappings.hostname_to_idx[sess.hostname]
                is_root = hasattr(sess, "username") and sess.username in ("root", "SYSTEM")
                if is_root:
                    assert bool(const.host_active[host_idx]), f"Red foothold host {sess.hostname} not active in JAX"

    def test_adjacency_is_not_all_true(self, jax_const):
        adj = np.array(jax_const.subnet_adjacency)
        active = adj[:NUM_SUBNETS, :NUM_SUBNETS]
        assert not np.all(active), "Subnet adjacency should not be all-true"

    def test_adjacency_is_symmetric_where_expected(self, jax_const):
        adj = np.array(jax_const.subnet_adjacency)
        assert adj[SUBNET_IDS["RESTRICTED_ZONE_A"], SUBNET_IDS["OPERATIONAL_ZONE_A"]]
        assert adj[SUBNET_IDS["OPERATIONAL_ZONE_A"], SUBNET_IDS["RESTRICTED_ZONE_A"]]
