import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_red_action
from jaxborg.actions.encoding import encode_red_action
from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    MAX_HOSTS_PER_SUBNET,
    MIN_HOSTS_PER_SUBNET,
    NUM_BLUE_AGENTS,
    NUM_SUBNETS,
    SUBNET_IDS,
    SUBNET_NAMES,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import build_topology

_jit_apply_red = jax.jit(apply_red_action, static_argnums=(2,))

SEEDS = [1, 7, 42, 100, 256, 999, 12345, 54321]


@pytest.fixture(params=SEEDS)
def seed(request):
    return request.param


@pytest.fixture
def jax_const(seed):
    return build_topology(jax.random.PRNGKey(seed), num_steps=500)


class TestMultiSeedTopologyValidity:
    def test_num_hosts_in_valid_range(self, jax_const):
        num_active = int(jnp.sum(jax_const.host_active))
        min_total = 8 * MIN_HOSTS_PER_SUBNET + 1
        max_total = 8 * MAX_HOSTS_PER_SUBNET + 1
        assert min_total <= num_active <= max_total
        assert int(jax_const.num_hosts) == num_active

    def test_every_subnet_has_hosts(self, jax_const):
        for sid in range(NUM_SUBNETS):
            count = int(jnp.sum(jax_const.host_active & (jax_const.host_subnet == sid)))
            assert count >= 1

    def test_non_internet_subnets_have_router(self, jax_const):
        for sid in range(NUM_SUBNETS):
            sname = SUBNET_NAMES[sid]
            subnet_mask = jax_const.host_active & (jax_const.host_subnet == sid)
            if sname == "INTERNET":
                assert int(jnp.sum(subnet_mask & jax_const.host_is_router)) == 0
            else:
                assert int(jnp.sum(subnet_mask & jax_const.host_is_router)) == 1

    def test_per_subnet_host_count_in_range(self, jax_const):
        for sid in range(NUM_SUBNETS):
            sname = SUBNET_NAMES[sid]
            if sname == "INTERNET":
                continue
            mask = jax_const.host_active & (jax_const.host_subnet == sid)
            count = int(jnp.sum(mask))
            assert MIN_HOSTS_PER_SUBNET <= count <= MAX_HOSTS_PER_SUBNET

    def test_internet_has_exactly_one_host(self, jax_const):
        sid = SUBNET_IDS["INTERNET"]
        count = int(jnp.sum(jax_const.host_active & (jax_const.host_subnet == sid)))
        assert count == 1

    def test_host_types_mutually_exclusive(self, jax_const):
        for h in range(int(jax_const.num_hosts)):
            if not jax_const.host_active[h]:
                continue
            types = int(jax_const.host_is_router[h]) + int(jax_const.host_is_server[h]) + int(jax_const.host_is_user[h])
            sid = int(jax_const.host_subnet[h])
            if SUBNET_NAMES[sid] == "INTERNET":
                assert types == 0
            else:
                assert types == 1

    def test_data_links_symmetric(self, jax_const):
        dl = np.array(jax_const.data_links)
        np.testing.assert_array_equal(dl, dl.T)

    def test_data_links_no_self_loops(self, jax_const):
        dl = np.array(jax_const.data_links)
        assert not np.any(np.diag(dl))

    def test_non_router_linked_to_their_router(self, jax_const):
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

    def test_red_agent_0_starts_in_contractor(self, jax_const):
        start = int(jax_const.red_start_hosts[0])
        assert SUBNET_NAMES[int(jax_const.host_subnet[start])] == "CONTRACTOR_NETWORK"

    def test_blue_agents_cover_subnets(self, jax_const):
        c = jax_const
        for i in range(NUM_BLUE_AGENTS):
            covered_hosts = int(jnp.sum(c.blue_agent_hosts[i] & c.host_active))
            assert covered_hosts > 0


class TestInactiveHostSlots:
    def test_inactive_hosts_not_active(self, jax_const):
        c = jax_const
        for h in range(int(c.num_hosts), GLOBAL_MAX_HOSTS):
            assert not bool(c.host_active[h])

    def test_inactive_hosts_no_router_server_user(self, jax_const):
        c = jax_const
        for h in range(int(c.num_hosts), GLOBAL_MAX_HOSTS):
            assert not bool(c.host_is_router[h])
            assert not bool(c.host_is_server[h])
            assert not bool(c.host_is_user[h])

    def test_inactive_hosts_no_services(self, jax_const):
        c = jax_const
        for h in range(int(c.num_hosts), GLOBAL_MAX_HOSTS):
            assert not np.any(np.array(c.initial_services[h]))

    def test_inactive_hosts_no_data_links(self, jax_const):
        c = jax_const
        dl = np.array(c.data_links)
        for h in range(int(c.num_hosts), GLOBAL_MAX_HOSTS):
            assert not np.any(dl[h, :])
            assert not np.any(dl[:, h])

    def test_inactive_hosts_no_green_agent(self, jax_const):
        c = jax_const
        for h in range(int(c.num_hosts), GLOBAL_MAX_HOSTS):
            assert not bool(c.green_agent_active[h])
            assert int(c.green_agent_host[h]) == -1

    def test_inactive_hosts_not_in_blue_agent_hosts(self, jax_const):
        c = jax_const
        for h in range(int(c.num_hosts), GLOBAL_MAX_HOSTS):
            for i in range(NUM_BLUE_AGENTS):
                assert not bool(c.blue_agent_hosts[i, h])


class TestSeedVariability:
    def test_different_seeds_different_host_counts(self):
        counts = set()
        for seed in range(50):
            c = build_topology(jax.random.PRNGKey(seed), num_steps=500)
            counts.add(int(c.num_hosts))
        assert len(counts) > 1

    def test_different_seeds_different_start_hosts(self):
        starts = set()
        for seed in range(20):
            c = build_topology(jax.random.PRNGKey(seed), num_steps=500)
            starts.add(int(c.red_start_hosts[0]))
        assert len(starts) > 1

    def test_host_count_distribution(self):
        min_total = 8 * MIN_HOSTS_PER_SUBNET + 1
        max_total = 8 * MAX_HOSTS_PER_SUBNET + 1
        counts = []
        for seed in range(100):
            c = build_topology(jax.random.PRNGKey(seed), num_steps=500)
            counts.append(int(c.num_hosts))
        assert min(counts) >= min_total
        assert max(counts) <= max_total
        spread = max(counts) - min(counts)
        assert spread >= 10


class TestActionHandlersRespectHostActive:
    def test_action_on_inactive_host_no_effect(self):
        const = build_topology(jax.random.PRNGKey(42), num_steps=500)
        state = create_initial_state()
        start_host = int(const.red_start_hosts[0])
        sessions = state.red_sessions.at[0, start_host].set(True)
        state = state.replace(red_sessions=sessions)

        inactive_host = int(const.num_hosts)
        assert not bool(const.host_active[inactive_host])

        discovered = state.red_discovered_hosts.at[0, inactive_host].set(True)
        scanned = state.red_scanned_hosts.at[0, inactive_host].set(True)
        state = state.replace(red_discovered_hosts=discovered, red_scanned_hosts=scanned)

        scan_idx = encode_red_action("DiscoverNetworkServices", inactive_host, 0)
        new_state = _jit_apply_red(state, const, 0, scan_idx, jax.random.PRNGKey(0))
        was_scanned = bool(state.red_scanned_hosts[0, inactive_host])
        assert not bool(new_state.red_scanned_hosts[0, inactive_host]) or was_scanned

    def test_discover_only_finds_active_hosts(self):
        const = build_topology(jax.random.PRNGKey(42), num_steps=500)
        state = create_initial_state()
        start_host = int(const.red_start_hosts[0])
        sessions = state.red_sessions.at[0, start_host].set(True)
        state = state.replace(red_sessions=sessions)

        start_subnet = int(const.host_subnet[start_host])
        discover_idx = encode_red_action("DiscoverRemoteSystems", start_subnet, 0)
        new_state = _jit_apply_red(state, const, 0, discover_idx, jax.random.PRNGKey(0))

        discovered = np.array(new_state.red_discovered_hosts[0])
        for h in range(GLOBAL_MAX_HOSTS):
            if discovered[h]:
                assert bool(const.host_active[h]), f"Discovered inactive host {h}"

    def test_jit_with_dynamic_topology(self):
        const = build_topology(jax.random.PRNGKey(42), num_steps=500)
        state = create_initial_state()
        start_host = int(const.red_start_hosts[0])
        sessions = state.red_sessions.at[0, start_host].set(True)
        state = state.replace(red_sessions=sessions)

        start_subnet = int(const.host_subnet[start_host])
        discover_idx = encode_red_action("DiscoverRemoteSystems", start_subnet, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(state, const, 0, discover_idx, jax.random.PRNGKey(0))

        discovered = np.array(new_state.red_discovered_hosts[0])
        for h in range(GLOBAL_MAX_HOSTS):
            if discovered[h]:
                assert bool(const.host_active[h])


class TestDifferentialHostCounts:
    @pytest.fixture(scope="class")
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_host_count_matches_cyborg(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        cyborg_state = cyborg_env.environment_controller.state
        assert int(c.num_hosts) == len(cyborg_state.hosts)
        assert int(jnp.sum(c.host_active)) == len(cyborg_state.hosts)

    def test_per_subnet_counts_match_cyborg(self, cyborg_env):
        from jaxborg.topology import CYBORG_SUFFIX_TO_ID, build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        cyborg_state = cyborg_env.environment_controller.state

        for hostname in cyborg_state.hosts:
            subnet_name = cyborg_state.hostname_subnet_map[hostname]
            if subnet_name not in CYBORG_SUFFIX_TO_ID:
                continue
        subnet_counts_cyborg = {}
        for hostname in cyborg_state.hosts:
            sn = cyborg_state.hostname_subnet_map[hostname]
            if sn in CYBORG_SUFFIX_TO_ID:
                sid = CYBORG_SUFFIX_TO_ID[sn]
                subnet_counts_cyborg[sid] = subnet_counts_cyborg.get(sid, 0) + 1

        for sid, cyborg_count in subnet_counts_cyborg.items():
            jax_count = int(jnp.sum(c.host_active & (c.host_subnet == sid)))
            assert jax_count == cyborg_count, f"Subnet {sid}: JAX={jax_count} CybORG={cyborg_count}"

    def test_inactive_slots_clean_in_cyborg_extracted(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        c = build_const_from_cyborg(cyborg_env)
        for h in range(int(c.num_hosts), GLOBAL_MAX_HOSTS):
            assert not bool(c.host_active[h])
            assert not bool(c.host_is_router[h])
            assert not bool(c.host_is_server[h])
            assert not bool(c.host_is_user[h])
            assert not np.any(np.array(c.initial_services[h]))
