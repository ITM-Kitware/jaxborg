"""Tests for build_topology() JIT compatibility, structural invariants, and CybORG parity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    MAX_SERVER_HOSTS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
    SERVICE_IDS,
    SUBNET_IDS,
    SUBNET_NAMES,
)
from jaxborg.topology import (
    _ROUTER_LINKS,
    build_topology,
)


@pytest.fixture
def const():
    return build_topology(jax.random.PRNGKey(42), num_steps=100)


class TestJITCompatibility:
    def test_jit_compatible(self):
        jitted = jax.jit(build_topology, static_argnums=(1,))
        const = jitted(jax.random.PRNGKey(0), 500)
        assert int(const.num_hosts) >= 41

    def test_deterministic(self):
        key = jax.random.PRNGKey(123)
        c1 = build_topology(key, num_steps=100)
        c2 = build_topology(key, num_steps=100)
        np.testing.assert_array_equal(np.array(c1.host_active), np.array(c2.host_active))
        np.testing.assert_array_equal(np.array(c1.host_subnet), np.array(c2.host_subnet))
        np.testing.assert_array_equal(np.array(c1.red_start_hosts), np.array(c2.red_start_hosts))
        assert int(c1.num_hosts) == int(c2.num_hosts)


class TestHostCounts:
    def test_host_counts_in_range(self):
        for seed in range(20):
            c = build_topology(jax.random.PRNGKey(seed), num_steps=100)
            n = int(c.num_hosts)
            assert 41 <= n <= 137, f"seed={seed}: num_hosts={n}"

    def test_non_internet_subnets_have_5_to_17_hosts(self):
        for seed in range(10):
            c = build_topology(jax.random.PRNGKey(seed), num_steps=100)
            for sid, sname in enumerate(SUBNET_NAMES):
                if sname == "INTERNET":
                    continue
                count = int(jnp.sum(c.host_active & (c.host_subnet == sid)))
                assert 5 <= count <= 17, f"seed={seed} subnet {sname}: {count}"

    def test_internet_has_1_host(self):
        c = build_topology(jax.random.PRNGKey(0), num_steps=100)
        sid = SUBNET_IDS["INTERNET"]
        count = int(jnp.sum(c.host_active & (c.host_subnet == sid)))
        assert count == 1


class TestAlphabeticalOrdering:
    def test_host_types_ordered_within_subnet(self, const):
        for sid, sname in enumerate(SUBNET_NAMES):
            pattern = _dedup_type_sequence(const, sid)
            if sname == "INTERNET":
                assert pattern == ["internet"], f"{sname}: {pattern}"
            else:
                assert pattern == ["router", "server", "user"], f"{sname}: {pattern}"

    def test_ordering_consistent_across_seeds(self):
        for seed in range(10):
            c = build_topology(jax.random.PRNGKey(seed), num_steps=100)
            for sid, sname in enumerate(SUBNET_NAMES):
                pattern = _dedup_type_sequence(c, sid)
                if sname == "INTERNET":
                    assert pattern == ["internet"]
                else:
                    assert pattern == ["router", "server", "user"], f"seed={seed} {sname}"


class TestSubnetRouters:
    def test_every_subnet_has_router(self, const):
        for sid, sname in enumerate(SUBNET_NAMES):
            mask = const.host_active & (const.host_subnet == sid)
            if sname == "INTERNET":
                assert int(jnp.sum(mask)) == 1
            else:
                router_count = int(jnp.sum(mask & const.host_is_router))
                assert router_count == 1, f"{sname}: {router_count} routers"


class TestDataLinks:
    def test_regular_hosts_connect_to_router(self, const):
        n = int(const.num_hosts)
        for h in range(n):
            if bool(const.host_is_router[h]):
                continue
            sid = int(const.host_subnet[h])
            sname = SUBNET_NAMES[sid]
            if sname == "INTERNET":
                continue
            router_mask = const.host_active & const.host_is_router & (const.host_subnet == sid)
            router_idx = int(jnp.argmax(router_mask))
            assert bool(const.data_links[h, router_idx]), f"host {h} not linked to router {router_idx}"

    def test_router_links_match_topology(self, const):
        for src_name, neighbor_names in _ROUTER_LINKS.items():
            if src_name == "INTERNET":
                src_sid = SUBNET_IDS[src_name]
                src_mask = const.host_active & (const.host_subnet == src_sid)
                src_idx = int(jnp.argmax(src_mask))
            else:
                src_sid = SUBNET_IDS[src_name]
                src_mask = const.host_active & const.host_is_router & (const.host_subnet == src_sid)
                src_idx = int(jnp.argmax(src_mask))
            for dst_name in neighbor_names:
                if dst_name == "INTERNET":
                    dst_sid = SUBNET_IDS[dst_name]
                    dst_mask = const.host_active & (const.host_subnet == dst_sid)
                    dst_idx = int(jnp.argmax(dst_mask))
                else:
                    dst_sid = SUBNET_IDS[dst_name]
                    dst_mask = const.host_active & const.host_is_router & (const.host_subnet == dst_sid)
                    dst_idx = int(jnp.argmax(dst_mask))
                assert bool(const.data_links[src_idx, dst_idx]), (
                    f"{src_name}[{src_idx}] -> {dst_name}[{dst_idx}] missing"
                )


class TestServices:
    def test_servers_and_users_have_sshd(self, const):
        n = int(const.num_hosts)
        for h in range(n):
            if bool(const.host_is_server[h]) or bool(const.host_is_user[h]):
                assert bool(const.initial_services[h, SERVICE_IDS["SSHD"]])

    def test_operational_hosts_have_otservice(self, const):
        n = int(const.num_hosts)
        op_sids = {SUBNET_IDS["OPERATIONAL_ZONE_A"], SUBNET_IDS["OPERATIONAL_ZONE_B"]}
        for h in range(n):
            sid = int(const.host_subnet[h])
            if sid in op_sids and (bool(const.host_is_server[h]) or bool(const.host_is_user[h])):
                assert bool(const.initial_services[h, SERVICE_IDS["OTSERVICE"]])

    def test_addon_services_subset(self, const):
        addon_ids = {SERVICE_IDS["APACHE2"], SERVICE_IDS["MYSQLD"], SERVICE_IDS["SMTP"]}
        n = int(const.num_hosts)
        for h in range(n):
            if bool(const.host_is_router[h]):
                assert not jnp.any(const.initial_services[h])
                continue
            for svc_id in range(len(SERVICE_IDS)):
                if svc_id not in addon_ids and svc_id != SERVICE_IDS["SSHD"] and svc_id != SERVICE_IDS["OTSERVICE"]:
                    assert not bool(const.initial_services[h, svc_id])


class TestRedStartHosts:
    def test_red_start_hosts_valid(self, const):
        for r in range(NUM_RED_AGENTS):
            h = int(const.red_start_hosts[r])
            assert bool(const.host_active[h]), f"red {r}: start host {h} not active"
            assert not bool(const.host_is_router[h]), f"red {r}: start host {h} is router"
            sname = SUBNET_NAMES[int(const.host_subnet[h])]
            assert sname != "INTERNET", f"red {r}: start host in INTERNET"
            assert bool(const.red_agent_subnets[r, int(const.host_subnet[h])]), (
                f"red {r}: start host subnet {sname} not in allowed subnets"
            )


class TestObsHostMap:
    def test_obs_host_map_valid(self, const):
        for sid in range(NUM_SUBNETS):
            for slot in range(OBS_HOSTS_PER_SUBNET):
                h = int(const.obs_host_map[sid, slot])
                if h == GLOBAL_MAX_HOSTS:
                    continue
                assert bool(const.host_active[h]), f"obs[{sid},{slot}]={h} not active"
                assert int(const.host_subnet[h]) == sid, f"obs[{sid},{slot}]={h} wrong subnet"
                if slot < MAX_SERVER_HOSTS:
                    assert bool(const.host_is_server[h]), f"obs[{sid},{slot}]={h} expected server"
                else:
                    assert bool(const.host_is_user[h]), f"obs[{sid},{slot}]={h} expected user"


class TestTopologyVariation:
    def test_topology_varies_across_keys(self):
        counts = set()
        for seed in range(20):
            c = build_topology(jax.random.PRNGKey(seed), num_steps=100)
            counts.add(int(c.num_hosts))
        assert len(counts) > 1


class TestCybORGStructuralParity:
    @pytest.fixture
    def cyborg_env(self):
        from CybORG import CybORG
        from CybORG.Agents import SleepAgent
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_cyborg_const_alphabetical_ordering(self, cyborg_env):
        from jaxborg.topology import CYBORG_SUFFIX_TO_ID, build_const_from_cyborg

        state = cyborg_env.environment_controller.state
        sorted_hostnames = sorted(state.hosts.keys())
        const = build_const_from_cyborg(cyborg_env)

        assert int(const.num_hosts) == len(sorted_hostnames)
        for idx, hostname in enumerate(sorted_hostnames):
            subnet_name_cyborg = state.hostname_subnet_map[hostname]
            expected_sid = CYBORG_SUFFIX_TO_ID[subnet_name_cyborg]
            assert int(const.host_subnet[idx]) == expected_sid

    def test_both_builders_same_type_ordering(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        cyborg_const = build_const_from_cyborg(cyborg_env)
        pure_const = build_topology(jax.random.PRNGKey(0), num_steps=100)

        for sid, sname in enumerate(SUBNET_NAMES):
            cyborg_pattern = _dedup_type_sequence(cyborg_const, sid)
            pure_pattern = _dedup_type_sequence(pure_const, sid)
            assert cyborg_pattern == pure_pattern, f"Subnet {sname}: cyborg={cyborg_pattern}, pure={pure_pattern}"

    def test_blue_red_agent_subnets_match(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        cyborg_const = build_const_from_cyborg(cyborg_env)
        pure_const = build_topology(jax.random.PRNGKey(0), num_steps=100)

        np.testing.assert_array_equal(
            np.array(cyborg_const.blue_agent_subnets), np.array(pure_const.blue_agent_subnets)
        )
        np.testing.assert_array_equal(np.array(cyborg_const.red_agent_subnets), np.array(pure_const.red_agent_subnets))

    def test_services_from_same_set(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        cyborg_const = build_const_from_cyborg(cyborg_env)
        for h in range(int(cyborg_const.num_hosts)):
            if bool(cyborg_const.host_is_server[h]) or bool(cyborg_const.host_is_user[h]):
                assert bool(cyborg_const.initial_services[h, SERVICE_IDS["SSHD"]])

    def test_structural_parity_multiple_seeds(self):
        from CybORG import CybORG
        from CybORG.Agents import SleepAgent
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.topology import build_const_from_cyborg

        for seed in [0, 1, 2, 3, 4]:
            sg = EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=SleepAgent,
                red_agent_class=SleepAgent,
            )
            env = CybORG(scenario_generator=sg, seed=seed)
            cyborg_const = build_const_from_cyborg(env)

            for sid, sname in enumerate(SUBNET_NAMES):
                mask = cyborg_const.host_active & (cyborg_const.host_subnet == sid)
                count = int(jnp.sum(mask))
                assert count >= 1, f"seed={seed} subnet {sname} has no hosts"
                if sname != "INTERNET":
                    router_count = int(jnp.sum(mask & cyborg_const.host_is_router))
                    assert router_count == 1, f"seed={seed} subnet {sname}: {router_count} routers"


class TestAutoResetIntegration:
    def test_auto_reset_produces_new_topology(self):
        from jaxborg.env import CC4Env

        env = CC4Env(num_steps=1)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        original_num_hosts = int(state.const.num_hosts)

        topologies_seen = {original_num_hosts}
        for i in range(5):
            key, k_step = jax.random.split(key)
            actions = {agent: jnp.int32(0) for agent in env.agents}
            obs, state, rewards, dones, infos = env.step(k_step, state, actions)
            topologies_seen.add(int(state.const.num_hosts))

        assert len(topologies_seen) > 1, "Expected different topologies after auto-reset"


def _dedup_type_sequence(const, sid: int) -> list[str]:
    result = []
    n = int(const.num_hosts)
    for h in range(n):
        if int(const.host_subnet[h]) != sid:
            continue
        if bool(const.host_is_router[h]):
            t = "router"
        elif bool(const.host_is_server[h]):
            t = "server"
        elif bool(const.host_is_user[h]):
            t = "user"
        else:
            t = "internet"
        if not result or result[-1] != t:
            result.append(t)
    return result
