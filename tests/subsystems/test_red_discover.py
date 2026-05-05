import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Simulator.Actions import DiscoverRemoteSystems
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTrafficZone
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_red_action
from jaxborg.actions.encoding import (
    RED_DISCOVER_START,
    RED_SLEEP,
    decode_red_action,
    encode_red_action,
)
from jaxborg.actions.red_common import has_any_session
from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
    SUBNET_IDS,
)
from jaxborg.scenarios.cc4.topology import CYBORG_SUFFIX_TO_ID
from jaxborg.state import create_initial_state

_jit_apply_red = jax.jit(apply_red_action, static_argnums=(2,))


@pytest.fixture
def jax_state(jax_const):
    state = create_initial_state()
    start_host = int(jax_const.red_start_hosts[0])
    red_sessions = state.red_sessions.at[0, start_host].set(True)
    red_scan_anchor_host = state.red_scan_anchor_host.at[0].set(start_host)
    return state.replace(
        red_sessions=red_sessions,
        red_scan_anchor_host=red_scan_anchor_host,
    )


class TestEncoding:
    def test_sleep_encodes_to_zero(self):
        assert encode_red_action("Sleep", 0, 0) == RED_SLEEP

    def test_discover_encodes_per_subnet(self):
        for sid in range(NUM_SUBNETS):
            code = encode_red_action("DiscoverRemoteSystems", sid, 0)
            assert code == RED_DISCOVER_START + sid

    def test_decode_sleep(self, jax_const):
        action_type, target_subnet, target_host = decode_red_action(RED_SLEEP, 0, jax_const)
        assert int(action_type) == 0
        assert int(target_subnet) == -1
        assert int(target_host) == -1

    def test_decode_discover_roundtrip(self, jax_const):
        for sid in range(NUM_SUBNETS):
            code = encode_red_action("DiscoverRemoteSystems", sid, 0)
            action_type, target_subnet, target_host = decode_red_action(code, 0, jax_const)
            assert int(action_type) == 1
            assert int(target_subnet) == sid
            assert int(target_host) == -1


class TestHasAnySession:
    def test_has_session(self, jax_const, jax_state):
        session_hosts = jax_state.red_sessions[0]
        assert bool(has_any_session(session_hosts, jax_const))

    def test_no_session(self, jax_const):
        empty_sessions = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
        assert not bool(has_any_session(empty_sessions, jax_const))


class TestApplyDiscover:
    def test_sleep_no_state_change(self, jax_const, jax_state):
        new_state = _jit_apply_red(jax_state, jax_const, 0, RED_SLEEP, jax.random.PRNGKey(0))
        np.testing.assert_array_equal(
            np.array(new_state.red_discovered_hosts),
            np.array(jax_state.red_discovered_hosts),
        )

    def test_discover_own_subnet(self, jax_const, jax_state):
        start_host = int(jax_const.red_start_hosts[0])
        start_subnet = int(jax_const.host_subnet[start_host])
        action_idx = encode_red_action("DiscoverRemoteSystems", start_subnet, 0)
        new_state = _jit_apply_red(jax_state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        discovered = np.array(new_state.red_discovered_hosts[0])
        for h in range(int(jax_const.num_hosts)):
            if (
                jax_const.host_active[h]
                and int(jax_const.host_subnet[h]) == start_subnet
                and jax_const.host_respond_to_ping[h]
            ):
                assert discovered[h], f"Host {h} should be discovered"

    def test_discover_does_not_find_routers(self, jax_const, jax_state):
        start_host = int(jax_const.red_start_hosts[0])
        start_subnet = int(jax_const.host_subnet[start_host])
        action_idx = encode_red_action("DiscoverRemoteSystems", start_subnet, 0)
        new_state = _jit_apply_red(jax_state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        discovered = np.array(new_state.red_discovered_hosts[0])
        for h in range(int(jax_const.num_hosts)):
            if jax_const.host_is_router[h]:
                assert not discovered[h], f"Router host {h} should not be discovered"

    def test_discover_adjacent_subnet(self, jax_const, jax_state):
        start_host = int(jax_const.red_start_hosts[0])
        start_subnet = int(jax_const.host_subnet[start_host])
        adj = np.array(jax_const.subnet_adjacency)

        reachable = [sid for sid in range(NUM_SUBNETS) if adj[start_subnet, sid] and sid != start_subnet]
        assert len(reachable) > 0, "Need at least one adjacent subnet for test"

        target_subnet = reachable[0]
        action_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
        new_state = _jit_apply_red(jax_state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        discovered = np.array(new_state.red_discovered_hosts[0])
        for h in range(int(jax_const.num_hosts)):
            if (
                jax_const.host_active[h]
                and int(jax_const.host_subnet[h]) == target_subnet
                and jax_const.host_respond_to_ping[h]
            ):
                assert discovered[h], f"Host {h} in adjacent subnet should be discovered"

    def test_discover_without_session_no_change(self, jax_const):
        state = create_initial_state()
        action_idx = encode_red_action("DiscoverRemoteSystems", SUBNET_IDS["CONTRACTOR_NETWORK"], 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        np.testing.assert_array_equal(
            np.array(new_state.red_discovered_hosts),
            np.array(state.red_discovered_hosts),
        )

    def test_discover_idempotent(self, jax_const, jax_state):
        start_host = int(jax_const.red_start_hosts[0])
        start_subnet = int(jax_const.host_subnet[start_host])
        action_idx = encode_red_action("DiscoverRemoteSystems", start_subnet, 0)
        state1 = _jit_apply_red(jax_state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        state2 = _jit_apply_red(state1, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        np.testing.assert_array_equal(
            np.array(state1.red_discovered_hosts),
            np.array(state2.red_discovered_hosts),
        )

    def test_discover_does_not_affect_other_agents(self, jax_const, jax_state):
        start_host = int(jax_const.red_start_hosts[0])
        start_subnet = int(jax_const.host_subnet[start_host])
        action_idx = encode_red_action("DiscoverRemoteSystems", start_subnet, 0)
        new_state = _jit_apply_red(jax_state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        for agent in range(1, NUM_RED_AGENTS):
            np.testing.assert_array_equal(
                np.array(new_state.red_discovered_hosts[agent]),
                np.array(jax_state.red_discovered_hosts[agent]),
            )

    def test_jit_compatible(self, jax_const, jax_state):
        start_host = int(jax_const.red_start_hosts[0])
        start_subnet = int(jax_const.host_subnet[start_host])
        action_idx = encode_red_action("DiscoverRemoteSystems", start_subnet, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(jax_state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        discovered = np.array(new_state.red_discovered_hosts[0])
        assert np.any(discovered)


class TestDifferentialWithCybORG:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    @pytest.fixture
    def cyborg_and_jax(self, cyborg_env):
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        state = create_initial_state()
        start_host = int(const.red_start_hosts[0])
        red_sessions = state.red_sessions.at[0, start_host].set(True)
        red_scan_anchor_host = state.red_scan_anchor_host.at[0].set(start_host)
        state = state.replace(
            red_sessions=red_sessions,
            red_scan_anchor_host=red_scan_anchor_host,
        )
        return cyborg_env, const, state

    def test_discover_contractor_network(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state

        subnet_name = "contractor_network_subnet"
        subnet_cidr = cyborg_state.subnet_name_to_cidr[subnet_name]
        sid = CYBORG_SUFFIX_TO_ID[subnet_name]

        action = DiscoverRemoteSystems(subnet=subnet_cidr, session=0, agent="red_agent_0")
        action.duration = 1
        results = cyborg_env.step(agent="red_agent_0", action=action)
        obs = results.observation

        action_idx = encode_red_action("DiscoverRemoteSystems", sid, 0)
        jax_new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))

        cyborg_discovered_ips = set()
        for key, val in obs.items():
            if key in ("success", "action", "message"):
                continue
            if isinstance(val, dict) and "Interface" in val:
                for iface in val["Interface"]:
                    if "ip_address" in iface:
                        cyborg_discovered_ips.add(str(iface["ip_address"]))

        sorted_hosts = sorted(cyborg_state.hosts.keys())
        jax_discovered = np.array(jax_new_state.red_discovered_hosts[0])

        jax_discovered_hostnames = {sorted_hosts[h] for h in range(int(const.num_hosts)) if jax_discovered[h]}

        cyborg_discovered_hostnames = set()
        for ip_str in cyborg_discovered_ips:
            from ipaddress import IPv4Address

            ip = IPv4Address(ip_str)
            if ip in cyborg_state.ip_addresses:
                cyborg_discovered_hostnames.add(cyborg_state.ip_addresses[ip])

        assert jax_discovered_hostnames == cyborg_discovered_hostnames, (
            f"JAX: {jax_discovered_hostnames}, CybORG: {cyborg_discovered_hostnames}"
        )

    def test_discover_restricted_zone_a(self, cyborg_and_jax):
        """Compare JAX discover against CybORG's direct Pingsweep.execute()."""
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state

        subnet_name = "restricted_zone_a_subnet"
        subnet_cidr = cyborg_state.subnet_name_to_cidr[subnet_name]
        sid = CYBORG_SUFFIX_TO_ID[subnet_name]

        from CybORG.Simulator.Actions.ConcreteActions.Pingsweep import Pingsweep

        ps = Pingsweep(session=0, agent="red_agent_0", subnet=subnet_cidr)
        obs = ps.execute(cyborg_state)

        cyborg_discovered_hostnames = set()
        for key in obs.data:
            if key == "success":
                continue
            from ipaddress import IPv4Address

            ip = IPv4Address(key)
            if ip in cyborg_state.ip_addresses:
                cyborg_discovered_hostnames.add(cyborg_state.ip_addresses[ip])

        action_idx = encode_red_action("DiscoverRemoteSystems", sid, 0)
        jax_new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))

        sorted_hosts = sorted(cyborg_state.hosts.keys())
        jax_discovered = np.array(jax_new_state.red_discovered_hosts[0])
        jax_discovered_hostnames = {sorted_hosts[h] for h in range(int(const.num_hosts)) if jax_discovered[h]}

        assert jax_discovered_hostnames == cyborg_discovered_hostnames, (
            f"JAX: {jax_discovered_hostnames}, CybORG: {cyborg_discovered_hostnames}"
        )

    def test_discover_with_missing_primary_session_noops_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state

        start_host = int(const.red_start_hosts[0])
        start_subnet = int(const.host_subnet[start_host])
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        primary = cyborg_state.sessions["red_agent_0"].pop(0)
        primary.ident = 1
        cyborg_state.sessions["red_agent_0"][1] = primary

        subnet_name = f"{sorted_hosts[start_host].split('_subnet')[0]}_subnet"
        subnet_cidr = cyborg_state.subnet_name_to_cidr[subnet_name]
        action = DiscoverRemoteSystems(subnet=subnet_cidr, session=0, agent="red_agent_0")
        obs = action.execute(cyborg_state)

        cyborg_new = set()
        for key, val in obs.data.items():
            if key in ("success", "action", "message"):
                continue
            if isinstance(val, dict) and "Interface" in val:
                for iface in val["Interface"]:
                    ip_addr = iface.get("ip_address")
                    if ip_addr in cyborg_state.ip_addresses:
                        cyborg_new.add(cyborg_state.ip_addresses[ip_addr])
        assert str(obs.success).upper() == "FALSE"
        assert not cyborg_new

        jax_state = state.replace(
            red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(-1),
        )
        action_idx = encode_red_action("DiscoverRemoteSystems", start_subnet, 0)
        jax_after = _jit_apply_red(jax_state, const, 0, action_idx, jax.random.PRNGKey(0))

        before = set(np.where(np.array(jax_state.red_discovered_hosts[0]))[0].tolist())
        after = set(np.where(np.array(jax_after.red_discovered_hosts[0]))[0].tolist())
        assert after == before

    def test_discover_ignores_blocked_subnet_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        start_host = int(const.red_start_hosts[0])
        start_hostname = sorted_hosts[start_host]
        start_subnet = int(const.host_subnet[start_host])
        start_subnet_name = cyborg_state.hostname_subnet_map[start_hostname].value

        target_subnet_name = None
        target_subnet = None
        for subnet_name, subnet_cidr in sorted(cyborg_state.subnet_name_to_cidr.items()):
            sid = CYBORG_SUFFIX_TO_ID[subnet_name]
            if sid == start_subnet or subnet_name.startswith("operational_zone_"):
                continue
            has_pingable_host = any(
                bool(const.host_active[h])
                and int(const.host_subnet[h]) == sid
                and bool(const.host_respond_to_ping[h])
                and not bool(const.host_is_router[h])
                for h in range(int(const.num_hosts))
            )
            if has_pingable_host:
                target_subnet_name = subnet_name
                target_subnet = sid
                target_cidr = subnet_cidr
                break
        if target_subnet_name is None or target_subnet is None:
            pytest.fail("Need a non-operational target subnet with a pingable host")

        block_obs = BlockTrafficZone(
            session=0,
            agent="blue_agent_0",
            from_subnet=start_subnet_name,
            to_subnet=target_subnet_name,
        ).execute(cyborg_state)
        assert str(block_obs.success).upper() == "TRUE"

        cyborg_obs = DiscoverRemoteSystems(subnet=target_cidr, session=0, agent="red_agent_0").execute(cyborg_state)
        assert str(cyborg_obs.success).upper() == "TRUE"

        blocked = state.blocked_zones.at[target_subnet, start_subnet].set(True)
        jax_state = state.replace(blocked_zones=blocked)

        action_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
        jax_after = _jit_apply_red(jax_state, const, 0, action_idx, jax.random.PRNGKey(0))

        discovered = np.array(jax_after.red_discovered_hosts[0])
        assert np.any(
            discovered & (np.array(const.host_subnet) == target_subnet) & np.array(const.host_respond_to_ping)
        )
