import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Shared.Session import RedAbstractSession, Session
from CybORG.Simulator.Actions import DiscoverRemoteSystems, Restore
from CybORG.Simulator.Actions.AbstractActions.DiscoverNetworkServices import (
    AggressiveServiceDiscovery,
    StealthServiceDiscovery,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.duration import process_red_with_duration
from jaxborg.actions.encoding import (
    ACTION_TYPE_SCAN,
    RED_SCAN_START,
    decode_red_action,
    encode_blue_action,
    encode_red_action,
)
from jaxborg.actions.pending_source import (
    PENDING_SOURCE_KIND_BOUND_NONE,
    PENDING_SOURCE_KIND_HOST,
    PENDING_SOURCE_KIND_NONE,
    PENDING_SOURCE_KIND_SESSION_BINDING,
)
from jaxborg.actions.red_common import can_reach_subnet, select_new_primary_session_host
from jaxborg.actions.rng import rng_impls
from jaxborg.constants import (
    ACTIVITY_SCAN,
    GLOBAL_MAX_HOSTS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
)
from jaxborg.scenarios.cc4.topology import CYBORG_SUFFIX_TO_ID
from jaxborg.state import create_initial_state
from tests.differential.parity_rng_replay import RNGTape

_jit_apply_red = jax.jit(apply_red_action, static_argnums=(2,))
_jit_apply_blue = jax.jit(apply_blue_action, static_argnums=(2,))


@pytest.fixture(scope="module")
def jax_state_with_discovered(jax_const):
    state = create_initial_state()
    start_host = int(jax_const.red_start_hosts[0])
    state = state.replace(
        red_sessions=state.red_sessions.at[0, start_host].set(True),
        red_session_is_abstract=state.red_session_is_abstract.at[0, start_host].set(True),
        red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(start_host),
    )

    start_subnet = int(jax_const.host_subnet[start_host])
    discover_idx = encode_red_action("DiscoverRemoteSystems", start_subnet, 0)
    state = _jit_apply_red(state, jax_const, 0, discover_idx, jax.random.PRNGKey(0))
    state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
    return state


def _first_discovered_non_router(jax_const, state, agent_id=0):
    discovered = np.array(state.red_discovered_hosts[agent_id])
    for h in range(int(jax_const.num_hosts)):
        if discovered[h] and not jax_const.host_is_router[h]:
            return h
    return None


class TestScanEncoding:
    def test_scan_encodes_per_host(self):
        for h in range(10):
            code = encode_red_action("DiscoverNetworkServices", h, 0)
            assert code == RED_SCAN_START + h

    def test_decode_scan_roundtrip(self, jax_const):
        for h in [0, 5, 50, GLOBAL_MAX_HOSTS - 1]:
            code = encode_red_action("DiscoverNetworkServices", h, 0)
            action_type, target_subnet, target_host = decode_red_action(code, 0, jax_const)
            assert int(action_type) == ACTION_TYPE_SCAN
            assert int(target_subnet) == -1
            assert int(target_host) == h

    def test_scan_range_does_not_overlap_discover(self):
        from jaxborg.actions.encoding import RED_DISCOVER_END

        assert RED_SCAN_START == RED_DISCOVER_END


class TestCanReachSubnet:
    def test_can_reach_own_subnet(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        start_host = int(jax_const.red_start_hosts[0])
        start_subnet = int(jax_const.host_subnet[start_host])
        assert bool(can_reach_subnet(state, jax_const, 0, start_subnet))

    def test_cannot_reach_when_no_session(self, jax_const):
        state = create_initial_state()
        assert not bool(can_reach_subnet(state, jax_const, 0, 0))

    def test_blocked_zone_prevents_reach(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        start_host = int(jax_const.red_start_hosts[0])
        start_subnet = int(jax_const.host_subnet[start_host])

        target_subnet = (start_subnet + 1) % NUM_SUBNETS
        blocked = state.blocked_zones.at[target_subnet, start_subnet].set(True)
        blocked_state = state.replace(blocked_zones=blocked)

        can_reach_unblocked = bool(can_reach_subnet(state, jax_const, 0, target_subnet))
        can_reach_blocked = bool(can_reach_subnet(blocked_state, jax_const, 0, target_subnet))

        if can_reach_unblocked:
            assert not can_reach_blocked


class TestApplyScan:
    def test_scan_discovered_host_marks_scanned(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert bool(new_state.red_scanned_hosts[0, target])

    def test_scan_sets_activity(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert int(new_state.red_activity_this_step[target]) == ACTIVITY_SCAN

    def test_scan_only_affects_target(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        for h in range(int(jax_const.num_hosts)):
            if h != target:
                assert not bool(new_state.red_scanned_hosts[0, h])

    def test_scan_undiscovered_host_no_change(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        discovered = np.array(state.red_discovered_hosts[0])
        undiscovered = None
        for h in range(int(jax_const.num_hosts)):
            if jax_const.host_active[h] and not discovered[h]:
                undiscovered = h
                break
        if undiscovered is None:
            pytest.fail("All hosts discovered")

        action_idx = encode_red_action("DiscoverNetworkServices", undiscovered, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert not bool(new_state.red_scanned_hosts[0, undiscovered])

    def test_scan_without_session_no_change(self, jax_const):
        state = create_initial_state()
        discovered = state.red_discovered_hosts.at[0, 5].set(True)
        state = state.replace(red_discovered_hosts=discovered)

        action_idx = encode_red_action("DiscoverNetworkServices", 5, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert not bool(new_state.red_scanned_hosts[0, 5])

    def test_scan_idempotent(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        state1 = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        state2 = _jit_apply_red(state1, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        np.testing.assert_array_equal(
            np.array(state1.red_scanned_hosts),
            np.array(state2.red_scanned_hosts),
        )

    def test_scan_does_not_affect_other_agents(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        for agent in range(1, NUM_RED_AGENTS):
            np.testing.assert_array_equal(
                np.array(new_state.red_scanned_hosts[agent]),
                np.array(state.red_scanned_hosts[agent]),
            )

    def test_scan_does_not_change_discovered(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        np.testing.assert_array_equal(
            np.array(new_state.red_discovered_hosts),
            np.array(state.red_discovered_hosts),
        )

    def test_scan_blocked_zone_no_change(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        start_host = int(jax_const.red_start_hosts[0])
        int(jax_const.host_subnet[start_host])
        int(jax_const.host_subnet[target])

        blocked = jnp.ones((NUM_SUBNETS, NUM_SUBNETS), dtype=jnp.bool_)
        state_blocked = state.replace(blocked_zones=blocked)

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = _jit_apply_red(state_blocked, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert not bool(new_state.red_scanned_hosts[0, target])

    def test_jit_compatible(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert bool(new_state.red_scanned_hosts[0, target])


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
        state = state.replace(
            red_sessions=state.red_sessions.at[0, start_host].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[0, start_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(start_host),
        )
        return cyborg_env, const, state

    def test_scan_host_services_detected(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state

        subnet_name = "contractor_network_subnet"
        subnet_cidr = cyborg_state.subnet_name_to_cidr[subnet_name]
        sid = CYBORG_SUFFIX_TO_ID[subnet_name]

        discover_action = DiscoverRemoteSystems(subnet=subnet_cidr, session=0, agent="red_agent_0")
        discover_action.duration = 1
        cyborg_env.step(agent="red_agent_0", action=discover_action)

        discover_idx = encode_red_action("DiscoverRemoteSystems", sid, 0)
        state = _jit_apply_red(state, const, 0, discover_idx, jax.random.PRNGKey(0))

        sorted_hosts = sorted(cyborg_state.hosts.keys())
        discovered_jax = np.array(state.red_discovered_hosts[0])
        discovered_hosts = [h for h in range(int(const.num_hosts)) if discovered_jax[h] and not const.host_is_router[h]]
        assert len(discovered_hosts) > 0

        target_h = discovered_hosts[0]
        target_hostname = sorted_hosts[target_h]
        target_ip = None
        for ip, hostname in cyborg_state.ip_addresses.items():
            if hostname == target_hostname:
                target_ip = ip
                break
        assert target_ip is not None

        scan_action = AggressiveServiceDiscovery(session=0, agent="red_agent_0", ip_address=target_ip)
        scan_action.duration = 1
        cyborg_env.step(agent="red_agent_0", action=scan_action)

        scan_idx = encode_red_action("DiscoverNetworkServices", target_h, 0)
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
        new_state = _jit_apply_red(state, const, 0, scan_idx, jax.random.PRNGKey(0))

        assert bool(new_state.red_scanned_hosts[0, target_h]), (
            f"JAX should mark host {target_h} ({target_hostname}) as scanned"
        )

    def test_restore_of_unrelated_abstract_session_keeps_scan_memory_matches_cyborg(self):
        """Regression: scan ownership should not drift to lowest abstract host index.

        If scan memory belongs to one abstract session and a different abstract
        session is restored, CybORG keeps the scanned host. JAX must do the same.
        """
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=5)
        cyborg_env.reset()

        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)
        cy_state = cyborg_env.environment_controller.state
        controller = cyborg_env.environment_controller

        red_agent_id = 3
        red_agent_name = f"red_agent_{red_agent_id}"

        choice = None
        for subnet_id in range(NUM_SUBNETS):
            if not bool(const.red_agent_subnets[red_agent_id, subnet_id]):
                continue
            subnet_hosts = [
                h
                for h in range(int(const.num_hosts))
                if bool(const.host_active[h])
                and not bool(const.host_is_router[h])
                and int(const.host_subnet[h]) == subnet_id
            ]
            if len(subnet_hosts) < 4:
                continue
            subnet_hosts = sorted(subnet_hosts)
            removable_hosts = [h for h in subnet_hosts if any(bool(const.blue_agent_hosts[b, h]) for b in range(5))]
            if not removable_hosts:
                continue
            low_host = removable_hosts[0]
            high_host = subnet_hosts[-1] if subnet_hosts[-1] != low_host else subnet_hosts[-2]
            remaining = [h for h in subnet_hosts if h not in {low_host, high_host}]
            if len(remaining) < 2:
                continue
            anchor_host = remaining[0]
            target_host = remaining[1]
            choice = (subnet_id, low_host, high_host, anchor_host, target_host)
            break

        assert choice is not None, "Need a subnet with removable host and multiple red-reachable targets"
        subnet_id, low_host, high_host, anchor_host, target_host = choice

        cy_state.add_session(
            Session(
                ident=None,
                hostname=mappings.idx_to_hostname[anchor_host],
                username="user",
                agent=red_agent_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[high_host],
                username="user",
                agent=red_agent_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[low_host],
                username="user",
                agent=red_agent_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )

        red_sessions = cy_state.sessions[red_agent_name]
        anchor_sid = next(
            sid
            for sid, sess in red_sessions.items()
            if sess.hostname == mappings.idx_to_hostname[anchor_host] and type(sess).__name__ == "Session"
        )
        high_sid = next(
            sid
            for sid, sess in red_sessions.items()
            if sess.hostname == mappings.idx_to_hostname[high_host] and type(sess).__name__ == "RedAbstractSession"
        )
        low_sid = next(
            sid
            for sid, sess in red_sessions.items()
            if sess.hostname == mappings.idx_to_hostname[low_host] and type(sess).__name__ == "RedAbstractSession"
        )
        iface = controller.agent_interfaces[red_agent_name]
        for sid in (anchor_sid, high_sid, low_sid):
            iface.action_space.client_session[sid] = True
            iface.action_space.server_session[sid] = True

        seed_host = anchor_host
        seed_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[seed_host]]
        red_sessions[high_sid].addport(seed_ip, 22)

        subnet_name = next(name for name, sid in CYBORG_SUFFIX_TO_ID.items() if sid == subnet_id)
        subnet_cidr = cy_state.subnet_name_to_cidr[subnet_name]
        discover_action = DiscoverRemoteSystems(subnet=subnet_cidr, session=high_sid, agent=red_agent_name)
        discover_action.duration = 1
        cyborg_env.step(agent=red_agent_name, action=discover_action)

        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]
        scan_action = AggressiveServiceDiscovery(session=high_sid, agent=red_agent_name, ip_address=target_ip)
        scan_action.duration = 1
        cyborg_env.step(agent=red_agent_name, action=scan_action)

        def _cy_scanned_hosts():
            scanned = set()
            for sess in cy_state.sessions[red_agent_name].values():
                for ip in getattr(sess, "ports", {}).keys():
                    hostname = cy_state.ip_addresses.get(ip)
                    if hostname in mappings.hostname_to_idx:
                        scanned.add(mappings.hostname_to_idx[hostname])
            return scanned

        assert target_host in _cy_scanned_hosts(), "CybORG setup failed to scan target host"

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        for h in (anchor_host, high_host, low_host):
            state = state.replace(
                red_sessions=state.red_sessions.at[red_agent_id, h].set(True),
                red_session_count=state.red_session_count.at[red_agent_id, h].set(1),
                red_privilege=state.red_privilege.at[red_agent_id, h].set(1),
                red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, h].set(True),
            )
        state = state.replace(
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, anchor_host]
            .set(False)
            .at[red_agent_id, high_host]
            .set(True)
            .at[red_agent_id, low_host]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(high_host),
            red_scanned_hosts=state.red_scanned_hosts.at[red_agent_id, seed_host].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[red_agent_id, seed_host, high_host].set(True),
        )

        discover_idx = encode_red_action("DiscoverRemoteSystems", subnet_id, red_agent_id)
        state = _jit_apply_red(state, const, red_agent_id, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
        scan_idx = encode_red_action("AggressiveServiceDiscovery", target_host, red_agent_id)
        state = _jit_apply_red(state, const, red_agent_id, scan_idx, jax.random.PRNGKey(1))
        assert bool(state.red_scanned_hosts[red_agent_id, target_host]), "JAX setup failed to scan target host"

        blue_agent_id = next(b for b in range(5) if bool(const.blue_agent_hosts[b, low_host]))
        restore_action = Restore(
            session=0,
            agent=f"blue_agent_{blue_agent_id}",
            hostname=mappings.idx_to_hostname[low_host],
        )
        cyborg_env.step(agent=f"blue_agent_{blue_agent_id}", action=restore_action)

        blue_restore_idx = encode_blue_action("Restore", low_host, blue_agent_id, const=const)
        state = _jit_apply_blue(state, const, blue_agent_id, blue_restore_idx)

        cy_target_scanned = target_host in _cy_scanned_hosts()
        jax_target_scanned = bool(state.red_scanned_hosts[red_agent_id, target_host])
        assert cy_target_scanned, "CybORG should keep scan memory tied to the other abstract session"
        assert jax_target_scanned == cy_target_scanned

    def test_restore_source_host_keeps_target_scan_when_target_session_still_knows_ip_matches_cyborg(self):
        """Regression: rescans should not overwrite scan ownership away from a live target session."""
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=8)
        cyborg_env.reset()

        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)
        cy_state = cyborg_env.environment_controller.state
        controller = cyborg_env.environment_controller

        red_agent_id = 3
        red_agent_name = f"red_agent_{red_agent_id}"

        choice = None
        for subnet_id in range(NUM_SUBNETS):
            if not bool(const.red_agent_subnets[red_agent_id, subnet_id]):
                continue
            subnet_hosts = [
                h
                for h in range(int(const.num_hosts))
                if bool(const.host_active[h])
                and not bool(const.host_is_router[h])
                and int(const.host_subnet[h]) == subnet_id
            ]
            if len(subnet_hosts) < 2:
                continue
            restorable = [h for h in subnet_hosts if any(bool(const.blue_agent_hosts[b, h]) for b in range(5))]
            if len(restorable) < 2:
                continue
            restorable = sorted(restorable)
            target_host = restorable[0]
            source_host = restorable[-1]
            if source_host != target_host:
                choice = (target_host, source_host)
                break
        assert choice is not None, "Need two restorable hosts in a red-reachable subnet"
        target_host, source_host = choice

        pre_ids = set(cy_state.sessions[red_agent_name].keys())
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[source_host],
                username="user",
                agent=red_agent_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        source_sid = next(sid for sid in cy_state.sessions[red_agent_name] if sid not in pre_ids)

        pre_ids = set(cy_state.sessions[red_agent_name].keys())
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[target_host],
                username="user",
                agent=red_agent_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        target_sid = next(sid for sid in cy_state.sessions[red_agent_name] if sid not in pre_ids)

        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]
        source_sess = cy_state.sessions[red_agent_name][source_sid]
        target_sess = cy_state.sessions[red_agent_name][target_sid]
        source_sess.addport(target_ip, 22)
        target_sess.addport(target_ip, 22)

        iface = controller.agent_interfaces[red_agent_name]
        iface.action_space.client_session[source_sid] = True
        iface.action_space.server_session[source_sid] = True
        iface.action_space.client_session[target_sid] = True
        iface.action_space.server_session[target_sid] = True

        scan_action = AggressiveServiceDiscovery(session=source_sid, agent=red_agent_name, ip_address=target_ip)
        scan_action.duration = 1
        cyborg_env.step(agent=red_agent_name, action=scan_action)

        blue_agent_id = next(b for b in range(5) if bool(const.blue_agent_hosts[b, source_host]))
        restore_action = Restore(
            session=0,
            agent=f"blue_agent_{blue_agent_id}",
            hostname=mappings.idx_to_hostname[source_host],
        )
        cyborg_env.step(agent=f"blue_agent_{blue_agent_id}", action=restore_action)

        def _cy_scanned_hosts():
            scanned = set()
            for sess in cy_state.sessions[red_agent_name].values():
                for ip in getattr(sess, "ports", {}).keys():
                    hostname = cy_state.ip_addresses.get(ip)
                    if hostname in mappings.hostname_to_idx:
                        scanned.add(mappings.hostname_to_idx[hostname])
            return scanned

        assert target_host in _cy_scanned_hosts(), "CybORG should retain scan via surviving target session"

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, source_host]
            .set(True)
            .at[red_agent_id, target_host]
            .set(True),
            red_session_count=state.red_session_count.at[red_agent_id, source_host]
            .set(1)
            .at[red_agent_id, target_host]
            .set(1),
            red_privilege=state.red_privilege.at[red_agent_id, source_host].set(1).at[red_agent_id, target_host].set(1),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, source_host]
            .set(True)
            .at[red_agent_id, target_host]
            .set(True),
            red_scanned_hosts=state.red_scanned_hosts.at[red_agent_id, target_host].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[red_agent_id, target_host, target_host].set(
                True
            ),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(source_host),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, source_host]
            .set(True)
            .at[red_agent_id, target_host]
            .set(True),
        )

        scan_idx = encode_red_action("AggressiveServiceDiscovery", target_host, red_agent_id)
        state = _jit_apply_red(state, const, red_agent_id, scan_idx, jax.random.PRNGKey(0))
        restore_idx = encode_blue_action("Restore", source_host, blue_agent_id, const=const)
        state = _jit_apply_blue(state, const, blue_agent_id, restore_idx)

        assert bool(state.red_scanned_hosts[red_agent_id, target_host]) == (target_host in _cy_scanned_hosts())


class TestScanRequiresAbstractSession:
    """CybORG gates DiscoverNetworkServices on RedAbstractSession.

    Sessions from green phishing reassignment are plain Sessions that cannot scan.
    JAX must replicate this by tracking red_session_is_abstract.
    """

    def test_scan_fails_without_abstract_session(self):
        """Scan must fail when agent only has non-abstract sessions (from phishing)."""
        from jaxborg.scenarios.cc4.topology import build_topology

        const = build_topology(jax.random.PRNGKey(42), num_steps=500)
        state = create_initial_state()

        agent_id = 0
        start_host = int(const.red_start_hosts[agent_id])
        target_subnet = int(const.host_subnet[start_host])

        # Give agent a session but NOT an abstract one (simulating phishing reassignment)
        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, start_host].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, start_host].set(False),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(start_host),
        )

        # Discover hosts first
        discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, agent_id)
        state = _jit_apply_red(state, const, agent_id, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        target = _first_discovered_non_router(const, state, agent_id)
        assert target is not None, "Need at least one discovered host to scan"

        scan_idx = encode_red_action("DiscoverNetworkServices", target, agent_id)
        new_state = _jit_apply_red(state, const, agent_id, scan_idx, jax.random.PRNGKey(1))

        assert not bool(new_state.red_scanned_hosts[agent_id, target]), (
            "Scan must fail when agent has no abstract session (CybORG RedAbstractSession check)"
        )

    def test_scan_succeeds_with_abstract_session(self):
        """Scan succeeds when agent has an abstract session (from exploit)."""
        from jaxborg.scenarios.cc4.topology import build_topology

        const = build_topology(jax.random.PRNGKey(42), num_steps=500)
        state = create_initial_state()

        agent_id = 0
        start_host = int(const.red_start_hosts[agent_id])
        target_subnet = int(const.host_subnet[start_host])

        # Give agent an abstract session (from exploit)
        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, start_host].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, start_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(start_host),
        )

        # Discover hosts
        discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, agent_id)
        state = _jit_apply_red(state, const, agent_id, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        target = _first_discovered_non_router(const, state, agent_id)
        assert target is not None

        scan_idx = encode_red_action("DiscoverNetworkServices", target, agent_id)
        new_state = _jit_apply_red(state, const, agent_id, scan_idx, jax.random.PRNGKey(1))

        assert bool(new_state.red_scanned_hosts[agent_id, target]), (
            "Scan should succeed when agent has an abstract session"
        )

    def test_exploit_does_not_grant_abstract_session(self):
        """CybORG ExploitAction creates plain Session, not RedAbstractSession.

        If blue kills the abstract session, the agent should lose scan/exploit
        ability even if exploit-created sessions remain.
        """
        from jaxborg.actions.red_common import apply_exploit_success
        from jaxborg.scenarios.cc4.topology import build_topology

        const = build_topology(jax.random.PRNGKey(42), num_steps=500)
        state = create_initial_state()

        agent_id = 0
        start_host = int(const.red_start_hosts[agent_id])
        target_subnet = int(const.host_subnet[start_host])

        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, start_host].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, start_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(start_host),
        )

        discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, agent_id)
        state = _jit_apply_red(state, const, agent_id, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        discovered = np.array(state.red_discovered_hosts[agent_id])
        target = None
        for h in range(int(const.num_hosts)):
            if discovered[h] and not const.host_is_router[h] and h != start_host:
                target = h
                break
        assert target is not None

        new_state = apply_exploit_success(
            state,
            const,
            agent_id,
            target,
            success=jnp.bool_(True),
            key=jax.random.PRNGKey(0),
        )

        assert bool(new_state.red_sessions[agent_id, target]), "Exploit should create session on target"
        assert not bool(new_state.red_session_is_abstract[agent_id, target]), (
            "Exploit-created sessions must NOT be abstract (CybORG creates plain Session, not RedAbstractSession)"
        )

    def test_stale_abstract_flag_without_live_session_does_not_allow_scan_matches_cyborg(self):
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=42)
        cyborg_env.reset()
        cy_state = cyborg_env.environment_controller.state

        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)

        target = next(
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h]) and not bool(const.host_is_router[h]) and h != int(const.red_start_hosts[0])
        )
        assert target is not None
        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target]]

        # CybORG baseline: queued scan with missing session id fails.
        del cy_state.sessions["red_agent_0"][0]
        cy_action = AggressiveServiceDiscovery(session=0, agent="red_agent_0", ip_address=target_ip)
        cy_obs = cy_action.execute(cy_state)
        assert str(cy_obs.success).upper() != "TRUE"

        # JAX mirror: stale abstract bit without an active session must not pass scan preconditions.
        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        start_host = int(const.red_start_hosts[0])
        state = state.replace(
            red_sessions=state.red_sessions.at[0, start_host].set(False),
            red_session_is_abstract=state.red_session_is_abstract.at[0, start_host].set(True),
            red_discovered_hosts=state.red_discovered_hosts.at[0, target].set(True),
        )
        scan_idx = encode_red_action("AggressiveServiceDiscovery", target, 0)
        new_state = _jit_apply_red(state, const, 0, scan_idx, jax.random.PRNGKey(0))

        assert not bool(new_state.red_scanned_hosts[0, target])


class TestDeferredScanSessionBinding:
    def test_deferred_scan_fails_if_bound_session_removed_before_execute_matches_cyborg(self):
        """Deferred scan stays bound to the queued session id in CybORG."""
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=42)
        cyborg_env.reset()
        controller = cyborg_env.environment_controller
        cy_state = controller.state

        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)

        red_agent_id = 0
        red_agent_name = "red_agent_0"
        anchor_host = int(const.red_start_hosts[red_agent_id])

        same_subnet_hosts = [
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h])
            and not bool(const.host_is_router[h])
            and int(const.host_subnet[h]) == int(const.host_subnet[anchor_host])
            and h != anchor_host
        ]
        assert len(same_subnet_hosts) >= 2, "Need two extra hosts in the anchor subnet"
        alt_host = same_subnet_hosts[0]
        target_host = same_subnet_hosts[1]

        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[alt_host],
                username="user",
                agent=red_agent_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )

        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]
        deferred = StealthServiceDiscovery(session=0, agent=red_agent_name, ip_address=target_ip)
        controller.actions_in_progress[red_agent_name] = {"action": deferred, "remaining_ticks": 1}

        # Mechanism under test: remove queued session id 0 before execution tick.
        del cy_state.sessions[red_agent_name][0]
        controller.step(actions={}, skip_valid_action_check=True)

        cy_scanned = False
        for sess in cy_state.sessions.get(red_agent_name, {}).values():
            if target_ip in getattr(sess, "ports", {}):
                cy_scanned = True
                break

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, anchor_host]
            .set(True)
            .at[red_agent_id, alt_host]
            .set(True),
            red_session_count=state.red_session_count.at[red_agent_id, anchor_host]
            .set(1)
            .at[red_agent_id, alt_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, anchor_host]
            .set(True)
            .at[red_agent_id, alt_host]
            .set(True),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(anchor_host),
            red_pending_ticks=state.red_pending_ticks.at[red_agent_id].set(1),
            red_pending_action=state.red_pending_action.at[red_agent_id].set(
                encode_red_action("StealthServiceDiscovery", target_host, red_agent_id)
            ),
            red_pending_key=state.red_pending_key.at[red_agent_id].set(jnp.array([1, 2], dtype=jnp.uint32)),
            red_pending_source_kind=state.red_pending_source_kind.at[red_agent_id].set(PENDING_SOURCE_KIND_HOST),
            red_pending_source_host=state.red_pending_source_host.at[red_agent_id].set(anchor_host),
        )

        # Mirror CybORG precondition: bound source session (anchor/session 0) is gone.
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, anchor_host].set(False),
            red_session_count=state.red_session_count.at[red_agent_id, anchor_host].set(0),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, anchor_host].set(False),
        )

        key = jax.random.PRNGKey(7)
        new_state = process_red_with_duration(state, const, red_agent_id, RED_SCAN_START + target_host, key)
        jax_scanned = bool(new_state.red_scanned_hosts[red_agent_id, target_host])

        assert not cy_scanned, "CybORG should fail deferred scan when queued session id is removed"
        assert jax_scanned == cy_scanned, "JAX must match CybORG for deferred scan session binding"

    def test_scan_does_not_rebind_when_bound_source_is_missing_matches_cyborg(self):
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=42)
        cyborg_env.reset()
        cy_state = cyborg_env.environment_controller.state

        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)

        red_agent_id = 0
        red_agent_name = "red_agent_0"
        start_host = int(const.red_start_hosts[red_agent_id])
        same_subnet_hosts = [
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h])
            and not bool(const.host_is_router[h])
            and int(const.host_subnet[h]) == int(const.host_subnet[start_host])
            and h != start_host
        ]
        assert len(same_subnet_hosts) >= 2
        alt_host = same_subnet_hosts[0]
        target_host = same_subnet_hosts[1]
        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]

        start_hostname = mappings.idx_to_hostname[start_host]
        cy_state.sessions[red_agent_name][0] = Session(
            ident=0,
            hostname=start_hostname,
            username="user",
            agent=red_agent_name,
            parent=None,
            session_type="shell",
            pid=None,
        )
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[alt_host],
                username="user",
                agent=red_agent_name,
                parent=None,
                session_type="shell",
                pid=None,
            )
        )
        cy_action = AggressiveServiceDiscovery(session=0, agent=red_agent_name, ip_address=target_ip)
        cy_obs = cy_action.execute(cy_state)
        assert str(cy_obs.success).upper() != "TRUE"

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, start_host].set(True).at[red_agent_id, alt_host].set(True),
            red_session_count=state.red_session_count.at[red_agent_id, start_host]
            .set(1)
            .at[red_agent_id, alt_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, alt_host].set(True),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            # Keep explicit "no bound source" for this deferred-action regression.
            red_pending_source_kind=state.red_pending_source_kind.at[red_agent_id].set(PENDING_SOURCE_KIND_BOUND_NONE),
            red_pending_source_host=state.red_pending_source_host.at[red_agent_id].set(-1),
        )

        action_idx = encode_red_action("AggressiveServiceDiscovery", target_host, red_agent_id)
        new_state = process_red_with_duration(state, const, red_agent_id, action_idx, jax.random.PRNGKey(0))
        assert not bool(new_state.red_scanned_hosts[red_agent_id, target_host])

    def test_scan_honors_prebound_source_during_execution_matches_cyborg(self):
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=42)
        cyborg_env.reset()
        cy_state = cyborg_env.environment_controller.state

        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)

        choice = None
        for red_agent_id in range(NUM_RED_AGENTS):
            start_host = int(const.red_start_hosts[red_agent_id])
            subnet_id = int(const.host_subnet[start_host])
            if not bool(const.red_agent_subnets[red_agent_id, subnet_id]):
                continue
            subnet_hosts = [
                h
                for h in range(int(const.num_hosts))
                if bool(const.host_active[h])
                and not bool(const.host_is_router[h])
                and int(const.host_subnet[h]) == subnet_id
            ]
            restorable_hosts = [h for h in subnet_hosts if any(bool(const.blue_agent_hosts[b, h]) for b in range(5))]
            if len(restorable_hosts) < 2:
                continue
            target_candidates = [h for h in subnet_hosts if h not in {restorable_hosts[0], restorable_hosts[1]}]
            if not target_candidates:
                continue
            choice = (red_agent_id, restorable_hosts[0], restorable_hosts[1], target_candidates[0])
            break
        if choice is None:
            pytest.fail("Need a subnet with source/anchor/target hosts")
        red_agent_id, source_host, anchor_host, target_host = choice
        red_agent_name = f"red_agent_{red_agent_id}"

        source_hostname = mappings.idx_to_hostname[source_host]
        anchor_hostname = mappings.idx_to_hostname[anchor_host]
        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]

        cy_state.sessions[red_agent_name][0] = RedAbstractSession(
            ident=0,
            hostname=source_hostname,
            username="user",
            agent=red_agent_name,
            parent=None,
            session_type="shell",
            pid=None,
        )
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=anchor_hostname,
                username="user",
                agent=red_agent_name,
                parent=None,
                session_type="shell",
                pid=None,
            )
        )
        cy_state.sessions[red_agent_name][0].addport(target_ip, 22)

        cy_scan = AggressiveServiceDiscovery(session=0, agent=red_agent_name, ip_address=target_ip)
        cy_scan.duration = 1
        cyborg_env.step(agent=red_agent_name, action=cy_scan)

        blue_agent_id = next(b for b in range(5) if bool(const.blue_agent_hosts[b, anchor_host]))
        cy_restore = Restore(session=0, agent=f"blue_agent_{blue_agent_id}", hostname=anchor_hostname)
        cyborg_env.step(agent=f"blue_agent_{blue_agent_id}", action=cy_restore)

        cy_scanned = False
        for sess in cy_state.sessions.get(red_agent_name, {}).values():
            if target_ip in getattr(sess, "ports", {}):
                cy_scanned = True
                break
        assert cy_scanned

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, source_host]
            .set(True)
            .at[red_agent_id, anchor_host]
            .set(True),
            red_session_count=state.red_session_count.at[red_agent_id, source_host]
            .set(1)
            .at[red_agent_id, anchor_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, source_host]
            .set(True)
            .at[red_agent_id, anchor_host]
            .set(True),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(anchor_host),
            red_pending_source_kind=state.red_pending_source_kind.at[red_agent_id].set(PENDING_SOURCE_KIND_HOST),
            red_pending_source_host=state.red_pending_source_host.at[red_agent_id].set(source_host),
        )

        scan_idx = encode_red_action("AggressiveServiceDiscovery", target_host, red_agent_id)
        state = process_red_with_duration(state, const, red_agent_id, scan_idx, jax.random.PRNGKey(0))
        restore_idx = encode_blue_action("Restore", anchor_host, blue_agent_id, const=const)
        state = _jit_apply_blue(state, const, blue_agent_id, restore_idx)

        assert bool(state.red_scanned_hosts[red_agent_id, target_host]) == cy_scanned

    def test_unset_anchor_deterministic_scan_source_not_forced_to_start_host(self, jax_const):
        choice = None
        for red_agent_id in range(NUM_RED_AGENTS):
            allowed_hosts = [
                h
                for h in range(int(jax_const.num_hosts))
                if bool(jax_const.host_active[h])
                and not bool(jax_const.host_is_router[h])
                and bool(jax_const.red_agent_subnets[red_agent_id, int(jax_const.host_subnet[h])])
            ]
            if len(allowed_hosts) < 3:
                continue
            allowed_hosts = sorted(allowed_hosts)
            source_host = allowed_hosts[0]
            start_host = allowed_hosts[1]
            target_host = allowed_hosts[2]
            choice = (red_agent_id, source_host, start_host, target_host)
            break
        assert choice is not None
        red_agent_id, source_host, start_host, target_host = choice

        const = jax_const.replace(red_start_hosts=jax_const.red_start_hosts.at[red_agent_id].set(start_host))
        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, source_host]
            .set(True)
            .at[red_agent_id, start_host]
            .set(True),
            red_session_count=state.red_session_count.at[red_agent_id, source_host]
            .set(1)
            .at[red_agent_id, start_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, source_host]
            .set(True)
            .at[red_agent_id, start_host]
            .set(True),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(-1),
        )

        scan_idx = encode_red_action("AggressiveServiceDiscovery", target_host, red_agent_id)
        state = process_red_with_duration(state, const, red_agent_id, scan_idx, jax.random.PRNGKey(0))

        assert bool(state.red_scanned_hosts[red_agent_id, target_host])
        assert bool(state.red_scanned_source_hosts[red_agent_id, target_host, source_host])

    def test_deferred_scan_uses_current_session_zero_host_when_prebound_source_stale_matches_cyborg(self):
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=42)
        cyborg_env.reset()

        controller = cyborg_env.environment_controller
        cy_state = controller.state
        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)

        red_agent_id = 0
        red_agent_name = "red_agent_0"
        stale_source_host = int(const.red_start_hosts[red_agent_id])
        subnet_id = int(const.host_subnet[stale_source_host])
        subnet_hosts = [
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h])
            and not bool(const.host_is_router[h])
            and int(const.host_subnet[h]) == subnet_id
            and h != stale_source_host
        ]
        assert len(subnet_hosts) >= 2
        session_zero_host = subnet_hosts[0]
        target_host = subnet_hosts[1]

        session_zero_hostname = mappings.idx_to_hostname[session_zero_host]
        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]

        # Explicit CybORG mechanism setup: deferred action bound to session 0, but
        # session 0 is now hosted on a different machine than the stale source host.
        for host in cy_state.hosts.values():
            host.sessions[red_agent_name] = [sid for sid in host.sessions.get(red_agent_name, []) if sid != 0]
        cy_state.sessions[red_agent_name][0] = RedAbstractSession(
            ident=0,
            hostname=session_zero_hostname,
            username="user",
            agent=red_agent_name,
            parent=None,
            session_type="shell",
            pid=None,
        )
        cy_state.hosts[session_zero_hostname].sessions.setdefault(red_agent_name, []).append(0)
        controller.actions_in_progress[red_agent_name] = {
            "action": StealthServiceDiscovery(session=0, agent=red_agent_name, ip_address=target_ip),
            "remaining_ticks": 1,
        }
        controller.step(actions={}, skip_valid_action_check=True)
        cy_scanned = any(
            target_ip in getattr(sess, "ports", {}) for sess in cy_state.sessions.get(red_agent_name, {}).values()
        )
        assert cy_scanned

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, session_zero_host].set(True),
            red_session_count=state.red_session_count.at[red_agent_id, session_zero_host].set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, session_zero_host].set(True),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(session_zero_host),
            red_pending_ticks=state.red_pending_ticks.at[red_agent_id].set(1),
            red_pending_action=state.red_pending_action.at[red_agent_id].set(
                encode_red_action("StealthServiceDiscovery", target_host, red_agent_id)
            ),
            red_pending_source_kind=state.red_pending_source_kind.at[red_agent_id].set(
                PENDING_SOURCE_KIND_SESSION_BINDING
            ),
            red_pending_source_host=state.red_pending_source_host.at[red_agent_id].set(-1),
        )

        new_state = process_red_with_duration(
            state,
            const,
            red_agent_id,
            RED_SCAN_START + target_host,
            jax.random.PRNGKey(0),
        )
        assert bool(new_state.red_scanned_hosts[red_agent_id, target_host]) == cy_scanned

    def test_deferred_scan_does_not_rebind_from_unset_source_when_new_abstract_appears_matches_cyborg(self):
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=42)
        cyborg_env.reset()

        controller = cyborg_env.environment_controller
        cy_state = controller.state
        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)

        red_agent_id = 0
        red_agent_name = "red_agent_0"
        start_host = int(const.red_start_hosts[red_agent_id])
        subnet_id = int(const.host_subnet[start_host])
        subnet_hosts = [
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h])
            and not bool(const.host_is_router[h])
            and int(const.host_subnet[h]) == subnet_id
            and h != start_host
        ]
        assert len(subnet_hosts) >= 2
        abstract_host = subnet_hosts[0]
        target_host = subnet_hosts[1]

        start_hostname = mappings.idx_to_hostname[start_host]
        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]

        # CybORG setup: queued scan is bound to session 0, but session 0 is non-abstract.
        # Even if another abstract session now exists, deferred action should still fail.
        cy_state.sessions[red_agent_name][0] = Session(
            ident=0,
            hostname=start_hostname,
            username="user",
            agent=red_agent_name,
            parent=None,
            session_type="shell",
            pid=None,
        )
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[abstract_host],
                username="user",
                agent=red_agent_name,
                parent=None,
                session_type="shell",
                pid=None,
            )
        )
        controller.actions_in_progress[red_agent_name] = {
            "action": StealthServiceDiscovery(session=0, agent=red_agent_name, ip_address=target_ip),
            "remaining_ticks": 1,
        }
        controller.step(actions={}, skip_valid_action_check=True)
        cy_scanned = any(
            target_ip in getattr(sess, "ports", {}) for sess in cy_state.sessions.get(red_agent_name, {}).values()
        )
        assert not cy_scanned

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, start_host]
            .set(True)
            .at[red_agent_id, abstract_host]
            .set(True),
            red_session_count=state.red_session_count.at[red_agent_id, start_host]
            .set(1)
            .at[red_agent_id, abstract_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, start_host]
            .set(False)
            .at[red_agent_id, abstract_host]
            .set(True),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            red_pending_ticks=state.red_pending_ticks.at[red_agent_id].set(1),
            red_pending_action=state.red_pending_action.at[red_agent_id].set(
                encode_red_action("StealthServiceDiscovery", target_host, red_agent_id)
            ),
            red_pending_source_kind=state.red_pending_source_kind.at[red_agent_id].set(PENDING_SOURCE_KIND_NONE),
            red_pending_source_host=state.red_pending_source_host.at[red_agent_id].set(-1),
        )

        new_state = process_red_with_duration(
            state,
            const,
            red_agent_id,
            RED_SCAN_START + target_host,
            jax.random.PRNGKey(0),
        )
        assert bool(new_state.red_scanned_hosts[red_agent_id, target_host]) == cy_scanned

    def test_deferred_scan_uses_anchor_session_when_source_is_unset_matches_cyborg(self):
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=42)
        cyborg_env.reset()

        controller = cyborg_env.environment_controller
        cy_state = controller.state
        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)

        red_agent_id = 0
        red_agent_name = "red_agent_0"
        anchor_host = int(const.red_start_hosts[red_agent_id])
        subnet_id = int(const.host_subnet[anchor_host])
        target_host = next(
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h])
            and not bool(const.host_is_router[h])
            and int(const.host_subnet[h]) == subnet_id
            and h != anchor_host
        )
        anchor_hostname = mappings.idx_to_hostname[anchor_host]
        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]

        cy_state.sessions[red_agent_name][0] = RedAbstractSession(
            ident=0,
            hostname=anchor_hostname,
            username="user",
            agent=red_agent_name,
            parent=None,
            session_type="shell",
            pid=None,
        )
        controller.actions_in_progress[red_agent_name] = {
            "action": StealthServiceDiscovery(session=0, agent=red_agent_name, ip_address=target_ip),
            "remaining_ticks": 1,
        }
        controller.step(actions={}, skip_valid_action_check=True)
        cy_scanned = any(
            target_ip in getattr(sess, "ports", {}) for sess in cy_state.sessions.get(red_agent_name, {}).values()
        )
        assert cy_scanned

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, anchor_host].set(True),
            red_session_count=state.red_session_count.at[red_agent_id, anchor_host].set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, anchor_host].set(True),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(anchor_host),
            red_pending_ticks=state.red_pending_ticks.at[red_agent_id].set(1),
            red_pending_action=state.red_pending_action.at[red_agent_id].set(
                encode_red_action("StealthServiceDiscovery", target_host, red_agent_id)
            ),
            red_pending_source_kind=state.red_pending_source_kind.at[red_agent_id].set(PENDING_SOURCE_KIND_NONE),
            red_pending_source_host=state.red_pending_source_host.at[red_agent_id].set(-1),
        )

        new_state = process_red_with_duration(
            state,
            const,
            red_agent_id,
            RED_SCAN_START + target_host,
            jax.random.PRNGKey(0),
        )
        assert bool(new_state.red_scanned_hosts[red_agent_id, target_host]) == cy_scanned

    def test_deferred_scan_does_not_rebind_stale_scan_memory_source_to_anchor_matches_cyborg(self):
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=42)
        cyborg_env.reset()

        controller = cyborg_env.environment_controller
        cy_state = controller.state
        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)

        red_agent_id = 0
        red_agent_name = "red_agent_0"
        anchor_host = int(const.red_start_hosts[red_agent_id])
        subnet_id = int(const.host_subnet[anchor_host])
        subnet_hosts = [
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h])
            and not bool(const.host_is_router[h])
            and int(const.host_subnet[h]) == subnet_id
            and h != anchor_host
        ]
        assert len(subnet_hosts) >= 3
        stale_scan_owner = subnet_hosts[0]
        fallback_abstract_host = subnet_hosts[1]
        target_host = subnet_hosts[2]

        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]
        cy_state.sessions[red_agent_name][0] = Session(
            ident=0,
            hostname=mappings.idx_to_hostname[anchor_host],
            username="user",
            agent=red_agent_name,
            parent=None,
            session_type="shell",
            pid=None,
        )
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[stale_scan_owner],
                username="user",
                agent=red_agent_name,
                parent=None,
                session_type="shell",
                pid=None,
            )
        )
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[fallback_abstract_host],
                username="user",
                agent=red_agent_name,
                parent=None,
                session_type="shell",
                pid=None,
            )
        )
        controller.actions_in_progress[red_agent_name] = {
            "action": StealthServiceDiscovery(session=0, agent=red_agent_name, ip_address=target_ip),
            "remaining_ticks": 1,
        }
        # Drop the stale scan owner before execution tick.
        for sid, sess in list(cy_state.sessions[red_agent_name].items()):
            if sess.hostname == mappings.idx_to_hostname[stale_scan_owner]:
                cy_state.sessions[red_agent_name].pop(sid)
                if sid in cy_state.hosts[sess.hostname].sessions.get(red_agent_name, []):
                    cy_state.hosts[sess.hostname].sessions[red_agent_name].remove(sid)
                break
        controller.step(actions={}, skip_valid_action_check=True)
        cy_scanned = any(
            target_ip in getattr(sess, "ports", {}) for sess in cy_state.sessions.get(red_agent_name, {}).values()
        )
        assert not cy_scanned

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, anchor_host]
            .set(True)
            .at[red_agent_id, fallback_abstract_host]
            .set(True),
            red_session_count=state.red_session_count.at[red_agent_id, anchor_host]
            .set(1)
            .at[red_agent_id, fallback_abstract_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, anchor_host]
            .set(False)
            .at[red_agent_id, fallback_abstract_host]
            .set(True),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            red_scanned_hosts=state.red_scanned_hosts.at[red_agent_id, target_host].set(False),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(fallback_abstract_host),
            red_pending_ticks=state.red_pending_ticks.at[red_agent_id].set(1),
            red_pending_action=state.red_pending_action.at[red_agent_id].set(
                encode_red_action("StealthServiceDiscovery", target_host, red_agent_id)
            ),
            red_pending_source_kind=state.red_pending_source_kind.at[red_agent_id].set(PENDING_SOURCE_KIND_HOST),
            red_pending_source_host=state.red_pending_source_host.at[red_agent_id].set(stale_scan_owner),
        )

        new_state = process_red_with_duration(
            state,
            const,
            red_agent_id,
            RED_SCAN_START + target_host,
            jax.random.PRNGKey(0),
        )
        assert bool(new_state.red_scanned_hosts[red_agent_id, target_host]) == cy_scanned

    def test_deferred_scan_fails_when_primary_session_remap_selects_non_abstract_host(self, jax_const):
        choice = None
        for red_agent_id in range(NUM_RED_AGENTS):
            start_host = int(jax_const.red_start_hosts[red_agent_id])
            subnet_id = int(jax_const.host_subnet[start_host])
            subnet_hosts = [
                h
                for h in range(int(jax_const.num_hosts))
                if bool(jax_const.host_active[h])
                and not bool(jax_const.host_is_router[h])
                and int(jax_const.host_subnet[h]) == subnet_id
                and h != start_host
            ]
            if len(subnet_hosts) < 3:
                continue
            choice = (red_agent_id, start_host, subnet_hosts[0], subnet_hosts[1], subnet_hosts[2])
            break
        assert choice is not None
        red_agent_id, stale_source_host, non_abstract_host, abstract_host, target_host = choice

        state = create_initial_state().replace(host_services=jnp.array(jax_const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, non_abstract_host]
            .set(True)
            .at[red_agent_id, abstract_host]
            .set(True),
            red_session_count=state.red_session_count.at[red_agent_id, non_abstract_host]
            .set(1)
            .at[red_agent_id, abstract_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, non_abstract_host]
            .set(False)
            .at[red_agent_id, abstract_host]
            .set(True),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(-1),
            red_pending_ticks=state.red_pending_ticks.at[red_agent_id].set(1),
            red_pending_action=state.red_pending_action.at[red_agent_id].set(
                encode_red_action("StealthServiceDiscovery", target_host, red_agent_id)
            ),
            red_pending_source_kind=state.red_pending_source_kind.at[red_agent_id].set(PENDING_SOURCE_KIND_HOST),
            red_pending_source_host=state.red_pending_source_host.at[red_agent_id].set(stale_source_host),
        )

        session_counts = state.red_session_count[red_agent_id]
        # This fixed key promotes the concrete session host for the deterministic
        # two-session setup above, without a brute-force seed search.
        chosen_key = jax.random.PRNGKey(0)
        promoted = int(
            select_new_primary_session_host(
                session_counts,
                jax_const.host_active,
                jax.random.fold_in(chosen_key, jnp.int32(931)),
            )
        )
        assert promoted == non_abstract_host

        new_state = process_red_with_duration(
            state,
            jax_const,
            red_agent_id,
            RED_SCAN_START + target_host,
            chosen_key,
        )
        assert int(new_state.red_scan_anchor_host[red_agent_id]) == non_abstract_host
        assert not bool(new_state.red_scanned_hosts[red_agent_id, target_host])

    def test_scan_bound_to_nonabstract_session_zero_does_not_fallback_to_other_abstract_host_matches_cyborg(self):
        """Session-0 scan cannot silently switch to another abstract session on a different host."""
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=42)
        cyborg_env.reset()

        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)
        cy_state = cyborg_env.environment_controller.state
        controller = cyborg_env.environment_controller

        red_agent_id = 0
        red_agent_name = "red_agent_0"
        primary = cy_state.sessions[red_agent_name][0]
        source_host = mappings.hostname_to_idx[primary.hostname]
        source_subnet = int(const.host_subnet[source_host])

        subnet_hosts = [
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h])
            and not bool(const.host_is_router[h])
            and int(const.host_subnet[h]) == source_subnet
            and h != source_host
        ]
        assert subnet_hosts, "Need another host in source subnet"
        abstract_host = subnet_hosts[0]
        target_host = abstract_host
        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]

        # Session 0 becomes non-abstract on source_host.
        cy_state.sessions[red_agent_name][0] = Session(
            ident=0,
            hostname=primary.hostname,
            username=getattr(primary, "username", "user") or "user",
            agent=red_agent_name,
            pid=getattr(primary, "pid", None),
            parent=getattr(primary, "parent", None),
            session_type="shell",
        )
        # A different abstract session exists on another host.
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[abstract_host],
                username="user",
                agent=red_agent_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        abstract_sid = max(cy_state.sessions[red_agent_name].keys())

        iface = controller.agent_interfaces[red_agent_name]
        iface.action_space.client_session[0] = True
        iface.action_space.server_session[0] = True
        iface.action_space.client_session[abstract_sid] = True
        iface.action_space.server_session[abstract_sid] = True

        subnet_name = next(name for name, sid in CYBORG_SUFFIX_TO_ID.items() if sid == source_subnet)
        subnet_cidr = cy_state.subnet_name_to_cidr[subnet_name]
        discover_action = DiscoverRemoteSystems(subnet=subnet_cidr, session=abstract_sid, agent=red_agent_name)
        discover_action.duration = 1
        cyborg_env.step(agent=red_agent_name, action=discover_action)

        scan_action = AggressiveServiceDiscovery(session=0, agent=red_agent_name, ip_address=target_ip)
        scan_action.duration = 1
        cyborg_env.step(agent=red_agent_name, action=scan_action)

        cy_target_scanned = any(
            target_ip in getattr(sess, "ports", {})
            for sess in cy_state.sessions[red_agent_name].values()
            if hasattr(sess, "ports")
        )
        assert not cy_target_scanned, "CybORG should fail non-abstract session-0 scan"

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, source_host]
            .set(True)
            .at[red_agent_id, abstract_host]
            .set(True),
            red_session_count=state.red_session_count.at[red_agent_id, source_host]
            .set(1)
            .at[red_agent_id, abstract_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, source_host]
            .set(False)
            .at[red_agent_id, abstract_host]
            .set(True),
            red_abstract_host_rank=state.red_abstract_host_rank.at[red_agent_id, abstract_host].set(abstract_sid),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(source_host),
        )

        scan_idx = encode_red_action("AggressiveServiceDiscovery", target_host, red_agent_id)
        new_state = process_red_with_duration(
            state,
            const,
            red_agent_id,
            scan_idx,
            jax.random.PRNGKey(0),
        )
        assert bool(new_state.red_scanned_hosts[red_agent_id, target_host]) == cy_target_scanned

    def test_forced_primary_host_overrides_stale_anchor_for_same_tick_scan_matches_cyborg(self):
        """Differential regression: current scan tick must use CybORG's current session-0 host."""
        from jaxborg.parity.translate import build_mappings_from_cyborg
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=42)
        cyborg_env.reset()

        const = build_const_from_cyborg(cyborg_env)
        mappings = build_mappings_from_cyborg(cyborg_env)
        cy_state = cyborg_env.environment_controller.state
        controller = cyborg_env.environment_controller

        red_agent_id = 0
        red_agent_name = "red_agent_0"
        source_host = mappings.hostname_to_idx[cy_state.sessions[red_agent_name][0].hostname]
        source_subnet = int(const.host_subnet[source_host])

        subnet_hosts = [
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h])
            and not bool(const.host_is_router[h])
            and int(const.host_subnet[h]) == source_subnet
            and h != source_host
        ]
        assert len(subnet_hosts) >= 2, "Need extra hosts in source subnet"
        stale_anchor_host, target_host = subnet_hosts[0], subnet_hosts[1]
        assert stale_anchor_host != source_host

        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=mappings.idx_to_hostname[stale_anchor_host],
                username="user",
                agent=red_agent_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        stale_sid = max(cy_state.sessions[red_agent_name].keys())
        iface = controller.agent_interfaces[red_agent_name]
        iface.action_space.client_session[stale_sid] = True
        iface.action_space.server_session[stale_sid] = True

        subnet_name = next(name for name, sid in CYBORG_SUFFIX_TO_ID.items() if sid == source_subnet)
        subnet_cidr = cy_state.subnet_name_to_cidr[subnet_name]
        discover_action = DiscoverRemoteSystems(subnet=subnet_cidr, session=0, agent=red_agent_name)
        discover_action.duration = 1
        cyborg_env.step(agent=red_agent_name, action=discover_action)

        target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]
        scan_action = AggressiveServiceDiscovery(session=0, agent=red_agent_name, ip_address=target_ip)
        scan_action.duration = 1
        cyborg_env.step(agent=red_agent_name, action=scan_action)

        cy_target_scanned = any(
            target_ip in getattr(sess, "ports", {})
            for sess in cy_state.sessions[red_agent_name].values()
            if hasattr(sess, "ports")
        )
        assert cy_target_scanned, "CybORG setup should scan target from session 0"

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, source_host]
            .set(True)
            .at[red_agent_id, stale_anchor_host]
            .set(True),
            red_session_count=state.red_session_count.at[red_agent_id, source_host]
            .set(1)
            .at[red_agent_id, stale_anchor_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, source_host]
            .set(True)
            .at[red_agent_id, stale_anchor_host]
            .set(True),
            red_abstract_host_rank=state.red_abstract_host_rank.at[red_agent_id, source_host]
            .set(0)
            .at[red_agent_id, stale_anchor_host]
            .set(stale_sid),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(stale_anchor_host),
            red_primary_is_abstract=state.red_primary_is_abstract.at[red_agent_id].set(True),
        )

        scan_idx = encode_red_action("AggressiveServiceDiscovery", target_host, red_agent_id)
        new_state = process_red_with_duration(
            state,
            const,
            red_agent_id,
            scan_idx,
            jax.random.PRNGKey(0),
            forced_primary_host=jnp.int32(source_host),
        )
        assert bool(new_state.red_scanned_hosts[red_agent_id, target_host]) == cy_target_scanned
        assert int(new_state.red_scan_anchor_host[red_agent_id]) == source_host

    def test_scan_fallback_source_prefers_lowest_abstract_rank_matches_session_identity(self, jax_const):
        choice = None
        for red_agent_id in range(NUM_RED_AGENTS):
            start_host = int(jax_const.red_start_hosts[red_agent_id])
            subnet_id = int(jax_const.host_subnet[start_host])
            subnet_hosts = [
                h
                for h in range(int(jax_const.num_hosts))
                if bool(jax_const.host_active[h])
                and not bool(jax_const.host_is_router[h])
                and int(jax_const.host_subnet[h]) == subnet_id
            ]
            if len(subnet_hosts) < 3:
                continue
            choice = (red_agent_id, subnet_hosts[0], subnet_hosts[1], subnet_hosts[2])
            break
        assert choice is not None
        red_agent_id, low_rank_host, high_rank_host, target_host = choice

        state = create_initial_state().replace(host_services=jnp.array(jax_const.initial_services))
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, low_rank_host]
            .set(True)
            .at[red_agent_id, high_rank_host]
            .set(True),
            red_session_count=state.red_session_count.at[red_agent_id, low_rank_host]
            .set(1)
            .at[red_agent_id, high_rank_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, low_rank_host]
            .set(True)
            .at[red_agent_id, high_rank_host]
            .set(True),
            red_abstract_host_rank=state.red_abstract_host_rank.at[red_agent_id, low_rank_host]
            .set(1)
            .at[red_agent_id, high_rank_host]
            .set(8),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, target_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(-1),
        )

        scan_idx = encode_red_action("AggressiveServiceDiscovery", target_host, red_agent_id)
        new_state = process_red_with_duration(state, jax_const, red_agent_id, scan_idx, jax.random.PRNGKey(0))
        assert bool(new_state.red_scanned_hosts[red_agent_id, target_host])
        assert bool(new_state.red_scanned_source_hosts[red_agent_id, target_host, low_rank_host])


class TestScanSourceBinding(TestDifferentialWithCybORG):
    """Tests for _compute_scan_source_binding used by FSM orchestration."""

    def test_none_without_anchor(self, cyborg_and_jax):
        """Returns PENDING_SOURCE_KIND_NONE when no bound anchor session exists."""
        from jaxborg.actions.pending_source import PENDING_SOURCE_KIND_NONE
        from jaxborg.scenarios.cc4.red_fsm import _compute_scan_source_binding

        _, jax_const, jax_state = cyborg_and_jax
        red_agent_id = 0
        target_host = int(jax_const.red_start_hosts[red_agent_id])
        scan_action = encode_red_action("StealthServiceDiscovery", target_host, red_agent_id)

        state = jax_state.replace(
            red_sessions=jax_state.red_sessions.at[red_agent_id].set(False),
            red_scan_anchor_host=jax_state.red_scan_anchor_host.at[red_agent_id].set(-1),
        )

        source_kind, source_host = _compute_scan_source_binding(state, jax_const, red_agent_id, jnp.int32(scan_action))
        assert int(source_kind) == int(PENDING_SOURCE_KIND_NONE)
        assert int(source_host) == -1

    def test_session_binding_with_anchor(self, cyborg_and_jax):
        """Returns PENDING_SOURCE_KIND_SESSION_BINDING when anchor session exists."""
        from jaxborg.actions.pending_source import PENDING_SOURCE_KIND_SESSION_BINDING
        from jaxborg.scenarios.cc4.red_fsm import _compute_scan_source_binding

        _, jax_const, jax_state = cyborg_and_jax
        red_agent_id = 0
        target_host = int(jax_const.red_start_hosts[red_agent_id])
        scan_action = encode_red_action("StealthServiceDiscovery", target_host, red_agent_id)

        state = jax_state.replace(
            red_scan_anchor_host=jax_state.red_scan_anchor_host.at[red_agent_id].set(target_host),
        )

        source_kind, _ = _compute_scan_source_binding(state, jax_const, red_agent_id, jnp.int32(scan_action))
        assert int(source_kind) == int(PENDING_SOURCE_KIND_SESSION_BINDING)


class TestPendingStealthScanSourceLostOnRestore:
    """When a pending stealth scan's source session is killed by blue Restore,
    JAX must invalidate the scan (no detection random consumed), matching CybORG.

    Regression for: seed=7 step=88 detection RNG sync mismatch (expected 4, consumed 5).
    Root cause: process_red_with_duration lost the pending source host for
    SESSION_BINDING kind actions, so the scan fell back to the current anchor
    host (which may be the target itself) instead of failing.
    """

    @pytest.fixture(scope="class")
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    @pytest.fixture
    def setup(self, cyborg_env):
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        state = create_initial_state()

        # Set up red_agent_0 with two abstract sessions:
        #   host A (source/anchor) and host B (target)
        # Then queue a stealth scan targeting B with source on A.
        red_agent_id = 0
        start_host = int(const.red_start_hosts[red_agent_id])
        start_subnet = int(const.host_subnet[start_host])

        # Pick two hosts in the same subnet
        host_a = start_host  # source/anchor
        host_b = -1  # target
        for h in range(int(const.num_hosts)):
            if (
                int(const.host_subnet[h]) == start_subnet
                and h != host_a
                and bool(const.host_active[h])
                and not bool(const.host_is_router[h])
            ):
                host_b = h
                break
        assert host_b >= 0, "Need at least two non-router hosts in start subnet"

        # Set up sessions on both hosts, anchor on host_a
        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, host_a].set(True).at[red_agent_id, host_b].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, host_a]
            .set(True)
            .at[red_agent_id, host_b]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[red_agent_id].set(host_a),
            red_discovered_hosts=state.red_discovered_hosts.at[red_agent_id, host_a]
            .set(True)
            .at[red_agent_id, host_b]
            .set(True),
        )

        # Queue a stealth scan on host_b with SESSION_BINDING source (bound to anchor = host_a)
        stealth_action = encode_red_action("StealthServiceDiscovery", host_b, red_agent_id)
        # Duration = 3 for stealth scan; start with ticks=2 so next call decrements to 1 (still pending)
        key = jax.random.PRNGKey(99)
        state = state.replace(
            red_pending_ticks=state.red_pending_ticks.at[red_agent_id].set(2),
            red_pending_action=state.red_pending_action.at[red_agent_id].set(stealth_action),
            red_pending_key=state.red_pending_key.at[red_agent_id].set(jnp.asarray(key, dtype=jnp.uint32)),
            red_pending_source_kind=state.red_pending_source_kind.at[red_agent_id].set(
                PENDING_SOURCE_KIND_SESSION_BINDING
            ),
            red_pending_source_host=state.red_pending_source_host.at[red_agent_id].set(host_a),
        )

        return state, const, red_agent_id, host_a, host_b, stealth_action

    def test_pending_source_preserved_for_session_binding(self, setup):
        """After one tick of pending, SESSION_BINDING source host must be preserved (not reset to -1)."""
        state, const, red_agent_id, host_a, host_b, stealth_action = setup

        # Process one tick (ticks goes from 2 -> 1, action still pending)
        key = jax.random.PRNGKey(99)
        new_state = process_red_with_duration(state, const, red_agent_id, stealth_action, key)

        # Source host should still be host_a (not lost to -1)
        assert int(new_state.red_pending_ticks[red_agent_id]) == 1
        assert int(new_state.red_pending_source_host[red_agent_id]) == host_a, (
            f"SESSION_BINDING source host lost: expected {host_a}, "
            f"got {int(new_state.red_pending_source_host[red_agent_id])}"
        )

    def test_stealth_scan_fails_after_source_restored(self, setup):
        """When blue Restore clears the source session, a pending stealth scan must not execute."""
        state, const, red_agent_id, host_a, host_b, stealth_action = setup

        # Simulate the effect of a blue Restore clearing host_a sessions.
        # Directly clear the session state as Restore would, since the
        # specific blue agent assignment is not important for this test.
        state_restored = state.replace(
            red_sessions=state.red_sessions.at[red_agent_id, host_a].set(False),
            red_session_is_abstract=state.red_session_is_abstract.at[red_agent_id, host_a].set(False),
        )

        # Verify host_a sessions are cleared
        assert not bool(state_restored.red_sessions[red_agent_id, host_a])

        # Now set pending_ticks=1 so action will try to execute this tick
        state_restored = state_restored.replace(
            red_pending_ticks=state_restored.red_pending_ticks.at[red_agent_id].set(1),
        )

        # Track detection-roll consumption via tape RNG.  No values are pushed
        # because the action should not execute and therefore should consume
        # no rolls; popping from an empty tape would surface that as a
        # RuntimeError instead of a silent regression.
        tape = RNGTape()

        key = jax.random.PRNGKey(99)
        with rng_impls(uniform=tape.uniform), jax.disable_jit():
            new_state = process_red_with_duration(state_restored, const, red_agent_id, stealth_action, key)

        # Stealth scan should NOT have executed (source was destroyed by Restore)
        del new_state
        assert tape.consumed == 0, (
            f"Stealth scan consumed {tape.consumed} detection random(s) "
            f"but should have failed because source session on host_a={host_a} was restored"
        )
