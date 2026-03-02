import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Shared.Session import RedAbstractSession, Session
from CybORG.Simulator.Actions import DiscoverRemoteSystems, PrivilegeEscalate
from CybORG.Simulator.Actions.AbstractActions.DiscoverNetworkServices import (
    AggressiveServiceDiscovery,
)
from CybORG.Simulator.Actions.AbstractActions.ExploitRemoteService import (
    ExploitRemoteService,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_red_action
from jaxborg.actions.encoding import (
    ACTION_TYPE_PRIVESC,
    RED_EXPLOIT_BLUEKEEP_END,
    RED_PRIVESC_END,
    RED_PRIVESC_START,
    decode_red_action,
    encode_red_action,
)
from jaxborg.constants import (
    ACTIVITY_EXPLOIT,
    ACTIVITY_NONE,
    COMPROMISE_NONE,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_RED_AGENTS,
    SERVICE_IDS,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import CYBORG_SUFFIX_TO_ID

_jit_apply_red = jax.jit(apply_red_action, static_argnums=(2,))

SSH_SVC = SERVICE_IDS["SSHD"]


def _setup_exploited_state(jax_const, target_host):
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(jax_const.initial_services))

    start_host = int(jax_const.red_start_hosts[0])
    red_sessions = state.red_sessions.at[0, start_host].set(True)
    red_session_is_abstract = state.red_session_is_abstract.at[0, start_host].set(True)
    state = state.replace(red_sessions=red_sessions, red_session_is_abstract=red_session_is_abstract)

    target_subnet = int(jax_const.host_subnet[target_host])
    discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
    state = _jit_apply_red(state, jax_const, 0, discover_idx, jax.random.PRNGKey(0))
    state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

    scan_idx = encode_red_action("DiscoverNetworkServices", target_host, 0)
    state = _jit_apply_red(state, jax_const, 0, scan_idx, jax.random.PRNGKey(0))
    state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

    exploit_idx = encode_red_action("ExploitRemoteService_cc4SSHBruteForce", target_host, 0)
    state = _jit_apply_red(state, jax_const, 0, exploit_idx, jax.random.PRNGKey(0))
    state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
    return state


def _find_exploitable_host(jax_const, exclude_start=True):
    for h in range(jax_const.num_hosts):
        if (
            jax_const.host_active[h]
            and not jax_const.host_is_router[h]
            and jax_const.initial_services[h, SSH_SVC]
            and jax_const.host_has_bruteforceable_user[h]
        ):
            if exclude_start and h == int(jax_const.red_start_hosts[0]):
                continue
            return h
    return None


@pytest.fixture(scope="module")
def exploited_host(jax_const):
    target = _find_exploitable_host(jax_const)
    if target is None:
        pytest.skip("No exploitable host found")
    return _setup_exploited_state(jax_const, target), target


class TestPrivescEncoding:
    def test_privesc_range_starts_after_bluekeep(self):
        assert RED_PRIVESC_START == RED_EXPLOIT_BLUEKEEP_END

    def test_privesc_range_is_max_hosts_wide(self):
        assert RED_PRIVESC_END - RED_PRIVESC_START == GLOBAL_MAX_HOSTS

    def test_encode_per_host(self):
        for h in [0, 5, 50, GLOBAL_MAX_HOSTS - 1]:
            code = encode_red_action("PrivilegeEscalate", h, 0)
            assert code == RED_PRIVESC_START + h

    def test_decode_roundtrip(self, jax_const):
        for h in [0, 5, 50, GLOBAL_MAX_HOSTS - 1]:
            code = encode_red_action("PrivilegeEscalate", h, 0)
            action_type, target_subnet, target_host = decode_red_action(code, 0, jax_const)
            assert int(action_type) == ACTION_TYPE_PRIVESC
            assert int(target_subnet) == -1
            assert int(target_host) == h


class TestApplyPrivesc:
    def test_privesc_upgrades_to_privileged(self, jax_const, exploited_host):
        state, target = exploited_host
        assert int(state.red_privilege[0, target]) == COMPROMISE_USER

        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert int(new_state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED
        assert int(new_state.host_compromised[target]) == COMPROMISE_PRIVILEGED

    def test_privesc_sets_activity(self, jax_const, exploited_host):
        state, target = exploited_host
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert int(new_state.red_activity_this_step[target]) == ACTIVITY_EXPLOIT

    def test_privesc_fails_without_session(self, jax_const):
        target = _find_exploitable_host(jax_const)
        if target is None:
            pytest.skip("No exploitable host found")

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))
        start_host = int(jax_const.red_start_hosts[0])
        red_sessions = state.red_sessions.at[0, start_host].set(True)
        state = state.replace(red_sessions=red_sessions)

        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE

    def test_privesc_noop_when_already_privileged(self, jax_const, exploited_host):
        state, target = exploited_host
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        state1 = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert int(state1.red_privilege[0, target]) == COMPROMISE_PRIVILEGED

        state1 = state1.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
        state2 = _jit_apply_red(state1, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert int(state2.red_activity_this_step[target]) == ACTIVITY_NONE

    def test_privesc_preserves_session(self, jax_const, exploited_host):
        state, target = exploited_host
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert bool(new_state.red_sessions[0, target])

    def test_privesc_discovers_host_info_links_on_success(self, jax_const):
        target = _find_exploitable_host(jax_const)
        if target is None:
            pytest.skip("No exploitable host found")

        linked_host = next((h for h in range(jax_const.num_hosts) if h != target and jax_const.host_active[h]), None)
        if linked_host is None:
            pytest.skip("No linked host candidate found")

        host_info_links = jnp.zeros_like(jax_const.host_info_links).at[target, linked_host].set(True)
        const = jax_const.replace(host_info_links=host_info_links)
        state = _setup_exploited_state(const, target)
        assert not bool(state.red_discovered_hosts[0, linked_host])

        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))
        assert bool(new_state.red_discovered_hosts[0, linked_host])

    def test_privesc_does_not_affect_other_agents(self, jax_const, exploited_host):
        state, target = exploited_host
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        for agent in range(1, NUM_RED_AGENTS):
            np.testing.assert_array_equal(
                np.array(new_state.red_sessions[agent]),
                np.array(state.red_sessions[agent]),
            )
            np.testing.assert_array_equal(
                np.array(new_state.red_privilege[agent]),
                np.array(state.red_privilege[agent]),
            )

    def test_privesc_only_affects_target(self, jax_const, exploited_host):
        state, target = exploited_host
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        for h in range(jax_const.num_hosts):
            if h == target:
                continue
            assert int(new_state.red_privilege[0, h]) == int(state.red_privilege[0, h])
            assert int(new_state.host_compromised[h]) == int(state.host_compromised[h])

    def test_jit_compatible(self, jax_const, exploited_host):
        state, target = exploited_host
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED


class TestPrivescChain:
    def test_discover_scan_exploit_privesc_chain(self, jax_const, exploited_host):
        state, target = exploited_host
        assert int(state.red_privilege[0, target]) == COMPROMISE_USER
        assert bool(state.red_sessions[0, target])

        privesc_idx = encode_red_action("PrivilegeEscalate", target, 0)
        state = _jit_apply_red(state, jax_const, 0, privesc_idx, jax.random.PRNGKey(0))

        assert int(state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED
        assert int(state.host_compromised[target]) == COMPROMISE_PRIVILEGED
        assert bool(state.red_sessions[0, target])
        assert int(state.red_activity_this_step[target]) == ACTIVITY_EXPLOIT


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
        from jaxborg.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(const.initial_services))
        start_host = int(const.red_start_hosts[0])
        red_sessions = state.red_sessions.at[0, start_host].set(True)
        red_session_is_abstract = state.red_session_is_abstract.at[0, start_host].set(True)
        state = state.replace(red_sessions=red_sessions, red_session_is_abstract=red_session_is_abstract)
        return cyborg_env, const, state

    def _find_and_exploit_host(self, cyborg_env, const, state):
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        subnet_name = "contractor_network_subnet"
        subnet_cidr = cyborg_state.subnet_name_to_cidr[subnet_name]
        sid = CYBORG_SUFFIX_TO_ID[subnet_name]

        discover_action = DiscoverRemoteSystems(subnet=subnet_cidr, session=0, agent="red_agent_0")
        discover_action.duration = 1
        cyborg_env.step(agent="red_agent_0", action=discover_action)

        discover_idx = encode_red_action("DiscoverRemoteSystems", sid, 0)
        state = _jit_apply_red(state, const, 0, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        discovered_jax = np.array(state.red_discovered_hosts[0])
        exploitable = [
            h
            for h in range(const.num_hosts)
            if (
                discovered_jax[h]
                and not const.host_is_router[h]
                and const.initial_services[h, SSH_SVC]
                and const.host_has_bruteforceable_user[h]
                and h != int(const.red_start_hosts[0])
            )
        ]
        assert len(exploitable) > 0, "No exploitable hosts found"
        target_h = exploitable[0]
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
        state = _jit_apply_red(state, const, 0, scan_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        exploit_action = ExploitRemoteService(ip_address=target_ip, session=0, agent="red_agent_0")
        exploit_action.duration = 1
        cyborg_result = cyborg_env.step(agent="red_agent_0", action=exploit_action)
        cyborg_exploit_success = cyborg_result.observation.get("success") == True  # noqa: E712

        exploit_idx = encode_red_action("ExploitRemoteService_cc4SSHBruteForce", target_h, 0)
        state = _jit_apply_red(state, const, 0, exploit_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        jax_exploit_success = bool(state.red_sessions[0, target_h])
        assert jax_exploit_success == cyborg_exploit_success
        assert cyborg_exploit_success, "CybORG exploit failed unexpectedly"

        return state, target_h, target_hostname, target_ip

    def test_privesc_success_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        state, target_h, target_hostname, target_ip = self._find_and_exploit_host(cyborg_env, const, state)

        privesc_action = PrivilegeEscalate(hostname=target_hostname, session=0, agent="red_agent_0")
        privesc_action.duration = 1
        cyborg_result = cyborg_env.step(agent="red_agent_0", action=privesc_action)
        cyborg_success = cyborg_result.observation.get("success") == True  # noqa: E712

        privesc_idx = encode_red_action("PrivilegeEscalate", target_h, 0)
        new_state = _jit_apply_red(state, const, 0, privesc_idx, jax.random.PRNGKey(0))

        jax_success = int(new_state.red_privilege[0, target_h]) == COMPROMISE_PRIVILEGED

        assert jax_success == cyborg_success, (
            f"JAX privesc success={jax_success} but CybORG={cyborg_success} for host {target_h} ({target_hostname})"
        )

        if cyborg_success:
            assert int(new_state.host_compromised[target_h]) == COMPROMISE_PRIVILEGED
            assert bool(new_state.red_sessions[0, target_h])

    def test_privesc_privilege_level_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        state, target_h, target_hostname, target_ip = self._find_and_exploit_host(cyborg_env, const, state)

        cyborg_state_obj = cyborg_env.environment_controller.state
        cyborg_state_obj.hosts[target_hostname]
        sessions_before = [
            s for s in cyborg_state_obj.sessions["red_agent_0"].values() if s.hostname == target_hostname
        ]
        assert len(sessions_before) > 0
        cyborg_priv_before = any(s.has_privileged_access() for s in sessions_before)
        jax_priv_before = int(state.red_privilege[0, target_h])
        assert (jax_priv_before >= COMPROMISE_PRIVILEGED) == cyborg_priv_before

        privesc_action = PrivilegeEscalate(hostname=target_hostname, session=0, agent="red_agent_0")
        privesc_action.duration = 1
        cyborg_env.step(agent="red_agent_0", action=privesc_action)

        privesc_idx = encode_red_action("PrivilegeEscalate", target_h, 0)
        new_state = _jit_apply_red(state, const, 0, privesc_idx, jax.random.PRNGKey(0))

        sessions_after = [s for s in cyborg_state_obj.sessions["red_agent_0"].values() if s.hostname == target_hostname]
        cyborg_priv_after = any(s.has_privileged_access() for s in sessions_after)
        jax_priv_after = int(new_state.red_privilege[0, target_h]) >= COMPROMISE_PRIVILEGED

        assert jax_priv_after == cyborg_priv_after, (
            f"JAX privileged={jax_priv_after} but CybORG privileged={cyborg_priv_after} "
            f"for host {target_h} ({target_hostname})"
        )

    def test_privesc_fails_without_session_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state_obj = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state_obj.hosts.keys())

        start_host = int(const.red_start_hosts[0])
        target_h = None
        for h in range(const.num_hosts):
            if (
                const.host_active[h]
                and not const.host_is_router[h]
                and h != start_host
                and not state.red_sessions[0, h]
            ):
                target_h = h
                break
        assert target_h is not None
        target_hostname = sorted_hosts[target_h]

        privesc_action = PrivilegeEscalate(hostname=target_hostname, session=0, agent="red_agent_0")
        privesc_action.duration = 1
        cyborg_result = cyborg_env.step(agent="red_agent_0", action=privesc_action)
        cyborg_success = cyborg_result.observation.get("success") == True  # noqa: E712

        privesc_idx = encode_red_action("PrivilegeEscalate", target_h, 0)
        new_state = _jit_apply_red(state, const, 0, privesc_idx, jax.random.PRNGKey(0))
        jax_success = int(new_state.red_privilege[0, target_h]) >= COMPROMISE_PRIVILEGED

        assert not jax_success and not cyborg_success

    def test_privesc_discovers_info_link_hosts_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        state, target_h, target_hostname, _ = self._find_and_exploit_host(cyborg_env, const, state)
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())
        hostname_to_idx = {hostname: idx for idx, hostname in enumerate(sorted_hosts)}

        linked_host_idxs = np.where(np.array(const.host_info_links[target_h]))[0].tolist()
        if not linked_host_idxs:
            pytest.skip("Target host has no info-link hosts")

        action_space = cyborg_env.environment_controller.agent_interfaces["red_agent_0"].action_space
        known_before = {
            hostname_to_idx[cyborg_state.ip_addresses[ip]]
            for ip, is_known in action_space.ip_address.items()
            if is_known and ip in cyborg_state.ip_addresses and cyborg_state.ip_addresses[ip] in hostname_to_idx
        }
        jax_known_before = set(np.where(np.array(state.red_discovered_hosts[0]))[0].tolist())

        unseen_linked = [h for h in linked_host_idxs if h not in known_before and h not in jax_known_before]
        if not unseen_linked:
            pytest.skip("No unseen info-link host available for differential assertion")

        privesc_action = PrivilegeEscalate(hostname=target_hostname, session=0, agent="red_agent_0")
        privesc_action.duration = 1
        cyborg_env.step(agent="red_agent_0", action=privesc_action)

        privesc_idx = encode_red_action("PrivilegeEscalate", target_h, 0)
        new_state = _jit_apply_red(state, const, 0, privesc_idx, jax.random.PRNGKey(0))

        action_space_after = cyborg_env.environment_controller.agent_interfaces["red_agent_0"].action_space
        known_after = {
            hostname_to_idx[cyborg_state.ip_addresses[ip]]
            for ip, is_known in action_space_after.ip_address.items()
            if is_known and ip in cyborg_state.ip_addresses and cyborg_state.ip_addresses[ip] in hostname_to_idx
        }
        jax_known_after = set(np.where(np.array(new_state.red_discovered_hosts[0]))[0].tolist())

        for linked_h in unseen_linked:
            assert linked_h in known_after
            assert linked_h in jax_known_after

    def test_privesc_requires_bound_source_session_to_be_abstract_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        source_host = int(const.red_start_hosts[0])
        target_host = next(
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h]) and not bool(const.host_is_router[h]) and h != source_host
        )
        abstract_host = next(
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h]) and not bool(const.host_is_router[h]) and h not in {source_host, target_host}
        )

        target_hostname = sorted_hosts[target_host]
        abstract_hostname = sorted_hosts[abstract_host]

        # Keep source session id=0 as plain Session (non-abstract), but add an abstract
        # session elsewhere to ensure "has any abstract session" is true.
        cy_state.add_session(
            Session(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent="red_agent_0",
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=abstract_hostname,
                username="user",
                agent="red_agent_0",
                parent=0,
                session_type="shell",
                pid=None,
            )
        )

        cy_action = PrivilegeEscalate(hostname=target_hostname, session=0, agent="red_agent_0")
        cy_action.duration = 1
        cy_result = cyborg_env.step(agent="red_agent_0", action=cy_action)
        cy_success = cy_result.observation.get("success") == True  # noqa: E712

        state = state.replace(
            red_sessions=state.red_sessions.at[0, source_host]
            .set(True)
            .at[0, target_host]
            .set(True)
            .at[0, abstract_host]
            .set(True),
            red_session_count=state.red_session_count.at[0, source_host]
            .set(1)
            .at[0, target_host]
            .set(1)
            .at[0, abstract_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[0, source_host]
            .set(False)
            .at[0, target_host]
            .set(False)
            .at[0, abstract_host]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(source_host),
            red_privilege=state.red_privilege.at[0, target_host].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target_host].set(COMPROMISE_USER),
        )
        jax_action = encode_red_action("PrivilegeEscalate", target_host, 0)
        new_state = _jit_apply_red(state, const, 0, jax_action, jax.random.PRNGKey(0))
        jax_success = int(new_state.red_privilege[0, target_host]) == COMPROMISE_PRIVILEGED

        assert not cy_success
        assert jax_success == cy_success

    def test_privesc_fails_when_bound_source_session_missing_even_if_other_abstract_exists_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        source_host = int(const.red_start_hosts[0])
        target_host = next(
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h]) and not bool(const.host_is_router[h]) and h != source_host
        )
        abstract_host = next(
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h]) and not bool(const.host_is_router[h]) and h not in {source_host, target_host}
        )
        target_hostname = sorted_hosts[target_host]
        abstract_hostname = sorted_hosts[abstract_host]

        # Remove source session 0 entirely, but keep another abstract session.
        del cy_state.sessions["red_agent_0"][0]
        source_hostname = sorted_hosts[source_host]
        if 0 in cy_state.hosts[source_hostname].sessions["red_agent_0"]:
            cy_state.hosts[source_hostname].sessions["red_agent_0"].remove(0)
        cy_state.add_session(
            Session(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent="red_agent_0",
                parent=None,
                session_type="shell",
                pid=None,
            )
        )
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=abstract_hostname,
                username="user",
                agent="red_agent_0",
                parent=None,
                session_type="shell",
                pid=None,
            )
        )

        cy_action = PrivilegeEscalate(hostname=target_hostname, session=0, agent="red_agent_0")
        cy_action.duration = 1
        cy_result = cyborg_env.step(agent="red_agent_0", action=cy_action)
        cy_success = cy_result.observation.get("success") == True  # noqa: E712

        state = state.replace(
            red_sessions=state.red_sessions.at[0, source_host]
            .set(False)
            .at[0, target_host]
            .set(True)
            .at[0, abstract_host]
            .set(True),
            red_session_count=state.red_session_count.at[0, source_host]
            .set(0)
            .at[0, target_host]
            .set(1)
            .at[0, abstract_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[0, target_host]
            .set(False)
            .at[0, abstract_host]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(-1),
            red_privilege=state.red_privilege.at[0, target_host].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target_host].set(COMPROMISE_USER),
        )
        jax_action = encode_red_action("PrivilegeEscalate", target_host, 0)
        new_state = _jit_apply_red(state, const, 0, jax_action, jax.random.PRNGKey(0))
        jax_success = int(new_state.red_privilege[0, target_host]) == COMPROMISE_PRIVILEGED

        assert not cy_success
        assert jax_success == cy_success

    def test_scan_recovers_bound_source_when_anchor_unset_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        agent_id = 0
        start_host = int(const.red_start_hosts[agent_id])
        subnet_id = int(const.host_subnet[start_host])
        subnet_hosts = [
            h
            for h in range(int(const.num_hosts))
            if bool(const.host_active[h])
            and not bool(const.host_is_router[h])
            and int(const.host_subnet[h]) == subnet_id
        ]
        if len(subnet_hosts) < 3:
            pytest.skip("Need three hosts in source subnet")
        source_host = next(h for h in subnet_hosts if h != start_host)
        target_host = next(
            h
            for h in subnet_hosts
            if h not in {start_host, source_host} and cy_state.hosts[sorted_hosts[h]].os_type.name == "LINUX"
        )
        target_hostname = sorted_hosts[target_host]

        red_name = "red_agent_0"
        sess0 = cy_state.sessions[red_name][0]
        old_hostname = sess0.hostname
        if 0 in cy_state.hosts[old_hostname].sessions[red_name]:
            cy_state.hosts[old_hostname].sessions[red_name].remove(0)
        sess0.hostname = sorted_hosts[source_host]
        cy_state.hosts[sorted_hosts[source_host]].sessions[red_name].append(0)
        cy_state.add_session(
            Session(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent=red_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )

        target_ip = cy_state.hostname_ip_map[target_hostname]
        scan_action = AggressiveServiceDiscovery(session=0, agent=red_name, ip_address=target_ip)
        scan_obs = scan_action.execute(cy_state)
        assert str(scan_obs.success).upper() == "TRUE"
        cy_action = PrivilegeEscalate(hostname=target_hostname, session=0, agent=red_name)
        cy_obs = cy_action.execute(cy_state)
        cy_success = str(cy_obs.success).upper() == "TRUE"

        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, start_host]
            .set(False)
            .at[agent_id, source_host]
            .set(True)
            .at[agent_id, target_host]
            .set(True),
            red_session_count=state.red_session_count.at[agent_id, start_host]
            .set(0)
            .at[agent_id, source_host]
            .set(1)
            .at[agent_id, target_host]
            .set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, start_host]
            .set(False)
            .at[agent_id, source_host]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(-1),
            red_discovered_hosts=state.red_discovered_hosts.at[agent_id, target_host].set(True),
            red_privilege=state.red_privilege.at[agent_id, target_host].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target_host].set(COMPROMISE_USER),
        )
        scan_idx = encode_red_action("AggressiveServiceDiscovery", target_host, agent_id)
        state_after_scan = _jit_apply_red(state, const, agent_id, scan_idx, jax.random.PRNGKey(0))
        assert int(state_after_scan.red_scan_anchor_host[agent_id]) == source_host

        privesc_idx = encode_red_action("PrivilegeEscalate", target_host, agent_id)
        new_state = _jit_apply_red(state_after_scan, const, agent_id, privesc_idx, jax.random.PRNGKey(1))
        jax_success = int(new_state.red_privilege[agent_id, target_host]) == COMPROMISE_PRIVILEGED

        assert cy_success
        assert jax_success == cy_success
