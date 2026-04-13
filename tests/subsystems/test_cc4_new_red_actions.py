import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Shared.Session import RedAbstractSession
from CybORG.Simulator.Actions import DegradeServices, DiscoverDeception
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTrafficZone
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyTomcat import DecoyTomcat
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_red_action
from jaxborg.actions.blue_decoys import apply_blue_decoy
from jaxborg.actions.encoding import (
    ACTION_TYPE_AGGRESSIVE_SCAN,
    ACTION_TYPE_DEGRADE,
    ACTION_TYPE_DISCOVER_DECEPTION,
    ACTION_TYPE_STEALTH_SCAN,
    ACTION_TYPE_WITHDRAW,
    RED_AGGRESSIVE_SCAN_START,
    RED_DEGRADE_START,
    RED_DISCOVER_DECEPTION_START,
    RED_STEALTH_SCAN_START,
    RED_WITHDRAW_START,
    decode_red_action,
    encode_red_action,
)
from jaxborg.agents.fsm_red import FSM_K, FSM_S
from jaxborg.constants import (
    ACTIVITY_SCAN,
    COMPROMISE_NONE,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    DECOY_IDS,
    GLOBAL_MAX_HOSTS,
    MAX_DETECTION_RANDOMS,
    NUM_SUBNETS,
    SERVICE_IDS,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg
from tests.conftest import setup_red_agent_session

_jit_apply_red = jax.jit(apply_red_action, static_argnums=(2,))


@pytest.fixture
def jax_state_with_discovered(jax_const):
    state = create_initial_state()
    start_host = int(jax_const.red_start_hosts[0])
    state = setup_red_agent_session(state, 0, start_host)
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


class TestAggressiveScanEncoding:
    def test_encode_decode_roundtrip(self, jax_const):
        for h in [0, 5, 50]:
            code = encode_red_action("AggressiveServiceDiscovery", h, 0)
            assert code == RED_AGGRESSIVE_SCAN_START + h
            action_type, _, target_host = decode_red_action(code, 0, jax_const)
            assert int(action_type) == ACTION_TYPE_AGGRESSIVE_SCAN
            assert int(target_host) == h

    def test_no_overlap_with_impact(self):
        from jaxborg.actions.encoding import RED_IMPACT_END

        assert RED_AGGRESSIVE_SCAN_START == RED_IMPACT_END


class TestApplyAggressiveScan:
    def test_marks_scanned_and_activity(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        randoms = jnp.full(MAX_DETECTION_RANDOMS, 0.1)
        const = jax_const.replace(
            detection_randoms=randoms,
            use_detection_randoms=jnp.array(True),
        )
        state = state.replace(
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        action_idx = encode_red_action("AggressiveServiceDiscovery", target, 0)
        new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))

        assert bool(new_state.red_scanned_hosts[0, target])
        assert int(new_state.red_activity_this_step[target]) == ACTIVITY_SCAN

    def test_undiscovered_no_effect(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        discovered = np.array(state.red_discovered_hosts[0])
        undiscovered = None
        for h in range(int(jax_const.num_hosts)):
            if jax_const.host_active[h] and not discovered[h]:
                undiscovered = h
                break
        if undiscovered is None:
            pytest.fail("All hosts discovered")

        action_idx = encode_red_action("AggressiveServiceDiscovery", undiscovered, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert not bool(new_state.red_scanned_hosts[0, undiscovered])

    def test_jit_compatible(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("AggressiveServiceDiscovery", target, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert bool(new_state.red_scanned_hosts[0, target])


class TestStealthScanEncoding:
    def test_encode_decode_roundtrip(self, jax_const):
        for h in [0, 5, 50]:
            code = encode_red_action("StealthServiceDiscovery", h, 0)
            assert code == RED_STEALTH_SCAN_START + h
            action_type, _, target_host = decode_red_action(code, 0, jax_const)
            assert int(action_type) == ACTION_TYPE_STEALTH_SCAN
            assert int(target_host) == h


class TestApplyStealthScan:
    def test_marks_scanned_no_activity_when_undetected(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        randoms = jnp.full(MAX_DETECTION_RANDOMS, 0.9)
        const = jax_const.replace(
            detection_randoms=randoms,
            use_detection_randoms=jnp.array(True),
        )
        state = state.replace(
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        action_idx = encode_red_action("StealthServiceDiscovery", target, 0)
        new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))

        assert bool(new_state.red_scanned_hosts[0, target])
        assert int(new_state.red_activity_this_step[target]) == 0

    def test_jit_compatible(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("StealthServiceDiscovery", target, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert bool(new_state.red_scanned_hosts[0, target])


class TestDiscoverDeceptionEncoding:
    def test_encode_decode_roundtrip(self, jax_const):
        for h in [0, 5, 50]:
            code = encode_red_action("DiscoverDeception", h, 0)
            assert code == RED_DISCOVER_DECEPTION_START + h
            action_type, _, target_host = decode_red_action(code, 0, jax_const)
            assert int(action_type) == ACTION_TYPE_DISCOVER_DECEPTION
            assert int(target_host) == h


class TestApplyDiscoverDeception:
    def test_detects_decoy_no_fsm_transition(self, jax_const, jax_state_with_discovered):
        """CybORG's DiscoverDeception does NOT change FSM state (column 3: S→S).

        FSM transitions are handled by _host_state_transition via the
        success/failure matrices, and DiscoverDeception's column keeps the
        current state on both success and failure.
        """
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        randoms = jnp.full(MAX_DETECTION_RANDOMS, 0.1)
        scanned = state.red_scanned_hosts.at[0, target].set(True)
        decoys = state.host_decoys.at[target, 0].set(True)
        fsm = state.fsm_host_states.at[0, target].set(FSM_S)
        const = jax_const.replace(
            detection_randoms=randoms,
            use_detection_randoms=jnp.array(True),
        )
        state = state.replace(
            red_scanned_hosts=scanned,
            host_decoys=decoys,
            fsm_host_states=fsm,
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        action_idx = encode_red_action("DiscoverDeception", target, 0)
        new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))

        assert int(new_state.fsm_host_states[0, target]) == FSM_S

    def test_no_decoys_no_transition(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        randoms = jnp.full(MAX_DETECTION_RANDOMS, 0.9)
        scanned = state.red_scanned_hosts.at[0, target].set(True)
        fsm = state.fsm_host_states.at[0, target].set(FSM_S)
        const = jax_const.replace(
            detection_randoms=randoms,
            use_detection_randoms=jnp.array(True),
        )
        state = state.replace(
            red_scanned_hosts=scanned,
            fsm_host_states=fsm,
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        action_idx = encode_red_action("DiscoverDeception", target, 0)
        new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))

        assert int(new_state.fsm_host_states[0, target]) == FSM_S

    def test_detects_decoy_without_prior_scan(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        randoms = jnp.full(MAX_DETECTION_RANDOMS, 0.1)
        decoys = state.host_decoys.at[target, 0].set(True)
        fsm = state.fsm_host_states.at[0, target].set(FSM_S)
        const = jax_const.replace(
            detection_randoms=randoms,
            use_detection_randoms=jnp.array(True),
        )
        state = state.replace(
            host_decoys=decoys,
            fsm_host_states=fsm,
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        action_idx = encode_red_action("DiscoverDeception", target, 0)
        new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))

        assert int(new_state.fsm_host_states[0, target]) == FSM_S

    def test_failed_discover_deception_does_not_consume_detection_randoms(self, jax_const, jax_state_with_discovered):
        state = create_initial_state()
        target = _first_discovered_non_router(jax_const, jax_state_with_discovered)
        assert target is not None

        randoms = jnp.full(MAX_DETECTION_RANDOMS, 0.1)
        const = jax_const.replace(
            detection_randoms=randoms,
            use_detection_randoms=jnp.array(True),
        )
        state = state.replace(
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        action_idx = encode_red_action("DiscoverDeception", target, 0)
        new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))

        assert int(new_state.detection_random_index) == 0

    def test_discover_deception_ignores_blocked_subnet_route_matches_cyborg(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=0)
        cyborg_env.reset()
        cyborg_state = cyborg_env.environment_controller.state
        const = build_const_from_cyborg(cyborg_env)
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        start_host = int(const.red_start_hosts[0])
        start_hostname = sorted_hosts[start_host]
        start_subnet = int(const.host_subnet[start_host])

        target = None
        target_hostname = None
        blue_agent_id = None
        for blue_agent in [f"blue_agent_{i}" for i in range(5)]:
            hostname = cyborg_state.sessions[blue_agent][0].hostname
            host_idx = sorted_hosts.index(hostname)
            if int(const.host_subnet[host_idx]) == start_subnet:
                continue
            decoy_action = DecoyTomcat(agent=blue_agent, session=0, hostname=hostname)
            if str(decoy_action.execute(cyborg_state).success).upper() == "TRUE":
                target = host_idx
                target_hostname = hostname
                blue_agent_id = int(blue_agent.split("_")[-1])
                break
        if target is None or target_hostname is None or blue_agent_id is None:
            pytest.fail("Need a cross-subnet blue host that accepts Tomcat decoy")

        target_subnet_name = cyborg_state.hostname_subnet_map[target_hostname].value
        start_subnet_name = cyborg_state.hostname_subnet_map[start_hostname].value
        block_obs = BlockTrafficZone(
            session=0,
            agent="blue_agent_0",
            from_subnet=start_subnet_name,
            to_subnet=target_subnet_name,
        ).execute(cyborg_state)
        assert str(block_obs.success).upper() == "TRUE"

        target_ip = next(ip for ip, hostname in cyborg_state.ip_addresses.items() if hostname == target_hostname)
        cyborg_obs = DiscoverDeception(session=0, agent="red_agent_0", ip_address=target_ip).execute(cyborg_state)
        assert str(cyborg_obs.success).upper() == "TRUE"

        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        state = setup_red_agent_session(state, 0, start_host)
        state = apply_blue_decoy(state, const, blue_agent_id, target, DECOY_IDS["Tomcat"])
        blocked = jnp.zeros((NUM_SUBNETS, NUM_SUBNETS), dtype=jnp.bool_)
        blocked = blocked.at[int(const.host_subnet[target]), start_subnet].set(True)
        const = const.replace(
            detection_randoms=jnp.full(MAX_DETECTION_RANDOMS, 0.1),
            use_detection_randoms=jnp.array(True),
        )
        state = state.replace(
            blocked_zones=blocked,
            fsm_host_states=state.fsm_host_states.at[0, target].set(FSM_S),
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        action_idx = encode_red_action("DiscoverDeception", target, 0)
        new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))

        assert int(new_state.fsm_host_states[0, target]) == FSM_S
        assert int(new_state.detection_random_index) == 2

    def test_jit_compatible(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        randoms = jnp.full(MAX_DETECTION_RANDOMS, 0.1)
        scanned = state.red_scanned_hosts.at[0, target].set(True)
        decoys = state.host_decoys.at[target, 0].set(True)
        fsm = state.fsm_host_states.at[0, target].set(FSM_K)
        const = jax_const.replace(
            detection_randoms=randoms,
            use_detection_randoms=jnp.array(True),
        )
        state = state.replace(
            red_scanned_hosts=scanned,
            host_decoys=decoys,
            fsm_host_states=fsm,
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )

        action_idx = encode_red_action("DiscoverDeception", target, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(state, const, 0, action_idx, jax.random.PRNGKey(0))
        assert int(new_state.fsm_host_states[0, target]) == FSM_K


class TestDegradeEncoding:
    def test_encode_decode_roundtrip(self, jax_const):
        for h in [0, 5, 50]:
            code = encode_red_action("DegradeServices", h, 0)
            assert code == RED_DEGRADE_START + h
            action_type, _, target_host = decode_red_action(code, 0, jax_const)
            assert int(action_type) == ACTION_TYPE_DEGRADE
            assert int(target_host) == h


class TestApplyDegrade:
    def test_degrade_with_priv_session_sets_activity(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        red_sessions = state.red_sessions.at[0, target].set(True)
        red_privilege = state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED)
        host_services = state.host_services.at[target, 0].set(True)
        state = state.replace(
            red_sessions=red_sessions,
            red_privilege=red_privilege,
            host_services=host_services,
        )

        action_idx = encode_red_action("DegradeServices", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert int(new_state.red_activity_this_step[target]) == 2

    def test_degrade_without_priv_no_effect(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        red_sessions = state.red_sessions.at[0, target].set(True)
        red_privilege = state.red_privilege.at[0, target].set(COMPROMISE_USER)
        host_services = state.host_services.at[target, 0].set(True)
        state = state.replace(
            red_sessions=red_sessions,
            red_privilege=red_privilege,
            host_services=host_services,
        )

        action_idx = encode_red_action("DegradeServices", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert int(new_state.red_activity_this_step[target]) == 0

    def test_jit_compatible(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        red_sessions = state.red_sessions.at[0, target].set(True)
        red_privilege = state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED)
        host_services = state.host_services.at[target, 0].set(True)
        state = state.replace(
            red_sessions=red_sessions,
            red_privilege=red_privilege,
            host_services=host_services,
        )

        action_idx = encode_red_action("DegradeServices", target, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert int(new_state.red_activity_this_step[target]) == 2


class TestWithdrawEncoding:
    def test_encode_decode_roundtrip(self, jax_const):
        for h in [0, 5, 50]:
            code = encode_red_action("Withdraw", h, 0)
            assert code == RED_WITHDRAW_START + h
            action_type, _, target_host = decode_red_action(code, 0, jax_const)
            assert int(action_type) == ACTION_TYPE_WITHDRAW
            assert int(target_host) == h


class TestApplyWithdraw:
    def test_withdraw_clears_session_and_privilege(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        red_sessions = state.red_sessions.at[0, target].set(True)
        red_session_count = state.red_session_count.at[0, target].set(1)
        red_privilege = state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED)
        host_compromised = state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED)
        state = state.replace(
            red_sessions=red_sessions,
            red_session_count=red_session_count,
            red_privilege=red_privilege,
            host_compromised=host_compromised,
        )

        action_idx = encode_red_action("Withdraw", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE

    def test_withdraw_no_session_no_effect(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        start_host = int(jax_const.red_start_hosts[0])
        discovered = np.array(state.red_discovered_hosts[0])
        target = None
        for h in range(int(jax_const.num_hosts)):
            if discovered[h] and not jax_const.host_is_router[h] and h != start_host:
                target = h
                break
        assert target is not None
        assert not bool(state.red_sessions[0, target])

        action_idx = encode_red_action("Withdraw", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE

    def test_withdraw_does_not_affect_other_agents(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        red_sessions = state.red_sessions.at[0, target].set(True)
        red_sessions = red_sessions.at[1, target].set(True)
        red_session_count = state.red_session_count.at[0, target].set(1).at[1, target].set(1)
        state = state.replace(red_sessions=red_sessions, red_session_count=red_session_count)

        action_idx = encode_red_action("Withdraw", target, 0)
        new_state = _jit_apply_red(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert not bool(new_state.red_sessions[0, target])
        assert bool(new_state.red_sessions[1, target])

    def test_jit_compatible(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        red_sessions = state.red_sessions.at[0, target].set(True)
        red_session_count = state.red_session_count.at[0, target].set(1)
        red_privilege = state.red_privilege.at[0, target].set(COMPROMISE_USER)
        state = state.replace(
            red_sessions=red_sessions,
            red_session_count=red_session_count,
            red_privilege=red_privilege,
        )

        action_idx = encode_red_action("Withdraw", target, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert not bool(new_state.red_sessions[0, target])


class TestDifferentialWithCybORG:
    @pytest.fixture
    def cyborg_and_jax(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        cyborg_env = CybORG(scenario_generator=sg, seed=42)
        const = build_const_from_cyborg(cyborg_env)
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(const.initial_services))
        start_host = int(const.red_start_hosts[0])
        state = state.replace(
            red_sessions=state.red_sessions.at[0, start_host].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[0, start_host].set(True),
        )
        return cyborg_env, const, state

    def test_degrade_only_changes_active_service_reliability_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = None
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
                continue
            if bool(np.any(np.array(const.initial_services[h]))):
                target = h
                break
        assert target is not None, "No host with active services found"
        target_hostname = sorted_hosts[target]

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="root",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
        )

        active_service_sids = set(np.where(np.array(const.initial_services[target]))[0].tolist())
        inactive_service_sid = next((sid for sid in SERVICE_IDS.values() if sid not in active_service_sids), None)
        if inactive_service_sid is None:
            pytest.fail("No inactive service slot available for this host")

        degrade_action = DegradeServices(hostname=target_hostname, session=0, agent="red_agent_0")
        degrade_action.duration = 1
        cyborg_obs = degrade_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_red_action("DegradeServices", target, 0)
        new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))

        cyborg_active_reliability = None
        active_sid = None
        for service_name, service in cyborg_state.hosts[target_hostname].services.items():
            svc_key = str(service_name).split(".")[-1] if "." in str(service_name) else str(service_name)
            sid = SERVICE_IDS.get(svc_key)
            if sid is None:
                continue
            active_sid = sid
            cyborg_active_reliability = int(service.get_service_reliability())
            break

        assert active_sid is not None and cyborg_active_reliability is not None
        assert int(new_state.host_service_reliability[target, active_sid]) == cyborg_active_reliability
        assert int(new_state.host_service_reliability[target, inactive_service_sid]) == 100

    def test_degrade_decoy_reliability_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target_hostname = None
        blue_agent_id = None
        for blue_agent in [f"blue_agent_{i}" for i in range(5)]:
            session = cyborg_state.sessions[blue_agent][0]
            hostname = session.hostname
            decoy_action = DecoyTomcat(agent=blue_agent, session=0, hostname=hostname)
            if str(decoy_action.execute(cyborg_state).success).upper() == "TRUE":
                target_hostname = hostname
                blue_agent_id = int(blue_agent.split("_")[-1])
                break
        if target_hostname is None:
            pytest.fail("No blue host accepted a Tomcat decoy")

        target = sorted_hosts.index(target_hostname)
        state = apply_blue_decoy(state, const, blue_agent_id, target, DECOY_IDS["Tomcat"])

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="root",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
        )

        degrade_action = DegradeServices(hostname=target_hostname, session=0, agent="red_agent_0")
        degrade_action.duration = 1
        cyborg_obs = degrade_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_red_action("DegradeServices", target, 0)
        new_state = _jit_apply_red(state, const, 0, action_idx, jax.random.PRNGKey(0))

        cyborg_decoy_reliability = int(cyborg_state.hosts[target_hostname].services["tomcat"].get_service_reliability())
        assert int(new_state.host_decoy_reliability[target, DECOY_IDS["Tomcat"]]) == cyborg_decoy_reliability
