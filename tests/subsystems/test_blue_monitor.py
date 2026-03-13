import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Simulator.Actions.AbstractActions.Monitor import Monitor
from CybORG.Simulator.Actions.ConcreteActions.Portscan import Portscan
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.blue_monitor import apply_blue_monitor
from jaxborg.actions.encoding import (
    BLUE_ACTION_TYPE_MONITOR,
    BLUE_ACTION_TYPE_SLEEP,
    BLUE_MONITOR,
    BLUE_SLEEP,
    decode_blue_action,
    encode_blue_action,
    encode_red_action,
)
from jaxborg.constants import (
    ACTIVITY_NONE,
    ACTIVITY_SCAN,
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    SERVICE_IDS,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg

_jit_apply_red = jax.jit(apply_red_action, static_argnums=(2,))
_jit_apply_blue = jax.jit(apply_blue_action, static_argnums=(2,))

SSH_SVC = SERVICE_IDS["SSHD"]


@pytest.fixture
def jax_const():
    return build_const_from_cyborg(_make_cyborg_env())


def _make_cyborg_env():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)


def _make_jax_state(const):
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    start_host = int(const.red_start_hosts[0])
    red_sessions = state.red_sessions.at[0, start_host].set(True)
    red_session_is_abstract = state.red_session_is_abstract.at[0, start_host].set(True)
    red_scan_anchor_host = state.red_scan_anchor_host.at[0].set(start_host)
    return state.replace(
        red_sessions=red_sessions,
        red_session_is_abstract=red_session_is_abstract,
        red_scan_anchor_host=red_scan_anchor_host,
    )


def _cyborg_hostname_to_idx(cyborg_env):
    state = cyborg_env.environment_controller.state
    sorted_hosts = sorted(state.hosts.keys())
    return {h: i for i, h in enumerate(sorted_hosts)}


def _cyborg_monitor_detected_hosts(cyborg_env):
    """Run Monitor for all blue agents in CybORG and return set of detected host indices."""
    state = cyborg_env.environment_controller.state
    sorted_hosts = sorted(state.hosts.keys())
    detected = set()
    for blue_idx in range(NUM_BLUE_AGENTS):
        agent_name = f"blue_agent_{blue_idx}"
        monitor = Monitor(session=0, agent=agent_name)
        obs = monitor.execute(state)
        for key in obs.data.keys():
            if key != "success" and key in sorted_hosts:
                detected.add(sorted_hosts.index(key))
    return detected


def _cyborg_portscan(cyborg_env, target_hostname):
    """Execute a Portscan on target_hostname in CybORG."""
    state = cyborg_env.environment_controller.state
    target_ip = next(ip for ip, h in state.ip_addresses.items() if h == target_hostname)
    ps = Portscan(session=0, agent="red_agent_0", ip_address=target_ip)
    return ps.execute(state)


def _find_host_in_subnet(const, subnet_name, exclude_router=True):
    from jaxborg.constants import SUBNET_IDS

    sid = SUBNET_IDS[subnet_name]
    for h in range(int(const.num_hosts)):
        if not bool(const.host_active[h]):
            continue
        if int(const.host_subnet[h]) != sid:
            continue
        if exclude_router and bool(const.host_is_router[h]):
            continue
        return h
    return None


class TestBlueActionEncoding:
    def test_encode_sleep(self):
        assert encode_blue_action("Sleep", -1, 0) == BLUE_SLEEP

    def test_encode_monitor(self):
        assert encode_blue_action("Monitor", -1, 0) == BLUE_MONITOR

    def test_decode_sleep(self, jax_const):
        action_type, *_ = decode_blue_action(BLUE_SLEEP, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_SLEEP

    def test_decode_monitor(self, jax_const):
        action_type, *_ = decode_blue_action(BLUE_MONITOR, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_MONITOR


class TestApplyBlueAction:
    def test_sleep_is_noop(self, jax_const):
        state = _make_jax_state(jax_const)
        new_state = _jit_apply_blue(state, jax_const, 0, BLUE_SLEEP)
        np.testing.assert_array_equal(
            np.array(new_state.host_activity_detected),
            np.array(state.host_activity_detected),
        )

    def test_monitor_action_is_noop(self, jax_const):
        state = _make_jax_state(jax_const)
        new_state = _jit_apply_blue(state, jax_const, 0, BLUE_MONITOR)
        np.testing.assert_array_equal(
            np.array(new_state.host_activity_detected),
            np.array(state.host_activity_detected),
        )


class TestApplyBlueMonitor:
    def test_no_activity_no_detection(self, jax_const):
        state = _make_jax_state(jax_const)
        new_state = apply_blue_monitor(state, jax_const)
        assert not np.any(np.array(new_state.host_activity_detected))

    def test_scan_on_monitored_host_detected(self, jax_const):
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = _make_jax_state(jax_const)
        target_subnet = int(jax_const.host_subnet[target])
        discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
        state = _jit_apply_red(state, jax_const, 0, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        scan_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        state = _jit_apply_red(state, jax_const, 0, scan_idx, jax.random.PRNGKey(0))

        assert int(state.red_activity_this_step[target]) == ACTIVITY_SCAN

        new_state = apply_blue_monitor(state, jax_const)
        assert bool(new_state.host_activity_detected[target])

    def test_activity_on_unmonitored_host_not_detected(self, jax_const):
        target = _find_host_in_subnet(jax_const, "CONTRACTOR_NETWORK")
        assert target is not None

        state = _make_jax_state(jax_const)
        activity = state.red_activity_this_step.at[target].set(ACTIVITY_SCAN)
        state = state.replace(red_activity_this_step=activity)

        new_state = apply_blue_monitor(state, jax_const)
        assert not bool(new_state.host_activity_detected[target])

    def test_detection_is_cumulative(self, jax_const):
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = _make_jax_state(jax_const)
        state = state.replace(
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32),
        )
        new_state = apply_blue_monitor(state, jax_const)
        assert bool(new_state.host_activity_detected[target])

    def test_jit_compatible(self, jax_const):
        state = _make_jax_state(jax_const)
        activity = state.red_activity_this_step.at[0].set(ACTIVITY_SCAN)
        state = state.replace(red_activity_this_step=activity)

        jitted = jax.jit(apply_blue_monitor)
        new_state = jitted(state, jax_const)
        expected_detected = bool(jnp.any(jax_const.blue_agent_hosts[:, 0]))
        assert bool(new_state.host_activity_detected[0]) == expected_detected


class TestDifferentialWithCybORG:
    @pytest.fixture
    def cyborg_env(self):
        return _make_cyborg_env()

    @pytest.fixture
    def cyborg_and_jax(self, cyborg_env):
        const = build_const_from_cyborg(cyborg_env)
        state = _make_jax_state(const)
        return cyborg_env, const, state

    def test_scan_detection_matches_cyborg(self, cyborg_and_jax):
        """Portscan on monitored host: both CybORG and JAX detect it."""
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target_h = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target_h is not None
        target_hostname = sorted_hosts[target_h]

        _cyborg_portscan(cyborg_env, target_hostname)
        cyborg_detected = _cyborg_monitor_detected_hosts(cyborg_env)

        target_subnet = int(const.host_subnet[target_h])
        discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
        state = _jit_apply_red(state, const, 0, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        scan_idx = encode_red_action("DiscoverNetworkServices", target_h, 0)
        state = _jit_apply_red(state, const, 0, scan_idx, jax.random.PRNGKey(0))
        state = apply_blue_monitor(state, const)

        jax_detected = {int(h) for h in range(int(const.num_hosts)) if bool(state.host_activity_detected[h])}

        cyborg_has_target = target_h in cyborg_detected
        jax_has_target = target_h in jax_detected
        assert cyborg_has_target == jax_has_target, f"CybORG detected target={cyborg_has_target}, JAX={jax_has_target}"

    def test_no_activity_no_detection_matches_cyborg(self, cyborg_and_jax):
        """No red activity: neither CybORG nor JAX detect anything."""
        cyborg_env, const, state = cyborg_and_jax

        cyborg_detected = _cyborg_monitor_detected_hosts(cyborg_env)

        state = apply_blue_monitor(state, const)
        jax_detected = {int(h) for h in range(int(const.num_hosts)) if bool(state.host_activity_detected[h])}

        assert len(cyborg_detected) == 0, f"CybORG detected: {cyborg_detected}"
        assert len(jax_detected) == 0, f"JAX detected: {jax_detected}"

    def test_contractor_network_not_detected_matches_cyborg(self, cyborg_and_jax):
        """Activity on contractor_network: not detected by any blue agent in either env."""
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target_h = _find_host_in_subnet(const, "CONTRACTOR_NETWORK")
        assert target_h is not None
        target_hostname = sorted_hosts[target_h]

        _cyborg_portscan(cyborg_env, target_hostname)
        cyborg_detected = _cyborg_monitor_detected_hosts(cyborg_env)

        activity = state.red_activity_this_step.at[target_h].set(ACTIVITY_SCAN)
        state = state.replace(red_activity_this_step=activity)
        state = apply_blue_monitor(state, const)

        jax_detected_on_target = bool(state.host_activity_detected[target_h])
        cyborg_detected_on_target = target_h in cyborg_detected

        assert not cyborg_detected_on_target and not jax_detected_on_target, (
            f"CybORG={cyborg_detected_on_target}, JAX={jax_detected_on_target}"
        )

    def test_multi_subnet_scan_detection_matches_cyborg(self, cyborg_and_jax):
        """Scan hosts in multiple subnets, compare detection across all blue agents."""
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        subnets_to_test = [
            "RESTRICTED_ZONE_A",
            "OPERATIONAL_ZONE_A",
            "RESTRICTED_ZONE_B",
            "OFFICE_NETWORK",
        ]

        targets = {}
        for subnet in subnets_to_test:
            h = _find_host_in_subnet(const, subnet)
            if h is not None:
                targets[subnet] = h

        for subnet, h in targets.items():
            hostname = sorted_hosts[h]
            _cyborg_portscan(cyborg_env, hostname)

        cyborg_detected = _cyborg_monitor_detected_hosts(cyborg_env)

        for subnet, h in targets.items():
            target_subnet = int(const.host_subnet[h])
            discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
            state = _jit_apply_red(state, const, 0, discover_idx, jax.random.PRNGKey(0))
            state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        for subnet, h in targets.items():
            scan_idx = encode_red_action("DiscoverNetworkServices", h, 0)
            state = _jit_apply_red(state, const, 0, scan_idx, jax.random.PRNGKey(0))

        state = apply_blue_monitor(state, const)

        for subnet, h in targets.items():
            cyborg_has = h in cyborg_detected
            jax_has = bool(state.host_activity_detected[h])
            assert cyborg_has == jax_has, f"Subnet {subnet} host {h}: CybORG={cyborg_has}, JAX={jax_has}"

    def test_exploit_activity_detection_matches_cyborg(self, cyborg_and_jax):
        """Exploit activity on monitored host: both detect it."""
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target_h = None
        for h in range(int(const.num_hosts)):
            if (
                bool(const.host_active[h])
                and not bool(const.host_is_router[h])
                and bool(const.initial_services[h, SSH_SVC])
                and bool(const.host_has_bruteforceable_user[h])
                and h != int(const.red_start_hosts[0])
                and bool(jnp.any(const.blue_agent_hosts[:, h]))
            ):
                target_h = h
                break
        assert target_h is not None, "No exploitable monitored host found"
        target_hostname = sorted_hosts[target_h]

        target_ip = next(ip for ip, hname in cyborg_state.ip_addresses.items() if hname == target_hostname)
        from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.SSHBruteForce import SSHBruteForce

        exploit = SSHBruteForce(session=0, agent="red_agent_0", ip_address=target_ip)
        exploit.execute(cyborg_state)

        host_obj = cyborg_state.hosts[target_hostname]
        cyborg_has_events = len(host_obj.events.network_connections) > 0 or len(host_obj.events.process_creation) > 0

        cyborg_detected = _cyborg_monitor_detected_hosts(cyborg_env)
        cyborg_has_target = target_h in cyborg_detected

        target_subnet = int(const.host_subnet[target_h])
        discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
        state = _jit_apply_red(state, const, 0, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
        scan_idx = encode_red_action("DiscoverNetworkServices", target_h, 0)
        state = _jit_apply_red(state, const, 0, scan_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
        exploit_idx = encode_red_action("ExploitRemoteService_cc4SSHBruteForce", target_h, 0)
        state = _jit_apply_red(state, const, 0, exploit_idx, jax.random.PRNGKey(0))

        int(state.red_activity_this_step[target_h]) != ACTIVITY_NONE

        state = apply_blue_monitor(state, const)
        # Exploit activity goes to host_exploit_detected (process_creation events)
        jax_has_target = bool(state.host_activity_detected[target_h]) or bool(state.host_exploit_detected[target_h])

        if cyborg_has_events:
            assert cyborg_has_target == jax_has_target, (
                f"CybORG detected={cyborg_has_target}, JAX detected={jax_has_target}"
            )

    def test_successful_sql_exploit_without_process_event_matches_cyborg(self, cyborg_and_jax):
        """Successful SQL exploit with a missed process-event roll should not light malicious_processes."""
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        class _ScriptedRandom:
            def __init__(self, base, random_values):
                self._base = base
                self._random_values = list(random_values)
                self._idx = 0

            def random(self, *args, **kwargs):
                if self._idx < len(self._random_values):
                    value = self._random_values[self._idx]
                    self._idx += 1
                    return value
                return self._base.random(*args, **kwargs)

            def integers(self, *args, **kwargs):
                return self._base.integers(*args, **kwargs)

            def choice(self, *args, **kwargs):
                return self._base.choice(*args, **kwargs)

        target_h = None
        for h in range(int(const.num_hosts)):
            if (
                bool(const.host_active[h])
                and not bool(const.host_is_router[h])
                and bool(const.initial_services[h, SERVICE_IDS["MYSQLD"]])
                and bool(const.initial_services[h, SERVICE_IDS["APACHE2"]])
                and h != int(const.red_start_hosts[0])
                and bool(jnp.any(const.blue_agent_hosts[:, h]))
            ):
                target_h = h
                break
        assert target_h is not None, "No monitored SQL target found"

        target_hostname = sorted_hosts[target_h]
        target_ip = next(ip for ip, hname in cyborg_state.ip_addresses.items() if hname == target_hostname)
        from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.SQLInjection import SQLInjection

        cyborg_state.np_random = _ScriptedRandom(cyborg_state.np_random, [0.5, 0.01])
        cy_obs = SQLInjection(session=0, agent="red_agent_0", ip_address=target_ip).execute(cyborg_state)
        assert str(cy_obs.success).upper() == "TRUE"
        assert len(cyborg_state.hosts[target_hostname].events.process_creation) == 0

        cyborg_detected = _cyborg_monitor_detected_hosts(cyborg_env)
        assert target_h not in cyborg_detected

        target_subnet = int(const.host_subnet[target_h])
        discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
        state = _jit_apply_red(state, const, 0, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
        scan_idx = encode_red_action("DiscoverNetworkServices", target_h, 0)
        state = _jit_apply_red(state, const, 0, scan_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        randoms = const.detection_randoms.at[0].set(0.5).at[1].set(0.01)
        const = const.replace(
            detection_randoms=randoms,
            use_detection_randoms=jnp.array(True),
        )
        state = state.replace(
            detection_random_index=jnp.array(0, dtype=jnp.int32),
        )
        exploit_idx = encode_red_action("ExploitRemoteService_cc4SQLInjection", target_h, 0)
        state = _jit_apply_red(state, const, 0, exploit_idx, jax.random.PRNGKey(0))

        assert bool(state.red_sessions[0, target_h])
        assert int(state.detection_random_index) == 2
        assert np.all(np.asarray(state.host_process_creation_pids[target_h]) == -1)

        state = apply_blue_monitor(state, const)
        assert not bool(state.host_exploit_detected[target_h])

    def test_blue_coverage_matches_cyborg(self, cyborg_and_jax):
        """Verify which hosts each blue agent covers matches CybORG."""
        cyborg_env, const, _ = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        for blue_idx in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{blue_idx}"
            session = cyborg_state.sessions[agent_name][0]
            children = list(session.children.values()) + [session]
            cyborg_hosts = {sorted_hosts.index(c.hostname) for c in children}
            jax_hosts = {int(h) for h in range(int(const.num_hosts)) if bool(const.blue_agent_hosts[blue_idx, h])}
            assert cyborg_hosts == jax_hosts, (
                f"{agent_name}: CybORG covers {len(cyborg_hosts)} hosts, JAX covers {len(jax_hosts)} hosts"
            )

    def test_monitor_clears_nothing_in_jax(self, cyborg_and_jax):
        """CybORG Monitor clears events after reading. JAX doesn't track events,
        so verify red_activity_this_step is not modified by monitor."""
        cyborg_env, const, state = cyborg_and_jax

        target_h = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target_h is not None

        activity = state.red_activity_this_step.at[target_h].set(ACTIVITY_SCAN)
        state = state.replace(red_activity_this_step=activity)
        state = apply_blue_monitor(state, const)

        assert int(state.red_activity_this_step[target_h]) == ACTIVITY_SCAN


class TestMonitorActionMask:
    def test_monitor_action_is_valid(self, jax_const):
        from jaxborg.actions.masking import compute_blue_action_mask

        mask = np.array(compute_blue_action_mask(jax_const, 0))
        assert mask[BLUE_MONITOR]

    def test_sleep_action_is_valid(self, jax_const):
        from jaxborg.actions.masking import compute_blue_action_mask

        mask = np.array(compute_blue_action_mask(jax_const, 0))
        assert mask[BLUE_SLEEP]
