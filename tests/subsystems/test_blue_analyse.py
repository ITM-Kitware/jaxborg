import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper
from CybORG.Simulator.Actions.AbstractActions.Analyse import Analyse
from CybORG.Simulator.Actions.AbstractActions.Monitor import Monitor
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.blue_analyse import apply_blue_analyse
from jaxborg.actions.encoding import (
    BLUE_ACTION_TYPE_ANALYSE,
    decode_blue_action,
    encode_blue_action,
    encode_red_action,
)
from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    SERVICE_IDS,
)
from jaxborg.observations import get_blue_obs
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg

_jit_apply_red = jax.jit(apply_red_action, static_argnums=(2,))
_jit_apply_blue = jax.jit(apply_blue_action, static_argnums=(2,))

SSH_SVC = SERVICE_IDS["SSHD"]


def _make_cyborg_env():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)


@pytest.fixture
def jax_const():
    return build_const_from_cyborg(_make_cyborg_env())


def _make_jax_state(const):
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    start_host = int(const.red_start_hosts[0])
    red_sessions = state.red_sessions.at[0, start_host].set(True)
    return state.replace(red_sessions=red_sessions)


def _clear_transient_obs(state, const):
    any_covered = jnp.any(const.blue_agent_hosts, axis=0)
    return state.replace(
        red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32),
        host_activity_detected=jnp.where(any_covered, False, state.host_activity_detected),
        host_exploit_detected=jnp.where(any_covered, False, state.host_exploit_detected),
    )


def _assert_activity_unchanged(before_state, after_state):
    np.testing.assert_array_equal(
        np.array(after_state.host_activity_detected),
        np.array(before_state.host_activity_detected),
    )


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


def _find_exploitable_monitored_host(const):
    for h in range(int(const.num_hosts)):
        if (
            bool(const.host_active[h])
            and not bool(const.host_is_router[h])
            and bool(const.initial_services[h, SSH_SVC])
            and bool(const.host_has_bruteforceable_user[h])
            and h != int(const.red_start_hosts[0])
            and bool(jnp.any(const.blue_agent_hosts[:, h]))
        ):
            return h
    return None


class TestBlueAnalyseEncoding:
    def test_encode_analyse(self, jax_const):
        action_idx = encode_blue_action("Analyse", 5, 0, const=jax_const)
        action_type, target_host, *_ = decode_blue_action(action_idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_ANALYSE
        assert int(target_host) == 5

    def test_decode_analyse(self, jax_const):
        action_idx = encode_blue_action("Analyse", 5, 0, const=jax_const)
        action_type, target_host, *_ = decode_blue_action(action_idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_ANALYSE
        assert int(target_host) == 5

    def test_roundtrip(self, jax_const):
        for h in range(min(int(jax_const.num_hosts), 20)):
            if not bool(jax_const.host_active[h]) or bool(jax_const.host_is_router[h]):
                continue
            action_idx = encode_blue_action("Analyse", h, 0, const=jax_const)
            action_type, target_host, *_ = decode_blue_action(action_idx, 0, jax_const)
            assert int(action_type) == BLUE_ACTION_TYPE_ANALYSE
            assert int(target_host) == h


class TestApplyBlueAnalyse:
    def test_no_malware_is_noop(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None
        new_state = apply_blue_analyse(state, jax_const, 0, target)
        _assert_activity_unchanged(state, new_state)

    def test_malware_on_covered_host_is_still_noop(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(host_has_malware=state.host_has_malware.at[target].set(True))

        blue_idx = None
        for b in range(NUM_BLUE_AGENTS):
            if bool(jax_const.blue_agent_hosts[b, target]):
                blue_idx = b
                break
        assert blue_idx is not None

        new_state = apply_blue_analyse(state, jax_const, blue_idx, target)
        _assert_activity_unchanged(state, new_state)

    def test_malware_on_uncovered_host_is_still_noop(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(host_has_malware=state.host_has_malware.at[target].set(True))

        uncovering_blue = None
        for b in range(NUM_BLUE_AGENTS):
            if not bool(jax_const.blue_agent_hosts[b, target]):
                uncovering_blue = b
                break
        assert uncovering_blue is not None

        new_state = apply_blue_analyse(state, jax_const, uncovering_blue, target)
        _assert_activity_unchanged(state, new_state)

    def test_noop_preserves_existing_detections(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        other = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        assert target is not None and other is not None

        state = state.replace(
            host_activity_detected=state.host_activity_detected.at[other].set(True),
            host_has_malware=state.host_has_malware.at[target].set(True),
        )

        blue_idx = None
        for b in range(NUM_BLUE_AGENTS):
            if bool(jax_const.blue_agent_hosts[b, target]):
                blue_idx = b
                break
        assert blue_idx is not None

        new_state = apply_blue_analyse(state, jax_const, blue_idx, target)
        _assert_activity_unchanged(state, new_state)

    def test_jit_compatible(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(host_has_malware=state.host_has_malware.at[target].set(True))

        blue_idx = None
        for b in range(NUM_BLUE_AGENTS):
            if bool(jax_const.blue_agent_hosts[b, target]):
                blue_idx = b
                break
        assert blue_idx is not None

        jitted = jax.jit(apply_blue_analyse, static_argnums=(2, 3))
        new_state = jitted(state, jax_const, blue_idx, target)
        _assert_activity_unchanged(state, new_state)


class TestApplyBlueActionDispatch:
    def test_analyse_via_dispatch(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(host_has_malware=state.host_has_malware.at[target].set(True))

        blue_idx = None
        for b in range(NUM_BLUE_AGENTS):
            if bool(jax_const.blue_agent_hosts[b, target]):
                blue_idx = b
                break
        assert blue_idx is not None

        action_idx = encode_blue_action("Analyse", target, blue_idx, const=jax_const)
        new_state = _jit_apply_blue(state, jax_const, blue_idx, action_idx)
        _assert_activity_unchanged(state, new_state)

    def test_sleep_still_noop(self, jax_const):
        state = _make_jax_state(jax_const)
        new_state = _jit_apply_blue(state, jax_const, 0, 0)
        np.testing.assert_array_equal(
            np.array(new_state.host_activity_detected),
            np.array(state.host_activity_detected),
        )


class TestDifferentialWithCybORG:
    @pytest.fixture
    def cyborg_env(self):
        return _make_cyborg_env()

    @pytest.fixture
    def cyborg_and_jax(self, cyborg_env):
        const = build_const_from_cyborg(cyborg_env)
        state = _make_jax_state(const)
        return cyborg_env, const, state

    def test_analyse_clean_host_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        wrapper = BlueFlatWrapper(cyborg_env, pad_spaces=True)
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target_h = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target_h is not None
        target_hostname = sorted_hosts[target_h]

        blue_idx = None
        for b in range(NUM_BLUE_AGENTS):
            if bool(const.blue_agent_hosts[b, target_h]):
                blue_idx = b
                break
        assert blue_idx is not None
        agent_name = f"blue_agent_{blue_idx}"
        cyborg_before = wrapper.observation_change(agent_name, cyborg_env.get_observation(agent_name))
        Analyse(session=0, agent=agent_name, hostname=target_hostname).execute(cyborg_state)
        cyborg_after = wrapper.observation_change(agent_name, cyborg_env.get_observation(agent_name))

        new_state = apply_blue_analyse(state, const, blue_idx, target_h)
        jax_before = np.asarray(get_blue_obs(state, const, blue_idx))
        jax_after = np.asarray(get_blue_obs(new_state, const, blue_idx))

        np.testing.assert_array_equal(cyborg_after, cyborg_before)
        np.testing.assert_array_equal(jax_after, jax_before)

    def test_analyse_after_exploit_does_not_change_flat_obs_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        wrapper = BlueFlatWrapper(cyborg_env, pad_spaces=True)
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target_h = _find_exploitable_monitored_host(const)
        assert target_h is not None
        target_hostname = sorted_hosts[target_h]

        target_ip = next(ip for ip, hname in cyborg_state.ip_addresses.items() if hname == target_hostname)
        from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.SSHBruteForce import SSHBruteForce

        exploit = SSHBruteForce(session=0, agent="red_agent_0", ip_address=target_ip)
        exploit.execute(cyborg_state)
        for b in range(NUM_BLUE_AGENTS):
            Monitor(session=0, agent=f"blue_agent_{b}").execute(cyborg_state)

        target_subnet = int(const.host_subnet[target_h])
        discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
        state = _jit_apply_red(state, const, 0, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
        scan_idx = encode_red_action("DiscoverNetworkServices", target_h, 0)
        state = _jit_apply_red(state, const, 0, scan_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
        exploit_idx = encode_red_action("ExploitRemoteService_cc4SSHBruteForce", target_h, 0)
        state = _jit_apply_red(state, const, 0, exploit_idx, jax.random.PRNGKey(0))
        state = _clear_transient_obs(state, const)

        blue_idx = None
        for b in range(NUM_BLUE_AGENTS):
            if bool(const.blue_agent_hosts[b, target_h]):
                blue_idx = b
                break
        assert blue_idx is not None
        agent_name = f"blue_agent_{blue_idx}"
        cyborg_before = wrapper.observation_change(agent_name, cyborg_env.get_observation(agent_name))
        Analyse(session=0, agent=agent_name, hostname=target_hostname).execute(cyborg_state)
        cyborg_after = wrapper.observation_change(agent_name, cyborg_env.get_observation(agent_name))

        new_state = apply_blue_analyse(state, const, blue_idx, target_h)
        jax_before = np.asarray(get_blue_obs(state, const, blue_idx))
        jax_after = np.asarray(get_blue_obs(new_state, const, blue_idx))

        np.testing.assert_array_equal(cyborg_after, cyborg_before)
        np.testing.assert_array_equal(jax_after, jax_before)
