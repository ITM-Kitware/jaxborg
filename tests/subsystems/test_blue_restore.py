import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Shared.Session import RedAbstractSession
from CybORG.Simulator.Actions import Restore
from CybORG.Simulator.Actions.ConcreteActions.RedSessionCheck import RedSessionCheck
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action
from jaxborg.actions.blue_restore import apply_blue_restore
from jaxborg.actions.encoding import (
    BLUE_ACTION_TYPE_RESTORE,
    decode_blue_action,
    encode_blue_action,
)
from jaxborg.constants import (
    COMPROMISE_NONE,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    NUM_BLUE_AGENTS,
    NUM_DECOY_TYPES,
)
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg
from jaxborg.state import create_initial_state

_jit_apply_blue = jax.jit(apply_blue_action, static_argnums=(2,))


def _make_cyborg_env():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)


@pytest.fixture(scope="module")
def jax_const():
    return build_const_from_cyborg(_make_cyborg_env())


def _make_jax_state(const):
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    return state


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


def _find_blue_for_host(const, host):
    for b in range(NUM_BLUE_AGENTS):
        if bool(const.blue_agent_hosts[b, host]):
            return b
    return None


class TestBlueRestoreEncoding:
    def _agent0_host(self, jax_const):
        """Find a non-router host in agent 0's observed subnets."""
        active = np.array(jax_const.host_active, dtype=bool)
        ctrl = (
            np.array(jax_const.blue_agent_hosts[0], dtype=bool)
            & active
            & ~np.array(jax_const.host_is_router, dtype=bool)
        )
        return int(np.flatnonzero(ctrl)[0])

    def test_encode_restore(self, jax_const):
        h = self._agent0_host(jax_const)
        action_idx = encode_blue_action("Restore", h, 0, const=jax_const)
        action_type, target_host, *_ = decode_blue_action(action_idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_RESTORE
        assert int(target_host) == h

    def test_decode_restore(self, jax_const):
        h = self._agent0_host(jax_const)
        action_idx = encode_blue_action("Restore", h, 0, const=jax_const)
        action_type, target_host, *_ = decode_blue_action(action_idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_RESTORE
        assert int(target_host) == h


class TestApplyBlueRestore:
    def test_restore_clears_all_red_sessions(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True).at[1, target].set(True),
            red_privilege=state.red_privilege.at[0, target]
            .set(COMPROMISE_USER)
            .at[1, target]
            .set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
            host_has_malware=state.host_has_malware.at[target].set(True),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert not bool(new_state.red_sessions[1, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE
        assert int(new_state.red_privilege[1, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE
        assert not bool(new_state.host_has_malware[target])

    def test_restore_clears_decoys(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(host_decoys=state.host_decoys.at[target].set(jnp.ones(NUM_DECOY_TYPES, dtype=jnp.bool_)))

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        assert not np.any(np.array(new_state.host_decoys[target]))

    def test_restore_resets_services_to_initial(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(host_services=state.host_services.at[target].set(False))

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        np.testing.assert_array_equal(
            np.array(new_state.host_services[target]),
            np.array(jax_const.initial_services[target]),
        )

    def test_restore_clears_activity_detected(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(host_activity_detected=state.host_activity_detected.at[target].set(True))

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        assert not bool(new_state.host_activity_detected[target])

    def test_restore_clears_ot_service_stopped(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(ot_service_stopped=state.ot_service_stopped.at[target].set(True))

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        assert not bool(new_state.ot_service_stopped[target])

    def test_restore_does_not_affect_other_hosts(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        other = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        assert target is not None and other is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True).at[0, other].set(True),
            red_privilege=state.red_privilege.at[0, target]
            .set(COMPROMISE_PRIVILEGED)
            .at[0, other]
            .set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target]
            .set(COMPROMISE_PRIVILEGED)
            .at[other]
            .set(COMPROMISE_USER),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert bool(new_state.red_sessions[0, other])
        assert int(new_state.red_privilege[0, other]) == COMPROMISE_USER

    def test_jit_compatible(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        jitted = jax.jit(apply_blue_restore, static_argnums=(2, 3))
        new_state = jitted(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE


class TestRestoreViaDispatch:
    def test_restore_dispatched(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
            host_has_malware=state.host_has_malware.at[target].set(True),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        action_idx = encode_blue_action("Restore", target, blue_idx, const=jax_const)
        new_state = _jit_apply_blue(state, jax_const, blue_idx, action_idx)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE
        assert not bool(new_state.host_has_malware[target])


class TestDifferentialWithCybORG:
    @pytest.fixture
    def cyborg_and_jax(self):
        cyborg_env = _make_cyborg_env()
        const = build_const_from_cyborg(cyborg_env)
        state = _make_jax_state(const)
        return cyborg_env, const, state

    def test_restore_clears_scan_memory_when_restored_host_session_is_removed_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        target = _find_host_in_subnet(const, "OPERATIONAL_ZONE_A")
        other = _find_host_in_subnet(const, "OPERATIONAL_ZONE_B")
        assert target is not None and other is not None

        target_hostname = sorted_hosts[target]
        other_hostname = sorted_hosts[other]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_target = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_2",
            parent=0,
            session_type="shell",
            pid=None,
        )
        red_other = RedAbstractSession(
            ident=None,
            hostname=other_hostname,
            username="user",
            agent="red_agent_2",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cy_state.add_session(red_target)
        cy_state.add_session(red_other)

        target_session = next(
            sess for sess in cy_state.sessions["red_agent_2"].values() if sess.hostname == target_hostname
        )
        target_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == target_hostname)
        other_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == other_hostname)
        target_session.addport(target_ip, 22)
        target_session.addport(other_ip, 443)

        state = state.replace(
            red_sessions=state.red_sessions.at[2, target].set(True).at[2, other].set(True),
            red_session_count=state.red_session_count.at[2, target].set(1).at[2, other].set(1),
            red_privilege=state.red_privilege.at[2, target].set(COMPROMISE_USER).at[2, other].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[2, target].set(True).at[2, other].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[2, target, target]
            .set(True)
            .at[2, other, target]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[2].set(target),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER).at[other].set(COMPROMISE_USER),
        )

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        action_idx = encode_blue_action("Restore", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_2"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[2, h])}
        assert cy_scanned == set()
        assert jax_scanned == cy_scanned

    def test_restore_clears_scan_memory_for_non_anchor_stale_session_with_activity_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        target = _find_host_in_subnet(const, "OPERATIONAL_ZONE_A")
        anchor = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None and anchor is not None and target != anchor

        target_hostname = sorted_hosts[target]
        anchor_hostname = sorted_hosts[anchor]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_target = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_2",
            parent=0,
            session_type="shell",
            pid=None,
        )
        red_anchor = RedAbstractSession(
            ident=None,
            hostname=anchor_hostname,
            username="user",
            agent="red_agent_2",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cy_state.add_session(red_target)
        cy_state.add_session(red_anchor)

        target_session = next(
            sess for sess in cy_state.sessions["red_agent_2"].values() if sess.hostname == target_hostname
        )
        target_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == target_hostname)
        target_session.addport(target_ip, 22)

        state = state.replace(
            red_sessions=state.red_sessions.at[2, target].set(True).at[2, anchor].set(True),
            red_session_count=state.red_session_count.at[2, target].set(1).at[2, anchor].set(1),
            red_privilege=state.red_privilege.at[2, target].set(COMPROMISE_USER).at[2, anchor].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[2, target].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[2, target, target].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[2].set(anchor),
            red_suspicious_process_count=state.red_suspicious_process_count.at[2, anchor].set(1).at[2, target].set(0),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER).at[anchor].set(COMPROMISE_USER),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
        )

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        action_idx = encode_blue_action("Restore", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_2"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[2, h])}
        assert cy_scanned == set()
        assert jax_scanned == cy_scanned

    def test_restore_clears_scan_memory_for_non_unique_stale_target_with_remote_scan_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        subnet_hosts = []
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
                continue
            if int(const.host_subnet[h]) != int(const.host_subnet[_find_host_in_subnet(const, "OPERATIONAL_ZONE_A")]):
                continue
            subnet_hosts.append(h)
            if len(subnet_hosts) == 2:
                break
        assert len(subnet_hosts) == 2
        target, keep = subnet_hosts
        remote = _find_host_in_subnet(const, "OPERATIONAL_ZONE_B")
        assert remote is not None and remote not in {target, keep}

        target_hostname = sorted_hosts[target]
        keep_hostname = sorted_hosts[keep]
        remote_hostname = sorted_hosts[remote]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_target = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_2",
            parent=0,
            session_type="shell",
            pid=None,
        )
        red_keep = RedAbstractSession(
            ident=None,
            hostname=keep_hostname,
            username="user",
            agent="red_agent_2",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cy_state.add_session(red_target)
        cy_state.add_session(red_keep)

        target_session = next(
            sess for sess in cy_state.sessions["red_agent_2"].values() if sess.hostname == target_hostname
        )
        target_session.addport(next(ip for ip, host in cy_state.ip_addresses.items() if host == remote_hostname), 22)
        target_session.addport(next(ip for ip, host in cy_state.ip_addresses.items() if host == keep_hostname), 443)

        state = state.replace(
            red_sessions=state.red_sessions.at[2, target].set(True).at[2, keep].set(True),
            red_session_count=state.red_session_count.at[2, target].set(1).at[2, keep].set(1),
            red_privilege=state.red_privilege.at[2, target].set(COMPROMISE_USER).at[2, keep].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[2, remote].set(True).at[2, keep].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[2, remote, target]
            .set(True)
            .at[2, keep, target]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[2].set(target),
            red_suspicious_process_count=state.red_suspicious_process_count.at[2, target].set(0).at[2, keep].set(0),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER).at[keep].set(COMPROMISE_USER),
        )

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        action_idx = encode_blue_action("Restore", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_2"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[2, h])}
        assert cy_scanned == set()
        assert jax_scanned == cy_scanned

    def test_restore_does_not_overclear_scan_memory_when_target_is_scanned_but_owner_session_remains_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        keep = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        remote = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        assert target is not None and keep is not None and remote is not None
        if keep == target:
            for h in range(int(const.num_hosts)):
                if (
                    bool(const.host_active[h])
                    and not bool(const.host_is_router[h])
                    and int(const.host_subnet[h]) == int(const.host_subnet[target])
                    and h != target
                ):
                    keep = h
                    break
        assert keep != target

        target_hostname = sorted_hosts[target]
        keep_hostname = sorted_hosts[keep]
        remote_hostname = sorted_hosts[remote]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_target = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_1",
            parent=0,
            session_type="shell",
            pid=None,
        )
        red_keep = RedAbstractSession(
            ident=None,
            hostname=keep_hostname,
            username="user",
            agent="red_agent_1",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cy_state.add_session(red_target)
        cy_state.add_session(red_keep)

        keep_session = next(
            sess for sess in cy_state.sessions["red_agent_1"].values() if sess.hostname == keep_hostname
        )
        keep_session.addport(next(ip for ip, host in cy_state.ip_addresses.items() if host == target_hostname), 22)
        keep_session.addport(next(ip for ip, host in cy_state.ip_addresses.items() if host == remote_hostname), 443)

        state = state.replace(
            red_sessions=state.red_sessions.at[1, target].set(True).at[1, keep].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[1, target].set(True).at[1, keep].set(True),
            red_privilege=state.red_privilege.at[1, target].set(COMPROMISE_USER).at[1, keep].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[1, target].set(True).at[1, remote].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[1, target, keep]
            .set(True)
            .at[1, remote, keep]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[1].set(keep),
            red_suspicious_process_count=state.red_suspicious_process_count.at[1, target].set(0).at[1, keep].set(0),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER).at[keep].set(COMPROMISE_USER),
        )

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        action_idx = encode_blue_action("Restore", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_1"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[1, h])}
        assert cy_scanned == {target, remote}
        assert jax_scanned == cy_scanned

    def test_restore_does_not_overclear_scan_memory_when_target_not_scanned_and_owner_session_remains_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        target = _find_host_in_subnet(const, "OPERATIONAL_ZONE_A")
        keep = target
        for h in range(int(const.num_hosts)):
            if (
                bool(const.host_active[h])
                and not bool(const.host_is_router[h])
                and int(const.host_subnet[h]) == int(const.host_subnet[target])
                and h != target
            ):
                keep = h
                break
        remote = _find_host_in_subnet(const, "OPERATIONAL_ZONE_B")
        assert target is not None and keep is not None and keep != target and remote is not None

        target_hostname = sorted_hosts[target]
        keep_hostname = sorted_hosts[keep]
        remote_hostname = sorted_hosts[remote]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_target = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_2",
            parent=0,
            session_type="shell",
            pid=None,
        )
        red_keep = RedAbstractSession(
            ident=None,
            hostname=keep_hostname,
            username="user",
            agent="red_agent_2",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cy_state.add_session(red_target)
        cy_state.add_session(red_keep)

        keep_session = next(
            sess for sess in cy_state.sessions["red_agent_2"].values() if sess.hostname == keep_hostname
        )
        keep_session.addport(next(ip for ip, host in cy_state.ip_addresses.items() if host == keep_hostname), 22)
        keep_session.addport(next(ip for ip, host in cy_state.ip_addresses.items() if host == remote_hostname), 443)

        state = state.replace(
            red_sessions=state.red_sessions.at[2, target].set(True).at[2, keep].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[2, target].set(True).at[2, keep].set(True),
            red_privilege=state.red_privilege.at[2, target].set(COMPROMISE_USER).at[2, keep].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[2, keep].set(True).at[2, remote].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[2, keep, keep]
            .set(True)
            .at[2, remote, keep]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[2].set(keep),
            red_suspicious_process_count=state.red_suspicious_process_count.at[2, target].set(0).at[2, keep].set(0),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER).at[keep].set(COMPROMISE_USER),
        )

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        action_idx = encode_blue_action("Restore", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_2"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[2, h])}
        assert cy_scanned == {keep, remote}
        assert jax_scanned == cy_scanned

    def test_restore_clears_scan_memory_after_primary_session_reassignment_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        base = _find_host_in_subnet(const, "OPERATIONAL_ZONE_A")
        assert base is not None
        subnet_id = int(const.host_subnet[base])
        subnet_hosts = []
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
                continue
            if int(const.host_subnet[h]) != subnet_id:
                continue
            subnet_hosts.append(h)
        assert len(subnet_hosts) >= 5
        target, keep_1, keep_2, keep_3, promoted = subnet_hosts[:5]

        target_hostname = sorted_hosts[target]
        keep_1_hostname = sorted_hosts[keep_1]
        keep_2_hostname = sorted_hosts[keep_2]
        keep_3_hostname = sorted_hosts[keep_3]
        promoted_hostname = sorted_hosts[promoted]

        blue_target = _find_blue_for_host(const, target)
        assert blue_target is not None

        for ident, hostname in (
            (0, target_hostname),
            (1, keep_1_hostname),
            (3, keep_2_hostname),
            (4, keep_3_hostname),
            (7, promoted_hostname),
        ):
            cy_state.add_session(
                RedAbstractSession(
                    ident=ident,
                    hostname=hostname,
                    username="user",
                    agent="red_agent_2",
                    parent=0 if ident != 0 else None,
                    session_type="shell",
                    pid=None,
                )
            )

        state = state.replace(
            red_sessions=state.red_sessions.at[2, target]
            .set(True)
            .at[2, keep_1]
            .set(True)
            .at[2, keep_2]
            .set(True)
            .at[2, keep_3]
            .set(True)
            .at[2, promoted]
            .set(True),
            red_session_count=state.red_session_count.at[2, target]
            .set(1)
            .at[2, keep_1]
            .set(1)
            .at[2, keep_2]
            .set(1)
            .at[2, keep_3]
            .set(1)
            .at[2, promoted]
            .set(1),
            red_privilege=state.red_privilege.at[2, target]
            .set(COMPROMISE_USER)
            .at[2, keep_1]
            .set(COMPROMISE_USER)
            .at[2, keep_2]
            .set(COMPROMISE_USER)
            .at[2, keep_3]
            .set(COMPROMISE_USER)
            .at[2, promoted]
            .set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target]
            .set(COMPROMISE_USER)
            .at[keep_1]
            .set(COMPROMISE_USER)
            .at[keep_2]
            .set(COMPROMISE_USER)
            .at[keep_3]
            .set(COMPROMISE_USER)
            .at[promoted]
            .set(COMPROMISE_USER),
            red_scan_anchor_host=state.red_scan_anchor_host.at[2].set(target),
        )

        restore_1 = Restore(session=0, agent=f"blue_agent_{blue_target}", hostname=target_hostname)
        restore_1.duration = 1
        cy_obs_1 = restore_1.execute(cy_state)
        assert cy_obs_1.success

        action_1 = encode_blue_action("Restore", target, blue_target, const=const)
        state = _jit_apply_blue(state, const, blue_target, action_1)

        # CybORG promotes a new primary session when id=0 is removed; make this deterministic.
        cy_state.np_random = np.random.default_rng(0)
        RedSessionCheck(session=0, agent="red_agent_2").execute(cy_state)
        promoted_now = sorted_hosts.index(cy_state.sessions["red_agent_2"][0].hostname)
        assert promoted_now == promoted

        target_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == target_hostname)
        keep_1_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == keep_1_hostname)
        cy_state.sessions["red_agent_2"][0].addport(target_ip, 22)
        cy_state.sessions["red_agent_2"][0].addport(keep_1_ip, 443)
        state = state.replace(
            red_scanned_hosts=state.red_scanned_hosts.at[2, target].set(True).at[2, keep_1].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[2, target, promoted]
            .set(True)
            .at[2, keep_1, promoted]
            .set(True),
        )

        blue_promoted = _find_blue_for_host(const, promoted_now)
        assert blue_promoted is not None
        restore_2 = Restore(session=0, agent=f"blue_agent_{blue_promoted}", hostname=sorted_hosts[promoted_now])
        restore_2.duration = 1
        cy_obs_2 = restore_2.execute(cy_state)
        assert cy_obs_2.success

        action_2 = encode_blue_action("Restore", promoted_now, blue_promoted, const=const)
        new_state = _jit_apply_blue(state, const, blue_promoted, action_2)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_2"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[2, h])}
        assert cy_scanned == set()
        assert jax_scanned == cy_scanned

    def test_restore_preserves_scan_memory_when_alternate_abstract_source_still_knows_target_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        restore_host = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        keep_host = restore_host
        for h in range(int(const.num_hosts)):
            if (
                bool(const.host_active[h])
                and not bool(const.host_is_router[h])
                and int(const.host_subnet[h]) == int(const.host_subnet[restore_host])
                and h != restore_host
            ):
                keep_host = h
                break
        scanned_target = _find_host_in_subnet(const, "OPERATIONAL_ZONE_B")
        assert (
            restore_host is not None
            and keep_host is not None
            and keep_host != restore_host
            and scanned_target is not None
        )

        restore_hostname = sorted_hosts[restore_host]
        keep_hostname = sorted_hosts[keep_host]
        scanned_hostname = sorted_hosts[scanned_target]
        scan_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == scanned_hostname)

        blue_idx = _find_blue_for_host(const, restore_host)
        assert blue_idx is not None

        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=restore_hostname,
                username="user",
                agent="red_agent_2",
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        cy_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=keep_hostname,
                username="user",
                agent="red_agent_2",
                parent=0,
                session_type="shell",
                pid=None,
            )
        )

        restore_sess = next(
            sess for sess in cy_state.sessions["red_agent_2"].values() if sess.hostname == restore_hostname
        )
        keep_sess = next(sess for sess in cy_state.sessions["red_agent_2"].values() if sess.hostname == keep_hostname)
        restore_sess.addport(scan_ip, 22)
        keep_sess.addport(scan_ip, 443)

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_idx}", hostname=restore_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_2"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))
        assert scanned_target in cy_scanned

        state = state.replace(
            red_sessions=state.red_sessions.at[2, restore_host].set(True).at[2, keep_host].set(True),
            red_session_count=state.red_session_count.at[2, restore_host].set(1).at[2, keep_host].set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[2, restore_host]
            .set(True)
            .at[2, keep_host]
            .set(True),
            red_privilege=state.red_privilege.at[2, restore_host]
            .set(COMPROMISE_USER)
            .at[2, keep_host]
            .set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[restore_host]
            .set(COMPROMISE_USER)
            .at[keep_host]
            .set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[2, scanned_target].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[2, scanned_target, restore_host]
            .set(True)
            .at[2, scanned_target, keep_host]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[2].set(restore_host),
        )

        action_idx = encode_blue_action("Restore", restore_host, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)
        assert bool(new_state.red_scanned_hosts[2, scanned_target]) == (scanned_target in cy_scanned)

    def test_restore_reassignment_between_two_remaining_sessions_clears_scan_memory_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        base = _find_host_in_subnet(const, "OPERATIONAL_ZONE_A")
        assert base is not None
        subnet_id = int(const.host_subnet[base])
        subnet_hosts = []
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
                continue
            if int(const.host_subnet[h]) != subnet_id:
                continue
            subnet_hosts.append(h)
        assert len(subnet_hosts) >= 4
        promoted, target, keep = subnet_hosts[0], subnet_hosts[2], subnet_hosts[3]

        promoted_hostname = sorted_hosts[promoted]
        target_hostname = sorted_hosts[target]
        keep_hostname = sorted_hosts[keep]

        blue_target = _find_blue_for_host(const, target)
        assert blue_target is not None

        for ident, hostname in ((0, target_hostname), (2, keep_hostname), (3, promoted_hostname)):
            cy_state.add_session(
                RedAbstractSession(
                    ident=ident,
                    hostname=hostname,
                    username="user",
                    agent="red_agent_2",
                    parent=0 if ident != 0 else None,
                    session_type="shell",
                    pid=None,
                )
            )

        state = state.replace(
            red_sessions=state.red_sessions.at[2, promoted].set(True).at[2, target].set(True).at[2, keep].set(True),
            red_session_count=state.red_session_count.at[2, promoted].set(1).at[2, target].set(1).at[2, keep].set(1),
            red_privilege=state.red_privilege.at[2, promoted]
            .set(COMPROMISE_USER)
            .at[2, target]
            .set(COMPROMISE_USER)
            .at[2, keep]
            .set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[promoted]
            .set(COMPROMISE_USER)
            .at[target]
            .set(COMPROMISE_USER)
            .at[keep]
            .set(COMPROMISE_USER),
            red_scan_anchor_host=state.red_scan_anchor_host.at[2].set(target),
        )

        restore_1 = Restore(session=0, agent=f"blue_agent_{blue_target}", hostname=target_hostname)
        restore_1.duration = 1
        cy_obs_1 = restore_1.execute(cy_state)
        assert cy_obs_1.success
        restore_act = encode_blue_action("Restore", target, blue_target, const=const)
        state = _jit_apply_blue(state, const, blue_target, restore_act)

        # Deterministic primary-session promotion to id=0 after removing the original primary.
        cy_state.np_random = np.random.default_rng(0)
        RedSessionCheck(session=0, agent="red_agent_2").execute(cy_state)
        promoted_now = sorted_hosts.index(cy_state.sessions["red_agent_2"][0].hostname)
        assert promoted_now == promoted

        keep_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == keep_hostname)
        cy_state.sessions["red_agent_2"][0].addport(keep_ip, 22)
        state = state.replace(
            red_scanned_hosts=state.red_scanned_hosts.at[2, keep].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[2, keep, promoted].set(True),
        )

        blue_promoted = _find_blue_for_host(const, promoted_now)
        assert blue_promoted is not None
        restore_2 = Restore(session=0, agent=f"blue_agent_{blue_promoted}", hostname=sorted_hosts[promoted_now])
        restore_2.duration = 1
        cy_obs_2 = restore_2.execute(cy_state)
        assert cy_obs_2.success

        new_state = _jit_apply_blue(
            state, const, blue_promoted, encode_blue_action("Restore", promoted_now, blue_promoted, const=const)
        )

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_2"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[2, h])}
        assert cy_scanned == set()
        assert jax_scanned == cy_scanned

    def test_restore_with_stale_anchor_and_single_remote_scan_clears_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        base = _find_host_in_subnet(const, "OPERATIONAL_ZONE_A")
        assert base is not None
        subnet_id = int(const.host_subnet[base])
        subnet_hosts = []
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
                continue
            if int(const.host_subnet[h]) != subnet_id:
                continue
            subnet_hosts.append(h)
        assert len(subnet_hosts) >= 4
        other_a, target, scanned_host, stale_anchor = subnet_hosts[:4]

        other_a_hostname = sorted_hosts[other_a]
        target_hostname = sorted_hosts[target]
        scanned_hostname = sorted_hosts[scanned_host]
        stale_anchor_hostname = sorted_hosts[stale_anchor]

        blue_target = _find_blue_for_host(const, target)
        assert blue_target is not None

        for ident, hostname in (
            (0, target_hostname),
            (1, other_a_hostname),
            (3, stale_anchor_hostname),
            (4, scanned_hostname),
        ):
            cy_state.add_session(
                RedAbstractSession(
                    ident=ident,
                    hostname=hostname,
                    username="user",
                    agent="red_agent_2",
                    parent=0 if ident != 0 else None,
                    session_type="shell",
                    pid=None,
                )
            )

        scanned_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == scanned_hostname)
        cy_state.sessions["red_agent_2"][0].addport(scanned_ip, 22)

        state = state.replace(
            red_sessions=state.red_sessions.at[2, target]
            .set(True)
            .at[2, other_a]
            .set(True)
            .at[2, stale_anchor]
            .set(True)
            .at[2, scanned_host]
            .set(True),
            red_session_count=state.red_session_count.at[2, target]
            .set(1)
            .at[2, other_a]
            .set(1)
            .at[2, stale_anchor]
            .set(1)
            .at[2, scanned_host]
            .set(1),
            red_privilege=state.red_privilege.at[2, target]
            .set(COMPROMISE_USER)
            .at[2, other_a]
            .set(COMPROMISE_USER)
            .at[2, stale_anchor]
            .set(COMPROMISE_USER)
            .at[2, scanned_host]
            .set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target]
            .set(COMPROMISE_USER)
            .at[other_a]
            .set(COMPROMISE_USER)
            .at[stale_anchor]
            .set(COMPROMISE_USER)
            .at[scanned_host]
            .set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[2, scanned_host].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[2, scanned_host, target].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[2].set(stale_anchor),
        )

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_target}", hostname=target_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        restore_act = encode_blue_action("Restore", target, blue_target, const=const)
        new_state = _jit_apply_blue(state, const, blue_target, restore_act)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_2"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[2, h])}
        assert cy_scanned == set()
        assert jax_scanned == cy_scanned

    def test_restore_unique_stale_non_scanned_target_does_not_clear_scan_memory_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        anchor = _find_host_in_subnet(const, "OPERATIONAL_ZONE_B")
        target = _find_host_in_subnet(const, "OPERATIONAL_ZONE_B")
        if anchor == target:
            for h in range(int(const.num_hosts)):
                if (
                    bool(const.host_active[h])
                    and not bool(const.host_is_router[h])
                    and int(const.host_subnet[h]) == int(const.host_subnet[anchor])
                    and h != anchor
                ):
                    target = h
                    break
        remote = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        assert anchor is not None and target is not None and target != anchor and remote is not None

        anchor_hostname = sorted_hosts[anchor]
        target_hostname = sorted_hosts[target]
        remote_hostname = sorted_hosts[remote]

        blue_target = _find_blue_for_host(const, target)
        assert blue_target is not None

        for ident, hostname in ((0, anchor_hostname), (1, anchor_hostname), (2, target_hostname)):
            cy_state.add_session(
                RedAbstractSession(
                    ident=ident,
                    hostname=hostname,
                    username="user",
                    agent="red_agent_3",
                    parent=0 if ident != 0 else None,
                    session_type="shell",
                    pid=None,
                )
            )

        anchor_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == anchor_hostname)
        remote_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == remote_hostname)
        cy_state.sessions["red_agent_3"][0].addport(anchor_ip, 22)
        cy_state.sessions["red_agent_3"][0].addport(remote_ip, 443)

        state = state.replace(
            red_sessions=state.red_sessions.at[3, anchor].set(True).at[3, target].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[3, anchor].set(True).at[3, target].set(True),
            red_session_count=state.red_session_count.at[3, anchor].set(2).at[3, target].set(1),
            red_privilege=state.red_privilege.at[3, anchor].set(COMPROMISE_USER).at[3, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[anchor].set(COMPROMISE_USER).at[target].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[3, anchor].set(True).at[3, remote].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[3, anchor, anchor]
            .set(True)
            .at[3, remote, anchor]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[3].set(anchor),
            red_suspicious_process_count=state.red_suspicious_process_count.at[3, anchor].set(1).at[3, target].set(0),
        )

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_target}", hostname=target_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        restore_act = encode_blue_action("Restore", target, blue_target, const=const)
        new_state = _jit_apply_blue(state, const, blue_target, restore_act)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_3"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[3, h])}
        assert cy_scanned == {anchor, remote}
        assert jax_scanned == cy_scanned

    def test_restore_unscanned_stale_target_with_nonstale_session_keeps_scan_memory_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        subnet_hosts = []
        base = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert base is not None
        subnet_id = int(const.host_subnet[base])
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
                continue
            if int(const.host_subnet[h]) != subnet_id:
                continue
            subnet_hosts.append(h)
            if len(subnet_hosts) == 4:
                break
        assert len(subnet_hosts) == 4
        anchor, target, nonstale_host, scanned_host = subnet_hosts

        anchor_hostname = sorted_hosts[anchor]
        target_hostname = sorted_hosts[target]
        nonstale_hostname = sorted_hosts[nonstale_host]
        scanned_hostname = sorted_hosts[scanned_host]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        for ident, hostname in ((0, anchor_hostname), (1, target_hostname), (2, nonstale_hostname)):
            cy_state.add_session(
                RedAbstractSession(
                    ident=ident,
                    hostname=hostname,
                    username="user",
                    agent="red_agent_1",
                    parent=0 if ident != 0 else None,
                    session_type="shell",
                    pid=None,
                )
            )

        scanned_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == scanned_hostname)
        cy_state.sessions["red_agent_1"][0].addport(scanned_ip, 22)

        state = state.replace(
            red_sessions=state.red_sessions.at[1, anchor]
            .set(True)
            .at[1, target]
            .set(True)
            .at[1, nonstale_host]
            .set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[1, anchor]
            .set(True)
            .at[1, target]
            .set(True)
            .at[1, nonstale_host]
            .set(True),
            red_privilege=state.red_privilege.at[1, anchor]
            .set(COMPROMISE_USER)
            .at[1, target]
            .set(COMPROMISE_USER)
            .at[1, nonstale_host]
            .set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[anchor]
            .set(COMPROMISE_USER)
            .at[target]
            .set(COMPROMISE_USER)
            .at[nonstale_host]
            .set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[1, scanned_host].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[1, scanned_host, anchor].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[1].set(anchor),
            red_suspicious_process_count=state.red_suspicious_process_count.at[1, anchor]
            .set(0)
            .at[1, target]
            .set(0)
            .at[1, nonstale_host]
            .set(1),
        )

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        restore_act = encode_blue_action("Restore", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, restore_act)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_1"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[1, h])}
        assert cy_scanned == {scanned_host}
        assert jax_scanned == cy_scanned

    def test_restore_unique_stale_scanned_target_with_multi_scan_memory_keeps_scans_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        subnet_hosts = []
        base = _find_host_in_subnet(const, "OPERATIONAL_ZONE_A")
        assert base is not None
        subnet_id = int(const.host_subnet[base])
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
                continue
            if int(const.host_subnet[h]) != subnet_id:
                continue
            subnet_hosts.append(h)
            if len(subnet_hosts) == 3:
                break
        assert len(subnet_hosts) == 3
        anchor, target, keep = subnet_hosts

        remote = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        assert remote is not None and remote not in {anchor, target, keep}

        anchor_hostname = sorted_hosts[anchor]
        target_hostname = sorted_hosts[target]
        keep_hostname = sorted_hosts[keep]
        remote_hostname = sorted_hosts[remote]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        for ident, hostname in ((0, anchor_hostname), (1, target_hostname), (2, keep_hostname)):
            cy_state.add_session(
                RedAbstractSession(
                    ident=ident,
                    hostname=hostname,
                    username="user",
                    agent="red_agent_2",
                    parent=0 if ident != 0 else None,
                    session_type="shell",
                    pid=None,
                )
            )

        target_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == target_hostname)
        keep_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == keep_hostname)
        remote_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == remote_hostname)
        cy_state.sessions["red_agent_2"][0].addport(target_ip, 22)
        cy_state.sessions["red_agent_2"][0].addport(keep_ip, 443)
        cy_state.sessions["red_agent_2"][0].addport(remote_ip, 80)

        state = state.replace(
            red_sessions=state.red_sessions.at[2, anchor].set(True).at[2, target].set(True).at[2, keep].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[2, anchor]
            .set(True)
            .at[2, target]
            .set(True)
            .at[2, keep]
            .set(True),
            red_privilege=state.red_privilege.at[2, anchor]
            .set(COMPROMISE_USER)
            .at[2, target]
            .set(COMPROMISE_USER)
            .at[2, keep]
            .set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[anchor]
            .set(COMPROMISE_USER)
            .at[target]
            .set(COMPROMISE_USER)
            .at[keep]
            .set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[2, target]
            .set(True)
            .at[2, keep]
            .set(True)
            .at[2, remote]
            .set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[2, target, anchor]
            .set(True)
            .at[2, keep, anchor]
            .set(True)
            .at[2, remote, anchor]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[2].set(anchor),
            red_suspicious_process_count=state.red_suspicious_process_count.at[2, target]
            .set(0)
            .at[2, anchor]
            .set(1)
            .at[2, keep]
            .set(1),
        )

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        restore_act = encode_blue_action("Restore", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, restore_act)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_2"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[2, h])}
        assert cy_scanned == {target, keep, remote}
        assert jax_scanned == cy_scanned

    def test_restore_removing_stale_scan_owner_session_clears_scans_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        subnet_hosts = []
        base = _find_host_in_subnet(const, "OPERATIONAL_ZONE_B")
        assert base is not None
        subnet_id = int(const.host_subnet[base])
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
                continue
            if int(const.host_subnet[h]) != subnet_id:
                continue
            subnet_hosts.append(h)
            if len(subnet_hosts) == 4:
                break
        assert len(subnet_hosts) == 4
        target, stale_anchor, nonstale_host, keep_host = subnet_hosts

        remote_1 = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        remote_2 = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert remote_1 is not None and remote_2 is not None

        target_hostname = sorted_hosts[target]
        stale_anchor_hostname = sorted_hosts[stale_anchor]
        nonstale_hostname = sorted_hosts[nonstale_host]
        keep_hostname = sorted_hosts[keep_host]
        remote_1_hostname = sorted_hosts[remote_1]
        remote_2_hostname = sorted_hosts[remote_2]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        for ident, hostname in (
            (0, target_hostname),
            (1, stale_anchor_hostname),
            (2, nonstale_hostname),
            (3, keep_hostname),
        ):
            cy_state.add_session(
                RedAbstractSession(
                    ident=ident,
                    hostname=hostname,
                    username="user",
                    agent="red_agent_4",
                    parent=0 if ident != 0 else None,
                    session_type="shell",
                    pid=None,
                )
            )

        owner = cy_state.sessions["red_agent_4"][0]
        owner.addport(next(ip for ip, host in cy_state.ip_addresses.items() if host == stale_anchor_hostname), 22)
        owner.addport(next(ip for ip, host in cy_state.ip_addresses.items() if host == nonstale_hostname), 443)
        owner.addport(next(ip for ip, host in cy_state.ip_addresses.items() if host == keep_hostname), 80)
        owner.addport(next(ip for ip, host in cy_state.ip_addresses.items() if host == remote_1_hostname), 25)
        owner.addport(next(ip for ip, host in cy_state.ip_addresses.items() if host == remote_2_hostname), 3390)

        state = state.replace(
            red_sessions=state.red_sessions.at[4, target]
            .set(True)
            .at[4, stale_anchor]
            .set(True)
            .at[4, nonstale_host]
            .set(True)
            .at[4, keep_host]
            .set(True),
            red_session_count=state.red_session_count.at[4, target]
            .set(1)
            .at[4, stale_anchor]
            .set(1)
            .at[4, nonstale_host]
            .set(1)
            .at[4, keep_host]
            .set(1),
            red_privilege=state.red_privilege.at[4, target]
            .set(COMPROMISE_USER)
            .at[4, stale_anchor]
            .set(COMPROMISE_USER)
            .at[4, nonstale_host]
            .set(COMPROMISE_USER)
            .at[4, keep_host]
            .set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target]
            .set(COMPROMISE_USER)
            .at[stale_anchor]
            .set(COMPROMISE_USER)
            .at[nonstale_host]
            .set(COMPROMISE_USER)
            .at[keep_host]
            .set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[4, stale_anchor]
            .set(True)
            .at[4, nonstale_host]
            .set(True)
            .at[4, keep_host]
            .set(True)
            .at[4, remote_1]
            .set(True)
            .at[4, remote_2]
            .set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[4, stale_anchor, target]
            .set(True)
            .at[4, nonstale_host, target]
            .set(True)
            .at[4, keep_host, target]
            .set(True)
            .at[4, remote_1, target]
            .set(True)
            .at[4, remote_2, target]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[4].set(stale_anchor),
            red_suspicious_process_count=state.red_suspicious_process_count.at[4, target]
            .set(0)
            .at[4, stale_anchor]
            .set(0)
            .at[4, nonstale_host]
            .set(1)
            .at[4, keep_host]
            .set(0),
        )

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        restore_act = encode_blue_action("Restore", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, restore_act)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_4"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[4, h])}
        assert cy_scanned == set()
        assert jax_scanned == cy_scanned

    def test_restore_does_not_remap_single_scan_candidate_when_source_session_removed_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        base = _find_host_in_subnet(const, "OPERATIONAL_ZONE_A")
        assert base is not None
        target_subnet = int(const.host_subnet[base])

        same_subnet_hosts = []
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
                continue
            if int(const.host_subnet[h]) != target_subnet:
                continue
            same_subnet_hosts.append(h)
            if len(same_subnet_hosts) == 3:
                break
        assert len(same_subnet_hosts) == 3
        target, keep, candidate = same_subnet_hosts

        remote_hosts = []
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]) or bool(const.host_is_router[h]):
                continue
            if h in {target, keep, candidate}:
                continue
            if int(const.host_subnet[h]) == target_subnet:
                continue
            remote_hosts.append(h)
            if len(remote_hosts) == 3:
                break
        assert len(remote_hosts) == 3
        remote_a, remote_b, remote_c = remote_hosts

        target_hostname = sorted_hosts[target]
        keep_hostname = sorted_hosts[keep]
        candidate_hostname = sorted_hosts[candidate]
        remote_a_hostname = sorted_hosts[remote_a]
        remote_b_hostname = sorted_hosts[remote_b]
        remote_c_hostname = sorted_hosts[remote_c]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        for ident, hostname in ((0, target_hostname), (1, keep_hostname)):
            cy_state.add_session(
                RedAbstractSession(
                    ident=ident,
                    hostname=hostname,
                    username="user",
                    agent="red_agent_3",
                    parent=0 if ident != 0 else None,
                    session_type="shell",
                    pid=None,
                )
            )

        owner = next(sess for sess in cy_state.sessions["red_agent_3"].values() if sess.hostname == target_hostname)
        for scan_hostname, scan_port in (
            (candidate_hostname, 22),
            (remote_a_hostname, 443),
            (remote_b_hostname, 80),
            (remote_c_hostname, 25),
        ):
            scan_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == scan_hostname)
            owner.addport(scan_ip, scan_port)

        state = state.replace(
            red_sessions=state.red_sessions.at[3, target].set(True).at[3, keep].set(True),
            red_session_count=state.red_session_count.at[3, target].set(1).at[3, keep].set(1),
            red_privilege=state.red_privilege.at[3, target].set(COMPROMISE_USER).at[3, keep].set(COMPROMISE_USER),
            red_session_is_abstract=state.red_session_is_abstract.at[3, target].set(True).at[3, keep].set(True),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER).at[keep].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[3, candidate]
            .set(True)
            .at[3, remote_a]
            .set(True)
            .at[3, remote_b]
            .set(True)
            .at[3, remote_c]
            .set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[3, candidate, target]
            .set(True)
            .at[3, remote_a, target]
            .set(True)
            .at[3, remote_b, target]
            .set(True)
            .at[3, remote_c, target]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[3].set(keep),
        )

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        restore_action.duration = 1
        cy_obs = restore_action.execute(cy_state)
        assert cy_obs.success

        restore_act = encode_blue_action("Restore", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, restore_act)

        cy_scanned = set()
        for sess in cy_state.sessions["red_agent_3"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[3, h])}
        assert cy_scanned == set()
        assert jax_scanned == cy_scanned


class TestRestoreOnNonFoothold:
    def test_restore_on_non_foothold_clears_sessions(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        blue_agent = _find_blue_for_host(jax_const, target)
        assert blue_agent is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
        )

        action = encode_blue_action("Restore", target, blue_agent, const=jax_const)
        state = _jit_apply_blue(state, jax_const, blue_agent, action)

        assert int(state.red_sessions[0, target]) == 0
        assert int(state.red_privilege[0, target]) == COMPROMISE_NONE


class TestRestorePreservesOtherSessionScanData:
    def test_restore_on_non_anchor_host_preserves_scan_data(self):
        cyborg_env = _make_cyborg_env()
        const = build_const_from_cyborg(cyborg_env)
        state = _make_jax_state(const)
        cy_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cy_state.hosts.keys())

        from jaxborg.constants import SUBNET_IDS

        sid = SUBNET_IDS["RESTRICTED_ZONE_A"]
        subnet_hosts = []
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]):
                continue
            if int(const.host_subnet[h]) != sid:
                continue
            if bool(const.host_is_router[h]):
                continue
            subnet_hosts.append(h)
        assert len(subnet_hosts) >= 3

        anchor_host = subnet_hosts[0]
        other_host = subnet_hosts[1]
        scanned_target = subnet_hosts[2]

        anchor_hostname = sorted_hosts[anchor_host]
        other_hostname = sorted_hosts[other_host]
        scanned_hostname = sorted_hosts[scanned_target]
        scanned_ip = next(ip for ip, host in cy_state.ip_addresses.items() if host == scanned_hostname)

        red_anchor = RedAbstractSession(
            ident=None,
            hostname=anchor_hostname,
            username="user",
            agent="red_agent_1",
            parent=0,
            session_type="shell",
            pid=None,
        )
        red_other = RedAbstractSession(
            ident=None,
            hostname=other_hostname,
            username="user",
            agent="red_agent_1",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cy_state.add_session(red_anchor)
        cy_state.add_session(red_other)

        anchor_sess = next(
            sess
            for sess in cy_state.sessions["red_agent_1"].values()
            if sess.hostname == anchor_hostname and isinstance(sess, RedAbstractSession)
        )
        anchor_sess.addport(scanned_ip, 22)

        state = state.replace(
            red_sessions=state.red_sessions.at[1, anchor_host].set(True).at[1, other_host].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[1, anchor_host]
            .set(True)
            .at[1, other_host]
            .set(True),
            red_privilege=state.red_privilege.at[1, anchor_host]
            .set(COMPROMISE_USER)
            .at[1, other_host]
            .set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[1, scanned_target].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[1, scanned_target, anchor_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[1].set(anchor_host),
            host_compromised=state.host_compromised.at[anchor_host]
            .set(COMPROMISE_USER)
            .at[other_host]
            .set(COMPROMISE_USER),
        )

        blue_idx = _find_blue_for_host(const, other_host)
        assert blue_idx is not None

        restore_action = Restore(session=0, agent=f"blue_agent_{blue_idx}", hostname=other_hostname)
        restore_action.duration = 1
        restore_action.execute(cy_state)

        action_idx = encode_blue_action("Restore", other_host, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cy_scanned = set()
        for sess in cy_state.sessions.get("red_agent_1", {}).values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cy_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[1, h])}
        assert scanned_target in cy_scanned, "CybORG should preserve scan data on surviving anchor session"
        assert jax_scanned == cy_scanned, (
            f"JAX scanned hosts should match CybORG: jax={jax_scanned} cyborg={cy_scanned}"
        )
