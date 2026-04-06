import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Shared.Session import RedAbstractSession, Session
from CybORG.Simulator.Actions import Remove
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.blue_remove import apply_blue_remove
from jaxborg.actions.encoding import (
    BLUE_ACTION_TYPE_REMOVE,
    decode_blue_action,
    encode_blue_action,
    encode_red_action,
)
from jaxborg.actions.pids import append_pid_to_row
from jaxborg.constants import (
    ACTIVITY_EXPLOIT,
    COMPROMISE_NONE,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    MAX_TRACKED_SESSION_PIDS,
    MAX_TRACKED_SUSPICIOUS_PIDS,
    NUM_BLUE_AGENTS,
    SERVICE_IDS,
)
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


@pytest.fixture(scope="module")
def jax_const():
    return build_const_from_cyborg(_make_cyborg_env())


def _make_jax_state(const):
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    start_host = int(const.red_start_hosts[0])
    red_sessions = state.red_sessions.at[0, start_host].set(True)
    red_session_count = state.red_session_count.at[0, start_host].set(1)
    red_privilege = state.red_privilege.at[0, start_host].set(COMPROMISE_USER)
    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_privilege=red_privilege,
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


def _find_blue_for_host(const, host):
    for b in range(NUM_BLUE_AGENTS):
        if bool(const.blue_agent_hosts[b, host]):
            return b
    return None


def _setup_exploit(state, const, target_h):
    target_subnet = int(const.host_subnet[target_h])
    discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
    state = _jit_apply_red(state, const, 0, discover_idx, jax.random.PRNGKey(0))
    state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
    scan_idx = encode_red_action("DiscoverNetworkServices", target_h, 0)
    state = _jit_apply_red(state, const, 0, scan_idx, jax.random.PRNGKey(0))
    state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
    exploit_idx = encode_red_action("ExploitRemoteService_cc4SSHBruteForce", target_h, 0)
    state = _jit_apply_red(state, const, 0, exploit_idx, jax.random.PRNGKey(0))
    return state


def _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts):
    host_to_idx = {h: i for i, h in enumerate(sorted_hosts)}

    modeled_hosts = state.red_sessions | (state.red_session_count > 0)

    cy_session_count = jnp.zeros_like(state.red_session_count)
    cy_session_pids = jnp.full_like(state.red_session_pids, -1)
    cy_privilege = jnp.zeros_like(state.red_privilege)
    max_pid = int(state.red_next_pid)
    for r in range(6):
        agent_name = f"red_agent_{r}"
        for sess in cyborg_state.sessions.get(agent_name, {}).values():
            if sess.hostname not in host_to_idx:
                continue
            hidx = host_to_idx[sess.hostname]
            cy_session_count = cy_session_count.at[r, hidx].add(1)
            pid = int(getattr(sess, "pid", -1))
            if pid >= 0:
                row = cy_session_pids[r, hidx]
                row_has_empty = bool(jnp.any(row < 0))
                row_has_pid = bool(jnp.any(row == pid))
                if not row_has_empty and not row_has_pid:
                    raise RuntimeError(
                        "Test setup overflow while syncing red session pids for "
                        f"{agent_name} host={sess.hostname}: "
                        f"MAX_TRACKED_SESSION_PIDS={MAX_TRACKED_SESSION_PIDS}"
                    )
                cy_session_pids = cy_session_pids.at[r, hidx].set(append_pid_to_row(row, pid))
                max_pid = max(max_pid, pid + 1)
            level = 2 if getattr(sess, "username", "") in ("root", "SYSTEM") else 1
            cy_privilege = cy_privilege.at[r, hidx].set(jnp.maximum(cy_privilege[r, hidx], level))

    red_session_count = jnp.where(modeled_hosts, cy_session_count, state.red_session_count)
    red_sessions = jnp.where(modeled_hosts, red_session_count > 0, state.red_sessions)
    red_privilege = jnp.where(modeled_hosts, cy_privilege, state.red_privilege)
    red_session_pids = jnp.where(modeled_hosts[:, :, None], cy_session_pids, state.red_session_pids)

    blue_suspicious_pids = state.blue_suspicious_pids
    for b in range(NUM_BLUE_AGENTS):
        agent_name = f"blue_agent_{b}"
        for blue_sess in cyborg_state.sessions.get(agent_name, {}).values():
            sus_pids = getattr(blue_sess, "sus_pids", {})
            for hostname, pid_list in sus_pids.items():
                if hostname not in host_to_idx:
                    continue
                hidx = host_to_idx[hostname]
                if len(pid_list) > MAX_TRACKED_SUSPICIOUS_PIDS:
                    raise RuntimeError(
                        "Test setup overflow while syncing blue suspicious pids for "
                        f"{agent_name} host={hostname}: observed {len(pid_list)} "
                        f"> MAX_TRACKED_SUSPICIOUS_PIDS={MAX_TRACKED_SUSPICIOUS_PIDS}"
                    )
                row = jnp.full(MAX_TRACKED_SUSPICIOUS_PIDS, -1, dtype=jnp.int32)
                for i, pid in enumerate(pid_list):
                    row = row.at[i].set(int(pid))
                blue_suspicious_pids = blue_suspicious_pids.at[b, hidx].set(row)

    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_privilege=red_privilege,
        red_session_pids=red_session_pids,
        red_next_pid=jnp.int32(max_pid),
        blue_suspicious_pids=blue_suspicious_pids,
    )


class TestBlueRemoveEncoding:
    def _agent0_host(self, jax_const):
        """Find a non-router host in agent 0's observed subnets."""
        active = np.array(jax_const.host_active, dtype=bool)
        ctrl = (
            np.array(jax_const.blue_agent_hosts[0], dtype=bool)
            & active
            & ~np.array(jax_const.host_is_router, dtype=bool)
        )
        return int(np.flatnonzero(ctrl)[0])

    def test_encode_remove(self, jax_const):
        h = self._agent0_host(jax_const)
        action_idx = encode_blue_action("Remove", h, 0, const=jax_const)
        action_type, target_host, *_ = decode_blue_action(action_idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_REMOVE
        assert int(target_host) == h

    def test_decode_remove(self, jax_const):
        h = self._agent0_host(jax_const)
        action_idx = encode_blue_action("Remove", h, 0, const=jax_const)
        action_type, target_host, *_ = decode_blue_action(action_idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_REMOVE
        assert int(target_host) == h


class TestApplyBlueRemove:
    def test_remove_clears_user_session(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(1),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            red_suspicious_process_count=state.red_suspicious_process_count.at[0, target].set(1),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
            red_activity_this_step=state.red_activity_this_step.at[target].set(ACTIVITY_EXPLOIT),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None
        test_pid = 6001
        state = state.replace(
            red_session_pids=state.red_session_pids.at[0, target, 0].set(test_pid),
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue_idx, target, 0].set(test_pid),
        )

        new_state = apply_blue_remove(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE
        assert bool(new_state.host_has_malware[target])

    def test_remove_does_not_clear_privileged_session(self, jax_const):
        """CybORG StopProcess skips PIDs whose process user is root/SYSTEM.
        JAX tracks this via red_session_privileged_pids."""
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        test_pid = 6002
        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(1),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
            host_has_malware=state.host_has_malware.at[target].set(True),
            # PID is tracked in both session pids and privileged pids (mirrors privesc)
            red_session_pids=state.red_session_pids.at[0, target, 0].set(test_pid).at[1, target, 0].set(7001),
            red_session_privileged_pids=state.red_session_privileged_pids.at[0, target, 0].set(test_pid),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None
        state = state.replace(
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue_idx, target, 0].set(test_pid),
        )

        new_state = apply_blue_remove(state, jax_const, blue_idx, target)
        assert bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED
        assert bool(new_state.host_has_malware[target])

    def test_remove_on_clean_host_is_noop(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None
        test_pid = 6002
        state = state.replace(
            red_session_pids=state.red_session_pids.at[0, target, 0].set(test_pid).at[1, target, 0].set(7001),
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue_idx, target, 0].set(test_pid),
        )

        new_state = apply_blue_remove(state, jax_const, blue_idx, target)
        np.testing.assert_array_equal(np.array(new_state.red_sessions), np.array(state.red_sessions))

    def test_remove_on_non_anchor_host_preserves_primary_session(self, jax_const):
        state = _make_jax_state(jax_const)
        agent_id = 0
        blue_idx = 0

        covered_hosts = [
            h
            for h in range(int(jax_const.num_hosts))
            if bool(jax_const.host_active[h]) and bool(jax_const.blue_agent_hosts[blue_idx, h])
        ]
        assert len(covered_hosts) >= 2
        anchor_host = covered_hosts[0]
        target_host = covered_hosts[1]

        primary_pid = 6003
        other_pid = 7003
        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, anchor_host].set(True).at[agent_id, target_host].set(True),
            red_session_count=state.red_session_count.at[agent_id, anchor_host].set(1).at[agent_id, target_host].set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, anchor_host]
            .set(True)
            .at[agent_id, target_host]
            .set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(jnp.int32(anchor_host)),
            red_primary_is_abstract=state.red_primary_is_abstract.at[agent_id].set(True),
            red_primary_pid=state.red_primary_pid.at[agent_id].set(jnp.int32(primary_pid)),
            red_session_pids=state.red_session_pids.at[agent_id, anchor_host, 0]
            .set(primary_pid)
            .at[agent_id, target_host, 0]
            .set(other_pid),
            red_session_abstract_pids=state.red_session_abstract_pids.at[agent_id, anchor_host, 0]
            .set(primary_pid)
            .at[agent_id, target_host, 0]
            .set(other_pid),
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue_idx, target_host, 0].set(other_pid),
        )

        new_state = apply_blue_remove(state, jax_const, blue_idx, target_host)
        assert not bool(new_state.red_sessions[agent_id, target_host])
        assert int(new_state.red_primary_pid[agent_id]) == primary_pid
        assert bool(new_state.red_primary_is_abstract[agent_id])
        assert int(new_state.red_scan_anchor_host[agent_id]) == anchor_host

    def test_remove_clears_user_but_leaves_other_red_agents_privileged(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True).at[1, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(1).at[1, target].set(1),
            red_privilege=state.red_privilege.at[0, target]
            .set(COMPROMISE_USER)
            .at[1, target]
            .set(COMPROMISE_PRIVILEGED),
            red_suspicious_process_count=state.red_suspicious_process_count.at[0, target].set(1),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
            red_activity_this_step=state.red_activity_this_step.at[target].set(ACTIVITY_EXPLOIT),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None
        test_pid = 6002
        state = state.replace(
            red_session_pids=state.red_session_pids.at[0, target, 0].set(test_pid).at[1, target, 0].set(7001),
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue_idx, target, 0].set(test_pid),
        )

        new_state = apply_blue_remove(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE
        assert bool(new_state.red_sessions[1, target])
        assert int(new_state.red_privilege[1, target]) == COMPROMISE_PRIVILEGED
        assert bool(new_state.host_has_malware[target])

    def test_jit_compatible(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(1),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            red_suspicious_process_count=state.red_suspicious_process_count.at[0, target].set(1),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
            red_activity_this_step=state.red_activity_this_step.at[target].set(ACTIVITY_EXPLOIT),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None
        test_pid = 6003
        state = state.replace(
            red_session_pids=state.red_session_pids.at[0, target, 0].set(test_pid),
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue_idx, target, 0].set(test_pid),
        )

        jitted = jax.jit(apply_blue_remove, static_argnums=(2, 3))
        new_state = jitted(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE


class TestRemoveViaDispatch:
    def test_remove_dispatched(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(1),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            red_suspicious_process_count=state.red_suspicious_process_count.at[0, target].set(1),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
            red_activity_this_step=state.red_activity_this_step.at[target].set(ACTIVITY_EXPLOIT),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None
        test_pid = 6004
        state = state.replace(
            red_session_pids=state.red_session_pids.at[0, target, 0].set(test_pid),
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue_idx, target, 0].set(test_pid),
        )

        action_idx = encode_blue_action("Remove", target, blue_idx, const=jax_const)
        new_state = _jit_apply_blue(state, jax_const, blue_idx, action_idx)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE


class TestDifferentialWithCybORG:
    @pytest.fixture
    def cyborg_and_jax(self):
        cyborg_env = _make_cyborg_env()
        const = build_const_from_cyborg(cyborg_env)
        state = _make_jax_state(const)
        return cyborg_env, const, state

    def test_remove_without_suspicious_process_does_not_clear_user_session_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_red_sessions = [
            s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_red_sessions)
        cyborg_host_compromised = 1 if cyborg_has_user_session else 0

        assert bool(new_state.red_sessions[0, target]) == cyborg_has_user_session
        expected_priv = COMPROMISE_USER if cyborg_has_user_session else COMPROMISE_NONE
        assert int(new_state.red_privilege[0, target]) == expected_priv
        assert int(new_state.host_compromised[target]) == cyborg_host_compromised

    def test_remove_with_stale_suspicious_pid_does_not_clear_user_session_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        stale_pid = 999999
        assert cyborg_state.hosts[target_hostname].get_process(stale_pid) is None
        blue_parent.add_sus_pids(hostname=target_hostname, pid=stale_pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(False),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_red_sessions = [
            s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_red_sessions)

        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[0, target]) == cyborg_has_user_session

    def test_remove_with_valid_suspicious_pid_clears_user_session_without_fresh_activity_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        cy_red_sess = next(s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname)
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=cy_red_sess.pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            red_suspicious_process_count=state.red_suspicious_process_count.at[0, target].set(1),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(False),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
            red_activity_this_step=state.red_activity_this_step.at[target].set(0),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_red_sessions = [
            s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_red_sessions)

        assert not cyborg_has_user_session
        assert bool(new_state.red_sessions[0, target]) == cyborg_has_user_session
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE

    def test_remove_with_multi_pid_budget_and_malware_clears_user_session_without_live_suspicious_flag_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "OPERATIONAL_ZONE_B")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_4",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        cy_red_sess = next(s for s in cyborg_state.sessions["red_agent_4"].values() if s.hostname == target_hostname)
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=cy_red_sess.pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[4, target].set(True),
            red_privilege=state.red_privilege.at[4, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(False),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            red_suspicious_process_count=state.red_suspicious_process_count.at[4, target].set(0),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_red_sessions = [
            s for s in cyborg_state.sessions["red_agent_4"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_red_sessions)

        assert not cyborg_has_user_session
        assert bool(new_state.red_sessions[4, target]) == cyborg_has_user_session
        assert int(new_state.red_privilege[4, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE

    def test_remove_with_single_stale_budget_and_malware_does_not_clear_user_session_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_1",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999999)

        state = state.replace(
            red_sessions=state.red_sessions.at[1, target].set(True),
            red_privilege=state.red_privilege.at[1, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(False),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            red_suspicious_process_count=state.red_suspicious_process_count.at[1, target].set(0),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_red_sessions = [
            s for s in cyborg_state.sessions["red_agent_1"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_red_sessions)

        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[1, target]) == cyborg_has_user_session

    def test_remove_with_stale_multi_budget_on_scanned_target_does_not_clear_user_session_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "ADMIN_NETWORK")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_5",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999991)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999992)

        state = state.replace(
            red_sessions=state.red_sessions.at[5, target].set(True),
            red_privilege=state.red_privilege.at[5, target].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[5, target].set(True),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(False),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            red_suspicious_process_count=state.red_suspicious_process_count.at[5, target].set(0),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_red_sessions = [
            s for s in cyborg_state.sessions["red_agent_5"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_red_sessions)

        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[5, target]) == cyborg_has_user_session

    def test_remove_with_live_suspicious_pid_removes_user_session_even_when_malware_flag_is_false_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_1",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        cy_red_sess = next(s for s in cyborg_state.sessions["red_agent_1"].values() if s.hostname == target_hostname)
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=cy_red_sess.pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[1, target].set(True),
            red_privilege=state.red_privilege.at[1, target].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[1, target].set(True),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(False),
            host_suspicious_process=state.host_suspicious_process.at[target].set(False),
            red_suspicious_process_count=state.red_suspicious_process_count.at[1, target].set(0),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_red_sessions = [
            s for s in cyborg_state.sessions["red_agent_1"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_red_sessions)

        assert not cyborg_has_user_session
        assert bool(new_state.red_sessions[1, target]) == cyborg_has_user_session

    def test_remove_clears_anchor_host_when_suspicious_pids_cover_all_user_sessions_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        cyborg_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent="red_agent_1",
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        cyborg_state.add_session(
            Session(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent="red_agent_1",
                parent=0,
                session_type="shell",
                pid=None,
            )
        )

        cy_sessions = [s for s in cyborg_state.sessions["red_agent_1"].values() if s.hostname == target_hostname]
        assert len(cy_sessions) == 2
        cy_pids = [int(s.pid) for s in cy_sessions]
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        for pid in cy_pids:
            blue_parent.add_sus_pids(hostname=target_hostname, pid=pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[1, target].set(True),
            red_session_count=state.red_session_count.at[1, target].set(2),
            red_session_is_abstract=state.red_session_is_abstract.at[1, target].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[1].set(target),
            red_privilege=state.red_privilege.at[1, target].set(COMPROMISE_USER),
            red_suspicious_process_count=state.red_suspicious_process_count.at[1, target].set(2),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )
        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cy_remaining = [s for s in cyborg_state.sessions["red_agent_1"].values() if s.hostname == target_hostname]
        assert len(cy_remaining) == 0
        assert int(new_state.red_session_count[1, target]) == 0
        assert not bool(new_state.red_sessions[1, target])
        assert int(new_state.red_privilege[1, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE
        assert int(new_state.red_privilege[1, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE

    def test_remove_with_exact_budget_on_scanned_activity_abstract_session_clears_user_session_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_3",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        cy_red_sess = next(s for s in cyborg_state.sessions["red_agent_3"].values() if s.hostname == target_hostname)
        target_ip = next(ip for ip, host in cyborg_state.ip_addresses.items() if host == target_hostname)
        cy_red_sess.addport(target_ip, 22)
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=cy_red_sess.pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[3, target].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[3, target].set(True),
            red_privilege=state.red_privilege.at[3, target].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[3, target].set(True),
            red_pending_ticks=state.red_pending_ticks.at[3].set(3),
            red_pending_action=state.red_pending_action.at[3].set(
                encode_red_action("ExploitRemoteService_cc4SSHBruteForce", target, 3)
            ),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(False),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(False),
            red_suspicious_process_count=state.red_suspicious_process_count.at[3, target].set(0),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_red_sessions = [
            s for s in cyborg_state.sessions["red_agent_3"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_red_sessions)

        assert not cyborg_has_user_session
        assert bool(new_state.red_sessions[3, target]) == cyborg_has_user_session
        assert int(new_state.red_privilege[3, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE

    def test_remove_with_stale_multi_budget_on_unscanned_non_malware_host_keeps_user_session_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_3",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999971)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999972)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999973)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999974)

        state = state.replace(
            red_sessions=state.red_sessions.at[3, target].set(True),
            red_privilege=state.red_privilege.at[3, target].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[3, target].set(False),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(False),
            host_activity_detected=state.host_activity_detected.at[target].set(False),
            host_suspicious_process=state.host_suspicious_process.at[target].set(False),
            red_suspicious_process_count=state.red_suspicious_process_count.at[3, target].set(0),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_red_sessions = [
            s for s in cyborg_state.sessions["red_agent_3"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_red_sessions)

        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[3, target]) == cyborg_has_user_session

    def test_remove_clearing_last_session_clears_scanned_hosts_memory_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None
        red_agent_idx = next(r for r in range(6) if int(const.red_start_hosts[r]) != target)
        red_agent_name = f"red_agent_{red_agent_idx}"

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent=red_agent_name,
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        cy_red_sess = next(s for s in cyborg_state.sessions[red_agent_name].values() if s.hostname == target_hostname)
        target_ip = next(ip for ip, host in cyborg_state.ip_addresses.items() if host == target_hostname)
        cy_red_sess.addport(target_ip, 22)
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=cy_red_sess.pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[red_agent_idx].set(False).at[red_agent_idx, target].set(True),
            red_session_count=state.red_session_count.at[red_agent_idx].set(0).at[red_agent_idx, target].set(1),
            red_privilege=state.red_privilege.at[red_agent_idx]
            .set(COMPROMISE_NONE)
            .at[red_agent_idx, target]
            .set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[red_agent_idx].set(False).at[red_agent_idx, target].set(True),
            red_suspicious_process_count=state.red_suspicious_process_count.at[red_agent_idx]
            .set(0)
            .at[red_agent_idx, target]
            .set(1),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cy_scanned_hosts = set()
        for sess in cyborg_state.sessions[red_agent_name].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cyborg_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned_hosts.add(sorted_hosts.index(host))

        assert cy_scanned_hosts == set()
        jax_scanned_hosts = {
            h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[red_agent_idx, h])
        }
        assert jax_scanned_hosts == cy_scanned_hosts

    def test_remove_with_multiple_user_sessions_removes_one_not_all_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "OFFICE_NETWORK")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session_a = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
        red_session_b = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session_a)
        cyborg_state.add_session(red_session_b)

        cy_red_sessions = [s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
        assert len(cy_red_sessions) >= 2
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=cy_red_sessions[0].pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[0].set(False).at[0, target].set(True),
            red_suspicious_process_count=state.red_suspicious_process_count.at[0].set(0).at[0, target].set(1),
            red_privilege=state.red_privilege.at[0].set(COMPROMISE_NONE).at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_remaining = [s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_remaining)
        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[0, target]) == cyborg_has_user_session

    def test_remove_with_mixed_live_and_stale_budget_clears_all_user_sessions_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "PUBLIC_ACCESS_ZONE")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        agent_name = "red_agent_5"
        cyborg_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent=agent_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        cy_red_sess = next(s for s in cyborg_state.sessions[agent_name].values() if s.hostname == target_hostname)
        assert cy_red_sess.pid is not None

        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=cy_red_sess.pid)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999991)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999992)

        state = state.replace(
            red_sessions=state.red_sessions.at[5, target].set(True),
            red_session_count=state.red_session_count.at[5, target].set(3),
            red_session_is_abstract=state.red_session_is_abstract.at[5, target].set(True),
            red_privilege=state.red_privilege.at[5, target].set(COMPROMISE_USER),
            red_suspicious_process_count=state.red_suspicious_process_count.at[5, target].set(2),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_remaining = [s for s in cyborg_state.sessions[agent_name].values() if s.hostname == target_hostname]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_remaining)

        assert not cyborg_has_user_session
        assert bool(new_state.red_sessions[5, target]) == cyborg_has_user_session
        assert int(new_state.red_privilege[5, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE

    def test_remove_with_exact_two_pid_budget_and_one_live_signal_clears_user_session_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        agent_name = "red_agent_3"
        cyborg_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent=agent_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        cy_red_sess = next(s for s in cyborg_state.sessions[agent_name].values() if s.hostname == target_hostname)
        assert cy_red_sess.pid is not None

        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=cy_red_sess.pid)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999981)

        state = state.replace(
            red_sessions=state.red_sessions.at[3, target].set(True),
            red_session_count=state.red_session_count.at[3, target].set(2),
            red_session_is_abstract=state.red_session_is_abstract.at[3, target].set(True),
            red_privilege=state.red_privilege.at[3, target].set(COMPROMISE_USER),
            red_suspicious_process_count=state.red_suspicious_process_count.at[3, target].set(1),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_remaining = [s for s in cyborg_state.sessions[agent_name].values() if s.hostname == target_hostname]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_remaining)

        assert not cyborg_has_user_session
        assert bool(new_state.red_sessions[3, target]) == cyborg_has_user_session
        assert int(new_state.red_privilege[3, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE

    def test_remove_with_three_stale_budget_on_scanned_non_malware_host_keeps_user_session_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "OPERATIONAL_ZONE_B")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        agent_name = "red_agent_4"
        cyborg_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent=agent_name,
                parent=0,
                session_type="shell",
                pid=None,
            )
        )

        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999961)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999962)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=999963)

        state = state.replace(
            red_sessions=state.red_sessions.at[4, target].set(True),
            red_session_count=state.red_session_count.at[4, target].set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[4, target].set(True),
            red_privilege=state.red_privilege.at[4, target].set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[4, target].set(True),
            red_suspicious_process_count=state.red_suspicious_process_count.at[4, target].set(0),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(False),
            host_suspicious_process=state.host_suspicious_process.at[target].set(False),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_remaining = [s for s in cyborg_state.sessions[agent_name].values() if s.hostname == target_hostname]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_remaining)

        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[4, target]) == cyborg_has_user_session
        assert int(new_state.red_privilege[4, target]) == COMPROMISE_USER
        assert int(new_state.host_compromised[target]) == COMPROMISE_USER

    def test_remove_with_many_user_sessions_and_multiple_suspicious_pids_keeps_one_session_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        for _ in range(4):
            red_session = RedAbstractSession(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent="red_agent_0",
                parent=0,
                session_type="shell",
                pid=None,
            )
            cyborg_state.add_session(red_session)

        cy_red_sessions = [s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
        assert len(cy_red_sessions) >= 4
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        for sess in cy_red_sessions[:3]:
            blue_parent.add_sus_pids(hostname=target_hostname, pid=sess.pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[0].set(False).at[0, target].set(True),
            red_suspicious_process_count=state.red_suspicious_process_count.at[0].set(0).at[0, target].set(2),
            red_privilege=state.red_privilege.at[0].set(COMPROMISE_NONE).at[0, target].set(COMPROMISE_USER),
            red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(target),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_remaining = [s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_remaining)
        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[0, target]) == cyborg_has_user_session
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_USER
        assert int(new_state.red_session_count[0, target]) == 1

    def test_remove_with_many_sessions_on_non_anchor_host_clears_target_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "OFFICE_NETWORK")
        anchor = _find_host_in_subnet(const, "ADMIN_NETWORK")
        assert target is not None and anchor is not None

        target_hostname = sorted_hosts[target]
        anchor_hostname = sorted_hosts[anchor]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        anchor_session = RedAbstractSession(
            ident=None,
            hostname=anchor_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(anchor_session)

        for _ in range(3):
            target_session = RedAbstractSession(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent="red_agent_0",
                parent=0,
                session_type="shell",
                pid=None,
            )
            cyborg_state.add_session(target_session)

        target_sessions = [s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
        assert len(target_sessions) >= 3
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        for sess in target_sessions:
            blue_parent.add_sus_pids(hostname=target_hostname, pid=sess.pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[0].set(False).at[0, anchor].set(True).at[0, target].set(True),
            red_session_count=state.red_session_count.at[0].set(0).at[0, anchor].set(1).at[0, target].set(3),
            red_suspicious_process_count=state.red_suspicious_process_count.at[0].set(0).at[0, target].set(3),
            red_privilege=state.red_privilege.at[0]
            .set(COMPROMISE_NONE)
            .at[0, anchor]
            .set(COMPROMISE_USER)
            .at[0, target]
            .set(COMPROMISE_USER),
            red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(anchor),
            host_compromised=state.host_compromised.at[anchor].set(COMPROMISE_USER).at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_target_remaining = [
            s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname
        ]
        assert not cyborg_target_remaining
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE

    def test_remove_with_many_sessions_non_anchor_and_stale_signal_keeps_user_session_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        anchor = _find_host_in_subnet(const, "ADMIN_NETWORK")
        assert target is not None and anchor is not None and target != anchor

        target_hostname = sorted_hosts[target]
        anchor_hostname = sorted_hosts[anchor]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        anchor_session = RedAbstractSession(
            ident=None,
            hostname=anchor_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(anchor_session)

        for _ in range(5):
            target_session = RedAbstractSession(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent="red_agent_0",
                parent=0,
                session_type="shell",
                pid=None,
            )
            cyborg_state.add_session(target_session)

        target_sessions = [s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
        assert len(target_sessions) >= 5
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        for sess in target_sessions[:4]:
            blue_parent.add_sus_pids(hostname=target_hostname, pid=sess.pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[0].set(False).at[0, anchor].set(True).at[0, target].set(True),
            red_suspicious_process_count=state.red_suspicious_process_count.at[0].set(0).at[0, target].set(2),
            red_privilege=state.red_privilege.at[0]
            .set(COMPROMISE_NONE)
            .at[0, anchor]
            .set(COMPROMISE_USER)
            .at[0, target]
            .set(COMPROMISE_USER),
            red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(anchor),
            host_compromised=state.host_compromised.at[anchor].set(COMPROMISE_USER).at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(False),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_target_remaining = [
            s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_target_remaining)
        cyborg_host_compromised = COMPROMISE_USER if cyborg_has_user_session else COMPROMISE_NONE

        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[0, target]) == cyborg_has_user_session
        assert int(new_state.red_privilege[0, target]) == cyborg_host_compromised
        assert int(new_state.host_compromised[target]) == cyborg_host_compromised

    def test_remove_with_two_sessions_non_anchor_and_stale_signal_keeps_one_user_session_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        anchor = _find_host_in_subnet(const, "ADMIN_NETWORK")
        assert target is not None and anchor is not None and target != anchor

        target_hostname = sorted_hosts[target]
        anchor_hostname = sorted_hosts[anchor]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        anchor_session = RedAbstractSession(
            ident=None,
            hostname=anchor_hostname,
            username="user",
            agent="red_agent_1",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(anchor_session)

        for _ in range(2):
            target_session = RedAbstractSession(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent="red_agent_1",
                parent=0,
                session_type="shell",
                pid=None,
            )
            cyborg_state.add_session(target_session)

        target_sessions = [s for s in cyborg_state.sessions["red_agent_1"].values() if s.hostname == target_hostname]
        assert len(target_sessions) >= 2
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        valid_pid = target_sessions[0].pid
        stale_pid = 999999
        assert cyborg_state.hosts[target_hostname].get_process(stale_pid) is None
        blue_parent.add_sus_pids(hostname=target_hostname, pid=valid_pid)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=stale_pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[1].set(False).at[1, anchor].set(True).at[1, target].set(True),
            red_session_count=state.red_session_count.at[1].set(0).at[1, anchor].set(1).at[1, target].set(2),
            red_suspicious_process_count=state.red_suspicious_process_count.at[1].set(0).at[1, target].set(1),
            red_privilege=state.red_privilege.at[1]
            .set(COMPROMISE_NONE)
            .at[1, anchor]
            .set(COMPROMISE_USER)
            .at[1, target]
            .set(COMPROMISE_USER),
            red_scan_anchor_host=state.red_scan_anchor_host.at[1].set(anchor),
            host_compromised=state.host_compromised.at[anchor].set(COMPROMISE_USER).at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_target_remaining = [
            s for s in cyborg_state.sessions["red_agent_1"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_target_remaining)

        assert len(cyborg_target_remaining) == 1
        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[1, target]) == cyborg_has_user_session
        assert int(new_state.red_session_count[1, target]) == len(cyborg_target_remaining)
        assert int(new_state.red_privilege[1, target]) == COMPROMISE_USER
        assert int(new_state.host_compromised[target]) == COMPROMISE_USER

    def test_remove_with_two_sessions_and_two_valid_suspicious_pids_clears_target_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "OFFICE_NETWORK")
        anchor = _find_host_in_subnet(const, "ADMIN_NETWORK")
        assert target is not None and anchor is not None and target != anchor

        target_hostname = sorted_hosts[target]
        anchor_hostname = sorted_hosts[anchor]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        anchor_session = RedAbstractSession(
            ident=None,
            hostname=anchor_hostname,
            username="user",
            agent="red_agent_5",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(anchor_session)

        for _ in range(2):
            target_session = RedAbstractSession(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent="red_agent_5",
                parent=0,
                session_type="shell",
                pid=None,
            )
            cyborg_state.add_session(target_session)

        target_sessions = [s for s in cyborg_state.sessions["red_agent_5"].values() if s.hostname == target_hostname]
        assert len(target_sessions) >= 2
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=target_sessions[0].pid)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=target_sessions[1].pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[5].set(False).at[5, anchor].set(True).at[5, target].set(True),
            red_session_count=state.red_session_count.at[5].set(0).at[5, anchor].set(1).at[5, target].set(2),
            red_suspicious_process_count=state.red_suspicious_process_count.at[5].set(0).at[5, target].set(2),
            red_privilege=state.red_privilege.at[5]
            .set(COMPROMISE_NONE)
            .at[5, anchor]
            .set(COMPROMISE_USER)
            .at[5, target]
            .set(COMPROMISE_USER),
            red_scan_anchor_host=state.red_scan_anchor_host.at[5].set(anchor),
            host_compromised=state.host_compromised.at[anchor].set(COMPROMISE_USER).at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_target_remaining = [
            s for s in cyborg_state.sessions["red_agent_5"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_target_remaining)

        assert not cyborg_has_user_session
        assert not bool(new_state.red_sessions[5, target])
        assert int(new_state.red_session_count[5, target]) == len(cyborg_target_remaining)
        assert int(new_state.red_privilege[5, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE

    def test_remove_with_four_sessions_and_three_valid_suspicious_pids_keeps_one_user_session_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "ADMIN_NETWORK")
        anchor = _find_host_in_subnet(const, "OPERATIONAL_ZONE_A")
        assert target is not None and anchor is not None and target != anchor

        target_hostname = sorted_hosts[target]
        anchor_hostname = sorted_hosts[anchor]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        anchor_session = RedAbstractSession(
            ident=None,
            hostname=anchor_hostname,
            username="user",
            agent="red_agent_5",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(anchor_session)

        for _ in range(4):
            target_session = RedAbstractSession(
                ident=None,
                hostname=target_hostname,
                username="user",
                agent="red_agent_5",
                parent=0,
                session_type="shell",
                pid=None,
            )
            cyborg_state.add_session(target_session)

        target_sessions = [s for s in cyborg_state.sessions["red_agent_5"].values() if s.hostname == target_hostname]
        assert len(target_sessions) >= 4
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        for sess in target_sessions[:3]:
            blue_parent.add_sus_pids(hostname=target_hostname, pid=sess.pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[5].set(False).at[5, anchor].set(True).at[5, target].set(True),
            red_session_count=state.red_session_count.at[5].set(0).at[5, anchor].set(1).at[5, target].set(4),
            red_suspicious_process_count=state.red_suspicious_process_count.at[5].set(0).at[5, target].set(3),
            red_privilege=state.red_privilege.at[5]
            .set(COMPROMISE_NONE)
            .at[5, anchor]
            .set(COMPROMISE_USER)
            .at[5, target]
            .set(COMPROMISE_USER),
            red_scan_anchor_host=state.red_scan_anchor_host.at[5].set(anchor),
            host_compromised=state.host_compromised.at[anchor].set(COMPROMISE_USER).at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_target_remaining = [
            s for s in cyborg_state.sessions["red_agent_5"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_target_remaining)

        assert len(cyborg_target_remaining) == 1
        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[5, target]) == cyborg_has_user_session
        assert int(new_state.red_session_count[5, target]) == len(cyborg_target_remaining)
        assert int(new_state.red_privilege[5, target]) == COMPROMISE_USER
        assert int(new_state.host_compromised[target]) == COMPROMISE_USER

    def test_remove_with_budget_above_session_count_but_fewer_valid_pids_keeps_one_user_session_matches_cyborg(
        self, cyborg_and_jax
    ):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "OPERATIONAL_ZONE_A")
        anchor = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        assert target is not None and anchor is not None and target != anchor

        target_hostname = sorted_hosts[target]
        anchor_hostname = sorted_hosts[anchor]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        cyborg_state.add_session(
            RedAbstractSession(
                ident=None,
                hostname=anchor_hostname,
                username="user",
                agent="red_agent_2",
                parent=0,
                session_type="shell",
                pid=None,
            )
        )
        for _ in range(3):
            cyborg_state.add_session(
                RedAbstractSession(
                    ident=None,
                    hostname=target_hostname,
                    username="user",
                    agent="red_agent_2",
                    parent=0,
                    session_type="shell",
                    pid=None,
                )
            )

        target_sessions = [s for s in cyborg_state.sessions["red_agent_2"].values() if s.hostname == target_hostname]
        assert len(target_sessions) >= 3
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=target_sessions[0].pid)
        blue_parent.add_sus_pids(hostname=target_hostname, pid=target_sessions[1].pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[2].set(False).at[2, anchor].set(True).at[2, target].set(True),
            red_session_count=state.red_session_count.at[2].set(0).at[2, anchor].set(1).at[2, target].set(3),
            red_suspicious_process_count=state.red_suspicious_process_count.at[2].set(0).at[2, target].set(2),
            red_privilege=state.red_privilege.at[2]
            .set(COMPROMISE_NONE)
            .at[2, anchor]
            .set(COMPROMISE_USER)
            .at[2, target]
            .set(COMPROMISE_USER),
            red_scan_anchor_host=state.red_scan_anchor_host.at[2].set(anchor),
            host_compromised=state.host_compromised.at[anchor].set(COMPROMISE_USER).at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(False),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = apply_blue_action(state, const, blue_idx, action_idx)

        cyborg_target_remaining = [
            s for s in cyborg_state.sessions["red_agent_2"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_target_remaining)

        assert len(cyborg_target_remaining) == 1
        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[2, target]) == cyborg_has_user_session
        assert int(new_state.red_session_count[2, target]) == len(cyborg_target_remaining)
        assert int(new_state.red_privilege[2, target]) == COMPROMISE_USER
        assert int(new_state.host_compromised[target]) == COMPROMISE_USER

    def test_remove_clears_scan_memory_when_unique_stale_session_host_is_removed_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        other = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        assert target is not None and other is not None and target != other

        target_hostname = sorted_hosts[target]
        other_hostname = sorted_hosts[other]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        target_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_3",
            parent=0,
            session_type="shell",
            pid=None,
        )
        other_session = RedAbstractSession(
            ident=None,
            hostname=other_hostname,
            username="user",
            agent="red_agent_3",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(target_session)
        cyborg_state.add_session(other_session)

        cy_target_session = next(
            sess for sess in cyborg_state.sessions["red_agent_3"].values() if sess.hostname == target_hostname
        )
        target_ip = next(ip for ip, host in cyborg_state.ip_addresses.items() if host == target_hostname)
        cy_target_session.addport(target_ip, 22)

        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        blue_parent.add_sus_pids(hostname=target_hostname, pid=cy_target_session.pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[3].set(False).at[3, target].set(True).at[3, other].set(True),
            red_session_count=state.red_session_count.at[3].set(0).at[3, target].set(1).at[3, other].set(1),
            red_suspicious_process_count=state.red_suspicious_process_count.at[3].set(0).at[3, other].set(1),
            red_privilege=state.red_privilege.at[3]
            .set(COMPROMISE_NONE)
            .at[3, target]
            .set(COMPROMISE_USER)
            .at[3, other]
            .set(COMPROMISE_USER),
            red_scanned_hosts=state.red_scanned_hosts.at[3].set(False).at[3, target].set(True),
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[3].set(False).at[3, target, target].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[3].set(other),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER).at[other].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cy_scanned = set()
        for sess in cyborg_state.sessions["red_agent_3"].values():
            for ip in getattr(sess, "ports", {}).keys():
                host = cyborg_state.ip_addresses.get(ip)
                if host is not None:
                    cy_scanned.add(sorted_hosts.index(host))

        jax_scanned = {h for h in range(int(const.num_hosts)) if bool(new_state.red_scanned_hosts[3, h])}
        assert cy_scanned == set()
        assert jax_scanned == cy_scanned

    def test_remove_uses_blue_pid_budget_when_it_exceeds_jax_suspicious_count_matches_cyborg(self, cyborg_and_jax):
        """If CybORG has more valid suspicious PIDs than JAX suspicious count, remove should follow CybORG PIDs."""
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        for _ in range(5):
            cyborg_state.add_session(
                RedAbstractSession(
                    ident=None,
                    hostname=target_hostname,
                    username="user",
                    agent="red_agent_3",
                    parent=0,
                    session_type="shell",
                    pid=None,
                )
            )

        cy_target_sessions = [s for s in cyborg_state.sessions["red_agent_3"].values() if s.hostname == target_hostname]
        assert len(cy_target_sessions) == 5
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        for sess in cy_target_sessions:
            blue_parent.add_sus_pids(hostname=target_hostname, pid=sess.pid)
        assert len(blue_parent.sus_pids[target_hostname]) == 5

        state = state.replace(
            red_sessions=state.red_sessions.at[3, target].set(True),
            red_session_count=state.red_session_count.at[3, target].set(5),
            red_session_is_abstract=state.red_session_is_abstract.at[3, target].set(True),
            red_privilege=state.red_privilege.at[3, target].set(COMPROMISE_USER),
            # Reproduces mismatch where JAX suspicious count underestimates true blue PID budget.
            red_suspicious_process_count=state.red_suspicious_process_count.at[3, target].set(3),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_remaining = [s for s in cyborg_state.sessions["red_agent_3"].values() if s.hostname == target_hostname]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_remaining)
        expected_priv = COMPROMISE_USER if cyborg_has_user_session else COMPROMISE_NONE

        assert not cyborg_has_user_session
        assert bool(new_state.red_sessions[3, target]) == cyborg_has_user_session
        assert int(new_state.red_privilege[3, target]) == expected_priv
        assert int(new_state.host_compromised[target]) == expected_priv

    def test_remove_with_nine_live_suspicious_pids_clears_all_sessions_matches_cyborg(self, cyborg_and_jax):
        """Regression for seed=5 step=198: Remove must process all live suspicious PIDs on the host."""
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_B")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        for _ in range(9):
            cyborg_state.add_session(
                RedAbstractSession(
                    ident=None,
                    hostname=target_hostname,
                    username="user",
                    agent="red_agent_3",
                    parent=0,
                    session_type="shell",
                    pid=None,
                )
            )

        cy_target_sessions = [s for s in cyborg_state.sessions["red_agent_3"].values() if s.hostname == target_hostname]
        assert len(cy_target_sessions) == 9
        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        for sess in cy_target_sessions:
            blue_parent.add_sus_pids(hostname=target_hostname, pid=sess.pid)
        assert len(blue_parent.sus_pids[target_hostname]) == 9

        state = state.replace(
            red_sessions=state.red_sessions.at[3, target].set(True),
            red_session_count=state.red_session_count.at[3, target].set(9),
            red_session_is_abstract=state.red_session_is_abstract.at[3, target].set(True),
            red_privilege=state.red_privilege.at[3, target].set(COMPROMISE_USER),
            red_suspicious_process_count=state.red_suspicious_process_count.at[3, target].set(9),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_suspicious_process=state.host_suspicious_process.at[target].set(True),
        )

        state = _inject_pid_model_from_cyborg(state, cyborg_state, sorted_hosts)

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx, const=const)
        new_state = _jit_apply_blue(state, const, blue_idx, action_idx)

        cyborg_remaining = [s for s in cyborg_state.sessions["red_agent_3"].values() if s.hostname == target_hostname]
        assert len(cyborg_remaining) == 0
        assert not bool(new_state.red_sessions[3, target])
        assert int(new_state.red_session_count[3, target]) == 0
        assert int(new_state.red_privilege[3, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE


class TestSessionCheckClearsScanMemoryOnAnchorChange:
    """CybORG scan memory (ports dict) lives on session 0 specifically.

    When session 0 is destroyed and RedSessionCheck promotes a new primary
    on a different host, the old session's ports dict is permanently lost.
    apply_red_session_check must clear scan memory when the anchor host changes.
    """

    def test_session_check_clears_scan_memory_when_anchor_moves_to_new_host(self, jax_const):
        """When RedSessionCheck promotes a primary on a different host,
        scan memory must be cleared — CybORG session 0's ports are lost."""
        from jaxborg.actions.red_common import apply_red_session_check

        state = _make_jax_state(jax_const)
        anchor_host = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert anchor_host is not None
        scan_target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_B")
        assert scan_target is not None

        # Find a second host in a different subnet for the new primary
        new_primary_host = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        assert new_primary_host is not None
        assert new_primary_host != anchor_host

        # Red_agent_0 has session on anchor_host (abstract) + second host
        state = state.replace(
            red_sessions=state.red_sessions.at[0, anchor_host].set(True).at[0, new_primary_host].set(True),
            red_session_count=state.red_session_count.at[0, anchor_host].set(1).at[0, new_primary_host].set(1),
            red_session_is_abstract=state.red_session_is_abstract.at[0, anchor_host]
            .set(True)
            .at[0, new_primary_host]
            .set(False),
            red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(anchor_host),
            red_primary_is_abstract=state.red_primary_is_abstract.at[0].set(True),
            # Scan memory sourced from anchor_host
            red_scanned_source_hosts=state.red_scanned_source_hosts.at[0, scan_target, anchor_host].set(True),
            red_scanned_hosts=state.red_scanned_hosts.at[0, scan_target].set(True),
        )

        # Simulate session 0 destruction: CybORG promotes new_primary_host.
        # Use forced_primary_host to move the anchor (as the harness does).
        new_state = apply_red_session_check(
            state,
            jax_const,
            agent_id=0,
            key=jax.random.PRNGKey(42),
            forced_primary_host=jnp.int32(new_primary_host),
        )

        # Anchor moved
        assert int(new_state.red_scan_anchor_host[0]) == new_primary_host

        # Scan memory must be cleared
        assert not bool(new_state.red_scanned_hosts[0, scan_target]), (
            "red_scanned_hosts should be cleared when anchor moves to a "
            "different host — CybORG session 0's ports dict is lost"
        )
        assert not bool(new_state.red_scanned_source_hosts[0, scan_target, anchor_host])
