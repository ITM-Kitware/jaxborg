"""Multi-session stress test.

Constructs scenarios with 3+ sessions per host from different red agents,
then Remove/Restore. Stresses session count tracking, PID overflow,
and privilege downgrade edge cases.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.blue_remove import apply_blue_remove
from jaxborg.actions.blue_restore import apply_blue_restore
from jaxborg.actions.pids import append_pid_to_row
from jaxborg.constants import (
    COMPROMISE_NONE,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    MAX_TRACKED_SESSION_PIDS,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
    SUBNET_IDS,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import build_topology


@pytest.fixture(scope="module")
def jax_const():
    return build_topology(jax.random.PRNGKey(42), num_steps=100)


def _find_host_in_subnet(const, subnet_name, exclude_router=True):
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


def _find_blue_for_host(const, host_idx):
    for b in range(NUM_BLUE_AGENTS):
        if bool(const.blue_agent_hosts[b, host_idx]):
            return b
    return None


def _setup_multi_session_host(const, target_h, num_agents=3, sessions_per_agent=1, privileged_agents=None):
    """Create state with multiple red agents having sessions on the same host."""
    if privileged_agents is None:
        privileged_agents = set()

    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))

    base_pid = 5000
    pid_counter = 0

    for r in range(min(num_agents, NUM_RED_AGENTS)):
        level = COMPROMISE_PRIVILEGED if r in privileged_agents else COMPROMISE_USER

        for s in range(sessions_per_agent):
            pid = base_pid + pid_counter
            pid_counter += 1

            # Set session
            state = state.replace(
                red_sessions=state.red_sessions.at[r, target_h].set(True),
                red_session_count=state.red_session_count.at[r, target_h].set(state.red_session_count[r, target_h] + 1),
                red_privilege=state.red_privilege.at[r, target_h].set(
                    jnp.maximum(state.red_privilege[r, target_h], level)
                ),
                host_compromised=state.host_compromised.at[target_h].set(
                    jnp.maximum(state.host_compromised[target_h], level)
                ),
                red_session_pids=state.red_session_pids.at[r, target_h].set(
                    append_pid_to_row(state.red_session_pids[r, target_h], pid)
                ),
            )

    return state


class TestMultiAgentMultiSession:
    """Multiple red agents with sessions on same host."""

    def test_three_agents_one_session_each(self, jax_const):
        target = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        assert target is not None
        blue = _find_blue_for_host(jax_const, target)
        assert blue is not None

        state = _setup_multi_session_host(jax_const, target, num_agents=3)

        # Verify setup
        for r in range(3):
            assert bool(state.red_sessions[r, target])
            assert int(state.red_session_count[r, target]) == 1
        assert int(state.host_compromised[target]) == COMPROMISE_USER

    def test_three_agents_remove_one_preserves_compromise(self, jax_const):
        target = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        blue = _find_blue_for_host(jax_const, target)
        if blue is None:
            pytest.skip("No blue agent covers target")

        state = _setup_multi_session_host(jax_const, target, num_agents=3)

        # Add PID 5000 (agent 0's session) to blue suspicious PIDs
        state = state.replace(
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue, target].set(
                append_pid_to_row(state.blue_suspicious_pids[blue, target], 5000)
            )
        )

        new_state = apply_blue_remove(state, jax_const, blue, target)

        # Agent 0's session should be removed
        assert int(new_state.red_session_count[0, target]) == 0
        # Agents 1 and 2 still have sessions
        assert bool(new_state.red_sessions[1, target])
        assert bool(new_state.red_sessions[2, target])
        # Host still compromised (other agents have sessions)
        assert int(new_state.host_compromised[target]) >= COMPROMISE_USER

    def test_mixed_privilege_remove_preserves_max(self, jax_const):
        target = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        blue = _find_blue_for_host(jax_const, target)
        if blue is None:
            pytest.skip("No blue agent covers target")

        # Agent 0: USER, Agent 1: PRIVILEGED
        state = _setup_multi_session_host(jax_const, target, num_agents=2, privileged_agents={1})

        # Add agent 0's PID to suspicious
        state = state.replace(
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue, target].set(
                append_pid_to_row(state.blue_suspicious_pids[blue, target], 5000)
            )
        )

        new_state = apply_blue_remove(state, jax_const, blue, target)

        # Even after removing agent 0, host is still PRIVILEGED via agent 1
        assert int(new_state.host_compromised[target]) == COMPROMISE_PRIVILEGED

    def test_restore_clears_all_agents(self, jax_const):
        target = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        blue = _find_blue_for_host(jax_const, target)
        if blue is None:
            pytest.skip("No blue agent covers target")

        state = _setup_multi_session_host(jax_const, target, num_agents=3)

        new_state = apply_blue_restore(state, jax_const, blue, target)

        # Restore should clear ALL sessions
        for r in range(3):
            assert not bool(new_state.red_sessions[r, target])
            assert int(new_state.red_session_count[r, target]) == 0
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE


class TestMultiSessionSameAgent:
    """Single red agent with multiple sessions on same host."""

    def test_two_sessions_same_agent(self, jax_const):
        target = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        assert target is not None

        state = _setup_multi_session_host(jax_const, target, num_agents=1, sessions_per_agent=2)

        assert int(state.red_session_count[0, target]) == 2
        pids = np.asarray(state.red_session_pids[0, target])
        live_pids = [int(p) for p in pids if p >= 0]
        assert len(live_pids) == 2
        assert 5000 in live_pids
        assert 5001 in live_pids

    def test_remove_one_of_two_sessions(self, jax_const):
        target = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        blue = _find_blue_for_host(jax_const, target)
        if blue is None:
            pytest.skip("No blue agent covers target")

        state = _setup_multi_session_host(jax_const, target, num_agents=1, sessions_per_agent=2)

        # Add only PID 5000 to suspicious
        state = state.replace(
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue, target].set(
                append_pid_to_row(state.blue_suspicious_pids[blue, target], 5000)
            )
        )

        new_state = apply_blue_remove(state, jax_const, blue, target)

        # Should still have one session
        assert bool(new_state.red_sessions[0, target])
        assert int(new_state.red_session_count[0, target]) >= 1
        # Host still compromised
        assert int(new_state.host_compromised[target]) >= COMPROMISE_USER


class TestPIDCapacityStress:
    """Test behavior near PID tracking capacity limits."""

    def test_many_sessions_near_capacity(self, jax_const):
        target = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        assert target is not None

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Fill PID slots to near capacity
        n_pids = min(MAX_TRACKED_SESSION_PIDS - 2, 20)
        pid_row = state.red_session_pids[0, target]
        for i in range(n_pids):
            pid_row = append_pid_to_row(pid_row, 5000 + i)

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(n_pids),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            red_session_pids=state.red_session_pids.at[0, target].set(pid_row),
        )

        # Verify all PIDs are stored
        pids = np.asarray(state.red_session_pids[0, target])
        live_pids = [int(p) for p in pids if p >= 0]
        assert len(live_pids) == n_pids, f"Expected {n_pids} PIDs, got {len(live_pids)}"

    def test_restore_clears_all_pids(self, jax_const):
        target = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        blue = _find_blue_for_host(jax_const, target)
        if blue is None:
            pytest.skip("No blue agent covers target")

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Add 10 PIDs
        pid_row = state.red_session_pids[0, target]
        for i in range(10):
            pid_row = append_pid_to_row(pid_row, 5000 + i)

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(10),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            red_session_pids=state.red_session_pids.at[0, target].set(pid_row),
        )

        new_state = apply_blue_restore(state, jax_const, blue, target)

        # All PIDs should be cleared
        pids = np.asarray(new_state.red_session_pids[0, target])
        live_pids = [int(p) for p in pids if p >= 0]
        assert len(live_pids) == 0, f"Expected 0 PIDs after restore, got {len(live_pids)}"
        assert int(new_state.red_session_count[0, target]) == 0
