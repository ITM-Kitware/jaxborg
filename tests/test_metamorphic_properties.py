"""Pure-JAX metamorphic property tests for simulation invariants.

No CybORG dependency — uses build_topology() directly to create a JAX topology.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.blue_monitor import apply_blue_monitor
from jaxborg.actions.blue_restore import apply_blue_restore
from jaxborg.actions.encoding import (
    encode_red_action,
)
from jaxborg.constants import (
    COMPROMISE_NONE,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
    SUBNET_IDS,
)
from jaxborg.rewards import compute_rewards
from jaxborg.state import create_initial_state
from jaxborg.topology import build_topology

_jit_apply_red = jax.jit(apply_red_action, static_argnums=(2,))
_jit_apply_blue = jax.jit(apply_blue_action, static_argnums=(2,))


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


def _make_state(const):
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


# ---------------------------------------------------------------------------
# 1. Restore on a clean host is a no-op
# ---------------------------------------------------------------------------


def test_restore_clean_host_is_noop(jax_const):
    """Restoring an uncompromised host should leave state unchanged."""
    const = jax_const
    state = _make_state(const)

    # Find an uncompromised host covered by some blue agent
    target = None
    blue_idx = None
    for h in range(int(const.num_hosts)):
        if not bool(const.host_active[h]):
            continue
        if bool(const.host_is_router[h]):
            continue
        # Must not be the red start host
        if int(state.red_sessions[0, h]):
            continue
        b = _find_blue_for_host(const, h)
        if b is not None:
            target = h
            blue_idx = b
            break
    assert target is not None, "No clean, blue-covered host found"

    # Confirm host is clean
    assert int(state.host_compromised[target]) == COMPROMISE_NONE

    state2 = apply_blue_restore(state, const, blue_idx, target)

    np.testing.assert_array_equal(np.array(state.red_sessions), np.array(state2.red_sessions))
    np.testing.assert_array_equal(np.array(state.host_compromised), np.array(state2.host_compromised))
    np.testing.assert_array_equal(np.array(state.host_services), np.array(state2.host_services))


# ---------------------------------------------------------------------------
# 2. Reward computation is deterministic
# ---------------------------------------------------------------------------


def test_reward_deterministic(jax_const):
    """Computing rewards twice with identical inputs must produce equal results."""
    const = jax_const
    state = _make_state(const)

    r1 = compute_rewards(
        state,
        const,
        state.red_impact_attempted,
        state.green_lwf_this_step,
        state.green_asf_this_step,
    )
    r2 = compute_rewards(
        state,
        const,
        state.red_impact_attempted,
        state.green_lwf_this_step,
        state.green_asf_this_step,
    )

    np.testing.assert_array_equal(np.array(r1), np.array(r2))


# ---------------------------------------------------------------------------
# 3. Session count bounded by number of exploit attempts
# ---------------------------------------------------------------------------


def test_session_count_bounded_by_exploit_successes(jax_const):
    """After N exploit attempts, session_count[agent, host] <= N."""
    const = jax_const
    state = _make_state(const)

    agent_id = 0
    start_host = int(const.red_start_hosts[agent_id])

    # Find a target host that is scanned and reachable: set up scan memory
    # so exploit preconditions pass. Pick a host in the same subnet.
    start_subnet = int(const.host_subnet[start_host])
    target = None
    for h in range(int(const.num_hosts)):
        if not bool(const.host_active[h]):
            continue
        if h == start_host:
            continue
        if int(const.host_subnet[h]) != start_subnet:
            continue
        if bool(const.host_is_router[h]):
            continue
        target = h
        break

    if target is None:
        pytest.skip("No suitable target host in start subnet")

    # Manually set up scan and discovery state so exploit preconditions pass
    state = state.replace(
        red_discovered_hosts=state.red_discovered_hosts.at[agent_id, target].set(True),
        red_scanned_hosts=state.red_scanned_hosts.at[agent_id, target].set(True),
        red_scanned_source_hosts=state.red_scanned_source_hosts.at[agent_id, target, start_host].set(True),
        red_session_is_abstract=state.red_session_is_abstract.at[agent_id, start_host].set(True),
        red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(start_host),
    )

    N = 5
    action_idx = encode_red_action("ExploitRemoteService_cc4SSHBruteForce", target, agent_id)
    key = jax.random.PRNGKey(99)

    for i in range(N):
        key, subkey = jax.random.split(key)
        state = _jit_apply_red(state, const, agent_id, action_idx, subkey)

    session_count = int(state.red_session_count[agent_id, target])
    assert session_count <= N, f"session_count={session_count} exceeds {N} exploit attempts"


# ---------------------------------------------------------------------------
# 4. Zero sessions across all red agents implies host_compromised == 0
# ---------------------------------------------------------------------------


def test_removing_all_sessions_clears_compromise(jax_const):
    """If red_session_count is 0 for all agents on a host, host_compromised must be 0.

    Tests this as a property across multiple manually constructed states.
    """
    const = jax_const

    for compromise_level in (COMPROMISE_USER, COMPROMISE_PRIVILEGED):
        # Build a state where a host is marked compromised but all sessions are gone
        state = _make_state(const)

        target = None
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]):
                continue
            if bool(const.host_is_router[h]):
                continue
            if int(state.red_sessions[0, h]):
                continue
            target = h
            break
        assert target is not None

        # Manually compromise, then clear sessions — simulating blue Remove
        state = state.replace(
            host_compromised=state.host_compromised.at[target].set(compromise_level),
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(1),
            red_privilege=state.red_privilege.at[0, target].set(compromise_level),
        )

        # Now use Restore to clear everything (a blue agent that covers this host)
        blue_idx = _find_blue_for_host(const, target)
        if blue_idx is None:
            continue  # skip if no blue covers this host

        state2 = apply_blue_restore(state, const, blue_idx, target)

        # Verify invariant: no sessions => no compromise
        for r in range(NUM_RED_AGENTS):
            assert int(state2.red_session_count[r, target]) == 0
        assert int(state2.host_compromised[target]) == COMPROMISE_NONE, (
            f"host_compromised should be 0 after all sessions cleared, got {int(state2.host_compromised[target])}"
        )


# ---------------------------------------------------------------------------
# 5. Monitor on uncovered host is a no-op for that host
# ---------------------------------------------------------------------------


def test_monitor_on_uncovered_host_is_noop(jax_const):
    """Monitor by an agent should not change detection state on uncovered hosts."""
    const = jax_const
    state = _make_state(const)

    # Pick blue agent 0 and find a host it does NOT cover
    agent_id = 0
    uncovered_host = None
    for h in range(int(const.num_hosts)):
        if not bool(const.host_active[h]):
            continue
        if not bool(const.blue_agent_hosts[agent_id, h]):
            uncovered_host = h
            break
    if uncovered_host is None:
        pytest.skip("All active hosts are covered by blue_agent_0")

    # Set some detection flags on the uncovered host
    state = state.replace(
        red_activity_this_step=state.red_activity_this_step.at[uncovered_host].set(1),
        host_activity_detected=state.host_activity_detected.at[uncovered_host].set(True),
        host_exploit_detected=state.host_exploit_detected.at[uncovered_host].set(True),
    )

    state2 = apply_blue_monitor(state, const, agent_id=agent_id)

    # The uncovered host's old detection state should not have been modified
    # by a monitor action from agent 0. CybORG only ages events on covered hosts.
    # The host_exploit_detected flag updates globally (based on process creation
    # events), but old_host_exploit_detected should not be set for uncovered hosts.
    assert not bool(state2.old_host_activity_detected[uncovered_host]), (
        "Monitor should not age activity for uncovered host"
    )


# ---------------------------------------------------------------------------
# 6. Blocked zones prevent cross-subnet exploit
# ---------------------------------------------------------------------------


def test_blocking_prevents_cross_subnet_exploit(jax_const):
    """With blocked_zones[dst, src] = True, exploit from src to dst should fail."""
    const = jax_const
    agent_id = 0
    state = _make_state(const)

    start_host = int(const.red_start_hosts[agent_id])
    src_subnet = int(const.host_subnet[start_host])

    # Find a target in a different subnet
    target = None
    dst_subnet = None
    for h in range(int(const.num_hosts)):
        if not bool(const.host_active[h]):
            continue
        if bool(const.host_is_router[h]):
            continue
        h_subnet = int(const.host_subnet[h])
        if h_subnet == src_subnet:
            continue
        target = h
        dst_subnet = h_subnet
        break

    if target is None:
        pytest.skip("No target host in a different subnet from red start")

    # Set up scan/discovery prereqs so the exploit would normally be attempted
    state = state.replace(
        red_discovered_hosts=state.red_discovered_hosts.at[agent_id, target].set(True),
        red_scanned_hosts=state.red_scanned_hosts.at[agent_id, target].set(True),
        red_scanned_source_hosts=state.red_scanned_source_hosts.at[agent_id, target, start_host].set(True),
        red_session_is_abstract=state.red_session_is_abstract.at[agent_id, start_host].set(True),
        red_scan_anchor_host=state.red_scan_anchor_host.at[agent_id].set(start_host),
    )

    # Block traffic from src_subnet to dst_subnet
    blocked = state.blocked_zones.at[dst_subnet, src_subnet].set(True)
    state = state.replace(blocked_zones=blocked)

    # Remember session count before
    sessions_before = int(state.red_session_count[agent_id, target])

    # Attempt SSH exploit
    action_idx = encode_red_action("ExploitRemoteService_cc4SSHBruteForce", target, agent_id)
    key = jax.random.PRNGKey(123)
    state2 = _jit_apply_red(state, const, agent_id, action_idx, key)

    sessions_after = int(state2.red_session_count[agent_id, target])
    assert sessions_after == sessions_before, (
        f"Exploit should fail when zone is blocked: sessions went from {sessions_before} to {sessions_after}"
    )
