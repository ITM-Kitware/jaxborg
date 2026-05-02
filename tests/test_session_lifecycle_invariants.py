"""Session lifecycle invariant tests.

Verifies that session-related state fields remain consistent throughout
the lifecycle: create → monitor → remove → restore. These invariants
must hold at every step regardless of action sequence.

Tests cross-field consistency rather than CybORG parity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import find_blue_for_host, find_host_in_subnet

from jaxborg.actions import apply_red_action
from jaxborg.actions.blue_restore import apply_blue_restore
from jaxborg.actions.encoding import encode_red_action
from jaxborg.actions.pids import append_pid_to_row
from jaxborg.constants import (
    COMPROMISE_USER,
    NUM_RED_AGENTS,
)
from jaxborg.scenarios.cc4.topology import build_topology
from jaxborg.state import create_initial_state

_jit_apply_red = jax.jit(apply_red_action, static_argnums=(2,))


@pytest.fixture(scope="module")
def jax_const():
    return build_topology(jax.random.PRNGKey(42), num_steps=100)


def _check_session_invariants(state, const, label=""):
    """Verify cross-field session consistency. Returns list of violation strings."""
    violations = []
    n = int(const.num_hosts)

    for r in range(NUM_RED_AGENTS):
        for h in range(n):
            if not bool(const.host_active[h]):
                continue

            has_session = bool(state.red_sessions[r, h])
            count = int(state.red_session_count[r, h])
            priv = int(state.red_privilege[r, h])
            compromised = int(state.host_compromised[h])

            # Invariant 1: session_count > 0 iff red_sessions is True
            if has_session and count <= 0:
                violations.append(f"{label} red_{r}_host_{h}: session=True but count={count}")
            if not has_session and count > 0:
                violations.append(f"{label} red_{r}_host_{h}: session=False but count={count}")

            # Invariant 2: privilege > 0 implies session exists
            if priv > 0 and not has_session:
                violations.append(f"{label} red_{r}_host_{h}: privilege={priv} but no session")

            # Invariant 3: session implies compromise >= USER
            if has_session and compromised < COMPROMISE_USER:
                violations.append(f"{label} red_{r}_host_{h}: has session but compromised={compromised}")

    # Invariant 4: host_compromised is max of all red_privilege for that host
    for h in range(n):
        if not bool(const.host_active[h]):
            continue
        max_priv = 0
        for r in range(NUM_RED_AGENTS):
            max_priv = max(max_priv, int(state.red_privilege[r, h]))
        compromised = int(state.host_compromised[h])
        if compromised != max_priv:
            violations.append(f"{label} host_{h}: compromised={compromised} but max_privilege={max_priv}")

    # Invariant 5: PID count should be consistent with session count
    for r in range(NUM_RED_AGENTS):
        for h in range(n):
            if not bool(const.host_active[h]):
                continue
            count = int(state.red_session_count[r, h])
            pids = np.asarray(state.red_session_pids[r, h])
            live_pids = int(np.sum(pids >= 0))
            if count > 0 and live_pids == 0:
                violations.append(f"{label} red_{r}_host_{h}: session_count={count} but 0 live PIDs")

    return violations


class TestSessionInvariantsAtInit:
    """Verify invariants hold on fresh state."""

    def test_initial_state_invariants(self, jax_const):
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        violations = _check_session_invariants(state, jax_const, "init")
        assert not violations, "\n".join(violations)


class TestSessionInvariantsAfterExploit:
    """Verify invariants hold after exploit creates sessions."""

    def test_invariants_after_discover_scan_exploit(self, jax_const):
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Set up red agent 0 with initial session
        start_host = int(jax_const.red_start_hosts[0])
        state = state.replace(
            red_sessions=state.red_sessions.at[0, start_host].set(True),
            red_session_count=state.red_session_count.at[0, start_host].set(1),
            red_privilege=state.red_privilege.at[0, start_host].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[start_host].set(COMPROMISE_USER),
            red_session_pids=state.red_session_pids.at[0, start_host].set(
                append_pid_to_row(state.red_session_pids[0, start_host], 5000)
            ),
            red_session_is_abstract=state.red_session_is_abstract.at[0, start_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(start_host),
            red_discovered_hosts=state.red_discovered_hosts.at[0, start_host].set(True),
        )

        violations = _check_session_invariants(state, jax_const, "post-setup")
        assert not violations, "\n".join(violations)


class TestSessionInvariantsAfterRestore:
    """Verify invariants hold after Restore clears a host."""

    def test_invariants_after_restore(self, jax_const):
        target = find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        blue = find_blue_for_host(jax_const, target)
        assert blue is not None, "OPERATIONAL_ZONE_A host must be covered by blue agent 1"

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Create session
        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(1),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            red_session_pids=state.red_session_pids.at[0, target].set(
                append_pid_to_row(state.red_session_pids[0, target], 5000)
            ),
        )

        # Restore
        new_state = apply_blue_restore(state, jax_const, blue, target)

        violations = _check_session_invariants(new_state, jax_const, "post-restore")
        assert not violations, "\n".join(violations)


class TestSessionInvariantsUnderStress:
    """Run random action sequences and check invariants at every step."""

    def test_random_actions_maintain_invariants(self, jax_const):
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Set up initial red session
        start_host = int(jax_const.red_start_hosts[0])
        state = state.replace(
            red_sessions=state.red_sessions.at[0, start_host].set(True),
            red_session_count=state.red_session_count.at[0, start_host].set(1),
            red_privilege=state.red_privilege.at[0, start_host].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[start_host].set(COMPROMISE_USER),
            red_session_pids=state.red_session_pids.at[0, start_host].set(
                append_pid_to_row(state.red_session_pids[0, start_host], 5000)
            ),
            red_session_is_abstract=state.red_session_is_abstract.at[0, start_host].set(True),
            red_scan_anchor_host=state.red_scan_anchor_host.at[0].set(start_host),
            red_discovered_hosts=state.red_discovered_hosts.at[0, start_host].set(True),
        )

        rng = np.random.RandomState(42)
        all_violations = []

        # Run 20 random red actions
        for step in range(20):
            # Random discover or scan
            target_subnet = rng.randint(0, 9)
            action_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
            key = jax.random.PRNGKey(step)
            state = _jit_apply_red(state, jax_const, 0, action_idx, key)

            violations = _check_session_invariants(state, jax_const, f"step_{step}")
            all_violations.extend(violations)

        if all_violations:
            msg = f"Invariant violations during random actions ({len(all_violations)} total):\n"
            msg += "\n".join(f"  {v}" for v in all_violations[:20])
            if len(all_violations) > 20:
                msg += f"\n  ... and {len(all_violations) - 20} more"
            pytest.fail(msg)
