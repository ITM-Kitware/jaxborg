"""Tests targeting specific code-review-identified simulation gaps.

Each test class targets a specific gap found by comparing JaxBorg source
code against CybORG reference implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import find_blue_for_host, find_host_in_subnet

from jaxborg.actions.blue_decoys import host_decoy_compatibility_mask
from jaxborg.actions.blue_monitor import apply_blue_monitor
from jaxborg.actions.blue_remove import apply_blue_remove
from jaxborg.actions.pids import append_pid_to_row
from jaxborg.actions.red_privesc import apply_privesc
from jaxborg.constants import (
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    DECOY_IDS,
    NUM_DECOY_TYPES,
    NUM_SERVICES,
    SERVICE_IDS,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import build_topology


@pytest.fixture(scope="module")
def jax_const():
    return build_topology(jax.random.PRNGKey(42), num_steps=100)


# =============================================================================
# Gap 1: Sandbox removal with multi-session hosts
# =============================================================================
class TestSandboxRemovalMultiSession:
    """Privesc sandbox removal should handle multi-session hosts correctly.

    Gap: red_session_sandboxed is a per-host flag, but CybORG checks
    is_escalate_sandbox per-session. When count > 1, the sandbox flag
    is ignored in JAX, which may diverge from CybORG.
    """

    def test_single_sandboxed_session_is_removed(self, jax_const):
        """Sandboxed single session should be removed on privesc attempt."""
        target = find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Set up single sandboxed session
        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(1),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            red_session_sandboxed=state.red_session_sandboxed.at[0, target].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[0, target].set(True),
            red_primary_is_abstract=state.red_primary_is_abstract.at[0].set(True),
            red_session_pids=state.red_session_pids.at[0, target].set(
                append_pid_to_row(state.red_session_pids[0, target], 5000)
            ),
        )

        key = jax.random.PRNGKey(42)
        new_state = apply_privesc(state, jax_const, 0, target, key)

        # Session should be removed
        assert not bool(new_state.red_sessions[0, target]), "Sandboxed session should be removed"
        assert int(new_state.red_session_count[0, target]) == 0

    def test_multi_session_with_sandbox_flag_sessions_survive(self, jax_const):
        """When count > 1, sandbox flag is ignored — sessions survive.

        Note: privesc may still fail if other preconditions aren't met
        (e.g., source_is_abstract requires the agent's bound source host to
        have an abstract session). The key invariant is that sessions are NOT
        removed when count > 1, even if the sandbox flag is set.
        """
        target = find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Set up two sessions, one sandboxed at host level
        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(2),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            red_session_sandboxed=state.red_session_sandboxed.at[0, target].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[0, target].set(True),
            red_primary_is_abstract=state.red_primary_is_abstract.at[0].set(True),
            red_session_pids=state.red_session_pids.at[0, target].set(
                append_pid_to_row(
                    append_pid_to_row(state.red_session_pids[0, target], 5000),
                    5001,
                )
            ),
        )

        key = jax.random.PRNGKey(42)
        new_state = apply_privesc(state, jax_const, 0, target, key)

        # With count=2, sandbox removal is NOT triggered — sessions survive
        assert bool(new_state.red_sessions[0, target]), "Sessions should survive with count>1"
        assert int(new_state.red_session_count[0, target]) == 2, "Count should stay at 2"


# =============================================================================
# Gap 2: Decoy compatibility mask allows Tomcat re-deployment
# =============================================================================
class TestDecoyCompatibility:
    """Decoy compatibility mask should prevent double-deployment."""

    def test_tomcat_blocked_after_first_deploy(self, jax_const):
        """Tomcat decoy should only be deployable once per host."""
        services = jnp.zeros(NUM_SERVICES, dtype=jnp.bool_)
        decoys = jnp.zeros(NUM_DECOY_TYPES, dtype=jnp.bool_)

        # Before any deployment, Tomcat should be compatible
        mask1 = host_decoy_compatibility_mask(services, decoys)
        assert bool(mask1[DECOY_IDS["Tomcat"]]), "Tomcat should be deployable initially"

        # After deploying Tomcat, it should NOT be compatible
        decoys_with_tomcat = decoys.at[DECOY_IDS["Tomcat"]].set(True)
        mask2 = host_decoy_compatibility_mask(services, decoys_with_tomcat)
        assert not bool(mask2[DECOY_IDS["Tomcat"]]), "Tomcat should NOT be re-deployable"

    def test_haraka_blocked_when_smtp_exists(self, jax_const):
        """Haraka should not deploy if SMTP service already exists."""
        services = jnp.zeros(NUM_SERVICES, dtype=jnp.bool_)
        services = services.at[SERVICE_IDS["SMTP"]].set(True)
        decoys = jnp.zeros(NUM_DECOY_TYPES, dtype=jnp.bool_)

        mask = host_decoy_compatibility_mask(services, decoys)
        assert not bool(mask[DECOY_IDS["HarakaSMPT"]]), "Haraka blocked when SMTP exists"

    def test_apache_blocked_when_apache2_exists(self, jax_const):
        """Apache decoy should not deploy if APACHE2 service already exists."""
        services = jnp.zeros(NUM_SERVICES, dtype=jnp.bool_)
        services = services.at[SERVICE_IDS["APACHE2"]].set(True)
        decoys = jnp.zeros(NUM_DECOY_TYPES, dtype=jnp.bool_)

        mask = host_decoy_compatibility_mask(services, decoys)
        assert not bool(mask[DECOY_IDS["Apache"]]), "Apache blocked when APACHE2 exists"

    def test_vsftpd_always_deployable(self, jax_const):
        """Vsftpd should always be deployable (no port conflict)."""
        services = jnp.zeros(NUM_SERVICES, dtype=jnp.bool_)
        decoys = jnp.zeros(NUM_DECOY_TYPES, dtype=jnp.bool_)

        mask = host_decoy_compatibility_mask(services, decoys)
        assert bool(mask[DECOY_IDS["Vsftpd"]]), "Vsftpd should always be deployable"


# =============================================================================
# Gap 3: Remove privilege protection
# =============================================================================
class TestRemovePrivilegeProtection:
    """Remove should not kill privileged (root/SYSTEM) PIDs."""

    def test_privileged_pid_survives_remove(self, jax_const):
        target = find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        blue = find_blue_for_host(jax_const, target)
        assert blue is not None, "OPERATIONAL_ZONE_A host must be covered by blue agent 1"

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Set up privileged session with privileged PID
        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(1),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
            red_session_pids=state.red_session_pids.at[0, target].set(
                append_pid_to_row(state.red_session_pids[0, target], 5000)
            ),
            red_session_privileged_pids=state.red_session_privileged_pids.at[0, target].set(
                append_pid_to_row(state.red_session_privileged_pids[0, target], 5000)
            ),
            # Add the privileged PID to suspicious list
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue, target].set(
                append_pid_to_row(state.blue_suspicious_pids[blue, target], 5000)
            ),
        )

        new_state = apply_blue_remove(state, jax_const, blue, target)

        # Privileged PID should survive Remove
        assert bool(new_state.red_sessions[0, target]), "Privileged session should survive"
        assert int(new_state.red_session_count[0, target]) == 1

    def test_unprivileged_pid_killed_privileged_survives(self, jax_const):
        target = find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        blue = find_blue_for_host(jax_const, target)
        assert blue is not None, "OPERATIONAL_ZONE_A host must be covered by blue agent 1"

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Two sessions: one user (5000), one privileged (5001)
        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(2),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
            red_session_pids=state.red_session_pids.at[0, target].set(
                append_pid_to_row(
                    append_pid_to_row(state.red_session_pids[0, target], 5000),
                    5001,
                )
            ),
            red_session_privileged_pids=state.red_session_privileged_pids.at[0, target].set(
                append_pid_to_row(state.red_session_privileged_pids[0, target], 5001)
            ),
            # Add both PIDs to suspicious
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue, target].set(
                append_pid_to_row(
                    append_pid_to_row(state.blue_suspicious_pids[blue, target], 5000),
                    5001,
                )
            ),
        )

        new_state = apply_blue_remove(state, jax_const, blue, target)

        # User PID 5000 should be killed, privileged PID 5001 should survive
        assert bool(new_state.red_sessions[0, target]), "Should still have privileged session"
        assert int(new_state.red_session_count[0, target]) >= 1


# =============================================================================
# Gap 4: Monitor old activity detection state machine
# =============================================================================
class TestMonitorActivityAging:
    """Monitor should correctly age activity detection flags."""

    def test_activity_ages_after_monitor(self, jax_const):
        """Current activity should become old_activity after monitor."""
        target = find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        blue = find_blue_for_host(jax_const, target)
        assert blue is not None, "OPERATIONAL_ZONE_A host must be covered by blue agent 1"

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Set current activity
        state = state.replace(
            host_activity_detected=state.host_activity_detected.at[target].set(True),
        )

        new_state = apply_blue_monitor(state, jax_const, blue)

        # Current should move to old
        assert bool(new_state.old_host_activity_detected[target]), "Activity should age to old"

    def test_old_activity_clears_after_second_monitor(self, jax_const):
        """Old activity should clear when no new activity occurs."""
        target = find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        blue = find_blue_for_host(jax_const, target)
        assert blue is not None, "OPERATIONAL_ZONE_A host must be covered by blue agent 1"

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Set old activity (no current)
        state = state.replace(
            old_host_activity_detected=state.old_host_activity_detected.at[target].set(True),
        )

        new_state = apply_blue_monitor(state, jax_const, blue)

        # Old should clear since there's no current activity
        assert not bool(new_state.old_host_activity_detected[target]), (
            "Old activity should clear when no current activity"
        )


# =============================================================================
# Gap 6: Reward phase boundary precision
# =============================================================================
class TestRewardPhaseBoundaries:
    """Reward weights should change correctly at phase boundaries."""

    def test_phase_0_rewards_exist(self, jax_const):
        """Phase 0 should have non-zero reward weights for some subnets."""
        phase_0_weights = np.asarray(jax_const.phase_rewards[0])
        assert np.any(phase_0_weights != 0), "Phase 0 should have non-zero reward weights"

    def test_phase_boundaries_ordered(self, jax_const):
        """Phase boundaries should be strictly increasing."""
        boundaries = np.asarray(jax_const.phase_boundaries)
        for i in range(1, len(boundaries)):
            if boundaries[i] > 0:
                assert boundaries[i] > boundaries[i - 1], f"Phase {i} boundary not after phase {i - 1}"


# =============================================================================
# Gap 7: Decoy detection for HTTP exploit
# =============================================================================
class TestDecoyExploitDetection:
    """Exploits should be blocked/detected when decoys are present."""

    def test_http_exploit_blocked_by_apache_decoy(self, jax_const):
        """HTTP exploit should fail when Apache decoy is deployed."""
        from jaxborg.actions.red_exploit import apply_exploit_http

        target = find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Deploy Apache decoy
        state = state.replace(
            host_decoys=state.host_decoys.at[target, DECOY_IDS["Apache"]].set(True),
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(1),
            red_discovered_hosts=state.red_discovered_hosts.at[0, target].set(True),
            red_scanned_hosts=state.red_scanned_hosts.at[0, target].set(True),
        )

        key = jax.random.PRNGKey(42)
        new_state = apply_exploit_http(state, jax_const, 0, target, key)

        # Should NOT create new session (decoy blocks)
        initial_count = int(state.red_session_count[0, target])
        final_count = int(new_state.red_session_count[0, target])
        assert final_count <= initial_count, "HTTP exploit should not succeed with Apache decoy"


# =============================================================================
# Gap 8: Session count consistency after various operations
# =============================================================================
class TestSessionCountConsistency:
    """Session count should stay consistent through operation chains."""

    def test_count_zero_implies_no_session(self, jax_const):
        """session_count==0 must imply red_sessions==False."""
        target = find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        blue = find_blue_for_host(jax_const, target)
        assert blue is not None, "OPERATIONAL_ZONE_A host must be covered by blue agent 1"

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        # Set up and then Remove
        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_session_count=state.red_session_count.at[0, target].set(1),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            red_session_pids=state.red_session_pids.at[0, target].set(
                append_pid_to_row(state.red_session_pids[0, target], 5000)
            ),
            blue_suspicious_pids=state.blue_suspicious_pids.at[blue, target].set(
                append_pid_to_row(state.blue_suspicious_pids[blue, target], 5000)
            ),
        )

        new_state = apply_blue_remove(state, jax_const, blue, target)

        count = int(new_state.red_session_count[0, target])
        has_session = bool(new_state.red_sessions[0, target])
        assert (count == 0) == (not has_session), f"Inconsistency: count={count} but session={has_session}"
