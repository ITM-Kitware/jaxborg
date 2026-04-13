"""Unit tests for green agent precomputed random mode (no CybORG dependency)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.green import (
    GREEN_ACCESS_SERVICE,
    GREEN_LOCAL_WORK,
    GREEN_SLEEP,
    NUM_GREEN_ACTIONS,
    apply_green_agents,
)
from jaxborg.constants import GLOBAL_MAX_HOSTS, MAX_STEPS, NUM_GREEN_RANDOM_FIELDS, NUM_SERVICES
from jaxborg.state import create_initial_state


@pytest.fixture
def jax_state(jax_const):
    state = create_initial_state()
    return state.replace(host_services=jax_const.initial_services)


def _first_active_green_host(const):
    for h in range(GLOBAL_MAX_HOSTS):
        if const.green_agent_active[h]:
            return h
    raise RuntimeError("No active green hosts")


def _make_precomputed_const(state, const, overrides=None):
    """Create a const with precomputed green randoms, optionally overriding specific fields."""
    randoms = np.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
    randoms[:, :, 0] = 0.5
    if overrides:
        for (t, h, f), val in overrides.items():
            randoms[t, h, f] = val
    return const.replace(
        green_randoms=jnp.array(randoms),
        use_green_randoms=jnp.array(True),
    )


class TestPrecomputedActionSelection:
    def test_sleep_selected(self, jax_const, jax_state):
        assert int(jnp.floor(jnp.array(0.1) * NUM_GREEN_ACTIONS)) == GREEN_SLEEP

    def test_local_work_selected(self, jax_const, jax_state):
        assert int(jnp.floor(jnp.array(0.5) * NUM_GREEN_ACTIONS)) == GREEN_LOCAL_WORK

    def test_access_service_selected(self, jax_const, jax_state):
        assert int(jnp.floor(jnp.array(0.8) * NUM_GREEN_ACTIONS)) == GREEN_ACCESS_SERVICE


class TestPrecomputedFP:
    def test_fp_triggers_when_roll_below_threshold(self, jax_const, jax_state):
        h = _first_active_green_host(jax_const)
        overrides = {
            (0, h, 0): (GREEN_LOCAL_WORK + 0.5) / NUM_GREEN_ACTIONS,
            (0, h, 2): 0.01,  # reliability passes (low roll)
            (0, h, 3): 0.001,  # FP roll below 0.01 threshold
        }
        has_service = bool(jnp.any(jax_state.host_services[h]))
        if not has_service:
            pytest.fail("Host has no services")
        const = _make_precomputed_const(jax_state, jax_const, overrides)
        result = apply_green_agents(jax_state, const, jax.random.PRNGKey(0))
        # GreenLocalWork FP creates process_creation events (host_exploit_detected)
        assert result.host_exploit_detected[h]

    def test_fp_does_not_trigger_when_roll_above_threshold(self, jax_const, jax_state):
        h = _first_active_green_host(jax_const)
        overrides = {
            (0, h, 0): (GREEN_LOCAL_WORK + 0.5) / NUM_GREEN_ACTIONS,
            (0, h, 2): 0.01,  # reliability passes
            (0, h, 3): 0.5,  # FP roll well above threshold
            (0, h, 4): 0.5,  # phishing roll well above threshold
        }
        has_service = bool(jnp.any(jax_state.host_services[h]))
        if not has_service:
            pytest.fail("Host has no services")
        const = _make_precomputed_const(jax_state, jax_const, overrides)
        result = apply_green_agents(jax_state, const, jax.random.PRNGKey(0))
        assert not result.host_activity_detected[h]


class TestPrecomputedPhishing:
    def test_phishing_creates_session_when_roll_below_threshold(self, jax_const, jax_state):
        h = _first_active_green_host(jax_const)
        start_host = int(jax_const.red_start_hosts[0])
        state = jax_state.replace(
            red_sessions=jax_state.red_sessions.at[0, start_host].set(True),
        )
        overrides = {
            (0, h, 0): (GREEN_LOCAL_WORK + 0.5) / NUM_GREEN_ACTIONS,
            (0, h, 2): 0.01,
            (0, h, 3): 0.5,
            (0, h, 4): 0.001,  # phishing triggers
        }
        has_service = bool(jnp.any(state.host_services[h]))
        if not has_service:
            pytest.fail("Host has no services")
        const = _make_precomputed_const(state, jax_const, overrides)
        result = apply_green_agents(state, const, jax.random.PRNGKey(0))
        new_sessions = np.array(result.red_sessions) & ~np.array(jax_state.red_sessions)
        if np.any(new_sessions[:, h]):
            assert True
        else:
            pytest.fail("Phishing red agent not reachable from this host")

    def test_phishing_does_not_trigger_when_roll_above_threshold(self, jax_const, jax_state):
        h = _first_active_green_host(jax_const)
        overrides = {
            (0, h, 0): (GREEN_LOCAL_WORK + 0.5) / NUM_GREEN_ACTIONS,
            (0, h, 2): 0.01,
            (0, h, 3): 0.5,
            (0, h, 4): 0.5,  # phishing roll above threshold
        }
        const = _make_precomputed_const(jax_state, jax_const, overrides)
        result = apply_green_agents(jax_state, const, jax.random.PRNGKey(0))
        np.testing.assert_array_equal(np.array(result.red_sessions), np.array(jax_state.red_sessions))


class TestPrecomputedReliability:
    def test_work_fails_when_roll_above_reliability(self, jax_const, jax_state):
        h = _first_active_green_host(jax_const)
        has_service = bool(jnp.any(jax_state.host_services[h]))
        if not has_service:
            pytest.fail("Host has no services")
        degraded_reliability = jax_state.host_service_reliability.at[h].set(jnp.full(NUM_SERVICES, 50, dtype=jnp.int32))
        state = jax_state.replace(host_service_reliability=degraded_reliability)
        overrides = {
            (0, h, 0): (GREEN_LOCAL_WORK + 0.5) / NUM_GREEN_ACTIONS,
            (0, h, 2): 0.99,  # floor(0.99 * 100) = 99 >= 50, so fails
        }
        const = _make_precomputed_const(state, jax_const, overrides)
        result = apply_green_agents(state, const, jax.random.PRNGKey(0))
        assert result.green_lwf_this_step[h]

    def test_work_succeeds_when_roll_below_reliability(self, jax_const, jax_state):
        h = _first_active_green_host(jax_const)
        overrides = {
            (0, h, 0): (GREEN_LOCAL_WORK + 0.5) / NUM_GREEN_ACTIONS,
            (0, h, 2): 0.01,  # low roll -> passes reliability
            (0, h, 3): 0.5,
            (0, h, 4): 0.5,
        }
        has_service = bool(jnp.any(jax_state.host_services[h]))
        if not has_service:
            pytest.fail("Host has no services")
        const = _make_precomputed_const(jax_state, jax_const, overrides)
        result = apply_green_agents(jax_state, const, jax.random.PRNGKey(0))
        assert not result.green_lwf_this_step[h]


class TestPrecomputedAccessServiceBlocked:
    def test_access_blocked_creates_event(self, jax_const, jax_state):
        from jaxborg.constants import NUM_SUBNETS

        h = _first_active_green_host(jax_const)
        blocked = jnp.ones((NUM_SUBNETS, NUM_SUBNETS), dtype=jnp.bool_)
        state = jax_state.replace(blocked_zones=blocked)
        overrides = {
            (0, h, 0): (GREEN_ACCESS_SERVICE + 0.5) / NUM_GREEN_ACTIONS,
            (0, h, 5): 0.5,  # some dest host
        }
        const = _make_precomputed_const(state, jax_const, overrides)
        result = apply_green_agents(state, const, jax.random.PRNGKey(0))
        assert jnp.any(result.green_asf_this_step) or jnp.any(result.host_activity_detected)


class TestJITCompatibility:
    def test_precomputed_mode_jit_compatible(self, jax_const, jax_state):
        const = _make_precomputed_const(jax_state, jax_const)
        jitted = jax.jit(apply_green_agents)
        result = jitted(jax_state, const, jax.random.PRNGKey(0))
        assert result.host_activity_detected.shape == (GLOBAL_MAX_HOSTS,)

    def test_precomputed_deterministic(self, jax_const, jax_state):
        const = _make_precomputed_const(jax_state, jax_const)
        jitted = jax.jit(apply_green_agents)
        r1 = jitted(jax_state, const, jax.random.PRNGKey(0))
        r2 = jitted(jax_state, const, jax.random.PRNGKey(999))
        np.testing.assert_array_equal(
            np.array(r1.host_activity_detected),
            np.array(r2.host_activity_detected),
        )
        np.testing.assert_array_equal(np.array(r1.red_sessions), np.array(r2.red_sessions))

    def test_precomputed_differs_from_rng(self, jax_const, jax_state):
        """Precomputed mode should ignore the JAX key."""
        overrides = {}
        for hh in range(GLOBAL_MAX_HOSTS):
            if jax_const.green_agent_active[hh]:
                overrides[(0, hh, 0)] = 0.1  # force all to SLEEP
        const = _make_precomputed_const(jax_state, jax_const, overrides)
        result = apply_green_agents(jax_state, const, jax.random.PRNGKey(0))
        assert not jnp.any(result.green_lwf_this_step)
        assert not jnp.any(result.green_asf_this_step)
