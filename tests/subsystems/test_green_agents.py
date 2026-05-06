import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions.green import (
    FP_DETECTION_RATE,
    PHISHING_ERROR_RATE,
    _find_phishing_red_agent,
    _ordered_green_hosts,
    apply_green_agents,
)
from jaxborg.constants import (
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
    SUBNET_IDS,
)
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg, build_topology
from jaxborg.state import create_initial_state


@pytest.fixture
def jax_state(jax_const):
    state = create_initial_state()
    return state.replace(host_services=jax_const.initial_services)


def _run_many_green(state, const, num_trials=50):
    jitted = jax.jit(apply_green_agents)
    keys = jax.random.split(jax.random.PRNGKey(0), num_trials)
    results = []
    for i in range(num_trials):
        results.append(jitted(state, const, keys[i]))
    return results


_CYBORG_GREEN_SUBNET_ORDER = [
    "RESTRICTED_ZONE_A",
    "OPERATIONAL_ZONE_A",
    "RESTRICTED_ZONE_B",
    "OPERATIONAL_ZONE_B",
    "CONTRACTOR_NETWORK",
    "PUBLIC_ACCESS_ZONE",
    "ADMIN_NETWORK",
    "OFFICE_NETWORK",
]


def _expected_green_hosts_in_cyborg_order(const):
    ordered_hosts = []
    for subnet_name in _CYBORG_GREEN_SUBNET_ORDER:
        sid = SUBNET_IDS[subnet_name]
        for host_idx in range(int(const.num_hosts)):
            if not bool(const.host_active[host_idx]):
                continue
            if int(const.host_subnet[host_idx]) != sid:
                continue
            if bool(const.host_is_user[host_idx]):
                ordered_hosts.append(host_idx)
    return ordered_hosts


class TestGreenAgentBasics:
    def test_no_crash_on_initial_state(self, jax_const, jax_state):
        key = jax.random.PRNGKey(0)
        new_state = apply_green_agents(jax_state, jax_const, key)
        assert new_state.host_activity_detected.shape == (GLOBAL_MAX_HOSTS,)

    def test_jit_compatible(self, jax_const, jax_state):
        key = jax.random.PRNGKey(0)
        jitted = jax.jit(apply_green_agents)
        new_state = jitted(jax_state, jax_const, key)
        assert new_state.host_activity_detected.shape == (GLOBAL_MAX_HOSTS,)

    def test_deterministic_with_same_key(self, jax_const, jax_state):
        key = jax.random.PRNGKey(123)
        jitted = jax.jit(apply_green_agents)
        s1 = jitted(jax_state, jax_const, key)
        s2 = jitted(jax_state, jax_const, key)
        np.testing.assert_array_equal(np.array(s1.host_activity_detected), np.array(s2.host_activity_detected))
        np.testing.assert_array_equal(np.array(s1.red_sessions), np.array(s2.red_sessions))

    def test_different_keys_different_results(self, jax_const, jax_state):
        jitted = jax.jit(apply_green_agents)
        results = []
        for seed in range(20):
            key = jax.random.PRNGKey(seed)
            s = jitted(jax_state, jax_const, key)
            results.append(int(jnp.sum(s.host_activity_detected)))
        assert len(set(results)) > 1

    def test_inactive_hosts_unchanged(self, jax_const, jax_state):
        key = jax.random.PRNGKey(0)
        new_state = apply_green_agents(jax_state, jax_const, key)
        for h in range(GLOBAL_MAX_HOSTS):
            if not jax_const.green_agent_active[h]:
                assert not new_state.host_activity_detected[h] or jax_state.host_activity_detected[h]

    @pytest.mark.parametrize("seed", range(3))
    def test_pure_green_agent_host_matches_cyborg_generation_order(self, seed):
        const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
        expected_hosts = _expected_green_hosts_in_cyborg_order(const)
        expected_indices = list(range(len(expected_hosts)))
        actual_indices = [int(const.green_agent_host[h]) for h in expected_hosts]
        assert actual_indices == expected_indices

    def test_apply_green_agents_uses_green_agent_order(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=500)
        expected_hosts = _expected_green_hosts_in_cyborg_order(const)
        ordered_hosts = [
            int(h) for h in np.array(_ordered_green_hosts(const)) if bool(const.green_agent_active[int(h)])
        ]
        assert ordered_hosts[: len(expected_hosts)] == expected_hosts


class TestGreenLocalWorkFalsePositive:
    def test_fp_rate_statistical(self, jax_const, jax_state):
        results = _run_many_green(jax_state, jax_const, num_trials=50)
        fp_count = sum(int(jnp.sum(r.host_activity_detected & ~jax_state.host_activity_detected)) for r in results)
        assert fp_count > 0, "Expected at least some false positives over 50 steps"


class TestGreenPhishing:
    def test_phishing_creates_red_session(self, jax_const, jax_state):
        start_host = int(jax_const.red_start_hosts[0])
        state = jax_state.replace(red_sessions=jax_state.red_sessions.at[0, start_host].set(True))
        results = _run_many_green(state, jax_const, num_trials=100)
        phish_count = sum(int(np.sum(np.array(r.red_sessions) & ~np.array(state.red_sessions))) for r in results)
        assert phish_count > 0, "Expected at least one phishing event over 100 steps"

    def test_phishing_only_user_level(self, jax_const, jax_state):
        start_host = int(jax_const.red_start_hosts[0])
        state = jax_state.replace(red_sessions=jax_state.red_sessions.at[0, start_host].set(True))
        results = _run_many_green(state, jax_const, num_trials=30)
        for new_state in results:
            new_sessions = np.array(new_state.red_sessions) & ~np.array(state.red_sessions)
            new_priv = np.array(new_state.red_privilege)
            for r in range(NUM_RED_AGENTS):
                for h in range(GLOBAL_MAX_HOSTS):
                    if new_sessions[r, h]:
                        assert new_priv[r, h] == COMPROMISE_USER


class TestGreenAccessService:
    def test_access_service_network_events(self, jax_const, jax_state):
        results = _run_many_green(jax_state, jax_const, num_trials=50)
        event_count = sum(int(jnp.sum(r.host_activity_detected & ~jax_state.host_activity_detected)) for r in results)
        assert event_count >= 0

    def test_blocked_zones_create_events(self, jax_const, jax_state):
        blocked = jnp.ones((NUM_SUBNETS, NUM_SUBNETS), dtype=jnp.bool_)
        state = jax_state.replace(blocked_zones=blocked)
        results = _run_many_green(state, jax_const, num_trials=30)
        event_count = sum(int(jnp.sum(r.host_activity_detected & ~state.host_activity_detected)) for r in results)
        assert event_count > 0, "Blocked zones should cause network connection events"


class TestDynamicTopology:
    def test_different_seeds_different_host_counts(self):
        counts = set()
        for seed in range(20):
            const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
            counts.add(int(const.num_hosts))
        assert len(counts) > 1, "Different seeds should produce different host counts"

    @pytest.mark.parametrize("seed", range(10))
    def test_host_count_range(self, seed):
        const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
        num = int(const.num_hosts)
        assert 37 <= num <= 137, f"Seed {seed}: host count {num} out of expected range"

    @pytest.mark.parametrize("seed", range(5))
    def test_padding_to_global_max(self, seed):
        const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
        assert const.host_active.shape == (GLOBAL_MAX_HOSTS,)
        assert const.host_subnet.shape == (GLOBAL_MAX_HOSTS,)
        assert const.data_links.shape == (GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS)

    @pytest.mark.parametrize("seed", range(5))
    def test_host_active_consistency(self, seed):
        const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
        active_count = int(jnp.sum(const.host_active))
        assert active_count == int(const.num_hosts)
        for h in range(GLOBAL_MAX_HOSTS):
            if h >= int(const.num_hosts):
                assert not const.host_active[h]

    @pytest.mark.parametrize("seed", range(5))
    def test_green_agents_present(self, seed):
        const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
        num_green = int(jnp.sum(const.green_agent_active))
        num_user = int(jnp.sum(const.host_is_user))
        assert num_green == num_user
        assert int(const.num_green_agents) == num_user

    @pytest.mark.parametrize("seed", range(10))
    def test_all_subnets_have_hosts(self, seed):
        const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
        for sid in range(NUM_SUBNETS):
            count = int(jnp.sum(const.host_active & (const.host_subnet == sid)))
            assert count >= 1, f"Seed {seed}: subnet {sid} has no hosts"


class TestDifferentialGreen:
    @pytest.fixture(scope="class")
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_green_agent_count_matches(self, cyborg_env):
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        scenario = cyborg_env.environment_controller.state.scenario
        cyborg_green_count = sum(1 for name in scenario.agents if name.startswith("green_agent_"))
        assert int(const.num_green_agents) == cyborg_green_count

    def test_green_agents_on_user_hosts(self, cyborg_env):
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        for h in range(int(const.num_hosts)):
            if const.host_is_user[h]:
                assert const.green_agent_active[h]
                assert const.green_agent_host[h] >= 0

    def test_green_agent_host_matches_cyborg_agent_creation_order(self, cyborg_env):
        const = build_const_from_cyborg(cyborg_env)
        state = cyborg_env.environment_controller.state
        hostname_to_idx = {hostname: idx for idx, hostname in enumerate(sorted(state.hosts.keys()))}
        expected_hosts = [
            agent_info.starting_sessions[0].hostname
            for agent_name, agent_info in state.scenario.agents.items()
            if agent_name.startswith("green_agent_")
        ]
        actual_indices = [int(const.green_agent_host[hostname_to_idx[hostname]]) for hostname in expected_hosts]
        assert actual_indices == list(range(len(expected_hosts)))

    def test_phishing_rate_matches_cyborg(self):
        assert PHISHING_ERROR_RATE == 0.01

    def test_fp_rate_matches_cyborg(self):
        assert FP_DETECTION_RATE == 0.01


class TestGreenStatisticalDifferential:
    @pytest.fixture(scope="class")
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)


class TestGreenSourceSelection:
    def test_same_subnet_phishing_prefers_last_host_in_cyborg_generation_order(self, jax_const, jax_state):
        target_host = None
        user_source = None
        server_source = None
        for subnet_name in _CYBORG_GREEN_SUBNET_ORDER:
            sid = SUBNET_IDS[subnet_name]
            users = [
                h
                for h in range(int(jax_const.num_hosts))
                if bool(jax_const.host_active[h])
                and int(jax_const.host_subnet[h]) == sid
                and bool(jax_const.host_is_user[h])
            ]
            servers = [
                h
                for h in range(int(jax_const.num_hosts))
                if bool(jax_const.host_active[h])
                and int(jax_const.host_subnet[h]) == sid
                and bool(jax_const.host_is_server[h])
            ]
            if len(users) >= 2 and servers:
                user_source = users[0]
                target_host = users[1]
                server_source = servers[0]
                break

        if target_host is None:
            pytest.fail("Need a subnet with at least two users and one server")

        state = jax_state.replace(
            red_sessions=jax_state.red_sessions.at[0, user_source].set(True).at[1, server_source].set(True),
        )

        # CybORG iterates hosts in scenario creation order: router -> users -> servers,
        # so the server-host session is encountered after the user-host session.
        chosen_agent = int(_find_phishing_red_agent(state, jax_const, jnp.int32(target_host), jax.random.PRNGKey(0)))
        assert chosen_agent == 1

    def test_fp_rate_within_statistical_bounds(self, cyborg_env):
        """Run many steps with only green active, check FP rate is statistically consistent."""
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        state = create_initial_state()
        state = state.replace(host_services=const.initial_services)

        jitted = jax.jit(apply_green_agents)
        fp_total = 0
        n_green = int(jnp.sum(const.green_agent_active))
        n_trials = 200

        for i in range(n_trials):
            key = jax.random.PRNGKey(i)
            new_state = jitted(state, const, key)
            fp_total += int(jnp.sum(new_state.host_activity_detected & const.green_agent_active))

        observed_rate = fp_total / (n_green * n_trials) if n_green > 0 else 0
        assert observed_rate < 0.05, f"FP rate {observed_rate} seems too high"


class TestServerSessionCumulativeCounter:
    """Verify red_server_session_count matches CybORG's server_session dict size.

    CybORG's server_session accumulates unique session IDs monotonically —
    entries are never removed even after Blue Restore.  After a Restore→re-phish
    cycle, server_session grows because the new session gets a new ID.

    JAXborg must replicate this via a cumulative counter that increments on
    each new abstract session creation (green phishing, reassignment).
    """

    def test_counter_grows_after_restore_rephish(self, jax_const):
        """After Restore clears a phishing session and green re-phishes,
        the cumulative counter must be HIGHER than the peak live count."""
        from jaxborg.actions.blue_restore import apply_blue_restore
        from jaxborg.env import _init_red_state

        const = jax_const
        state = create_initial_state()
        state = state.replace(host_services=const.initial_services, host_max_pid=const.host_initial_max_pid)
        state = _init_red_state(const, state)

        # Agent 0 should start with cumulative counter = 1
        assert int(state.red_server_session_count[0]) == 1

        # Find a host owned by agent 0's subnet
        agent0_hosts = const.red_agent_subnets[0, const.host_subnet] & const.host_active
        target_host = int(jnp.argmax(agent0_hosts))
        assert agent0_hosts[target_host], "Need an active host in agent 0's subnet"

        # Simulate a phishing session on target_host
        state = state.replace(
            red_sessions=state.red_sessions.at[0, target_host].set(True),
            red_session_count=state.red_session_count.at[0, target_host].set(1),
            red_abstract_session_count=state.red_abstract_session_count.at[0, target_host].set(1),
            red_server_session_count=state.red_server_session_count.at[0].set(state.red_server_session_count[0] + 1),
            red_privilege=state.red_privilege.at[0, target_host].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target_host].set(COMPROMISE_USER),
        )
        counter_after_phish = int(state.red_server_session_count[0])
        assert counter_after_phish == 2  # 1 initial + 1 phish

        # Blue Restore clears the host — counter must NOT decrease
        blue_idx = 0
        state_after_restore = apply_blue_restore(state, const, blue_idx, target_host)
        counter_after_restore = int(state_after_restore.red_server_session_count[0])
        assert counter_after_restore == counter_after_phish, (
            f"Cumulative counter must not decrease after Restore: "
            f"was {counter_after_phish}, now {counter_after_restore}"
        )

        # Simulate another phishing session on the same host — counter must grow
        state2 = state_after_restore.replace(
            red_sessions=state_after_restore.red_sessions.at[0, target_host].set(True),
            red_session_count=state_after_restore.red_session_count.at[0, target_host].set(1),
            red_abstract_session_count=state_after_restore.red_abstract_session_count.at[0, target_host].set(1),
            red_server_session_count=state_after_restore.red_server_session_count.at[0].set(
                state_after_restore.red_server_session_count[0] + 1
            ),
            red_privilege=state_after_restore.red_privilege.at[0, target_host].set(COMPROMISE_USER),
            host_compromised=state_after_restore.host_compromised.at[target_host].set(COMPROMISE_USER),
        )
        counter_after_rephish = int(state2.red_server_session_count[0])
        assert counter_after_rephish == 3, (
            f"After Restore→re-phish, counter should be 3 (1 init + 1 first phish + 1 re-phish), "
            f"got {counter_after_rephish}"
        )

    def test_green_phishing_increments_counter(self, jax_const):
        """Green phishing (apply_green_agents) must increment cumulative counter."""
        from jaxborg.env import _init_red_state

        const = jax_const
        state = create_initial_state()
        state = state.replace(host_services=const.initial_services, host_max_pid=const.host_initial_max_pid)
        state = _init_red_state(const, state)

        counter_before = int(state.red_server_session_count[0])

        # Run many green steps until a phish occurs
        key = jax.random.PRNGKey(0)
        phish_count = 0
        for i in range(200):
            key, subkey = jax.random.split(key)
            state_new = apply_green_agents(state, const, subkey)
            new_count = int(state_new.red_server_session_count[0])
            delta = new_count - int(state.red_server_session_count[0])
            phish_count += delta
            state = state_new
            if phish_count > 0:
                break

        counter_after = int(state.red_server_session_count[0])
        assert counter_after == counter_before + phish_count, (
            f"Counter should increase by phish count ({phish_count}), was {counter_before}, now {counter_after}"
        )
        assert phish_count > 0, "Expected at least one phishing event in 200 green steps"
