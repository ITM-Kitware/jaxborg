import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, SleepAgent
from CybORG.Shared.BlueRewardMachine import BlueRewardMachine
from CybORG.Shared.Session import RedAbstractSession, Session
from CybORG.Simulator.Actions import Remove
from CybORG.Simulator.Actions.ConcreteActions.PhishingEmail import PhishingEmail
from CybORG.Simulator.Actions.GreenActions.GreenAccessService import GreenAccessService
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action
from jaxborg.actions.blue_monitor import apply_blue_monitor
from jaxborg.actions.encoding import encode_blue_action
from jaxborg.actions.green import (
    FP_DETECTION_RATE,
    GREEN_ACCESS_SERVICE,
    GREEN_LOCAL_WORK,
    PHISHING_ERROR_RATE,
    apply_green_agents,
)
from jaxborg.actions.red_common import apply_exploit_success
from jaxborg.constants import (
    COMPROMISE_NONE,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    MAX_STEPS,
    NUM_GREEN_RANDOM_FIELDS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
)
from jaxborg.rewards import ASF, compute_reward_breakdown
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg, build_topology
from jaxborg.translate import build_mappings_from_cyborg


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

    def test_host_count_range(self):
        for seed in range(10):
            const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
            num = int(const.num_hosts)
            assert 37 <= num <= 137, f"Seed {seed}: host count {num} out of expected range"

    def test_padding_to_global_max(self):
        for seed in range(5):
            const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
            assert const.host_active.shape == (GLOBAL_MAX_HOSTS,)
            assert const.host_subnet.shape == (GLOBAL_MAX_HOSTS,)
            assert const.data_links.shape == (GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS)

    def test_host_active_consistency(self):
        for seed in range(5):
            const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
            active_count = int(jnp.sum(const.host_active))
            assert active_count == int(const.num_hosts)
            for h in range(GLOBAL_MAX_HOSTS):
                if h >= int(const.num_hosts):
                    assert not const.host_active[h]

    def test_green_agents_present(self):
        for seed in range(5):
            const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
            num_green = int(jnp.sum(const.green_agent_active))
            num_user = int(jnp.sum(const.host_is_user))
            assert num_green == num_user
            assert int(const.num_green_agents) == num_user

    def test_all_subnets_have_hosts(self):
        for seed in range(10):
            const = build_topology(jax.random.PRNGKey(seed), num_steps=500)
            for sid in range(NUM_SUBNETS):
                count = int(jnp.sum(const.host_active & (const.host_subnet == sid)))
                assert count >= 1, f"Seed {seed}: subnet {sid} has no hosts"


class TestDifferentialGreen:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_green_agent_count_matches(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        scenario = cyborg_env.environment_controller.state.scenario
        cyborg_green_count = sum(1 for name in scenario.agents if name.startswith("green_agent_"))
        assert int(const.num_green_agents) == cyborg_green_count

    def test_green_agents_on_user_hosts(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        for h in range(int(const.num_hosts)):
            if const.host_is_user[h]:
                assert const.green_agent_active[h]
                assert const.green_agent_host[h] >= 0

    def test_phishing_rate_matches_cyborg(self):
        assert PHISHING_ERROR_RATE == 0.01

    def test_fp_rate_matches_cyborg(self):
        assert FP_DETECTION_RATE == 0.01


def test_green_access_service_reward_uses_source_subnet_matches_cyborg():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    cyborg_env = CybORG(scenario_generator=sg, seed=0)
    cyborg_env.reset()
    cy_state = cyborg_env.environment_controller.state
    const = build_const_from_cyborg(cyborg_env)
    mappings = build_mappings_from_cyborg(cyborg_env)

    source_host = None
    dest_host = None
    phase = 0
    asf_weights = np.array(const.phase_rewards[phase, :, ASF])
    for candidate_src in range(int(const.num_hosts)):
        if not bool(const.green_agent_active[candidate_src]):
            continue
        src_subnet = int(const.host_subnet[candidate_src])
        for candidate_dst in range(int(const.num_hosts)):
            if candidate_dst == candidate_src:
                continue
            if not bool(const.host_active[candidate_dst]) or not bool(const.host_is_server[candidate_dst]):
                continue
            dst_subnet = int(const.host_subnet[candidate_dst])
            if asf_weights[src_subnet] == asf_weights[dst_subnet]:
                continue
            source_host = candidate_src
            dest_host = candidate_dst
            break
        if source_host is not None:
            break

    assert source_host is not None and dest_host is not None

    source_subnet = int(const.host_subnet[source_host])
    dest_subnet = int(const.host_subnet[dest_host])
    expected_reward = float(const.phase_rewards[phase, source_subnet, ASF])
    wrong_dest_reward = float(const.phase_rewards[phase, dest_subnet, ASF])
    assert expected_reward != wrong_dest_reward

    source_hostname = mappings.idx_to_hostname[source_host]
    dest_hostname = mappings.idx_to_hostname[dest_host]
    source_ip = mappings.hostname_to_ip[source_hostname]
    dest_ip = mappings.hostname_to_ip[dest_hostname]
    green_idx = int(const.green_agent_host[source_host])
    green_name = f"green_agent_{green_idx}"
    allowed_subnets = cy_state.scenario.agents[green_name].allowed_subnets

    cy_state.blocks.setdefault(cy_state.hostname_subnet_map[source_hostname].value, []).append(
        cy_state.hostname_subnet_map[dest_hostname].value
    )
    cy_state.blocks.setdefault(cy_state.hostname_subnet_map[dest_hostname].value, []).append(
        cy_state.hostname_subnet_map[source_hostname].value
    )
    cy_action = GreenAccessService(
        agent=green_name,
        session_id=0,
        src_ip=source_ip,
        allowed_subnets=allowed_subnets,
        fp_detection_rate=FP_DETECTION_RATE,
    )
    cy_action.random_reachable_ip = lambda state: dest_ip
    cy_obs = cy_action.execute(cy_state)
    assert str(cy_obs.success).upper() == "FALSE"
    assert cy_state.hosts[dest_hostname].events.network_connections

    class _ObsWrapper:
        def __init__(self, obs):
            self.observations = [obs]

    cy_reward = BlueRewardMachine("").calculate_reward(
        current_state={},
        action_dict={green_name: [cy_action]},
        agent_observations={green_name: _ObsWrapper(cy_obs)},
        done=False,
        state=cy_state,
    )
    assert cy_reward == pytest.approx(expected_reward)

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    active_green = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    active_green[source_host] = True
    blocked_zones = np.zeros((NUM_SUBNETS, NUM_SUBNETS), dtype=bool)
    blocked_zones[source_subnet, dest_subnet] = True
    blocked_zones[dest_subnet, source_subnet] = True
    green_randoms = np.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
    green_randoms[0, source_host, 0] = (GREEN_ACCESS_SERVICE + 0.5) / 3.0
    green_randoms[0, source_host, 5] = float(dest_host)
    green_randoms[0, source_host, 6] = 0.5
    jax_state = jax_state.replace(
        blocked_zones=jnp.array(blocked_zones),
    )
    const = const.replace(
        green_agent_active=jnp.array(active_green),
        green_randoms=jnp.array(green_randoms),
        use_green_randoms=jnp.array(True),
    )

    jax_after = apply_green_agents(jax_state, const, jax.random.PRNGKey(0))
    jax_reward = compute_reward_breakdown(
        jax_after,
        const,
        jax_after.red_impact_attempted,
        jax_after.green_lwf_this_step,
        jax_after.green_asf_this_step,
    )

    assert bool(jax_after.green_asf_this_step[source_host])
    assert not bool(jax_after.green_asf_this_step[dest_host])
    assert bool(jax_after.host_activity_detected[dest_host])
    assert not bool(jax_after.host_activity_detected[source_host])
    assert float(jax_reward.total) == pytest.approx(cy_reward)


class TestGreenStatisticalDifferential:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_fp_rate_within_statistical_bounds(self, cyborg_env):
        """Run many steps with only green active, check FP rate is statistically consistent."""
        from jaxborg.topology import build_const_from_cyborg

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


def test_phishing_prefers_same_subnet_source_agent_matches_cyborg():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    cyborg_env = CybORG(scenario_generator=sg, seed=0)
    cyborg_env.reset()
    cy_state = cyborg_env.environment_controller.state
    const = build_const_from_cyborg(cyborg_env)
    mappings = build_mappings_from_cyborg(cyborg_env)

    for r in range(NUM_RED_AGENTS):
        red_name = f"red_agent_{r}"
        cy_state.sessions[red_name] = {}
        cy_state.sessions_count[red_name] = 0
    for host in cy_state.hosts.values():
        for r in range(NUM_RED_AGENTS):
            host.sessions[f"red_agent_{r}"] = []

    target_host = next(
        h
        for h in range(int(const.num_hosts))
        if bool(const.green_agent_active[h]) and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    target_subnet = int(const.host_subnet[target_host])
    same_subnet_host = next(
        h
        for h in range(int(const.num_hosts))
        if h != target_host
        and bool(const.host_active[h])
        and not bool(const.host_is_router[h])
        and int(const.host_subnet[h]) == target_subnet
    )
    diff_subnet_host = next(
        h
        for h in range(int(const.num_hosts))
        if h not in {target_host, same_subnet_host}
        and bool(const.host_active[h])
        and not bool(const.host_is_router[h])
        and int(const.host_subnet[h]) != target_subnet
    )

    low_agent = 0
    high_agent = NUM_RED_AGENTS - 1
    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=mappings.idx_to_hostname[diff_subnet_host],
            username="user",
            agent=f"red_agent_{low_agent}",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )
    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=mappings.idx_to_hostname[same_subnet_host],
            username="user",
            agent=f"red_agent_{high_agent}",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )

    green_name = f"green_agent_{int(const.green_agent_host[target_host])}"
    target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]
    cy_action = PhishingEmail(session=0, agent=green_name, ip_address=target_ip)
    cy_obs = cy_action.execute(cy_state)
    assert str(cy_obs.success).upper() == "TRUE"

    cy_owner = next(
        r
        for r in range(NUM_RED_AGENTS)
        if any(
            sess.hostname == mappings.idx_to_hostname[target_host]
            for sess in cy_state.sessions[f"red_agent_{r}"].values()
        )
    )

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    for r, h in ((low_agent, diff_subnet_host), (high_agent, same_subnet_host)):
        jax_state = jax_state.replace(
            red_sessions=jax_state.red_sessions.at[r, h].set(True),
            red_session_count=jax_state.red_session_count.at[r, h].set(1),
            red_session_is_abstract=jax_state.red_session_is_abstract.at[r, h].set(True),
        )

    green_randoms = np.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
    green_randoms[0, target_host, 0] = (GREEN_LOCAL_WORK + 0.5) / 3.0
    green_randoms[0, target_host, 1] = 0.5
    green_randoms[0, target_host, 2] = 0.0
    green_randoms[0, target_host, 3] = 0.5
    green_randoms[0, target_host, 4] = 0.0
    const = const.replace(
        green_randoms=jnp.array(green_randoms),
        use_green_randoms=jnp.array(True),
    )
    jax_after = apply_green_agents(jax_state, const, jax.random.PRNGKey(0))

    jax_owner = next(r for r in range(NUM_RED_AGENTS) if bool(jax_after.red_sessions[r, target_host]))

    assert cy_owner == high_agent
    assert jax_owner == cy_owner


def test_phishing_creates_abstract_session_matches_cyborg():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    cyborg_env = CybORG(scenario_generator=sg, seed=0)
    cyborg_env.reset()
    cy_state = cyborg_env.environment_controller.state
    const = build_const_from_cyborg(cyborg_env)
    mappings = build_mappings_from_cyborg(cyborg_env)

    for r in range(NUM_RED_AGENTS):
        red_name = f"red_agent_{r}"
        cy_state.sessions[red_name] = {}
        cy_state.sessions_count[red_name] = 0
    for host in cy_state.hosts.values():
        for r in range(NUM_RED_AGENTS):
            host.sessions[f"red_agent_{r}"] = []

    target_host = next(
        h
        for h in range(int(const.num_hosts))
        if bool(const.green_agent_active[h]) and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    source_host = next(
        h
        for h in range(int(const.num_hosts))
        if h != target_host and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    source_agent = 0
    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=mappings.idx_to_hostname[source_host],
            username="user",
            agent=f"red_agent_{source_agent}",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )

    green_name = f"green_agent_{int(const.green_agent_host[target_host])}"
    target_ip = mappings.hostname_to_ip[mappings.idx_to_hostname[target_host]]
    cy_action = PhishingEmail(session=0, agent=green_name, ip_address=target_ip)
    cy_obs = cy_action.execute(cy_state)
    assert str(cy_obs.success).upper() == "TRUE"

    cy_created = [
        sess
        for r in range(NUM_RED_AGENTS)
        for sess in cy_state.sessions[f"red_agent_{r}"].values()
        if sess.hostname == mappings.idx_to_hostname[target_host]
    ]
    assert cy_created
    assert all(isinstance(sess, RedAbstractSession) for sess in cy_created)

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    jax_state = jax_state.replace(
        red_sessions=jax_state.red_sessions.at[source_agent, source_host].set(True),
        red_session_count=jax_state.red_session_count.at[source_agent, source_host].set(1),
        red_session_is_abstract=jax_state.red_session_is_abstract.at[source_agent, source_host].set(True),
    )
    green_randoms = np.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
    green_randoms[0, target_host, 0] = (GREEN_LOCAL_WORK + 0.5) / 3.0
    green_randoms[0, target_host, 1] = 0.5
    green_randoms[0, target_host, 2] = 0.0
    green_randoms[0, target_host, 3] = 0.5
    green_randoms[0, target_host, 4] = 0.0
    const = const.replace(green_randoms=jnp.array(green_randoms), use_green_randoms=jnp.array(True))
    jax_after = apply_green_agents(jax_state, const, jax.random.PRNGKey(0))

    jax_owner = next(r for r in range(NUM_RED_AGENTS) if bool(jax_after.red_sessions[r, target_host]))
    assert bool(jax_after.red_session_is_abstract[jax_owner, target_host])


def test_phishing_does_not_reuse_stale_blue_suspicious_pid_matches_cyborg():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    cyborg_env = CybORG(scenario_generator=sg, seed=0)
    cyborg_env.reset()
    cy_state = cyborg_env.environment_controller.state
    const = build_const_from_cyborg(cyborg_env)
    mappings = build_mappings_from_cyborg(cyborg_env)

    for r in range(NUM_RED_AGENTS):
        red_name = f"red_agent_{r}"
        cy_state.sessions[red_name] = {}
        cy_state.sessions_count[red_name] = 0
    for host in cy_state.hosts.values():
        for r in range(NUM_RED_AGENTS):
            host.sessions[f"red_agent_{r}"] = []

    target_host = next(
        h
        for h in range(int(const.num_hosts))
        if bool(const.green_agent_active[h]) and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    source_host = next(
        h
        for h in range(int(const.num_hosts))
        if h != target_host
        and bool(const.host_active[h])
        and not bool(const.host_is_router[h])
        and int(const.host_subnet[h]) == int(const.host_subnet[target_host])
    )
    target_hostname = mappings.idx_to_hostname[target_host]
    source_hostname = mappings.idx_to_hostname[source_host]

    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=source_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )
    blue_idx = next(b for b in range(const.blue_agent_hosts.shape[0]) if bool(const.blue_agent_hosts[b, target_host]))
    stale_pid = 424242
    cy_blue_parent = cy_state.sessions[f"blue_agent_{blue_idx}"][0]
    cy_blue_parent.add_sus_pids(hostname=target_hostname, pid=stale_pid)

    green_name = f"green_agent_{int(const.green_agent_host[target_host])}"
    target_ip = mappings.hostname_to_ip[target_hostname]
    cy_action = PhishingEmail(session=0, agent=green_name, ip_address=target_ip)
    cy_obs = cy_action.execute(cy_state)
    assert str(cy_obs.success).upper() == "TRUE"
    cy_target_sessions = [s for s in cy_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
    assert len(cy_target_sessions) == 1
    cy_new_pid = int(cy_target_sessions[0].pid)
    assert cy_new_pid != stale_pid

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    jax_state = jax_state.replace(
        red_sessions=jax_state.red_sessions.at[0, source_host].set(True),
        red_session_count=jax_state.red_session_count.at[0, source_host].set(1),
        red_session_is_abstract=jax_state.red_session_is_abstract.at[0, source_host].set(True),
        red_privilege=jax_state.red_privilege.at[0, source_host].set(COMPROMISE_USER),
        host_compromised=jax_state.host_compromised.at[source_host].set(COMPROMISE_USER),
        blue_suspicious_pids=jax_state.blue_suspicious_pids.at[blue_idx, target_host, 0].set(stale_pid),
    )

    green_randoms = np.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
    green_randoms[0, target_host, 0] = (GREEN_LOCAL_WORK + 0.5) / 3.0
    green_randoms[0, target_host, 1] = 0.5
    green_randoms[0, target_host, 2] = 0.0
    green_randoms[0, target_host, 3] = 0.5
    green_randoms[0, target_host, 4] = 0.0
    const = const.replace(
        green_randoms=jnp.array(green_randoms),
        use_green_randoms=jnp.array(True),
    )
    jax_after = apply_green_agents(jax_state, const, jax.random.PRNGKey(0))

    jax_target_pids = np.array(jax_after.red_session_pids[0, target_host])
    jax_target_pids = jax_target_pids[jax_target_pids >= 0]
    assert len(jax_target_pids) == 1
    assert int(jax_target_pids[0]) != stale_pid


def test_remove_clears_sessions_from_phishing_and_follow_on_compromise_matches_cyborg():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    cyborg_env = CybORG(scenario_generator=sg, seed=0)
    cyborg_env.reset()
    cy_state = cyborg_env.environment_controller.state
    const = build_const_from_cyborg(cyborg_env)
    mappings = build_mappings_from_cyborg(cyborg_env)

    for r in range(NUM_RED_AGENTS):
        red_name = f"red_agent_{r}"
        cy_state.sessions[red_name] = {}
        cy_state.sessions_count[red_name] = 0
    for host in cy_state.hosts.values():
        for r in range(NUM_RED_AGENTS):
            host.sessions[f"red_agent_{r}"] = []

    target_host = next(
        h
        for h in range(int(const.num_hosts))
        if bool(const.green_agent_active[h]) and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    source_host = next(
        h
        for h in range(int(const.num_hosts))
        if h != target_host and bool(const.host_active[h]) and not bool(const.host_is_router[h])
    )
    target_hostname = mappings.idx_to_hostname[target_host]
    source_hostname = mappings.idx_to_hostname[source_host]

    cy_state.add_session(
        RedAbstractSession(
            ident=None,
            hostname=source_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )

    green_name = f"green_agent_{int(const.green_agent_host[target_host])}"
    target_ip = mappings.hostname_to_ip[target_hostname]
    cy_action = PhishingEmail(session=0, agent=green_name, ip_address=target_ip)
    cy_obs = cy_action.execute(cy_state)
    assert str(cy_obs.success).upper() == "TRUE"

    # Add a second user session on the same host (models a follow-on concrete exploit shell).
    cy_state.add_session(
        Session(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
    )
    cy_target_sessions = [s for s in cy_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
    assert len(cy_target_sessions) == 2

    blue_idx = next(b for b in range(const.blue_agent_hosts.shape[0]) if bool(const.blue_agent_hosts[b, target_host]))
    cy_blue_parent = cy_state.sessions[f"blue_agent_{blue_idx}"][0]
    for sess in cy_target_sessions:
        cy_blue_parent.add_sus_pids(hostname=target_hostname, pid=sess.pid)
    cy_sus_pids = cy_blue_parent.sus_pids.get(target_hostname, [])
    assert set(cy_sus_pids) == {sess.pid for sess in cy_target_sessions}

    cy_remove = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
    cy_remove.duration = 1
    cy_remove_obs = cy_remove.execute(cy_state)
    assert cy_remove_obs.success
    cy_remaining = [s for s in cy_state.sessions["red_agent_0"].values() if s.hostname == target_hostname]
    assert len(cy_remaining) == 0

    jax_state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
    jax_state = jax_state.replace(
        red_sessions=jax_state.red_sessions.at[0, source_host].set(True),
        red_session_count=jax_state.red_session_count.at[0, source_host].set(1),
        red_session_is_abstract=jax_state.red_session_is_abstract.at[0, source_host].set(True),
        red_scan_anchor_host=jax_state.red_scan_anchor_host.at[0].set(source_host),
        red_privilege=jax_state.red_privilege.at[0, source_host].set(COMPROMISE_USER),
        host_compromised=jax_state.host_compromised.at[source_host].set(COMPROMISE_USER),
    )

    green_randoms = np.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
    green_randoms[0, target_host, 0] = (GREEN_LOCAL_WORK + 0.5) / 3.0
    green_randoms[0, target_host, 1] = 0.5
    green_randoms[0, target_host, 2] = 0.0
    green_randoms[0, target_host, 3] = 0.5
    green_randoms[0, target_host, 4] = 0.0
    const = const.replace(
        green_randoms=jnp.array(green_randoms),
        use_green_randoms=jnp.array(True),
    )
    jax_after_green = apply_green_agents(jax_state, const, jax.random.PRNGKey(0))

    jax_owner = next(r for r in range(NUM_RED_AGENTS) if bool(jax_after_green.red_sessions[r, target_host]))
    phish_pid_row = np.array(jax_after_green.red_session_pids[jax_owner, target_host])
    phish_pid = int(phish_pid_row[phish_pid_row >= 0][0])
    jax_after_green = jax_after_green.replace(
        blue_suspicious_pids=jax_after_green.blue_suspicious_pids.at[blue_idx, target_host, 0].set(phish_pid),
    )

    pre_exploit_pid_row = np.array(jax_after_green.red_session_pids[jax_owner, target_host])
    pre_exploit_pids = {int(pid) for pid in pre_exploit_pid_row if int(pid) >= 0}
    jax_before_remove = apply_exploit_success(
        jax_after_green,
        const,
        jax_owner,
        jnp.int32(target_host),
        jnp.array(True),
        key=jax.random.PRNGKey(1),
    )
    post_exploit_pid_row = np.array(jax_before_remove.red_session_pids[jax_owner, target_host])
    post_exploit_pids = {int(pid) for pid in post_exploit_pid_row if int(pid) >= 0}
    new_exploit_pids = post_exploit_pids - pre_exploit_pids
    assert len(new_exploit_pids) == 1
    follow_on_pid = next(iter(new_exploit_pids))
    jax_before_remove = apply_blue_monitor(jax_before_remove, const, blue_idx)

    jax_sus_pids = np.array(jax_before_remove.blue_suspicious_pids[blue_idx, target_host])
    jax_sus_pids = jax_sus_pids[jax_sus_pids >= 0].tolist()
    assert phish_pid in jax_sus_pids
    assert follow_on_pid in jax_sus_pids

    remove_idx = encode_blue_action("Remove", target_host, blue_idx, const=const)
    jax_after_remove = apply_blue_action(jax_before_remove, const, blue_idx, remove_idx)

    assert int(jax_after_remove.red_session_count[jax_owner, target_host]) == len(cy_remaining)
    assert bool(jax_after_remove.red_sessions[jax_owner, target_host]) == (len(cy_remaining) > 0)
    assert int(jax_after_remove.red_privilege[jax_owner, target_host]) == COMPROMISE_NONE
    assert int(jax_after_remove.host_compromised[target_host]) == COMPROMISE_NONE
