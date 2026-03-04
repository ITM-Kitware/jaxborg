import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Shared.BlueRewardMachine import BlueRewardMachine
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    MISSION_PHASES,
    NUM_SUBNETS,
    SUBNET_IDS,
)
from jaxborg.rewards import ASF, LWF, RIA, compute_rewards
from jaxborg.state import create_initial_state


class TestPhaseRewardsPopulated:
    def test_phase_rewards_nonzero(self, jax_const):
        pr = np.array(jax_const.phase_rewards)
        assert pr.shape == (MISSION_PHASES, NUM_SUBNETS, 3)
        assert np.any(pr != 0), "phase_rewards should not be all zeros"

    def test_phase_rewards_all_nonpositive(self, jax_const):
        pr = np.array(jax_const.phase_rewards)
        assert np.all(pr <= 0), "all phase rewards should be <= 0"

    def test_operational_zones_phase1_high_penalties(self, jax_const):
        pr = np.array(jax_const.phase_rewards)
        oa = SUBNET_IDS["OPERATIONAL_ZONE_A"]
        assert pr[1, oa, LWF] == -10
        assert pr[1, oa, RIA] == -10

    def test_operational_zones_phase2_high_penalties(self, jax_const):
        pr = np.array(jax_const.phase_rewards)
        ob = SUBNET_IDS["OPERATIONAL_ZONE_B"]
        assert pr[2, ob, LWF] == -10
        assert pr[2, ob, RIA] == -10

    def test_contractor_phase0_lwf_zero(self, jax_const):
        pr = np.array(jax_const.phase_rewards)
        cn = SUBNET_IDS["CONTRACTOR_NETWORK"]
        assert pr[0, cn, LWF] == 0
        assert pr[0, cn, ASF] == -5
        assert pr[0, cn, RIA] == -5

    def test_contractor_phase1_all_zero(self, jax_const):
        pr = np.array(jax_const.phase_rewards)
        cn = SUBNET_IDS["CONTRACTOR_NETWORK"]
        np.testing.assert_array_equal(pr[1, cn], [0, 0, 0])

    def test_internet_phase0(self, jax_const):
        pr = np.array(jax_const.phase_rewards)
        inet = SUBNET_IDS["INTERNET"]
        assert pr[0, inet, LWF] == 0
        assert pr[0, inet, ASF] == 0
        assert pr[0, inet, RIA] == -1


class TestPhaseRewardsMatchCybORG:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_phase_rewards_match_cyborg(self, cyborg_env):
        from jaxborg.topology import (
            CYBORG_SUFFIX_TO_ID,
            build_const_from_cyborg,
        )

        const = build_const_from_cyborg(cyborg_env)
        jax_pr = np.array(const.phase_rewards)

        brm = BlueRewardMachine("")
        for phase in range(MISSION_PHASES):
            table = brm.get_phase_rewards(phase)
            for cyborg_name, rewards in table.items():
                sid = CYBORG_SUFFIX_TO_ID[cyborg_name]
                assert jax_pr[phase, sid, LWF] == rewards["LWF"], f"phase={phase} subnet={cyborg_name} LWF mismatch"
                assert jax_pr[phase, sid, ASF] == rewards["ASF"], f"phase={phase} subnet={cyborg_name} ASF mismatch"
                assert jax_pr[phase, sid, RIA] == rewards["RIA"], f"phase={phase} subnet={cyborg_name} RIA mismatch"

    def test_pure_topology_matches_cyborg_topology(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg, build_topology

        cyborg_const = build_const_from_cyborg(cyborg_env)
        pure_const = build_topology(jax.random.PRNGKey(42), num_steps=500)

        np.testing.assert_array_equal(
            np.array(cyborg_const.phase_rewards),
            np.array(pure_const.phase_rewards),
        )


class TestComputeRewards:
    def test_zero_events_zero_reward(self, jax_const):
        state = create_initial_state()
        no_events = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
        reward = compute_rewards(state, jax_const, no_events, no_events, no_events)
        assert float(reward) == 0.0

    def test_single_impact_gives_ria_penalty(self, jax_const):
        state = create_initial_state()
        no_events = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)

        target = None
        for h in range(int(jax_const.num_hosts)):
            if jax_const.host_active[h] and not jax_const.host_is_router[h]:
                target = h
                break
        assert target is not None

        impact = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_).at[target].set(True)
        reward = compute_rewards(state, jax_const, impact, no_events, no_events)
        subnet = int(jax_const.host_subnet[target])
        expected_ria = float(jax_const.phase_rewards[0, subnet, RIA])
        assert float(reward) == pytest.approx(expected_ria)

    def test_single_lwf_gives_lwf_penalty(self, jax_const):
        state = create_initial_state()
        no_events = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)

        target = None
        for h in range(int(jax_const.num_hosts)):
            if jax_const.host_active[h] and not jax_const.host_is_router[h]:
                target = h
                break
        assert target is not None

        lwf = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_).at[target].set(True)
        reward = compute_rewards(state, jax_const, no_events, lwf, no_events)
        subnet = int(jax_const.host_subnet[target])
        expected = float(jax_const.phase_rewards[0, subnet, LWF])
        assert float(reward) == pytest.approx(expected)

    def test_single_asf_gives_asf_penalty(self, jax_const):
        state = create_initial_state()
        no_events = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)

        target = None
        for h in range(int(jax_const.num_hosts)):
            if jax_const.host_active[h] and not jax_const.host_is_router[h]:
                target = h
                break
        assert target is not None

        asf = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_).at[target].set(True)
        reward = compute_rewards(state, jax_const, no_events, no_events, asf)
        subnet = int(jax_const.host_subnet[target])
        expected = float(jax_const.phase_rewards[0, subnet, ASF])
        assert float(reward) == pytest.approx(expected)

    def test_multiple_events_sum(self, jax_const):
        state = create_initial_state()
        no_events = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)

        targets = []
        for h in range(int(jax_const.num_hosts)):
            if jax_const.host_active[h] and not jax_const.host_is_router[h]:
                targets.append(h)
            if len(targets) == 2:
                break

        impact = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
        expected = 0.0
        for t in targets:
            impact = impact.at[t].set(True)
            subnet = int(jax_const.host_subnet[t])
            expected += float(jax_const.phase_rewards[0, subnet, RIA])

        reward = compute_rewards(state, jax_const, impact, no_events, no_events)
        assert float(reward) == pytest.approx(expected)

    def test_phase_affects_reward(self, jax_const):
        no_events = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)

        oa_host = None
        oa_sid = SUBNET_IDS["OPERATIONAL_ZONE_A"]
        for h in range(int(jax_const.num_hosts)):
            if jax_const.host_active[h] and int(jax_const.host_subnet[h]) == oa_sid and not jax_const.host_is_router[h]:
                oa_host = h
                break
        if oa_host is None:
            pytest.skip("No operational_zone_a host found")

        impact = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_).at[oa_host].set(True)

        state_p0 = create_initial_state()
        r0 = float(compute_rewards(state_p0, jax_const, impact, no_events, no_events))

        state_p1 = create_initial_state().replace(mission_phase=jnp.int32(1))
        r1 = float(compute_rewards(state_p1, jax_const, impact, no_events, no_events))

        assert r0 == -1.0
        assert r1 == -10.0
        assert r1 < r0

    def test_inactive_host_no_reward(self, jax_const):
        state = create_initial_state()
        no_events = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)

        inactive_host = GLOBAL_MAX_HOSTS - 1
        assert not bool(jax_const.host_active[inactive_host])

        impact = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_).at[inactive_host].set(True)
        reward = compute_rewards(state, jax_const, impact, no_events, no_events)
        assert float(reward) == 0.0

    def test_jit_compatible(self, jax_const):
        import jax

        state = create_initial_state()
        no_events = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
        jitted = jax.jit(compute_rewards)
        reward = jitted(state, jax_const, no_events, no_events, no_events)
        assert float(reward) == 0.0


class TestRewardsDifferential:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_zero_reward_on_sleep_matches(self, cyborg_env):
        """Both envs should give 0 reward when all agents sleep."""
        from jaxborg.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        state = create_initial_state()
        no_events = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
        jax_reward = float(compute_rewards(state, const, no_events, no_events, no_events))
        assert jax_reward == 0.0
