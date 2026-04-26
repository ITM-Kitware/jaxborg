import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    MESSAGE_LENGTH,
    NUM_BLUE_AGENTS,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
    OBS_VECTOR_HOSTS_PER_SUBNET,
    SUBNET_IDS,
)
from jaxborg.observations import JAX_ID_TO_CYBORG_POS, get_blue_obs
from jaxborg.state import create_initial_state
from jaxborg.scenarios.cc4.topology import BLUE_AGENT_SUBNETS, build_const_from_cyborg

OBS_SIZE = 210
NUM_MESSAGES = 4
SUBNET_BLOCK_SIZE = 59
NUM_HQ_SUBNETS = 3
MESSAGE_SECTION_SIZE = NUM_MESSAGES * MESSAGE_LENGTH


@pytest.fixture
def jax_state(jax_const):
    state = create_initial_state()
    return state.replace(host_services=jax_const.initial_services)


def _get_subnet_block(obs, slot):
    start = 1 + slot * SUBNET_BLOCK_SIZE
    return np.array(obs[start : start + SUBNET_BLOCK_SIZE])


def _message_start(const, agent_id):
    num_subnets = int(np.sum(np.array(const.blue_obs_subnets[agent_id]) >= 0))
    return 1 + num_subnets * SUBNET_BLOCK_SIZE


class TestBlueObsShape:
    def test_obs_shape_all_agents(self, jax_const, jax_state):
        for agent_id in range(NUM_BLUE_AGENTS):
            obs = get_blue_obs(jax_state, jax_const, agent_id)
            assert obs.shape == (OBS_SIZE,), f"agent {agent_id}: got shape {obs.shape}"

    def test_obs_dtype_is_numeric(self, jax_const, jax_state):
        obs = get_blue_obs(jax_state, jax_const, 0)
        assert jnp.issubdtype(obs.dtype, jnp.number)


class TestBlueObsStructure:
    def test_mission_phase_is_first_element(self, jax_const, jax_state):
        obs = get_blue_obs(jax_state, jax_const, 0)
        assert int(obs[0]) == int(jax_state.mission_phase)

    def test_mission_phase_changes_with_state(self, jax_const, jax_state):
        state_p1 = jax_state.replace(mission_phase=jnp.int32(1))
        obs = get_blue_obs(state_p1, jax_const, 0)
        assert int(obs[0]) == 1

    def test_mission_phase_2(self, jax_const, jax_state):
        state_p2 = jax_state.replace(mission_phase=jnp.int32(2))
        obs = get_blue_obs(state_p2, jax_const, 0)
        assert int(obs[0]) == 2

    def test_messages_are_at_cyborg_position(self, jax_const, jax_state):
        obs = get_blue_obs(jax_state, jax_const, 0)
        start = _message_start(jax_const, 0)
        msg_section = obs[start : start + MESSAGE_SECTION_SIZE]
        np.testing.assert_array_equal(np.array(msg_section), np.zeros(MESSAGE_SECTION_SIZE))

    def test_messages_reflect_state(self, jax_const, jax_state):
        msgs = jax_state.messages.at[1, 0, :].set(jnp.ones(MESSAGE_LENGTH))
        state = jax_state.replace(messages=msgs)
        obs = get_blue_obs(state, jax_const, 0)
        start = _message_start(jax_const, 0)
        msg_section = np.array(obs[start : start + MESSAGE_SECTION_SIZE])
        assert np.any(msg_section != 0), "message should appear in observation"

    def test_padding_tail_zero_for_single_subnet_agents(self, jax_const, jax_state):
        obs = get_blue_obs(jax_state, jax_const, 0)
        start = _message_start(jax_const, 0)
        tail = np.array(obs[start + MESSAGE_SECTION_SIZE :])
        np.testing.assert_array_equal(tail, np.zeros_like(tail))


class TestSubnetOneHot:
    def test_agent0_single_subnet_rza(self, jax_const, jax_state):
        obs = get_blue_obs(jax_state, jax_const, 0)
        one_hot = _get_subnet_block(obs, 0)[:NUM_SUBNETS]
        cyborg_pos = int(JAX_ID_TO_CYBORG_POS[SUBNET_IDS["RESTRICTED_ZONE_A"]])
        expected = np.zeros(NUM_SUBNETS)
        expected[cyborg_pos] = 1.0
        np.testing.assert_array_equal(one_hot, expected)

    def test_agent0_unused_slots_zero(self, jax_const, jax_state):
        obs = get_blue_obs(jax_state, jax_const, 0)
        block1 = _get_subnet_block(obs, 1)
        block2 = _get_subnet_block(obs, 2)
        np.testing.assert_array_equal(block1, np.zeros(SUBNET_BLOCK_SIZE))
        np.testing.assert_array_equal(block2, np.zeros(SUBNET_BLOCK_SIZE))

    def test_agent4_three_subnets(self, jax_const, jax_state):
        obs = get_blue_obs(jax_state, jax_const, 4)
        expected_jax_ids = sorted(SUBNET_IDS[s] for s in BLUE_AGENT_SUBNETS[4])
        cyborg_positions = [int(JAX_ID_TO_CYBORG_POS[sid]) for sid in expected_jax_ids]

        for slot, cyborg_pos in enumerate(cyborg_positions):
            one_hot = _get_subnet_block(obs, slot)[:NUM_SUBNETS]
            assert int(np.argmax(one_hot)) == cyborg_pos, (
                f"slot {slot}: expected pos {cyborg_pos}, got {int(np.argmax(one_hot))}"
            )
            assert float(np.sum(one_hot)) == 1.0

    def test_each_agent_has_at_least_one_subnet(self, jax_const, jax_state):
        for agent_id in range(NUM_BLUE_AGENTS):
            obs = get_blue_obs(jax_state, jax_const, agent_id)
            one_hot = _get_subnet_block(obs, 0)[:NUM_SUBNETS]
            assert float(np.sum(one_hot)) == 1.0, f"agent {agent_id} should have subnet in slot 0"


class TestCommsPolicy:
    def test_comms_policy_changes_with_phase(self, jax_const, jax_state):
        obs_p0 = np.array(get_blue_obs(jax_state, jax_const, 0))
        state_p1 = jax_state.replace(mission_phase=jnp.int32(1))
        obs_p1 = np.array(get_blue_obs(state_p1, jax_const, 0))
        comms_p0 = obs_p0[1 + 2 * NUM_SUBNETS : 1 + 3 * NUM_SUBNETS]
        comms_p1 = obs_p1[1 + 2 * NUM_SUBNETS : 1 + 3 * NUM_SUBNETS]
        assert not np.array_equal(comms_p0, comms_p1), "comms should differ between phase 0 and 1"

    def test_self_not_connected_in_comms(self, jax_const, jax_state):
        for agent_id in range(NUM_BLUE_AGENTS):
            obs = np.array(get_blue_obs(jax_state, jax_const, agent_id))
            sid = int(jax_const.blue_obs_subnets[agent_id, 0])
            if sid < 0:
                continue
            cyborg_pos = int(JAX_ID_TO_CYBORG_POS[sid])
            comms_start = 1 + 2 * NUM_SUBNETS
            comms = obs[comms_start : comms_start + NUM_SUBNETS]
            assert comms[cyborg_pos] == 1.0, f"agent {agent_id}: self-position in comms should be 1 (not connected)"

    def test_oza_connected_only_to_rza_phase0(self, jax_const, jax_state):
        obs = np.array(get_blue_obs(jax_state, jax_const, 1))
        comms_start = 1 + 2 * NUM_SUBNETS
        comms = obs[comms_start : comms_start + NUM_SUBNETS]
        rza_cyborg_pos = int(JAX_ID_TO_CYBORG_POS[SUBNET_IDS["RESTRICTED_ZONE_A"]])
        assert comms[rza_cyborg_pos] == 0.0, "OZA should be connected to RZA in phase 0"
        num_connected = int(np.sum(comms == 0.0))
        assert num_connected == 1, f"OZA should be connected to exactly 1 subnet, got {num_connected}"


class TestBlockedSubnets:
    def test_blocked_zones_appear_in_obs(self, jax_const, jax_state):
        src = SUBNET_IDS["INTERNET"]
        dst = SUBNET_IDS["RESTRICTED_ZONE_A"]
        blocked = jax_state.blocked_zones.at[dst, src].set(True)
        state = jax_state.replace(blocked_zones=blocked)

        obs = np.array(get_blue_obs(state, jax_const, 0))
        blocked_start = 1 + NUM_SUBNETS
        blocked_vec = obs[blocked_start : blocked_start + NUM_SUBNETS]
        internet_cyborg_pos = int(JAX_ID_TO_CYBORG_POS[src])
        assert blocked_vec[internet_cyborg_pos] == 1.0, "internet should be blocked from reaching RZA"

    def test_no_blocks_initially(self, jax_const, jax_state):
        obs = np.array(get_blue_obs(jax_state, jax_const, 0))
        blocked_start = 1 + NUM_SUBNETS
        blocked_vec = obs[blocked_start : blocked_start + NUM_SUBNETS]
        np.testing.assert_array_equal(blocked_vec, np.zeros(NUM_SUBNETS))


class TestHostEvents:
    def test_malicious_processes_reflect_exploit_detected(self, jax_const, jax_state):
        sid = int(jax_const.blue_obs_subnets[0, 0])
        host_idx = int(jax_const.obs_host_map[sid, 0])
        if host_idx >= GLOBAL_MAX_HOSTS:
            pytest.fail("no active server host in first slot")

        exploit = jax_state.host_exploit_detected.at[host_idx].set(True)
        state = jax_state.replace(host_exploit_detected=exploit)

        obs = np.array(get_blue_obs(state, jax_const, 0))
        proc_start = 1 + 3 * NUM_SUBNETS
        proc_vec = obs[proc_start : proc_start + OBS_HOSTS_PER_SUBNET]
        assert proc_vec[0] == 1.0, "first host should show malicious process"

    def test_network_connections_reflect_activity(self, jax_const, jax_state):
        sid = int(jax_const.blue_obs_subnets[0, 0])
        host_idx = int(jax_const.obs_host_map[sid, 0])
        if host_idx >= GLOBAL_MAX_HOSTS:
            pytest.fail("no active server host in first slot")

        detected = jax_state.host_activity_detected.at[host_idx].set(True)
        state = jax_state.replace(host_activity_detected=detected)

        obs = np.array(get_blue_obs(state, jax_const, 0))
        conn_start = 1 + 3 * NUM_SUBNETS + OBS_VECTOR_HOSTS_PER_SUBNET
        conn_vec = obs[conn_start : conn_start + OBS_VECTOR_HOSTS_PER_SUBNET]
        assert conn_vec[0] == 1.0, "first host should show network connection"

    def test_inactive_host_slots_zero(self, jax_const, jax_state):
        exploit = jax_state.host_exploit_detected.at[:].set(True)
        state = jax_state.replace(host_exploit_detected=exploit)

        obs = np.array(get_blue_obs(state, jax_const, 0))
        proc_start = 1 + 3 * NUM_SUBNETS
        proc_vec = obs[proc_start : proc_start + OBS_HOSTS_PER_SUBNET]
        sid = int(jax_const.blue_obs_subnets[0, 0])
        host_map = np.array(jax_const.obs_host_map[sid])
        inactive_mask = host_map >= GLOBAL_MAX_HOSTS
        inactive_values = proc_vec[inactive_mask]
        if len(inactive_values) > 0:
            np.testing.assert_array_equal(
                inactive_values,
                np.zeros(len(inactive_values)),
            )


class TestBlueObsDeterministic:
    def test_same_state_same_obs(self, jax_const, jax_state):
        obs1 = get_blue_obs(jax_state, jax_const, 0)
        obs2 = get_blue_obs(jax_state, jax_const, 0)
        np.testing.assert_array_equal(np.array(obs1), np.array(obs2))


class TestBlueObsJIT:
    def test_jit_compatible(self, jax_const, jax_state):
        jitted = jax.jit(get_blue_obs, static_argnums=(2,))
        obs = jitted(jax_state, jax_const, 0)
        assert obs.shape == (OBS_SIZE,)

    def test_jit_matches_eager(self, jax_const, jax_state):
        eager_obs = get_blue_obs(jax_state, jax_const, 0)
        jitted = jax.jit(get_blue_obs, static_argnums=(2,))
        jit_obs = jitted(jax_state, jax_const, 0)
        np.testing.assert_array_equal(np.array(eager_obs), np.array(jit_obs))

    def test_jit_all_agents(self, jax_const, jax_state):
        for agent_id in range(NUM_BLUE_AGENTS):
            jitted = jax.jit(get_blue_obs, static_argnums=(2,))
            obs = jitted(jax_state, jax_const, agent_id)
            assert obs.shape == (OBS_SIZE,), f"agent {agent_id}: JIT shape mismatch"


class TestDifferentialWithCybORG:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    @pytest.fixture
    def wrapped_env(self, cyborg_env):
        return BlueFlatWrapper(cyborg_env, pad_spaces=True)

    @pytest.fixture
    def cyborg_and_jax(self, cyborg_env, wrapped_env):
        const = build_const_from_cyborg(cyborg_env)
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(const.initial_services))
        return wrapped_env, const, state

    def test_initial_obs_matches_cyborg(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        observations, _ = wrapped_env.reset()

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            cyborg_obs = observations[agent_name]
            jax_obs = np.array(get_blue_obs(state, const, agent_id))

            assert cyborg_obs.shape == jax_obs.shape, (
                f"{agent_name}: CybORG shape {cyborg_obs.shape} vs JAX shape {jax_obs.shape}"
            )

            np.testing.assert_array_equal(
                cyborg_obs,
                jax_obs,
                err_msg=f"{agent_name}: initial obs mismatch",
            )

    def test_obs_after_sleep_steps(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        observations, _ = wrapped_env.reset()

        for step in range(3):
            sleep_actions = {f"blue_agent_{i}": 0 for i in range(NUM_BLUE_AGENTS)}
            observations, *_ = wrapped_env.step(actions=sleep_actions)

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            cyborg_obs = observations[agent_name]
            jax_obs = np.array(get_blue_obs(state, const, agent_id))

            np.testing.assert_array_equal(
                cyborg_obs,
                jax_obs,
                err_msg=f"{agent_name} step {step}: obs mismatch",
            )

    def test_obs_size_is_210(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        observations, _ = wrapped_env.reset()

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            assert observations[agent_name].shape == (OBS_SIZE,), (
                f"{agent_name}: CybORG obs size {observations[agent_name].shape}"
            )
            jax_obs = get_blue_obs(state, const, agent_id)
            assert jax_obs.shape == (OBS_SIZE,), f"{agent_name}: JAX obs size {jax_obs.shape}"

    def test_mission_phase_element_matches(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        observations, _ = wrapped_env.reset()

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            cyborg_phase = int(observations[agent_name][0])
            jax_obs = get_blue_obs(state, const, agent_id)
            jax_phase = int(jax_obs[0])
            assert cyborg_phase == jax_phase, f"{agent_name}: phase CybORG={cyborg_phase} JAX={jax_phase}"

    def test_subnet_one_hot_matches_cyborg(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        observations, _ = wrapped_env.reset()

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            cyborg_obs = observations[agent_name]
            jax_obs = np.array(get_blue_obs(state, const, agent_id))

            for slot in range(NUM_HQ_SUBNETS):
                c_block = cyborg_obs[1 + slot * SUBNET_BLOCK_SIZE : 1 + (slot + 1) * SUBNET_BLOCK_SIZE]
                j_block = jax_obs[1 + slot * SUBNET_BLOCK_SIZE : 1 + (slot + 1) * SUBNET_BLOCK_SIZE]
                c_one_hot = c_block[:NUM_SUBNETS]
                j_one_hot = j_block[:NUM_SUBNETS]
                np.testing.assert_array_equal(
                    c_one_hot,
                    j_one_hot,
                    err_msg=f"{agent_name} slot {slot}: subnet one-hot mismatch",
                )

    def test_comms_policy_matches_cyborg(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        observations, _ = wrapped_env.reset()

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            cyborg_obs = observations[agent_name]
            jax_obs = np.array(get_blue_obs(state, const, agent_id))

            for slot in range(NUM_HQ_SUBNETS):
                start = 1 + slot * SUBNET_BLOCK_SIZE + 2 * NUM_SUBNETS
                c_comms = cyborg_obs[start : start + NUM_SUBNETS]
                j_comms = jax_obs[start : start + NUM_SUBNETS]
                np.testing.assert_array_equal(
                    c_comms,
                    j_comms,
                    err_msg=f"{agent_name} slot {slot}: comms policy mismatch",
                )

    def test_initial_obs_multiple_seeds(self):
        for seed in [42, 123, 999]:
            sg = EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=SleepAgent,
                red_agent_class=SleepAgent,
                steps=500,
            )
            cyborg_env = CybORG(scenario_generator=sg, seed=seed)
            wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
            observations, _ = wrapped.reset()

            const = build_const_from_cyborg(cyborg_env)
            state = create_initial_state()
            state = state.replace(host_services=jnp.array(const.initial_services))

            for agent_id in range(NUM_BLUE_AGENTS):
                agent_name = f"blue_agent_{agent_id}"
                cyborg_obs = observations[agent_name]
                jax_obs = np.array(get_blue_obs(state, const, agent_id))
                np.testing.assert_array_equal(
                    cyborg_obs,
                    jax_obs,
                    err_msg=f"seed={seed} {agent_name}: initial obs mismatch",
                )


class TestBlueObsDifferential:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_obs_structure_matches_across_seeds(self):
        """Verify obs structure matches CybORG for multiple seeds."""
        for seed in [42, 100, 200]:
            sg = EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=SleepAgent,
                red_agent_class=SleepAgent,
                steps=500,
            )
            env = CybORG(scenario_generator=sg, seed=seed)
            wrapped = BlueFlatWrapper(env, pad_spaces=True)
            observations, _ = wrapped.reset()
            const = build_const_from_cyborg(env)
            state = create_initial_state().replace(host_services=jnp.array(const.initial_services))

            for agent_id in range(NUM_BLUE_AGENTS):
                agent_name = f"blue_agent_{agent_id}"
                cyborg_obs = observations[agent_name]
                jax_obs = np.array(get_blue_obs(state, const, agent_id))
                assert cyborg_obs.shape == jax_obs.shape, f"seed={seed} {agent_name}: shape mismatch"


class TestActionMaskAcrossSteps:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_masks_match_after_5_steps(self, cyborg_env):
        from CybORG.Agents.Wrappers import BlueFlatWrapper

        from jaxborg.actions.masking import compute_blue_action_mask

        wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
        wrapped.reset()
        const = build_const_from_cyborg(cyborg_env)

        for step in range(5):
            actions = {agent: 0 for agent in wrapped.agents}
            wrapped.step(actions)

        for agent_idx in range(NUM_BLUE_AGENTS):
            jax_mask = np.array(compute_blue_action_mask(const, agent_idx))
            assert jax_mask.sum() > 0, f"Agent {agent_idx} has no valid actions after 5 steps"
