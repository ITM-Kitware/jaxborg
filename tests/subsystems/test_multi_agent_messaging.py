import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.constants import (
    MESSAGE_LENGTH,
    NUM_BLUE_AGENTS,
)
from jaxborg.observations import SUBNET_BLOCK_SIZE, get_blue_obs
from jaxborg.state import create_initial_state
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

OBS_SIZE = 210
NUM_MESSAGES = 4
MESSAGE_SECTION_SIZE = NUM_MESSAGES * MESSAGE_LENGTH


@pytest.fixture
def jax_state(jax_const):
    state = create_initial_state()
    return state.replace(host_services=jax_const.initial_services)


def _jax_msg_start(const, agent_id):
    num_subnets = int(np.sum(np.array(const.blue_obs_subnets[agent_id]) >= 0))
    return 1 + num_subnets * SUBNET_BLOCK_SIZE


def _jax_msg_section(obs, const, agent_id):
    start = _jax_msg_start(const, agent_id)
    return np.array(obs[start : start + MESSAGE_SECTION_SIZE])


class TestMessageSectionPosition:
    def test_message_section_has_expected_size(self, jax_const, jax_state):
        obs = get_blue_obs(jax_state, jax_const, 0)
        assert obs.shape == (OBS_SIZE,)
        msg_section = _jax_msg_section(obs, jax_const, 0)
        assert msg_section.shape == (MESSAGE_SECTION_SIZE,)

    def test_message_section_start_matches_subnet_count(self, jax_const, jax_state):
        _ = get_blue_obs(jax_state, jax_const, 0)
        assert _jax_msg_start(jax_const, 0) == 1 + SUBNET_BLOCK_SIZE
        assert _jax_msg_start(jax_const, 4) == 1 + 3 * SUBNET_BLOCK_SIZE

    def test_zero_messages_give_zero_section(self, jax_const, jax_state):
        for agent_id in range(NUM_BLUE_AGENTS):
            obs = get_blue_obs(jax_state, jax_const, agent_id)
            msg_section = _jax_msg_section(obs, jax_const, agent_id)
            np.testing.assert_array_equal(
                msg_section,
                np.zeros(MESSAGE_SECTION_SIZE),
                err_msg=f"agent {agent_id} should have zero messages",
            )

    def test_message_section_size(self, jax_const, jax_state):
        assert MESSAGE_SECTION_SIZE == 32


class TestMessageContent:
    def test_setting_message_changes_obs(self, jax_const, jax_state):
        msg_val = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        msgs = jax_state.messages.at[1, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 0)
        msg_section = _jax_msg_section(obs, jax_const, 0)
        assert np.any(msg_section != 0), "message from agent 1 should appear in agent 0 obs"

    def test_self_messages_excluded(self, jax_const, jax_state):
        msg_val = jnp.ones(MESSAGE_LENGTH)
        msgs = jax_state.messages.at[0, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 0)
        msg_section = _jax_msg_section(obs, jax_const, 0)
        np.testing.assert_array_equal(
            msg_section,
            np.zeros(MESSAGE_SECTION_SIZE),
            err_msg="self-message should not appear in own observation",
        )

    def test_self_messages_excluded_all_agents(self, jax_const, jax_state):
        for agent_id in range(NUM_BLUE_AGENTS):
            msgs = jax_state.messages.at[agent_id, agent_id, :].set(jnp.ones(MESSAGE_LENGTH))
            state = jax_state.replace(messages=msgs)
            obs = get_blue_obs(state, jax_const, agent_id)
            msg_section = _jax_msg_section(obs, jax_const, agent_id)
            np.testing.assert_array_equal(
                msg_section,
                np.zeros(MESSAGE_SECTION_SIZE),
                err_msg=f"agent {agent_id}: self-message should not appear",
            )

    def test_messages_from_all_others(self, jax_const, jax_state):
        msgs = jax_state.messages
        for sender in range(NUM_BLUE_AGENTS):
            if sender == 0:
                continue
            msg_val = jnp.full(MESSAGE_LENGTH, float(sender))
            msgs = msgs.at[sender, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 0)
        msg_section = _jax_msg_section(obs, jax_const, 0)
        nonzero_count = np.count_nonzero(msg_section)
        assert nonzero_count == NUM_MESSAGES * MESSAGE_LENGTH, (
            f"expected {NUM_MESSAGES * MESSAGE_LENGTH} nonzero, got {nonzero_count}"
        )

    def test_message_ordering(self, jax_const, jax_state):
        msgs = jax_state.messages
        other_agents = [i for i in range(NUM_BLUE_AGENTS) if i != 2]
        for slot_idx, sender in enumerate(other_agents):
            msg_val = jnp.full(MESSAGE_LENGTH, float(sender + 1))
            msgs = msgs.at[sender, 2, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 2)
        msg_section = _jax_msg_section(obs, jax_const, 2)

        for slot_idx in range(NUM_MESSAGES):
            slot = msg_section[slot_idx * MESSAGE_LENGTH : (slot_idx + 1) * MESSAGE_LENGTH]
            assert np.all(slot == slot[0]), f"slot {slot_idx} should be uniform: {slot}"

    def test_message_values_exact(self, jax_const, jax_state):
        msg_val = jnp.array([0.5, 1.0, 0.0, 0.25, 0.75, 0.0, 1.0, 0.5])
        msgs = jax_state.messages.at[3, 1, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 1)
        msg_section = _jax_msg_section(obs, jax_const, 1)
        slot2 = msg_section[2 * MESSAGE_LENGTH : 3 * MESSAGE_LENGTH]
        np.testing.assert_array_almost_equal(slot2, np.array(msg_val))

    def test_message_ordering_agent0(self, jax_const, jax_state):
        msgs = jax_state.messages
        for sender in range(1, NUM_BLUE_AGENTS):
            msg_val = jnp.full(MESSAGE_LENGTH, float(sender * 10))
            msgs = msgs.at[sender, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 0)
        msg_section = _jax_msg_section(obs, jax_const, 0)

        expected_senders = [1, 2, 3, 4]
        for slot_idx, sender in enumerate(expected_senders):
            slot = msg_section[slot_idx * MESSAGE_LENGTH : (slot_idx + 1) * MESSAGE_LENGTH]
            expected_val = float(sender * 10)
            np.testing.assert_array_almost_equal(
                slot,
                np.full(MESSAGE_LENGTH, expected_val),
                err_msg=f"slot {slot_idx}: expected value {expected_val}",
            )

    def test_message_ordering_agent4(self, jax_const, jax_state):
        msgs = jax_state.messages
        for sender in range(4):
            msg_val = jnp.full(MESSAGE_LENGTH, float(sender + 100))
            msgs = msgs.at[sender, 4, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 4)
        msg_section = _jax_msg_section(obs, jax_const, 4)

        expected_senders = [0, 1, 2, 3]
        for slot_idx, sender in enumerate(expected_senders):
            slot = msg_section[slot_idx * MESSAGE_LENGTH : (slot_idx + 1) * MESSAGE_LENGTH]
            expected_val = float(sender + 100)
            np.testing.assert_array_almost_equal(
                slot,
                np.full(MESSAGE_LENGTH, expected_val),
                err_msg=f"slot {slot_idx}: expected value {expected_val}",
            )

    def test_messages_do_not_leak_between_agents(self, jax_const, jax_state):
        msg_val = jnp.ones(MESSAGE_LENGTH) * 42.0
        msgs = jax_state.messages.at[1, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs_agent2 = np.array(get_blue_obs(state, jax_const, 2))
        msg_section_agent2 = _jax_msg_section(obs_agent2, jax_const, 2)
        np.testing.assert_array_equal(
            msg_section_agent2,
            np.zeros(MESSAGE_SECTION_SIZE),
            err_msg="message to agent 0 should not appear in agent 2 obs",
        )


class TestMessageJIT:
    def test_jit_with_messages(self, jax_const, jax_state):
        msg_val = jnp.ones(MESSAGE_LENGTH)
        msgs = jax_state.messages.at[1, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        jitted = jax.jit(get_blue_obs, static_argnums=(2,))
        obs = jitted(state, jax_const, 0)
        msg_section = _jax_msg_section(obs, jax_const, 0)
        assert np.any(msg_section != 0)

    def test_jit_all_agents_with_messages(self, jax_const, jax_state):
        msgs = jax_state.messages
        for sender in range(NUM_BLUE_AGENTS):
            for receiver in range(NUM_BLUE_AGENTS):
                if sender == receiver:
                    continue
                msg_val = jnp.full(MESSAGE_LENGTH, float(sender + receiver * 10))
                msgs = msgs.at[sender, receiver, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        for agent_id in range(NUM_BLUE_AGENTS):
            jitted = jax.jit(get_blue_obs, static_argnums=(2,))
            eager = np.array(get_blue_obs(state, jax_const, agent_id))
            jit_result = np.array(jitted(state, jax_const, agent_id))
            np.testing.assert_array_equal(
                eager,
                jit_result,
                err_msg=f"agent {agent_id}: JIT mismatch with messages",
            )


class TestDifferentialMessages:
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

    def test_initial_messages_zero_in_both(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        observations, _ = wrapped_env.reset()

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            msg_start = self._cyborg_msg_start(wrapped_env, agent_name)
            cyborg_msgs = observations[agent_name][msg_start : msg_start + MESSAGE_SECTION_SIZE]
            jax_obs = np.array(get_blue_obs(state, const, agent_id))
            jax_msgs = _jax_msg_section(jax_obs, const, agent_id)

            np.testing.assert_array_equal(
                cyborg_msgs,
                jax_msgs,
                err_msg=f"{agent_name}: initial message section mismatch",
            )

    def _cyborg_msg_start(self, wrapped_env, agent_name):
        num_subnets = len(wrapped_env.subnets(agent_name))
        return 1 + num_subnets * SUBNET_BLOCK_SIZE

    def test_message_delivery_matches(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        wrapped_env.reset()

        sent_messages = {}
        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            msg = np.zeros(MESSAGE_LENGTH, dtype=bool)
            for bit in range(min(agent_id + 1, MESSAGE_LENGTH)):
                msg[bit] = True
            sent_messages[agent_name] = msg

        sleep_actions = {f"blue_agent_{i}": 0 for i in range(NUM_BLUE_AGENTS)}
        cyborg_obs, *_ = wrapped_env.step(actions=sleep_actions, messages=sent_messages)

        jax_msgs = jnp.zeros_like(state.messages)
        for sender_id in range(NUM_BLUE_AGENTS):
            msg_float = jnp.array(sent_messages[f"blue_agent_{sender_id}"], dtype=jnp.float32)
            for receiver_id in range(NUM_BLUE_AGENTS):
                jax_msgs = jax_msgs.at[sender_id, receiver_id, :].set(msg_float)
        state_with_msgs = state.replace(messages=jax_msgs)

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            msg_start = self._cyborg_msg_start(wrapped_env, agent_name)
            cyborg_msg_section = cyborg_obs[agent_name][msg_start : msg_start + MESSAGE_SECTION_SIZE]

            assert np.any(cyborg_msg_section != 0), f"{agent_name}: CybORG delivered no messages"

            jax_obs = np.array(get_blue_obs(state_with_msgs, const, agent_id))
            jax_msg_section = _jax_msg_section(jax_obs, const, agent_id)

            np.testing.assert_array_equal(
                cyborg_msg_section,
                jax_msg_section,
                err_msg=f"{agent_name}: message section mismatch",
            )

    def test_varied_messages_match(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        wrapped_env.reset()

        sent_messages = {}
        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            msg = np.zeros(MESSAGE_LENGTH, dtype=bool)
            msg[agent_id % MESSAGE_LENGTH] = True
            msg[(agent_id + 3) % MESSAGE_LENGTH] = True
            sent_messages[agent_name] = msg

        sleep_actions = {f"blue_agent_{i}": 0 for i in range(NUM_BLUE_AGENTS)}
        cyborg_obs, *_ = wrapped_env.step(actions=sleep_actions, messages=sent_messages)

        jax_msgs = jnp.zeros_like(state.messages)
        for sender_id in range(NUM_BLUE_AGENTS):
            msg_float = jnp.array(sent_messages[f"blue_agent_{sender_id}"], dtype=jnp.float32)
            for receiver_id in range(NUM_BLUE_AGENTS):
                jax_msgs = jax_msgs.at[sender_id, receiver_id, :].set(msg_float)
        state_with_msgs = state.replace(messages=jax_msgs)

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            msg_start = self._cyborg_msg_start(wrapped_env, agent_name)
            cyborg_msg_section = cyborg_obs[agent_name][msg_start : msg_start + MESSAGE_SECTION_SIZE]
            jax_obs = np.array(get_blue_obs(state_with_msgs, const, agent_id))
            jax_msg_section = _jax_msg_section(jax_obs, const, agent_id)

            np.testing.assert_array_equal(
                cyborg_msg_section,
                jax_msg_section,
                err_msg=f"{agent_name}: varied message section mismatch",
            )

    def test_non_message_obs_match_after_step(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        wrapped_env.reset()

        sleep_actions = {f"blue_agent_{i}": 0 for i in range(NUM_BLUE_AGENTS)}
        observations, *_ = wrapped_env.step(actions=sleep_actions)

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            msg_start = self._cyborg_msg_start(wrapped_env, agent_name)
            cyborg_msg_end = msg_start + MESSAGE_SECTION_SIZE
            jax_obs = np.array(get_blue_obs(state, const, agent_id))
            jax_msg_start = _jax_msg_start(const, agent_id)
            jax_msg_end = jax_msg_start + MESSAGE_SECTION_SIZE

            np.testing.assert_array_equal(
                observations[agent_name][:msg_start],
                jax_obs[:jax_msg_start],
                err_msg=f"{agent_name}: pre-message obs mismatch after step",
            )
            np.testing.assert_array_equal(
                observations[agent_name][cyborg_msg_end:],
                jax_obs[jax_msg_end:],
                err_msg=f"{agent_name}: non-message obs mismatch after step",
            )


class TestMessageDifferential:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_message_section_position_matches(self, cyborg_env):
        """Verify message section is in same position in both implementations."""
        wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
        const = build_const_from_cyborg(cyborg_env)
        state = create_initial_state().replace(host_services=jnp.array(const.initial_services))
        observations, _ = wrapped.reset()

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            cyborg_obs = observations[agent_name]
            jax_obs = np.array(get_blue_obs(state, const, agent_id))
            cyborg_msg_start = 1 + len(wrapped.subnets(agent_name)) * SUBNET_BLOCK_SIZE
            cyborg_msgs = cyborg_obs[cyborg_msg_start : cyborg_msg_start + MESSAGE_SECTION_SIZE]
            jax_msgs = _jax_msg_section(jax_obs, const, agent_id)
            np.testing.assert_array_equal(cyborg_msgs, jax_msgs, err_msg=f"{agent_name}: message section mismatch")
