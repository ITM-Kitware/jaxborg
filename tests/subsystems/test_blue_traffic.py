import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action
from jaxborg.actions.blue_traffic import apply_allow_traffic, apply_block_traffic
from jaxborg.actions.encoding import (
    BLUE_ACTION_TYPE_ALLOW_TRAFFIC,
    BLUE_ACTION_TYPE_BLOCK_TRAFFIC,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_BLOCK_TRAFFIC_START,
    decode_blue_action,
    encode_blue_action,
)
from jaxborg.constants import BLUE_MAX_OBSERVED_SUBNETS, NUM_SUBNETS, SUBNET_IDS
from jaxborg.scenarios.cc4.topology import build_const_from_cyborg
from jaxborg.state import create_initial_state

_jit_apply_blue = jax.jit(apply_blue_action, static_argnums=(2,))


def _make_cyborg_env():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)


@pytest.fixture(scope="module")
def jax_const():
    return build_const_from_cyborg(_make_cyborg_env())


def _make_jax_state(const):
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    return state


# Agent 4 (HQ) observes 3 subnets: [5=ADMIN, 6=OFFICE, 7=PAZ]
# Use agent 4 for traffic tests since it has 3 observed subnets.
AGENT_ID = 4
SRC = SUBNET_IDS["INTERNET"]  # 8
DST = SUBNET_IDS["ADMIN_NETWORK"]  # 5 (relative index 0 for agent 4)


def _compress_src(src_subnet, dst_subnet):
    """Compress absolute source subnet to offset (skip self-loop)."""
    return src_subnet if src_subnet < dst_subnet else src_subnet - 1


class TestBlueTrafficEncoding:
    def test_encode_block(self, jax_const):
        rel_dst = int(np.flatnonzero(np.array(jax_const.blue_obs_subnets[AGENT_ID]) == DST)[0])
        idx = encode_blue_action("BlockTrafficZone", -1, AGENT_ID, const=jax_const, src_subnet=SRC, dst_subnet=DST)
        src_offset = _compress_src(SRC, DST)
        assert idx == BLUE_BLOCK_TRAFFIC_START + src_offset * BLUE_MAX_OBSERVED_SUBNETS + rel_dst

    def test_encode_allow(self, jax_const):
        rel_dst = int(np.flatnonzero(np.array(jax_const.blue_obs_subnets[AGENT_ID]) == DST)[0])
        idx = encode_blue_action("AllowTrafficZone", -1, AGENT_ID, const=jax_const, src_subnet=SRC, dst_subnet=DST)
        src_offset = _compress_src(SRC, DST)
        assert idx == BLUE_ALLOW_TRAFFIC_START + src_offset * BLUE_MAX_OBSERVED_SUBNETS + rel_dst

    def test_encode_self_loop_returns_sleep(self, jax_const):
        """Self-loop (src==dst) returns BLUE_SLEEP."""
        from jaxborg.actions.encoding import BLUE_SLEEP

        idx = encode_blue_action("BlockTrafficZone", -1, AGENT_ID, const=jax_const, src_subnet=DST, dst_subnet=DST)
        assert idx == BLUE_SLEEP

    def test_decode_block(self, jax_const):
        rel_dst = int(np.flatnonzero(np.array(jax_const.blue_obs_subnets[AGENT_ID]) == DST)[0])
        src_offset = _compress_src(SRC, DST)
        action_idx = BLUE_BLOCK_TRAFFIC_START + src_offset * BLUE_MAX_OBSERVED_SUBNETS + rel_dst
        action_type, _, _, src, dst = decode_blue_action(action_idx, AGENT_ID, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_BLOCK_TRAFFIC
        assert int(src) == SRC
        assert int(dst) == DST

    def test_decode_allow(self, jax_const):
        rel_dst = int(np.flatnonzero(np.array(jax_const.blue_obs_subnets[AGENT_ID]) == DST)[0])
        src_offset = _compress_src(SRC, DST)
        action_idx = BLUE_ALLOW_TRAFFIC_START + src_offset * BLUE_MAX_OBSERVED_SUBNETS + rel_dst
        action_type, _, _, src, dst = decode_blue_action(action_idx, AGENT_ID, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_ALLOW_TRAFFIC
        assert int(src) == SRC
        assert int(dst) == DST

    def test_roundtrip_block(self, jax_const):
        agent_obs = np.array(jax_const.blue_obs_subnets[AGENT_ID])
        for s in range(NUM_SUBNETS):
            for rel_d in range(BLUE_MAX_OBSERVED_SUBNETS):
                abs_d = int(agent_obs[rel_d])
                if abs_d < 0 or s == abs_d:
                    continue
                idx = encode_blue_action(
                    "BlockTrafficZone", -1, AGENT_ID, const=jax_const, src_subnet=s, dst_subnet=abs_d
                )
                action_type, _, _, src, dst = decode_blue_action(idx, AGENT_ID, jax_const)
                assert int(action_type) == BLUE_ACTION_TYPE_BLOCK_TRAFFIC
                assert int(src) == s
                assert int(dst) == abs_d


class TestApplyBlockTraffic:
    def test_block_sets_zone(self, jax_const):
        state = _make_jax_state(jax_const)
        assert not bool(state.blocked_zones[DST, SRC])

        new_state = apply_block_traffic(state, jax_const, AGENT_ID, SRC, DST)
        assert bool(new_state.blocked_zones[DST, SRC])

    def test_block_does_not_affect_other_pairs(self, jax_const):
        state = _make_jax_state(jax_const)
        new_state = apply_block_traffic(state, jax_const, AGENT_ID, SRC, DST)

        mask = np.zeros((NUM_SUBNETS, NUM_SUBNETS), dtype=bool)
        mask[DST, SRC] = True
        other_zones = np.array(new_state.blocked_zones) & ~mask
        assert not np.any(other_zones)

    def test_allow_clears_zone(self, jax_const):
        state = _make_jax_state(jax_const)
        state = state.replace(blocked_zones=state.blocked_zones.at[DST, SRC].set(True))
        assert bool(state.blocked_zones[DST, SRC])

        new_state = apply_allow_traffic(state, jax_const, AGENT_ID, SRC, DST)
        assert not bool(new_state.blocked_zones[DST, SRC])

    def test_allow_on_unblocked_is_noop(self, jax_const):
        state = _make_jax_state(jax_const)
        new_state = apply_allow_traffic(state, jax_const, AGENT_ID, SRC, DST)
        np.testing.assert_array_equal(
            np.array(new_state.blocked_zones),
            np.array(state.blocked_zones),
        )

    def test_block_then_allow_roundtrip(self, jax_const):
        state = _make_jax_state(jax_const)
        state = apply_block_traffic(state, jax_const, AGENT_ID, SRC, DST)
        assert bool(state.blocked_zones[DST, SRC])

        state = apply_allow_traffic(state, jax_const, AGENT_ID, SRC, DST)
        assert not bool(state.blocked_zones[DST, SRC])

    def test_jit_compatible(self, jax_const):
        state = _make_jax_state(jax_const)
        jitted_block = jax.jit(apply_block_traffic, static_argnums=(2, 3, 4))
        new_state = jitted_block(state, jax_const, AGENT_ID, SRC, DST)
        assert bool(new_state.blocked_zones[DST, SRC])

        jitted_allow = jax.jit(apply_allow_traffic, static_argnums=(2, 3, 4))
        new_state = jitted_allow(new_state, jax_const, AGENT_ID, SRC, DST)
        assert not bool(new_state.blocked_zones[DST, SRC])


class TestTrafficViaDispatch:
    def test_block_dispatched(self, jax_const):
        state = _make_jax_state(jax_const)
        action_idx = encode_blue_action(
            "BlockTrafficZone", -1, AGENT_ID, const=jax_const, src_subnet=SRC, dst_subnet=DST
        )
        new_state = _jit_apply_blue(state, jax_const, AGENT_ID, action_idx)
        assert bool(new_state.blocked_zones[DST, SRC])

    def test_allow_dispatched(self, jax_const):
        state = _make_jax_state(jax_const)
        state = state.replace(blocked_zones=state.blocked_zones.at[DST, SRC].set(True))
        action_idx = encode_blue_action(
            "AllowTrafficZone", -1, AGENT_ID, const=jax_const, src_subnet=SRC, dst_subnet=DST
        )
        new_state = _jit_apply_blue(state, jax_const, AGENT_ID, action_idx)
        assert not bool(new_state.blocked_zones[DST, SRC])


class TestDifferentialWithCybORG:
    @pytest.fixture
    def cyborg_env(self):
        return _make_cyborg_env()

    def test_block_matches_cyborg(self, cyborg_env, jax_const):
        cyborg_state = cyborg_env.environment_controller.state
        from_subnet_name = "internet_subnet"
        to_subnet_name = "admin_network_subnet"

        from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTrafficZone

        action = BlockTrafficZone(
            session=0, agent=f"blue_agent_{AGENT_ID}", from_subnet=from_subnet_name, to_subnet=to_subnet_name
        )
        cyborg_env.environment_controller.execute_action(action)

        state = _make_jax_state(jax_const)
        new_state = apply_block_traffic(state, jax_const, AGENT_ID, SRC, DST)

        cyborg_blocks = cyborg_state.blocks
        assert to_subnet_name in cyborg_blocks
        assert from_subnet_name in cyborg_blocks[to_subnet_name]
        assert bool(new_state.blocked_zones[DST, SRC])
