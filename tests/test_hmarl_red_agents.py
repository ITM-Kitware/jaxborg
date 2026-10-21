"""H-MARL red profiles preserve CC4 mechanics while specializing action choice."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG.Agents import FiniteStateRedAgent

from jaxborg.actions.encoding import (
    RED_AGGRESSIVE_SCAN_START,
    RED_IMPACT_START,
    RED_SLEEP,
    RED_STEALTH_SCAN_START,
    get_red_action_duration,
)
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_RED_AGENTS
from jaxborg.evaluation.cyborg_red_dispatch import cyborg_red_class
from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.scenarios.cc4.game_variants import variant_for_red
from jaxborg.scenarios.cc4.red_fsm import (
    FSM_ACT_AGGRESSIVE_SCAN,
    FSM_ACT_IMPACT,
    FSM_ACT_STEALTH_SCAN,
    FSM_KD,
    FSM_RD,
    PROBABILITY_MATRIX,
)
from jaxborg.scenarios.cc4.red_selectors import hmarl_red_probability_matrix, make_red_selector
from jaxborg.state import create_initial_state


@pytest.mark.parametrize(
    "name,changed_rows,expected",
    [
        ("aggressive", [0, 1], [[0.5, 0.5, 0, -1, -1, -1, -1, -1, -1], [-1, 1, 0, -1, -1, -1, -1, -1, -1]]),
        ("stealthy", [0, 1], [[0.5, 0, 0.5, -1, -1, -1, -1, -1, -1], [-1, 0, 1, -1, -1, -1, -1, -1, -1]]),
        ("impact", [6, 7], [[0.5, -1, -1, -1, -1, -1, 0.5, 0, 0], [-1, -1, -1, -1, -1, -1, 1, 0, 0]]),
    ],
)
def test_hmarl_probabilities_match_specification_and_cyborg(name, changed_rows, expected):
    matrix = np.asarray(hmarl_red_probability_matrix(name))
    np.testing.assert_array_equal(matrix[changed_rows], expected)
    unchanged = [i for i in range(8) if i not in changed_rows]
    np.testing.assert_array_equal(matrix[unchanged], np.asarray(PROBABILITY_MATRIX)[unchanged])
    np.testing.assert_array_equal(np.maximum(matrix, 0).sum(axis=1), np.ones(8))
    native = cyborg_red_class(name)().state_transitions_probability
    native_matrix = [[-1 if p is None else p for p in native[s]] for s in ("K", "KD", "S", "SD", "U", "UD", "R", "RD")]
    np.testing.assert_array_equal(matrix, native_matrix)
    stock = FiniteStateRedAgent().state_transitions_probability
    assert stock["K"][1:3] == [0.25, 0.25]
    assert stock["RD"][6:8] == [0.5, 0.5]


@pytest.mark.parametrize(
    "name,fsm_state,fsm_action,action_start,duration",
    [
        ("aggressive", FSM_KD, FSM_ACT_AGGRESSIVE_SCAN, RED_AGGRESSIVE_SCAN_START, 1),
        ("stealthy", FSM_KD, FSM_ACT_STEALTH_SCAN, RED_STEALTH_SCAN_START, 3),
        ("impact", FSM_RD, FSM_ACT_IMPACT, RED_IMPACT_START, 2),
    ],
)
def test_jitted_selectors_use_native_actions_and_preserve_pending(
    jax_const, name, fsm_state, fsm_action, action_start, duration
):
    host = int(jnp.argmax(jax_const.host_active & ~jax_const.host_is_router))
    state = create_initial_state()
    state = state.replace(
        red_agent_active=jnp.ones(NUM_RED_AGENTS, dtype=bool).at[2].set(False),
        red_discovered_hosts=state.red_discovered_hosts.at[:, host].set(True),
        fsm_host_entered=state.fsm_host_entered.at[:, host].set(True),
        fsm_host_states=state.fsm_host_states.at[:, host].set(fsm_state),
        red_scan_anchor_host=state.red_scan_anchor_host.at[:].set(host),
        red_sessions=state.red_sessions.at[:, host].set(True),
        red_session_is_abstract=state.red_session_is_abstract.at[:, host].set(True),
        red_pending_ticks=state.red_pending_ticks.at[1].set(2),
        red_pending_target_host=state.red_pending_target_host.at[1].set(host),
        red_pending_target_subnet=state.red_pending_target_subnet.at[1].set(jax_const.host_subnet[host]),
        red_pending_fsm_action=state.red_pending_fsm_action.at[1].set(FSM_ACT_STEALTH_SCAN),
        red_pending_source_kind=state.red_pending_source_kind.at[1].set(1),
        red_pending_source_host=state.red_pending_source_host.at[1].set(host),
    )
    selector = jax.jit(make_red_selector(name))
    keys = jax.random.split(jax.random.PRNGKey(12), NUM_RED_AGENTS)
    roles = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
    actions, hosts, subnets, fsm_actions, eligible, updated = selector(state, jax_const, roles, keys)
    assert int(actions[0]) == action_start + host
    assert int(fsm_actions[0]) == fsm_action
    assert int(get_red_action_duration(actions[0], jax_const)) == duration
    assert int(hosts[0]) == host
    assert bool(eligible[0])
    assert int(actions[1]) == int(actions[2]) == RED_SLEEP
    assert bool(eligible[1]) and not bool(eligible[2])
    for field in ("target_host", "target_subnet", "fsm_action", "source_kind", "source_host"):
        assert int(getattr(updated, f"red_pending_{field}")[1]) == int(getattr(state, f"red_pending_{field}")[1])
    # No role bias: assigning every host a CIA role cannot alter this profile.
    role_actions = selector(state, jax_const, roles + 1, keys)[0]
    np.testing.assert_array_equal(actions, role_actions)
    # An active agent with no observed hosts must sleep.
    empty = state.replace(fsm_host_entered=jnp.zeros_like(state.fsm_host_entered))
    assert int(selector(empty, jax_const, roles, keys)[0][0]) == RED_SLEEP


@pytest.mark.parametrize("name", ["aggressive", "stealthy", "impact"])
def test_hmarl_variant_constructs_env_and_preserves_blue_contract(name):
    variant = variant_for_red(name, resilience_roles=True, cage4_enhanced_obs=True, blue_block_policy="mission_safe")
    assert variant.red_agent == name
    assert variant.resilience_roles and variant.op_zone_servers == 3
    assert variant.cage4_enhanced_obs and variant.blue_block_policy == "mission_safe"
    assert make_jax_env(variant)._red_selector is not None
