import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.encoding import BLUE_SLEEP, RED_SLEEP, encode_blue_action
from jaxborg.actions.green import GREEN_ACCESS_SERVICE
from jaxborg.constants import GLOBAL_MAX_HOSTS, MAX_STEPS, NUM_BLUE_AGENTS, NUM_GREEN_RANDOM_FIELDS, NUM_RED_AGENTS
from jaxborg.env import (
    TOTAL_ACTION_ACTOR_SLOTS,
    CC4Env,
    _cyborg_priority_execution_order,
    apply_all_actions_in_order,
    apply_all_actions_typed,
)


def _move_slots_to_front(*front_slots: int) -> jnp.ndarray:
    tail = [slot for slot in range(TOTAL_ACTION_ACTOR_SLOTS) if slot not in front_slots]
    return jnp.array([*front_slots, *tail], dtype=jnp.int32)


@pytest.fixture(scope="module")
def env_state():
    env = CC4Env()
    _, out = env.reset(jax.random.PRNGKey(42))
    return out


def test_apply_all_actions_in_order_changes_green_access_outcome_when_block_moves_first(env_state):
    state = env_state.state.replace(
        blocked_zones=jnp.zeros_like(env_state.state.blocked_zones),
        green_asf_this_step=jnp.zeros_like(env_state.state.green_asf_this_step),
        host_activity_detected=jnp.zeros_like(env_state.state.host_activity_detected),
    )
    const = env_state.const

    chosen = None
    for source_host in range(int(const.num_hosts)):
        if not bool(const.green_agent_active[source_host]):
            continue
        src_sid = int(const.host_subnet[source_host])
        for dest_host in range(int(const.num_hosts)):
            if not bool(const.host_active[dest_host]) or not bool(const.host_is_server[dest_host]):
                continue
            dst_sid = int(const.host_subnet[dest_host])
            if src_sid == dst_sid:
                continue
            if not bool(const.allowed_subnet_pairs[0, src_sid, dst_sid]):
                continue
            blue_id = next((b for b in range(NUM_BLUE_AGENTS) if bool(const.blue_agent_subnets[b, dst_sid])), None)
            if blue_id is None:
                continue
            chosen = (blue_id, source_host, dest_host, src_sid, dst_sid)
            break
        if chosen is not None:
            break

    if chosen is None:
        pytest.fail("Need a green-access route whose destination subnet can be blocked by a blue agent")

    blue_id, source_host, dest_host, src_sid, dst_sid = chosen

    green_agent_active = jnp.zeros_like(const.green_agent_active).at[source_host].set(True)
    green_randoms = np.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
    green_randoms[0, source_host, 0] = (GREEN_ACCESS_SERVICE + 0.5) / 3.0
    green_randoms[0, source_host, 5] = float(dest_host)
    green_randoms[0, source_host, 6] = 0.99
    const = const.replace(
        green_agent_active=green_agent_active,
        num_green_agents=jnp.int32(1),
        green_randoms=jnp.array(green_randoms),
        use_green_randoms=jnp.array(True),
    )

    block_action = encode_blue_action(
        "BlockTrafficZone", -1, blue_id, const=const, src_subnet=src_sid, dst_subnet=dst_sid
    )
    blue_actions = jnp.full(NUM_BLUE_AGENTS, BLUE_SLEEP, dtype=jnp.int32).at[blue_id].set(block_action)
    red_actions = jnp.full(NUM_RED_AGENTS, RED_SLEEP, dtype=jnp.int32)
    forced_primary_hosts = jnp.full(NUM_RED_AGENTS, -2, dtype=jnp.int32)
    forced_primary_pids = jnp.full(NUM_RED_AGENTS, -2, dtype=jnp.int32)

    blue_slot = blue_id
    green_slot = NUM_BLUE_AGENTS + source_host

    blocked_first = apply_all_actions_in_order(
        state,
        const,
        blue_actions,
        red_actions,
        jax.random.PRNGKey(0),
        jax.random.split(jax.random.PRNGKey(1), NUM_RED_AGENTS),
        forced_primary_hosts,
        forced_primary_pids,
        _move_slots_to_front(blue_slot, green_slot),
    )
    green_first = apply_all_actions_in_order(
        state,
        const,
        blue_actions,
        red_actions,
        jax.random.PRNGKey(0),
        jax.random.split(jax.random.PRNGKey(1), NUM_RED_AGENTS),
        forced_primary_hosts,
        forced_primary_pids,
        _move_slots_to_front(green_slot, blue_slot),
    )

    assert bool(blocked_first.green_asf_this_step[source_host])
    assert not bool(green_first.green_asf_this_step[source_host])


def test_typed_path_matches_harness_path_execution_order(env_state):
    """Verify apply_all_actions_typed produces the same state as apply_all_actions_in_order.

    Regression test: the training path previously shuffled agent execution order
    within each phase, but CybORG uses a deterministic priority-sorted order.
    Both paths must produce identical results for the same inputs.
    """
    state = env_state.state
    const = env_state.const.replace(
        use_green_randoms=jnp.array(False),
    )

    blue_actions = jnp.full(NUM_BLUE_AGENTS, BLUE_SLEEP, dtype=jnp.int32)
    red_actions = jnp.full(NUM_RED_AGENTS, RED_SLEEP, dtype=jnp.int32)
    key_green = jax.random.PRNGKey(99)
    red_keys = jax.random.split(jax.random.PRNGKey(1), NUM_RED_AGENTS)
    forced_primary_hosts = jnp.full(NUM_RED_AGENTS, -2, dtype=jnp.int32)
    forced_primary_pids = jnp.full(NUM_RED_AGENTS, -2, dtype=jnp.int32)

    execution_order = _cyborg_priority_execution_order(blue_actions, TOTAL_ACTION_ACTOR_SLOTS)

    state_harness = apply_all_actions_in_order(
        state,
        const,
        blue_actions,
        red_actions,
        key_green,
        red_keys,
        forced_primary_hosts,
        forced_primary_pids,
        execution_order,
    )
    state_typed = apply_all_actions_typed(
        state,
        const,
        blue_actions,
        red_actions,
        key_green,
        red_keys,
        forced_primary_hosts,
        forced_primary_pids,
        execution_order,
        use_green_vmap=False,
    )

    # Compare key state fields
    fields_to_check = [
        "host_compromised",
        "red_sessions",
        "red_session_count",
        "green_lwf_this_step",
        "green_asf_this_step",
        "red_impact_attempted",
        "red_server_session_count",
        "fsm_host_entered",
        "host_service_reliability",
        "host_services",
    ]
    for field in fields_to_check:
        harness_val = np.array(getattr(state_harness, field))
        typed_val = np.array(getattr(state_typed, field))
        np.testing.assert_array_equal(
            harness_val,
            typed_val,
            err_msg=f"Execution order mismatch on field '{field}'",
        )
