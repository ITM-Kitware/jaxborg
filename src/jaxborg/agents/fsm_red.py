import jax
import jax.numpy as jnp

from jaxborg.actions.encoding import (
    ACTION_TYPE_AGGRESSIVE_SCAN,
    ACTION_TYPE_SCAN,
    ACTION_TYPE_STEALTH_SCAN,
    RED_AGGRESSIVE_SCAN_START,
    RED_DEGRADE_START,
    RED_DISCOVER_DECEPTION_START,
    RED_DISCOVER_START,
    RED_EXPLOIT_SSH_START,
    RED_IMPACT_START,
    RED_PRIVESC_START,
    RED_SLEEP,
    RED_STEALTH_SCAN_START,
    RED_WITHDRAW_START,
    decode_red_action,
)
from jaxborg.actions.pending_source import (
    PENDING_SOURCE_KIND_NONE,
    PENDING_SOURCE_KIND_SESSION_BINDING,
)
from jaxborg.actions.red_common import select_bound_source_host
from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
)
from jaxborg.state import CC4Const, CC4State

FSM_K = 0
FSM_KD = 1
FSM_S = 2
FSM_SD = 3
FSM_U = 4
FSM_UD = 5
FSM_R = 6
FSM_RD = 7
FSM_F = 8
NUM_FSM_STATES = 9

FSM_ACT_DISCOVER = 0
FSM_ACT_AGGRESSIVE_SCAN = 1
FSM_ACT_STEALTH_SCAN = 2
FSM_ACT_DISCOVER_DECEPTION = 3
FSM_ACT_EXPLOIT = 4
FSM_ACT_PRIVESC = 5
FSM_ACT_IMPACT = 6
FSM_ACT_DEGRADE = 7
FSM_ACT_WITHDRAW = 8
NUM_FSM_ACTIONS = 9

_SENTINEL = -1

TRANSITION_SUCCESS = jnp.array(
    [
        [FSM_KD, FSM_S, FSM_S, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_KD, FSM_SD, FSM_SD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_SD, _SENTINEL, _SENTINEL, FSM_S, FSM_U, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_SD, _SENTINEL, _SENTINEL, FSM_SD, FSM_UD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_UD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_R, _SENTINEL, _SENTINEL, FSM_S],
        [FSM_UD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_RD, _SENTINEL, _SENTINEL, FSM_SD],
        [FSM_RD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_R, FSM_R, FSM_S],
        [FSM_RD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_RD, FSM_RD, FSM_SD],
        [FSM_F, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
    ],
    dtype=jnp.int32,
)

TRANSITION_FAILURE = jnp.array(
    [
        [FSM_K, FSM_K, FSM_K, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_KD, FSM_KD, FSM_KD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_S, _SENTINEL, _SENTINEL, FSM_S, FSM_S, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_SD, _SENTINEL, _SENTINEL, FSM_SD, FSM_SD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_U, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_U, _SENTINEL, _SENTINEL, FSM_U],
        [FSM_UD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_UD, _SENTINEL, _SENTINEL, FSM_UD],
        [FSM_R, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_R, FSM_R, FSM_R],
        [FSM_RD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_RD, FSM_RD, FSM_RD],
        [FSM_F, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
    ],
    dtype=jnp.int32,
)

_prob_none = -1.0

PROBABILITY_MATRIX = jnp.array(
    [
        [0.5, 0.25, 0.25, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none],
        [_prob_none, 0.5, 0.5, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none],
        [0.25, _prob_none, _prob_none, 0.25, 0.5, _prob_none, _prob_none, _prob_none, _prob_none],
        [_prob_none, _prob_none, _prob_none, 0.25, 0.75, _prob_none, _prob_none, _prob_none, _prob_none],
        [0.5, _prob_none, _prob_none, _prob_none, _prob_none, 0.5, _prob_none, _prob_none, 0.0],
        [_prob_none, _prob_none, _prob_none, _prob_none, _prob_none, 1.0, _prob_none, _prob_none, 0.0],
        [0.5, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none, 0.25, 0.25, 0.0],
        [_prob_none, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none, 0.5, 0.5, 0.0],
    ],
    dtype=jnp.float32,
)

ACTION_VALID_MASK = PROBABILITY_MATRIX >= 0.0


def _fsm_action_to_jax_action(fsm_action, target_host, target_subnet):
    return jax.lax.switch(
        fsm_action,
        [
            lambda: RED_DISCOVER_START + target_subnet,
            lambda: RED_AGGRESSIVE_SCAN_START + target_host,
            lambda: RED_STEALTH_SCAN_START + target_host,
            lambda: RED_DISCOVER_DECEPTION_START + target_host,
            lambda: RED_EXPLOIT_SSH_START + target_host,
            lambda: RED_PRIVESC_START + target_host,
            lambda: RED_IMPACT_START + target_host,
            lambda: RED_DEGRADE_START + target_host,
            lambda: RED_WITHDRAW_START + target_host,
        ],
    )


def _pick_discover_subnet(state, const, agent_id, key):
    # CybORG's action space only marks subnets valid after observation (initial
    # session or green-phishing reassignment).  Match that by restricting to
    # subnets where the agent currently holds a session.
    session_hosts = state.red_sessions[agent_id] & const.host_active
    subnet_one_hot = jax.nn.one_hot(const.host_subnet, NUM_SUBNETS, dtype=jnp.bool_)
    session_subnets = jnp.any(session_hosts[:, None] & subnet_one_hot, axis=0)
    probs = jnp.where(session_subnets, 1.0, 0.0)
    probs = probs / jnp.maximum(jnp.sum(probs), 1e-8)
    return jax.random.choice(key, NUM_SUBNETS, p=probs)


def fsm_red_get_action_and_info(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    key: jax.Array,
) -> tuple:
    fsm_states = state.fsm_host_states[agent_id]
    discovered = state.red_discovered_hosts[agent_id]
    active = const.host_active

    eligible = discovered & active & (fsm_states != FSM_F)

    key1, key2, key3 = jax.random.split(key, 3)

    any_eligible = jnp.any(eligible)

    host_probs = jnp.where(eligible, 1.0, 0.0)
    host_total = jnp.sum(host_probs)
    host_probs = host_probs / jnp.maximum(host_total, 1e-8)
    chosen_host = jax.random.choice(key1, GLOBAL_MAX_HOSTS, p=host_probs)

    host_state = fsm_states[chosen_host]
    host_state_clamped = jnp.clip(host_state, 0, NUM_FSM_STATES - 2)

    action_probs_raw = PROBABILITY_MATRIX[host_state_clamped]
    valid_mask = ACTION_VALID_MASK[host_state_clamped]
    action_probs = jnp.where(valid_mask, jnp.maximum(action_probs_raw, 0.0), 0.0)
    action_total = jnp.sum(action_probs)
    action_probs = action_probs / jnp.maximum(action_total, 1e-8)
    chosen_fsm_action = jax.random.choice(key2, NUM_FSM_ACTIONS, p=action_probs)

    discover_subnet = _pick_discover_subnet(state, const, agent_id, key3)
    host_subnet = const.host_subnet[chosen_host]
    target_subnet = jnp.where(chosen_fsm_action == FSM_ACT_DISCOVER, discover_subnet, host_subnet)
    jax_action = _fsm_action_to_jax_action(chosen_fsm_action, chosen_host, target_subnet)

    return (
        jnp.where(any_eligible, jax_action, RED_SLEEP),
        chosen_host,
        chosen_fsm_action,
        any_eligible,
    )


def fsm_red_get_action(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    key: jax.Array,
) -> int:
    action, _, _, _ = fsm_red_get_action_and_info(state, const, agent_id, key)
    return action


def _compute_scan_source_binding(state, const, agent_id, action):
    """Compute pending_source_kind and pending_source_host for scan actions."""
    action_type, _, _ = decode_red_action(action, agent_id, const)
    is_scan_action = (
        (action_type == ACTION_TYPE_SCAN)
        | (action_type == ACTION_TYPE_AGGRESSIVE_SCAN)
        | (action_type == ACTION_TYPE_STEALTH_SCAN)
    )
    bound_anchor_source = select_bound_source_host(state, const, agent_id)
    source_kind = jnp.where(
        is_scan_action,
        jnp.where(
            bound_anchor_source >= 0,
            PENDING_SOURCE_KIND_SESSION_BINDING,
            PENDING_SOURCE_KIND_NONE,
        ),
        PENDING_SOURCE_KIND_NONE,
    )
    source_host = jnp.int32(-1)
    return source_kind, source_host


def _select_one_agent(state, const, r, key):
    """Select action for a single red agent, handling busy/inactive gating and scan source binding."""
    is_busy = state.red_pending_ticks[r] > 0
    is_active = state.red_agent_active[r]
    action, host, fsm_act, eligible = jax.lax.cond(
        is_busy | ~is_active,
        lambda: (jnp.int32(RED_SLEEP), jnp.int32(0), jnp.int32(0), jnp.bool_(False)),
        lambda: fsm_red_get_action_and_info(state, const, r, key),
    )
    eff_host = jnp.where(is_busy, state.red_pending_target_host[r], host)
    eff_fsm_act = jnp.where(is_busy, state.red_pending_fsm_action[r], fsm_act)
    eff_eligible = jnp.where(is_busy, jnp.bool_(True), eligible)

    # Scan source pre-binding: compute only when not busy
    new_source_kind, new_source_host = _compute_scan_source_binding(state, const, r, action)
    source_kind = jnp.where(is_busy, state.red_pending_source_kind[r], new_source_kind)
    source_host = jnp.where(is_busy, state.red_pending_source_host[r], new_source_host)

    return action, eff_host, eff_fsm_act, eff_eligible, source_kind, source_host


def fsm_red_select_actions(
    state: CC4State,
    const: CC4Const,
    red_keys: jax.Array,
) -> tuple:
    """Select FSM red actions for all agents. Shared by training env and differential harness.

    Returns (red_actions, target_hosts, fsm_actions, eligible_flags, updated_state)
    where red_actions is shape (NUM_RED_AGENTS,) int32 array, and the rest are
    (NUM_RED_AGENTS,) arrays. updated_state has pending fields written.
    """
    red_actions = jnp.zeros(NUM_RED_AGENTS, dtype=jnp.int32)
    target_hosts = jnp.zeros(NUM_RED_AGENTS, dtype=jnp.int32)
    fsm_actions = jnp.zeros(NUM_RED_AGENTS, dtype=jnp.int32)
    eligible_flags = jnp.zeros(NUM_RED_AGENTS, dtype=jnp.bool_)

    for r in range(NUM_RED_AGENTS):
        action, eff_host, eff_fsm_act, eff_eligible, source_kind, source_host = _select_one_agent(
            state, const, r, red_keys[r]
        )
        red_actions = red_actions.at[r].set(action)
        target_hosts = target_hosts.at[r].set(eff_host)
        fsm_actions = fsm_actions.at[r].set(eff_fsm_act)
        eligible_flags = eligible_flags.at[r].set(eff_eligible)

        state = state.replace(
            red_pending_fsm_action=state.red_pending_fsm_action.at[r].set(eff_fsm_act),
            red_pending_target_host=state.red_pending_target_host.at[r].set(eff_host),
            red_pending_source_kind=state.red_pending_source_kind.at[r].set(source_kind),
            red_pending_source_host=state.red_pending_source_host.at[r].set(source_host),
        )

    return red_actions, target_hosts, fsm_actions, eligible_flags, state


def determine_fsm_success(
    state_before: CC4State,
    state_after: CC4State,
    agent_id: int,
    target_host: jnp.ndarray,
    fsm_action: int,
) -> jnp.ndarray:
    return jax.lax.switch(
        fsm_action,
        [
            lambda: jnp.any(state_after.red_discovered_hosts[agent_id] & ~state_before.red_discovered_hosts[agent_id]),
            lambda: (
                state_after.red_scanned_hosts[agent_id, target_host]
                & ~state_before.red_scanned_hosts[agent_id, target_host]
            ),
            lambda: (
                state_after.red_scanned_hosts[agent_id, target_host]
                & ~state_before.red_scanned_hosts[agent_id, target_host]
            ),
            lambda: jnp.bool_(True),
            lambda: state_after.red_sessions[agent_id, target_host] & ~state_before.red_sessions[agent_id, target_host],
            lambda: (
                state_after.red_privilege[agent_id, target_host] > state_before.red_privilege[agent_id, target_host]
            ),
            lambda: state_after.ot_service_stopped[target_host] & ~state_before.ot_service_stopped[target_host],
            lambda: jnp.any(
                state_after.host_service_reliability[target_host] < state_before.host_service_reliability[target_host]
            ),
            lambda: ~state_after.red_sessions[agent_id, target_host] & state_before.red_sessions[agent_id, target_host],
        ],
    )


def fsm_red_update_state(
    fsm_states: jnp.ndarray,
    const: CC4Const,
    agent_id: int,
    target_host: jnp.ndarray,
    fsm_action: int,
    success: jnp.ndarray,
) -> jnp.ndarray:
    cur = fsm_states[agent_id, target_host]

    next_success = TRANSITION_SUCCESS[cur, fsm_action]
    next_failure = TRANSITION_FAILURE[cur, fsm_action]
    next_state = jnp.where(success, next_success, next_failure)

    valid = next_state != _SENTINEL
    new_state = jnp.where(valid, next_state, cur)

    # CybORG U→F guard: hosts outside agent's subnets can't reach user-level access
    host_subnet = const.host_subnet[target_host]
    in_allowed_subnets = const.red_agent_subnets[agent_id, host_subnet]
    new_state = jnp.where((new_state == FSM_U) & ~in_allowed_subnets, FSM_F, new_state)

    return fsm_states.at[agent_id, target_host].set(new_state)


def fsm_red_post_step_update(
    state_before: CC4State,
    state_after: CC4State,
    const: CC4Const,
    target_hosts: list,
    fsm_actions: list,
    eligible_flags: list,
    executed_flags: list | None = None,
) -> CC4State:
    fsm_states = state_after.fsm_host_states

    for r in range(NUM_RED_AGENTS):
        success = determine_fsm_success(state_before, state_after, r, target_hosts[r], fsm_actions[r])
        exec_flag = jnp.bool_(True) if executed_flags is None else executed_flags[r]
        skip = ~eligible_flags[r] | ~exec_flag | (fsm_actions[r] == FSM_ACT_DISCOVER_DECEPTION)
        updated = fsm_red_update_state(fsm_states, const, r, target_hosts[r], fsm_actions[r], success)
        fsm_states = jnp.where(skip, fsm_states, updated)

    for r in range(NUM_RED_AGENTS):
        agent_fsm = fsm_states[r]
        has_session = state_after.red_sessions[r]
        was_compromised = (agent_fsm == FSM_U) | (agent_fsm == FSM_UD) | (agent_fsm == FSM_R) | (agent_fsm == FSM_RD)
        lost_session = was_compromised & ~has_session
        fsm_states = fsm_states.at[r].set(jnp.where(lost_session, FSM_KD, agent_fsm))

    return state_after.replace(fsm_host_states=fsm_states)


def fsm_red_process_session_removal(
    state: CC4State,
    agent_id: int,
) -> jnp.ndarray:
    fsm_states = state.fsm_host_states[agent_id]
    has_session = state.red_sessions[agent_id]
    was_compromised = (fsm_states == FSM_U) | (fsm_states == FSM_UD) | (fsm_states == FSM_R) | (fsm_states == FSM_RD)
    lost_session = was_compromised & ~has_session

    new_states = jnp.where(lost_session, FSM_KD, fsm_states)
    return state.fsm_host_states.at[agent_id].set(new_states)


def fsm_red_init_states(
    const: CC4Const,
    agent_id: int,
) -> jnp.ndarray:
    start_host = const.red_start_hosts[agent_id]
    fsm = jnp.full(GLOBAL_MAX_HOSTS, FSM_K, dtype=jnp.int32)
    fsm = fsm.at[start_host].set(FSM_U)
    return fsm
