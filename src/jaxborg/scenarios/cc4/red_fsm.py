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
    RED_EXPLOIT_HARAKA_START,
    RED_EXPLOIT_HTTP_START,
    RED_EXPLOIT_SQL_START,
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
    SERVICE_IDS,
)
from jaxborg.state import SimulatorConst, SimulatorState

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

SSH_SERVICE_IDX = SERVICE_IDS["SSHD"]
APACHE_SERVICE_IDX = SERVICE_IDS["APACHE2"]
MYSQL_SERVICE_IDX = SERVICE_IDS["MYSQLD"]
SMTP_SERVICE_IDX = SERVICE_IDS["SMTP"]

GENERIC_EXPLOIT_STARTS = jnp.array(
    [
        RED_EXPLOIT_HTTP_START,
        RED_EXPLOIT_SSH_START,
        RED_EXPLOIT_SQL_START,
        RED_EXPLOIT_HARAKA_START,
    ],
    dtype=jnp.int32,
)
GENERIC_EXPLOIT_WEIGHTS = jnp.array([3.0, 0.1, 5.0, 6.0], dtype=jnp.float32)


def _uniform_choice_from_mask(mask, u):
    count = jnp.sum(mask.astype(jnp.int32))
    safe_count = jnp.maximum(count, 1)
    rank = jnp.minimum(jnp.floor(u * safe_count).astype(jnp.int32), safe_count - 1)
    ranks = jnp.cumsum(mask.astype(jnp.int32)) - 1
    chosen = jnp.argmax(jnp.where(mask & (ranks == rank), 1, 0))
    return jnp.where(count > 0, chosen, jnp.int32(0))


def _weighted_choice_from_probs(probs, u):
    cdf = jnp.cumsum(probs)
    return jnp.argmax(u < cdf)


def _decode_choice_token(u, total_count):
    total = jnp.maximum(jnp.int32(total_count), jnp.int32(1))
    return jnp.minimum(jnp.floor(u * total).astype(jnp.int32), total - 1)


def _pick_exploit_action(state, target_host, key):
    host_services = state.host_services[target_host]
    candidates = jnp.array(
        [
            host_services[APACHE_SERVICE_IDX],
            host_services[SSH_SERVICE_IDX],
            host_services[APACHE_SERVICE_IDX] & host_services[MYSQL_SERVICE_IDX],
            host_services[SMTP_SERVICE_IDX],
        ],
        dtype=jnp.bool_,
    )
    num_candidates = jnp.sum(candidates.astype(jnp.int32))
    fallback = RED_EXPLOIT_SSH_START + target_host

    def _choose_candidate(_):
        weights = jnp.where(candidates, GENERIC_EXPLOIT_WEIGHTS, 0.0)
        top_idx = jnp.argmax(weights)
        reduced = candidates.at[top_idx].set(False)

        def _single():
            return jnp.argmax(candidates.astype(jnp.int32))

        def _multi():
            probs = reduced.astype(jnp.float32)
            probs = probs / jnp.sum(probs)
            return jax.random.choice(key, len(GENERIC_EXPLOIT_STARTS), p=probs)

        choice_idx = jax.lax.cond(num_candidates > 1, _multi, _single)
        return GENERIC_EXPLOIT_STARTS[choice_idx] + target_host

    return jax.lax.cond(num_candidates > 0, _choose_candidate, lambda _: fallback, operand=None)


def _fsm_action_to_jax_action(fsm_action, target_host, target_subnet, exploit_action):
    return jax.lax.switch(
        fsm_action,
        [
            lambda: RED_DISCOVER_START + target_subnet,
            lambda: RED_AGGRESSIVE_SCAN_START + target_host,
            lambda: RED_STEALTH_SCAN_START + target_host,
            lambda: RED_DISCOVER_DECEPTION_START + target_host,
            lambda: exploit_action,
            lambda: RED_PRIVESC_START + target_host,
            lambda: RED_IMPACT_START + target_host,
            lambda: RED_DEGRADE_START + target_host,
            lambda: RED_WITHDRAW_START + target_host,
        ],
    )


def _pick_discover_subnet(state, const, agent_id, key):
    # CybORG samples DiscoverRemoteSystems subnets from the controller action
    # space, which is keyed by the red agent's allowed subnets, not by current
    # session placement.
    probs = jnp.where(const.red_agent_subnets[agent_id], 1.0, 0.0)
    probs = probs / jnp.maximum(jnp.sum(probs), 1e-8)
    return jax.random.choice(key, NUM_SUBNETS, p=probs)


def fsm_red_get_action_and_info(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    key: jax.Array,
) -> tuple:
    fsm_states = state.fsm_host_states[agent_id]
    discovered = state.red_discovered_hosts[agent_id]
    active = const.host_active

    # CybORG's FSM only acts on hosts in host_states (explicitly observed).
    # fsm_host_entered tracks which hosts have entered the FSM, equivalent
    # to CybORG's host_states dict membership.  Newly discovered hosts
    # enter as K (not KD) matching CybORG's _process_new_observations.
    fsm_known = state.fsm_host_entered[agent_id]
    eligible = discovered & active & fsm_known & (fsm_states != FSM_F)

    key1, key2, key3, key4 = jax.random.split(key, 4)

    any_eligible = jnp.any(eligible)

    host_probs = jnp.where(eligible, 1.0, 0.0).astype(jnp.float32)
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
    exploit_action = _pick_exploit_action(state, chosen_host, key4)
    host_subnet = const.host_subnet[chosen_host]
    target_subnet = jnp.where(chosen_fsm_action == FSM_ACT_DISCOVER, discover_subnet, host_subnet)
    jax_action = _fsm_action_to_jax_action(chosen_fsm_action, chosen_host, target_subnet, exploit_action)

    return (
        jnp.where(any_eligible, jax_action, RED_SLEEP),
        chosen_host,
        target_subnet,
        chosen_fsm_action,
        any_eligible,
    )


def fsm_red_get_action(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    key: jax.Array,
) -> int:
    action, _, _, _, _ = fsm_red_get_action_and_info(state, const, agent_id, key)
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
    source_host = jnp.where(
        is_scan_action & (bound_anchor_source >= 0),
        bound_anchor_source,
        jnp.int32(-1),
    )
    return source_kind, source_host


def _select_one_agent(state, const, r, key):
    """Select action for a single red agent, handling busy/inactive gating and scan source binding."""
    is_busy = state.red_pending_ticks[r] > 0
    is_active = state.red_agent_active[r]
    action, host, target_subnet, fsm_act, eligible = jax.lax.cond(
        is_busy | ~is_active,
        lambda: (jnp.int32(RED_SLEEP), jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.bool_(False)),
        lambda: fsm_red_get_action_and_info(state, const, r, key),
    )
    eff_host = jnp.where(is_busy, state.red_pending_target_host[r], host)
    eff_subnet = jnp.where(is_busy, state.red_pending_target_subnet[r], target_subnet)
    eff_fsm_act = jnp.where(is_busy, state.red_pending_fsm_action[r], fsm_act)
    eff_eligible = jnp.where(is_busy, jnp.bool_(True), eligible)

    # Scan source pre-binding: compute only when not busy
    new_source_kind, new_source_host = _compute_scan_source_binding(state, const, r, action)
    source_kind = jnp.where(is_busy, state.red_pending_source_kind[r], new_source_kind)
    source_host = jnp.where(is_busy, state.red_pending_source_host[r], new_source_host)

    return action, eff_host, eff_subnet, eff_fsm_act, eff_eligible, source_kind, source_host


def fsm_red_select_actions(
    state: SimulatorState,
    const: SimulatorConst,
    red_keys: jax.Array,
) -> tuple:
    """Select FSM red actions for all agents. Shared by training env and differential harness.

    Returns (red_actions, target_hosts, target_subnets, fsm_actions, eligible_flags, updated_state)
    where red_actions is shape (NUM_RED_AGENTS,) int32 array, and the rest are
    (NUM_RED_AGENTS,) arrays. updated_state has pending fields written.
    """
    # Each agent's selection reads only its own agent-indexed state slices, so
    # all 6 selections can be computed from the SAME initial state.  This lets
    # XLA parallelize the independent computations instead of chaining them
    # through 6 sequential state.replace calls.
    all_results = [_select_one_agent(state, const, r, red_keys[r]) for r in range(NUM_RED_AGENTS)]

    red_actions = jnp.array([r[0] for r in all_results], dtype=jnp.int32)
    target_hosts = jnp.array([r[1] for r in all_results], dtype=jnp.int32)
    target_subnets = jnp.array([r[2] for r in all_results], dtype=jnp.int32)
    fsm_actions = jnp.array([r[3] for r in all_results], dtype=jnp.int32)
    eligible_flags = jnp.array([r[4] for r in all_results], dtype=jnp.bool_)
    source_kinds = jnp.array([r[5] for r in all_results], dtype=jnp.int32)
    source_hosts = jnp.array([r[6] for r in all_results], dtype=jnp.int32)

    # Single batched state update instead of 6 sequential replaces.
    state = state.replace(
        red_pending_fsm_action=fsm_actions,
        red_pending_target_host=target_hosts,
        red_pending_target_subnet=target_subnets,
        red_pending_source_kind=source_kinds,
        red_pending_source_host=source_hosts,
    )

    return red_actions, target_hosts, target_subnets, fsm_actions, eligible_flags, state


def fsm_red_apply_delayed_update(state: SimulatorState) -> SimulatorState:
    """Apply the previously scheduled FSM hidden-state update at decision time."""

    def _apply(s):
        return s.replace(
            fsm_host_states=s.red_fsm_delayed_states,
            red_fsm_delayed_pending=jnp.array(False),
        )

    return jax.lax.cond(state.red_fsm_delayed_pending, _apply, lambda s: s, state)


def determine_fsm_success(
    state_before: SimulatorState,
    state_after: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    target_host: jnp.ndarray,
    target_subnet: jnp.ndarray,
    fsm_action: int,
) -> jnp.ndarray:
    return jax.lax.switch(
        fsm_action,
        [
            # CybORG reports discover success when the action executes (valid
            # source session exists), regardless of whether new hosts are found.
            # The old check (any discovered hosts in subnet) was wrong: it
            # returned True even when the action failed because the agent had
            # already-known hosts from prior steps.  Use the per-step flag set
            # by apply_discover, mirroring the scan/exploit pattern.
            lambda: state_after.red_discover_success[agent_id],
            # CybORG reports scan success when the action executes (session
            # available, target reachable), regardless of prior scan state.
            # A re-scan on an already-scanned host succeeds in CybORG but the
            # delta check (after & ~before) returns False because
            # red_scanned_hosts was already True.  Use the per-step scan
            # success flag set by the scan action module.
            lambda: state_after.red_scan_success[agent_id],
            lambda: state_after.red_scan_success[agent_id],
            lambda: jnp.bool_(True),
            # Use pre-reassignment success flag: CybORG's FSM processes the
            # action's success BEFORE sessions are reassigned.  Checking
            # session_count post-reassignment gives False when the session was
            # immediately moved to another agent.
            lambda: state_after.red_exploit_success[agent_id],
            lambda: (
                state_after.red_privilege[agent_id, target_host] > state_before.red_privilege[agent_id, target_host]
            ),
            lambda: state_after.ot_service_stopped[target_host] & ~state_before.ot_service_stopped[target_host],
            lambda: (
                jnp.any(
                    state_after.host_service_reliability[target_host]
                    < state_before.host_service_reliability[target_host]
                )
                | jnp.any(
                    state_after.host_decoy_reliability[target_host] < state_before.host_decoy_reliability[target_host]
                )
            ),
            lambda: (
                state_after.red_session_count[agent_id, target_host]
                < state_before.red_session_count[agent_id, target_host]
            ),
        ],
    )


def fsm_red_update_state(
    fsm_states: jnp.ndarray,
    const: SimulatorConst,
    agent_id: int,
    target_host: jnp.ndarray,
    discovered_hosts: jnp.ndarray,
    target_subnet: jnp.ndarray,
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
    discover_mask = discovered_hosts & const.host_active & (const.host_subnet == target_subnet)

    def _apply_discover(_):
        cur_row = fsm_states[agent_id]
        next_success_row = TRANSITION_SUCCESS[cur_row, fsm_action]
        next_failure_row = TRANSITION_FAILURE[cur_row, fsm_action]
        next_row = jnp.where(success, next_success_row, next_failure_row)
        valid_row = next_row != _SENTINEL
        new_row = jnp.where(valid_row, next_row, cur_row)
        allowed_row = const.red_agent_subnets[agent_id, const.host_subnet]
        new_row = jnp.where((new_row == FSM_U) & ~allowed_row, FSM_F, new_row)
        updated_row = jnp.where(discover_mask, new_row, cur_row)
        return fsm_states.at[agent_id].set(updated_row)

    def _apply_single(_):
        return fsm_states.at[agent_id, target_host].set(new_state)

    return jax.lax.cond(fsm_action == FSM_ACT_DISCOVER, _apply_discover, _apply_single, operand=None)


def fsm_red_post_step_update(
    state_before: SimulatorState,
    state_after: SimulatorState,
    const: SimulatorConst,
    target_hosts: list,
    target_subnets: list,
    fsm_actions: list,
    eligible_flags: list,
    executed_flags: list | None = None,
) -> SimulatorState:
    fsm_states = _compute_post_step_fsm_states(
        state_before,
        state_after,
        const,
        target_hosts,
        target_subnets,
        fsm_actions,
        eligible_flags,
        executed_flags,
    )
    return state_after.replace(fsm_host_states=fsm_states)


def _compute_post_step_fsm_states(
    state_before: SimulatorState,
    state_after: SimulatorState,
    const: SimulatorConst,
    target_hosts: list,
    target_subnets: list,
    fsm_actions: list,
    eligible_flags: list,
    executed_flags: list | None = None,
) -> jnp.ndarray:
    fsm_states = state_after.fsm_host_states

    # Each agent's FSM update reads/writes only its own row in fsm_states,
    # so all 6 agents can be computed from the same initial state.
    updated_rows = []
    skip_flags = []
    for r in range(NUM_RED_AGENTS):
        success = determine_fsm_success(
            state_before,
            state_after,
            const,
            r,
            target_hosts[r],
            target_subnets[r],
            fsm_actions[r],
        )
        exec_flag = jnp.bool_(True) if executed_flags is None else executed_flags[r]
        skip = (
            ~eligible_flags[r]
            | ~exec_flag
            | (fsm_actions[r] == FSM_ACT_DISCOVER_DECEPTION)
            | ~state_after.red_agent_active[r]
        )
        discovered_for_fsm = state_before.red_discovered_hosts[r] & state_before.fsm_host_entered[r]
        updated = fsm_red_update_state(
            fsm_states,
            const,
            r,
            target_hosts[r],
            discovered_for_fsm,
            target_subnets[r],
            fsm_actions[r],
            success,
        )
        updated_rows.append(updated[r])
        skip_flags.append(skip)

    # Batch merge: for each agent, select between original and updated row.
    updated_matrix = jnp.stack(updated_rows)  # (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS)
    skip_mask = jnp.stack(skip_flags)[:, None]  # (NUM_RED_AGENTS, 1)
    fsm_states = jnp.where(skip_mask, fsm_states, updated_matrix)

    # Session-loss recovery: each agent reads/writes only its own row.
    # Compute all recoveries from the same state, then batch apply.
    was_compromised = (fsm_states == FSM_U) | (fsm_states == FSM_UD) | (fsm_states == FSM_R) | (fsm_states == FSM_RD)
    lost_session = was_compromised & ~state_after.red_sessions & state_after.red_agent_active[:, None]
    fsm_states = jnp.where(lost_session, FSM_KD, fsm_states)

    return fsm_states


def fsm_red_schedule_post_step_update(
    state_before: SimulatorState,
    state_after: SimulatorState,
    const: SimulatorConst,
    target_hosts: list,
    target_subnets: list,
    fsm_actions: list,
    eligible_flags: list,
    executed_flags: list | None = None,
) -> SimulatorState:
    """Compute the next FSM hidden state and schedule it for the next decision step.

    CybORG updates FiniteStateRedAgent host_states when the agent processes the
    previous observation at the next action-selection point, not on the same
    simulation step that a duration action finishes.
    """

    next_fsm_states = _compute_post_step_fsm_states(
        state_before,
        state_after,
        const,
        target_hosts,
        target_subnets,
        fsm_actions,
        eligible_flags,
        executed_flags,
    )
    return state_after.replace(
        red_fsm_delayed_states=next_fsm_states,
        red_fsm_delayed_pending=jnp.any(next_fsm_states != state_after.fsm_host_states),
    )


def fsm_red_process_session_removal(
    state: SimulatorState,
    agent_id: int,
) -> jnp.ndarray:
    fsm_states = state.fsm_host_states[agent_id]
    has_session = state.red_sessions[agent_id]
    was_compromised = (fsm_states == FSM_U) | (fsm_states == FSM_UD) | (fsm_states == FSM_R) | (fsm_states == FSM_RD)
    lost_session = was_compromised & ~has_session

    new_states = jnp.where(lost_session, FSM_KD, fsm_states)
    return state.fsm_host_states.at[agent_id].set(new_states)


def fsm_red_init_states(
    const: SimulatorConst,
    agent_id: int,
) -> jnp.ndarray:
    start_host = const.red_start_hosts[agent_id]
    fsm = jnp.full(GLOBAL_MAX_HOSTS, FSM_K, dtype=jnp.int32)
    fsm = fsm.at[start_host].set(FSM_U)
    return fsm
