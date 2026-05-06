"""CIA-targeted and resilience-biased red FSM agents.

Drop-in replacements for ``fsm_red_select_actions``. All variants bias host
selection toward specific resilience-critical servers. The CIA-targeted variants
(c/i/a) additionally bias action selection at root-access states toward Impact
and DegradeServices.

Agents and their target server sets (from ``resilience_topology``):

  resilience_red_select_actions — all resilience-critical servers (AUTH + DB + WEB)
                                   with configurable weight; uses base FSM action probs.
  c_red_select_actions          — targets C: AUTH + DB servers
  i_red_select_actions          — targets I: AUTH + WEB servers
  a_red_select_actions          — targets A: AUTH + DB + WEB servers

Usage::

    from jaxborg.scenarios.cc4.resilience_topology import build_resilience_topology
    from jaxborg.scenarios.cc4.resilience_red_fsm import (
        resilience_red_select_actions,
        c_red_select_actions,
        i_red_select_actions,
        a_red_select_actions,
    )

    const, host_resilience_role = build_resilience_topology(key)
    red_actions, *rest, state = c_red_select_actions(
        state, const, host_resilience_role, red_keys
    )
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxborg.actions.encoding import RED_SLEEP
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_RED_AGENTS, NUM_SUBNETS
from jaxborg.scenarios.cc4.red_fsm import (
    ACTION_VALID_MASK,
    FSM_ACT_DISCOVER,
    FSM_F,
    NUM_FSM_ACTIONS,
    NUM_FSM_STATES,
    PROBABILITY_MATRIX,
    _compute_scan_source_binding,
    _decode_choice_token,
    _fsm_action_to_jax_action,
    _pick_discover_subnet,
    _pick_exploit_action,
    _uniform_choice_from_mask,
    _weighted_choice_from_probs,
)
from jaxborg.scenarios.cc4.resilience_topology import (
    RESILIENCE_ROLE_AUTH,
    RESILIENCE_ROLE_DB,
    RESILIENCE_ROLE_WEB,
)
from jaxborg.actions.rng import sample_red_policy_random
from jaxborg.state import SimulatorConst, SimulatorState

# Weight multiplier for targeted servers used by CIA-specific agents.
_CIA_TARGET_WEIGHT = 10.0

# Modified probability matrix for CIA-targeted agents: at FSM_R (root, undiscovered,
# state index 6) shift probability from Discover toward Impact + Degrade.
# Column order: DISCOVER AGGSCAN STEALTHSCAN DISC_DEC EXPLOIT PRIVESC IMPACT DEGRADE WITHDRAW
_p = -1.0  # sentinel for invalid actions
_TARGETED_PROBABILITY_MATRIX = PROBABILITY_MATRIX.at[6].set(
    jnp.array([0.1, _p, _p, _p, _p, _p, 0.45, 0.45, 0.0], dtype=jnp.float32)
)
_TARGETED_ACTION_VALID_MASK = _TARGETED_PROBABILITY_MATRIX >= 0.0


def _role_weights(
    host_resilience_role: jax.Array,
    role_set: tuple[int, ...],
    weight: float,
) -> jax.Array:
    """Return per-host weight array: weight for roles in role_set, else 1.0."""
    weights = jnp.ones(GLOBAL_MAX_HOSTS, dtype=jnp.float32)
    for role in role_set:
        weights = jnp.where(host_resilience_role == role, weight, weights)
    return weights


def _get_action_and_info(
    state: SimulatorState,
    const: SimulatorConst,
    host_weights: jax.Array,   # (GLOBAL_MAX_HOSTS,) float32
    prob_matrix: jax.Array,    # (NUM_FSM_STATES, NUM_FSM_ACTIONS) float32
    valid_mask: jax.Array,     # (NUM_FSM_STATES, NUM_FSM_ACTIONS) bool
    agent_id: int,
    key: jax.Array,
) -> tuple:
    fsm_states = state.fsm_host_states[agent_id]
    discovered = state.red_discovered_hosts[agent_id]
    active = const.host_active

    fsm_known = state.fsm_host_entered[agent_id]
    eligible = discovered & active & fsm_known & (fsm_states != FSM_F)

    key1, key2, key3, key4 = jax.random.split(key, 4)
    time_idx = jnp.minimum(jnp.int32(state.time), jnp.int32(const.red_policy_randoms.shape[0] - 1))

    any_eligible = jnp.any(eligible)

    raw = jnp.where(eligible, host_weights, 0.0)
    host_probs = raw / jnp.maximum(jnp.sum(raw), 1e-8)

    host_u = sample_red_policy_random(const, time_idx, agent_id, 0, key1)
    recorded_host = _decode_choice_token(host_u, GLOBAL_MAX_HOSTS)
    chosen_host = jax.lax.cond(
        const.use_red_policy_randoms,
        lambda _: jnp.where(
            eligible[recorded_host],
            recorded_host,
            _uniform_choice_from_mask(eligible, host_u),
        ),
        lambda _: jax.random.choice(key1, GLOBAL_MAX_HOSTS, p=host_probs),
        operand=None,
    )

    host_state = fsm_states[chosen_host]
    host_state_clamped = jnp.clip(host_state, 0, NUM_FSM_STATES - 2)

    action_probs_raw = prob_matrix[host_state_clamped]
    vmask = valid_mask[host_state_clamped]
    action_probs = jnp.where(vmask, jnp.maximum(action_probs_raw, 0.0), 0.0)
    action_probs = action_probs / jnp.maximum(jnp.sum(action_probs), 1e-8)

    action_u = sample_red_policy_random(const, time_idx, agent_id, 1, key2)
    recorded_action = _decode_choice_token(action_u, NUM_FSM_ACTIONS)
    chosen_fsm_action = jax.lax.cond(
        const.use_red_policy_randoms,
        lambda _: jnp.where(
            vmask[recorded_action],
            recorded_action,
            _weighted_choice_from_probs(action_probs, action_u),
        ),
        lambda _: jax.random.choice(key2, NUM_FSM_ACTIONS, p=action_probs),
        operand=None,
    )

    discover_u = sample_red_policy_random(const, time_idx, agent_id, 2, key3)
    recorded_subnet = _decode_choice_token(discover_u, jnp.int32(NUM_SUBNETS))
    discover_subnet = jax.lax.cond(
        const.use_red_policy_randoms,
        lambda _: jnp.where(
            const.red_agent_subnets[agent_id, recorded_subnet],
            recorded_subnet,
            _uniform_choice_from_mask(const.red_agent_subnets[agent_id], discover_u),
        ),
        lambda _: _pick_discover_subnet(state, const, agent_id, key3),
        operand=None,
    )

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


def _select_one_agent(
    state: SimulatorState,
    const: SimulatorConst,
    host_weights: jax.Array,
    prob_matrix: jax.Array,
    valid_mask: jax.Array,
    r: int,
    key: jax.Array,
) -> tuple:
    is_busy = state.red_pending_ticks[r] > 0
    is_active = state.red_agent_active[r]
    action, host, target_subnet, fsm_act, eligible = jax.lax.cond(
        is_busy | ~is_active,
        lambda: (jnp.int32(RED_SLEEP), jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.bool_(False)),
        lambda: _get_action_and_info(state, const, host_weights, prob_matrix, valid_mask, r, key),
    )
    eff_host    = jnp.where(is_busy, state.red_pending_target_host[r],   host)
    eff_subnet  = jnp.where(is_busy, state.red_pending_target_subnet[r], target_subnet)
    eff_fsm_act = jnp.where(is_busy, state.red_pending_fsm_action[r],    fsm_act)
    eff_eligible = jnp.where(is_busy, jnp.bool_(True), eligible)

    new_source_kind, new_source_host = _compute_scan_source_binding(state, const, r, action)
    source_kind = jnp.where(is_busy, state.red_pending_source_kind[r], new_source_kind)
    source_host = jnp.where(is_busy, state.red_pending_source_host[r], new_source_host)

    return action, eff_host, eff_subnet, eff_fsm_act, eff_eligible, source_kind, source_host


def _red_select_actions(
    state: SimulatorState,
    const: SimulatorConst,
    host_weights: jax.Array,
    prob_matrix: jax.Array,
    valid_mask: jax.Array,
    red_keys: jax.Array,
) -> tuple:
    all_results = [
        _select_one_agent(state, const, host_weights, prob_matrix, valid_mask, r, red_keys[r])
        for r in range(NUM_RED_AGENTS)
    ]

    red_actions    = jnp.array([r[0] for r in all_results], dtype=jnp.int32)
    target_hosts   = jnp.array([r[1] for r in all_results], dtype=jnp.int32)
    target_subnets = jnp.array([r[2] for r in all_results], dtype=jnp.int32)
    fsm_actions    = jnp.array([r[3] for r in all_results], dtype=jnp.int32)
    eligible_flags = jnp.array([r[4] for r in all_results], dtype=jnp.bool_)
    source_kinds   = jnp.array([r[5] for r in all_results], dtype=jnp.int32)
    source_hosts   = jnp.array([r[6] for r in all_results], dtype=jnp.int32)

    state = state.replace(
        red_pending_fsm_action=fsm_actions,
        red_pending_target_host=target_hosts,
        red_pending_target_subnet=target_subnets,
        red_pending_source_kind=source_kinds,
        red_pending_source_host=source_hosts,
    )

    return red_actions, target_hosts, target_subnets, fsm_actions, eligible_flags, state


# ---------------------------------------------------------------------------
# Public entry points.

def resilience_red_select_actions(
    state: SimulatorState,
    const: SimulatorConst,
    host_resilience_role: jax.Array,
    red_keys: jax.Array,
    target_weight: float = 5.0,
) -> tuple:
    """Red agent biased toward all resilience-critical servers (AUTH + DB + WEB).

    Uses the base FSM action probability matrix — no extra bias toward
    Impact/Degrade at root access. All resilience-critical servers are weighted
    ``target_weight`` times more likely to be targeted than ordinary hosts.
    """
    host_weights = _role_weights(
        host_resilience_role,
        (RESILIENCE_ROLE_AUTH, RESILIENCE_ROLE_DB, RESILIENCE_ROLE_WEB),
        target_weight,
    )
    return _red_select_actions(state, const, host_weights, PROBABILITY_MATRIX, ACTION_VALID_MASK, red_keys)


def c_red_select_actions(
    state: SimulatorState,
    const: SimulatorConst,
    host_resilience_role: jax.Array,
    red_keys: jax.Array,
) -> tuple:
    """Red agent targeting C: prefers AUTH and DB servers.

    Biases host selection toward RESILIENCE_ROLE_AUTH and RESILIENCE_ROLE_DB,
    and at root-access state heavily favors Impact/DegradeServices.
    """
    host_weights = _role_weights(
        host_resilience_role, (RESILIENCE_ROLE_AUTH, RESILIENCE_ROLE_DB), _CIA_TARGET_WEIGHT
    )
    return _red_select_actions(
        state, const, host_weights, _TARGETED_PROBABILITY_MATRIX, _TARGETED_ACTION_VALID_MASK, red_keys
    )


def i_red_select_actions(
    state: SimulatorState,
    const: SimulatorConst,
    host_resilience_role: jax.Array,
    red_keys: jax.Array,
) -> tuple:
    """Red agent targeting I: prefers AUTH and WEB servers.

    Biases host selection toward RESILIENCE_ROLE_AUTH and RESILIENCE_ROLE_WEB,
    and at root-access state heavily favors Impact/DegradeServices.
    """
    host_weights = _role_weights(
        host_resilience_role, (RESILIENCE_ROLE_AUTH, RESILIENCE_ROLE_WEB), _CIA_TARGET_WEIGHT
    )
    return _red_select_actions(
        state, const, host_weights, _TARGETED_PROBABILITY_MATRIX, _TARGETED_ACTION_VALID_MASK, red_keys
    )


def a_red_select_actions(
    state: SimulatorState,
    const: SimulatorConst,
    host_resilience_role: jax.Array,
    red_keys: jax.Array,
) -> tuple:
    """Red agent targeting A: prefers AUTH, DB, and WEB servers.

    Biases host selection toward all three resilience-critical servers,
    and at root-access state heavily favors Impact/DegradeServices.
    """
    host_weights = _role_weights(
        host_resilience_role,
        (RESILIENCE_ROLE_AUTH, RESILIENCE_ROLE_DB, RESILIENCE_ROLE_WEB),
        _CIA_TARGET_WEIGHT,
    )
    return _red_select_actions(
        state, const, host_weights, _TARGETED_PROBABILITY_MATRIX, _TARGETED_ACTION_VALID_MASK, red_keys
    )
