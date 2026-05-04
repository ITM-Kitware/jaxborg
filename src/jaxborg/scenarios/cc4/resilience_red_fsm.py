"""Resilience-aware red FSM agent.

Wraps the base ``red_fsm`` with biased host selection: when choosing which
eligible host to attack, resilience-critical servers (auth, db, web) are
weighted ``RESILIENCE_TARGET_WEIGHT`` times more likely to be selected than
ordinary hosts.

All FSM transition logic, action selection, and state-update functions are
inherited unchanged from ``red_fsm``.  Only host selection probability
changes.

Usage::

    from jaxborg.scenarios.cc4.resilience_topology import build_resilience_topology
    from jaxborg.scenarios.cc4.resilience_red_fsm import resilience_red_select_actions

    const, host_resilience_role = build_resilience_topology(key)
    red_actions, *rest, state = resilience_red_select_actions(
        state, const, host_resilience_role, red_keys
    )
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxborg.actions.encoding import RED_SLEEP
from jaxborg.actions.pending_source import (
    PENDING_SOURCE_KIND_NONE,
    PENDING_SOURCE_KIND_SESSION_BINDING,
)
from jaxborg.actions.red_common import select_bound_source_host

from jaxborg.actions.rng import sample_red_policy_random
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_RED_AGENTS
from jaxborg.scenarios.cc4.red_fsm import (
    ACTION_VALID_MASK,
    FSM_ACT_DISCOVER,
    NUM_FSM_ACTIONS,
    NUM_FSM_STATES,
    PROBABILITY_MATRIX,
    FSM_F,
    _compute_scan_source_binding,
    _decode_choice_token,
    _fsm_action_to_jax_action,
    _pick_discover_subnet,
    _pick_exploit_action,
    _uniform_choice_from_mask,
    _weighted_choice_from_probs,
)
from jaxborg.scenarios.cc4.resilience_topology import RESILIENCE_ROLE_NONE
from jaxborg.state import SimulatorConst, SimulatorState

# How much more likely resilience-critical servers are to be targeted.
RESILIENCE_TARGET_WEIGHT = 5.0


def _resilience_host_probs(
    eligible: jax.Array,
    host_resilience_role: jax.Array,
) -> jax.Array:
    """Return a probability vector over hosts, up-weighting resilience servers."""
    weight = jnp.where(host_resilience_role != RESILIENCE_ROLE_NONE, RESILIENCE_TARGET_WEIGHT, 1.0)
    probs = jnp.where(eligible, weight, 0.0).astype(jnp.float32)
    total = jnp.sum(probs)
    return probs / jnp.maximum(total, 1e-8)


def _resilience_get_action_and_info(
    state: SimulatorState,
    const: SimulatorConst,
    host_resilience_role: jax.Array,
    agent_id: int,
    key: jax.Array,
) -> tuple:
    """Select an action for one red agent with resilience-biased host targeting."""
    fsm_states = state.fsm_host_states[agent_id]
    discovered = state.red_discovered_hosts[agent_id]
    active = const.host_active

    fsm_known = state.fsm_host_entered[agent_id]
    eligible = discovered & active & fsm_known & (fsm_states != FSM_F)

    key1, key2, key3, key4 = jax.random.split(key, 4)
    time_idx = jnp.minimum(jnp.int32(state.time), jnp.int32(const.red_policy_randoms.shape[0] - 1))

    any_eligible = jnp.any(eligible)

    # Resilience-biased host probabilities (replaces uniform 1/n weighting).
    host_probs = _resilience_host_probs(eligible, host_resilience_role)

    host_u = sample_red_policy_random(const, time_idx, agent_id, 0, key1)
    recorded_host = _decode_choice_token(host_u, GLOBAL_MAX_HOSTS)
    # When using precomputed tapes (parity mode) fall back to the same
    # uniform-fallback logic as the base FSM — those tapes were generated with
    # the uniform policy.
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

    action_probs_raw = PROBABILITY_MATRIX[host_state_clamped]
    valid_mask = ACTION_VALID_MASK[host_state_clamped]
    action_probs = jnp.where(valid_mask, jnp.maximum(action_probs_raw, 0.0), 0.0)
    action_total = jnp.sum(action_probs)
    action_probs = action_probs / jnp.maximum(action_total, 1e-8)
    action_u = sample_red_policy_random(const, time_idx, agent_id, 1, key2)
    recorded_action = _decode_choice_token(action_u, NUM_FSM_ACTIONS)
    chosen_fsm_action = jax.lax.cond(
        const.use_red_policy_randoms,
        lambda _: jnp.where(
            valid_mask[recorded_action],
            recorded_action,
            _weighted_choice_from_probs(action_probs, action_u),
        ),
        lambda _: jax.random.choice(key2, NUM_FSM_ACTIONS, p=action_probs),
        operand=None,
    )

    discover_u = sample_red_policy_random(const, time_idx, agent_id, 2, key3)
    recorded_subnet = _decode_choice_token(discover_u, jnp.int32(const.red_agent_subnets.shape[1]))
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


def _resilience_select_one_agent(
    state: SimulatorState,
    const: SimulatorConst,
    host_resilience_role: jax.Array,
    r: int,
    key: jax.Array,
) -> tuple:
    is_busy = state.red_pending_ticks[r] > 0
    is_active = state.red_agent_active[r]
    action, host, target_subnet, fsm_act, eligible = jax.lax.cond(
        is_busy | ~is_active,
        lambda: (jnp.int32(RED_SLEEP), jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.bool_(False)),
        lambda: _resilience_get_action_and_info(state, const, host_resilience_role, r, key),
    )
    eff_host = jnp.where(is_busy, state.red_pending_target_host[r], host)
    eff_subnet = jnp.where(is_busy, state.red_pending_target_subnet[r], target_subnet)
    eff_fsm_act = jnp.where(is_busy, state.red_pending_fsm_action[r], fsm_act)
    eff_eligible = jnp.where(is_busy, jnp.bool_(True), eligible)

    new_source_kind, new_source_host = _compute_scan_source_binding(state, const, r, action)
    source_kind = jnp.where(is_busy, state.red_pending_source_kind[r], new_source_kind)
    source_host = jnp.where(is_busy, state.red_pending_source_host[r], new_source_host)

    return action, eff_host, eff_subnet, eff_fsm_act, eff_eligible, source_kind, source_host


def resilience_red_select_actions(
    state: SimulatorState,
    const: SimulatorConst,
    host_resilience_role: jax.Array,
    red_keys: jax.Array,
) -> tuple:
    """Select resilience-biased red actions for all agents.

    Drop-in replacement for ``fsm_red_select_actions`` — same return shape,
    same state update — but auth/db/web servers are ``RESILIENCE_TARGET_WEIGHT``
    times more likely to be targeted when the agent selects a host.

    Args:
        state:                Current simulator state.
        const:                Static topology constants.
        host_resilience_role: ``(GLOBAL_MAX_HOSTS,)`` int32 from
                              ``build_resilience_topology``.
        red_keys:             ``(NUM_RED_AGENTS,)`` JAX PRNG keys.

    Returns:
        (red_actions, target_hosts, target_subnets, fsm_actions,
         eligible_flags, updated_state)
    """
    all_results = [
        _resilience_select_one_agent(state, const, host_resilience_role, r, red_keys[r])
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
