"""Pluggable red action selectors.

A ``RedSelector`` is a callable wired into ``FsmRedCC4Env`` that chooses red
actions each step. All selectors share the same signature::

    selector(state, const, host_resilience_role, red_keys) -> (
        red_actions,        # (NUM_RED_AGENTS,) int32
        target_hosts,       # (NUM_RED_AGENTS,) int32
        target_subnets,     # (NUM_RED_AGENTS,) int32
        fsm_actions,        # (NUM_RED_AGENTS,) int32
        eligible_flags,     # (NUM_RED_AGENTS,) bool
        state,              # SimulatorState (with red_pending_* updated)
    )

``host_resilience_role`` is always passed (a zeros array when no role
assignment is in play); selectors that don't need it ignore the argument.

To add a new biased red:

1. Either parameterise :func:`role_biased_selector` (host bias plus an optional
   FSM action-prob matrix override) or write a fresh function with the
   signature above.
2. Add a name → factory mapping in :data:`REGISTRY`. Recipes pick by name.

If a new selector needs extras beyond ``host_resilience_role``, extend the
env-state extras schema (one place, in ``fsm_red_env``) rather than letting
parameters proliferate here.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from jaxborg.actions.encoding import RED_SLEEP
from jaxborg.actions.rng import sample_red_policy_choice
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_RED_AGENTS
from jaxborg.scenarios.cc4.red_fsm import (
    ACTION_VALID_MASK,
    FSM_ACT_DISCOVER,
    FSM_F,
    NUM_FSM_ACTIONS,
    NUM_FSM_STATES,
    PROBABILITY_MATRIX,
    RED_POLICY_FIELD_ACTION,
    RED_POLICY_FIELD_HOST,
    _compute_scan_source_binding,
    _fsm_action_to_jax_action,
    _pick_discover_subnet,
    _pick_exploit_action,
    fsm_red_select_actions,
)
from jaxborg.scenarios.cc4.topology_roles import ROLE_AUTH, ROLE_DB, ROLE_WEB
from jaxborg.state import SimulatorConst, SimulatorState

# A RedSelector takes (state, const, host_resilience_role, red_keys) and
# returns the 6-tuple shown in the module docstring.
RedSelector = Callable[..., tuple]


# ---------------------------------------------------------------------------
# Vanilla pass-through.


def fsm_selector(state, const, host_resilience_role, red_keys):
    """Vanilla CC4 finite-state red — biases nothing, ignores roles."""
    del host_resilience_role
    return fsm_red_select_actions(state, const, red_keys)


# ---------------------------------------------------------------------------
# Generic role-biased selector — supersedes the four hand-rolled C/I/A
# variants and the resilience selector. Differences between those collapse to
# (target_roles, target_weight, prob_matrix override).

# CIA-targeted action-prob matrix: at FSM_R (root, undiscovered, state index 6)
# shift mass from Discover toward Impact + Degrade.
# Column order: DISCOVER AGGSCAN STEALTHSCAN DISC_DEC EXPLOIT PRIVESC IMPACT DEGRADE WITHDRAW
_CIA_INVALID = -1.0
_CIA_PROB_MATRIX = PROBABILITY_MATRIX.at[6].set(
    jnp.array(
        [0.1, _CIA_INVALID, _CIA_INVALID, _CIA_INVALID, _CIA_INVALID, _CIA_INVALID, 0.45, 0.45, 0.0],
        dtype=jnp.float32,
    )
)
_CIA_VALID_MASK = _CIA_PROB_MATRIX >= 0.0


def role_biased_selector(
    *,
    target_roles: tuple[int, ...],
    target_weight: float = 5.0,
    prob_matrix: Optional[jax.Array] = None,
    valid_mask: Optional[jax.Array] = None,
) -> RedSelector:
    """Build a selector that biases host selection toward hosts in ``target_roles``.

    Empty ``target_roles`` is allowed and degrades gracefully to "uniform over
    eligible" — useful as a parity sanity check.
    """
    pm = PROBABILITY_MATRIX if prob_matrix is None else prob_matrix
    vm = ACTION_VALID_MASK if valid_mask is None else valid_mask

    def _selector(state, const, host_resilience_role, red_keys):
        host_weights = jnp.ones_like(host_resilience_role, dtype=jnp.float32)
        for role in target_roles:
            host_weights = jnp.where(host_resilience_role == role, target_weight, host_weights)
        return _select_all_red_agents(state, const, host_weights, pm, vm, red_keys)

    return _selector


# ---------------------------------------------------------------------------
# Internal FSM machinery — the "biased FSM" that all role-biased selectors
# share. These were previously in ``resilience_red_fsm.py`` (now deleted).
# Kept module-private; selectors above are the public interface.


def _get_action_and_info(
    state: SimulatorState,
    const: SimulatorConst,
    host_weights: jax.Array,
    prob_matrix: jax.Array,
    valid_mask: jax.Array,
    agent_id: int,
    key: jax.Array,
) -> tuple:
    fsm_states = state.fsm_host_states[agent_id]
    discovered = state.red_discovered_hosts[agent_id]
    fsm_known = state.fsm_host_entered[agent_id]
    eligible = discovered & const.host_active & fsm_known & (fsm_states != FSM_F)

    key1, key2, key3, key4 = jax.random.split(key, 4)
    any_eligible = jnp.any(eligible)

    # eligible-mask × per-host weights → categorical distribution over hosts.
    raw = jnp.where(eligible, host_weights, 0.0).astype(jnp.float32)
    host_probs = raw / jnp.maximum(jnp.sum(raw), 1e-8)
    chosen_host = sample_red_policy_choice(const, state.time, agent_id, RED_POLICY_FIELD_HOST, key1, host_probs)
    chosen_host = jnp.clip(chosen_host, 0, jnp.int32(GLOBAL_MAX_HOSTS - 1))

    host_state_clamped = jnp.clip(fsm_states[chosen_host], 0, NUM_FSM_STATES - 2)
    action_probs_raw = prob_matrix[host_state_clamped]
    vmask = valid_mask[host_state_clamped]
    action_probs = jnp.where(vmask, jnp.maximum(action_probs_raw, 0.0), 0.0)
    action_probs = action_probs / jnp.maximum(jnp.sum(action_probs), 1e-8)
    chosen_fsm_action = sample_red_policy_choice(
        const, state.time, agent_id, RED_POLICY_FIELD_ACTION, key2, action_probs
    )
    chosen_fsm_action = jnp.clip(chosen_fsm_action, 0, jnp.int32(NUM_FSM_ACTIONS - 1))

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
    eff_host = jnp.where(is_busy, state.red_pending_target_host[r], host)
    eff_subnet = jnp.where(is_busy, state.red_pending_target_subnet[r], target_subnet)
    eff_fsm_act = jnp.where(is_busy, state.red_pending_fsm_action[r], fsm_act)
    eff_eligible = jnp.where(is_busy, jnp.bool_(True), eligible)

    new_source_kind, new_source_host = _compute_scan_source_binding(state, const, r, action)
    source_kind = jnp.where(is_busy, state.red_pending_source_kind[r], new_source_kind)
    source_host = jnp.where(is_busy, state.red_pending_source_host[r], new_source_host)

    return action, eff_host, eff_subnet, eff_fsm_act, eff_eligible, source_kind, source_host


def _select_all_red_agents(
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
    red_actions = jnp.array([r[0] for r in all_results], dtype=jnp.int32)
    target_hosts = jnp.array([r[1] for r in all_results], dtype=jnp.int32)
    target_subnets = jnp.array([r[2] for r in all_results], dtype=jnp.int32)
    fsm_actions = jnp.array([r[3] for r in all_results], dtype=jnp.int32)
    eligible_flags = jnp.array([r[4] for r in all_results], dtype=jnp.bool_)
    source_kinds = jnp.array([r[5] for r in all_results], dtype=jnp.int32)
    source_hosts = jnp.array([r[6] for r in all_results], dtype=jnp.int32)

    state = state.replace(
        red_pending_fsm_action=fsm_actions,
        red_pending_target_host=target_hosts,
        red_pending_target_subnet=target_subnets,
        red_pending_source_kind=source_kinds,
        red_pending_source_host=source_hosts,
    )
    return red_actions, target_hosts, target_subnets, fsm_actions, eligible_flags, state


# ---------------------------------------------------------------------------
# Registry — name → factory(**kwargs) → RedSelector.


def _resilience(target_weight: float = 5.0, **_) -> RedSelector:
    return role_biased_selector(
        target_roles=(ROLE_AUTH, ROLE_DB, ROLE_WEB),
        target_weight=float(target_weight),
    )


def _cia_c(target_weight: float = 10.0, **_) -> RedSelector:
    return role_biased_selector(
        target_roles=(ROLE_AUTH, ROLE_DB),
        target_weight=float(target_weight),
        prob_matrix=_CIA_PROB_MATRIX,
        valid_mask=_CIA_VALID_MASK,
    )


def _cia_i(target_weight: float = 10.0, **_) -> RedSelector:
    return role_biased_selector(
        target_roles=(ROLE_AUTH, ROLE_WEB),
        target_weight=float(target_weight),
        prob_matrix=_CIA_PROB_MATRIX,
        valid_mask=_CIA_VALID_MASK,
    )


def _cia_a(target_weight: float = 10.0, **_) -> RedSelector:
    return role_biased_selector(
        target_roles=(ROLE_AUTH, ROLE_DB, ROLE_WEB),
        target_weight=float(target_weight),
        prob_matrix=_CIA_PROB_MATRIX,
        valid_mask=_CIA_VALID_MASK,
    )


REGISTRY: dict[str, Callable[..., RedSelector]] = {
    "fsm": lambda **_: fsm_selector,
    "finite_state": lambda **_: fsm_selector,  # CybORG-side agent name alias
    "resilience": _resilience,
    "cia_c": _cia_c,
    "c": _cia_c,
    "cia_i": _cia_i,
    "i": _cia_i,
    "cia_a": _cia_a,
    "a": _cia_a,
}


def make_red_selector(name: str, **kwargs) -> RedSelector:
    """Build a selector by name. Recipe-driven entry point.

    ``make_red_selector(cfg["red_agent"], target_weight=cfg["resilience_target_weight"])``
    — selector kwargs that don't apply to the chosen factory are ignored.
    """
    factory = REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unknown red selector {name!r}. Registered: {sorted(REGISTRY)}")
    return factory(**kwargs)
