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

``host_resilience_role`` is always passed (a zeros array when no role assignment
is in play); selectors that don't care about roles ignore it. This keeps the
selector signature uniform — the env wrapper doesn't have to know who needs
which extras.

To add a new biased red:

1. Either parameterise ``role_biased_selector`` (host-bias only, plus an
   optional FSM action-prob matrix override) or write a fresh function
   matching the signature above.
2. Add a name → factory mapping in ``REGISTRY`` so recipes can pick it up.
3. If the new selector needs extras beyond ``host_resilience_role``, extend
   the env-state extras schema in one place rather than parameter-creep here.

The four CIA-related selectors that PR #11 hand-rolled
(``resilience_red_select_actions``, ``c_/i_/a_red_select_actions``) all
collapse into one ``role_biased_selector`` factory call differing only by
``target_roles`` and (for CIA variants) a shifted action prob matrix. Adding
"target the database tier" or "ignore user hosts" becomes one registry entry.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from jaxborg.scenarios.cc4.red_fsm import (
    ACTION_VALID_MASK,
    PROBABILITY_MATRIX,
    fsm_red_select_actions,
)
from jaxborg.scenarios.cc4.topology_roles import ROLE_AUTH, ROLE_DB, ROLE_WEB

# A RedSelector takes (state, const, host_resilience_role, red_keys) and
# returns the 6-tuple shown in the module docstring.
RedSelector = Callable[..., tuple]


def fsm_selector(state, const, host_resilience_role, red_keys):
    """Vanilla CC4 finite-state red — biases nothing, ignores roles."""
    del host_resilience_role
    return fsm_red_select_actions(state, const, red_keys)


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

    Args:
        target_roles: role ints (see ``topology_roles``) that get up-weighted.
        target_weight: multiplier for hosts whose role is in ``target_roles``.
            All other eligible hosts keep weight 1.0.
        prob_matrix:   optional FSM action-probability override (e.g. CIA-shifted
            matrix that promotes Impact/Degrade at root access). Default:
            ``PROBABILITY_MATRIX``.
        valid_mask:    paired valid-action mask. Default: ``ACTION_VALID_MASK``.

    Returns:
        A ``RedSelector`` callable. Empty ``target_roles`` is allowed and degrades
        gracefully to "uniform over eligible" — useful as a parity sanity check.
    """
    from jaxborg.scenarios.cc4.resilience_red_fsm import _red_select_actions

    pm = PROBABILITY_MATRIX if prob_matrix is None else prob_matrix
    vm = ACTION_VALID_MASK if valid_mask is None else valid_mask

    def _selector(state, const, host_resilience_role, red_keys):
        host_weights = jnp.ones_like(host_resilience_role, dtype=jnp.float32)
        for role in target_roles:
            host_weights = jnp.where(host_resilience_role == role, target_weight, host_weights)
        return _red_select_actions(state, const, host_weights, pm, vm, red_keys)

    return _selector


# ---------------------------------------------------------------------------
# Registry — name → factory(**kwargs) → RedSelector
#
# Recipes refer to selectors by name; ippo_jax / eval scripts call
# ``make_red_selector(cfg["red_agent"], **cfg)`` and don't know any specifics.

_FIXED_CIA_TARGET_WEIGHT = 10.0


def _resilience(target_weight: float = 5.0, **_) -> RedSelector:
    return role_biased_selector(
        target_roles=(ROLE_AUTH, ROLE_DB, ROLE_WEB),
        target_weight=float(target_weight),
    )


def _cia_c(**_) -> RedSelector:
    return role_biased_selector(
        target_roles=(ROLE_AUTH, ROLE_DB),
        target_weight=_FIXED_CIA_TARGET_WEIGHT,
        prob_matrix=_CIA_PROB_MATRIX,
        valid_mask=_CIA_VALID_MASK,
    )


def _cia_i(**_) -> RedSelector:
    return role_biased_selector(
        target_roles=(ROLE_AUTH, ROLE_WEB),
        target_weight=_FIXED_CIA_TARGET_WEIGHT,
        prob_matrix=_CIA_PROB_MATRIX,
        valid_mask=_CIA_VALID_MASK,
    )


def _cia_a(**_) -> RedSelector:
    return role_biased_selector(
        target_roles=(ROLE_AUTH, ROLE_DB, ROLE_WEB),
        target_weight=_FIXED_CIA_TARGET_WEIGHT,
        prob_matrix=_CIA_PROB_MATRIX,
        valid_mask=_CIA_VALID_MASK,
    )


REGISTRY: dict[str, Callable[..., RedSelector]] = {
    "fsm": lambda **_: fsm_selector,
    "finite_state": lambda **_: fsm_selector,  # alias matching CybORG agent name
    "resilience": _resilience,
    "cia_c": _cia_c,
    "c": _cia_c,  # alias matching recipe shorthand
    "cia_i": _cia_i,
    "i": _cia_i,
    "cia_a": _cia_a,
    "a": _cia_a,
}


def make_red_selector(name: str, **kwargs) -> RedSelector:
    """Build a selector by name. Unknown names raise ``ValueError``.

    Recipe-driven entry point: ``make_red_selector(cfg["red_agent"],
    target_weight=cfg["resilience_target_weight"])``.
    """
    factory = REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown red selector {name!r}. Registered: {sorted(REGISTRY)}",
        )
    return factory(**kwargs)
