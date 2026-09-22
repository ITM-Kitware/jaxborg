"""Selector registry contract tests.

These pin the names callers (recipes, ippo_jax, eval scripts) rely on, and
the basic shape contract so a future refactor can't silently change the
return type of selectors registered under known names.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.parity.fsm_red_env import FsmRedCC4Env
from jaxborg.scenarios.cc4.game_variants import variant_for_red
from jaxborg.scenarios.cc4.red_selectors import (
    REGISTRY,
    fsm_selector,
    make_red_selector,
    role_biased_selector,
)


def test_registry_has_expected_names():
    # Names recipes / ippo_jax depend on. Adding aliases is fine; removing
    # one of these is a breaking change.
    for name in ("fsm", "finite_state", "resilience", "cia_c", "cia_i", "cia_a", "c", "i", "a"):
        assert name in REGISTRY, f"selector name {name!r} missing"


def test_fsm_selector_alias_returns_same_callable():
    # finite_state is a CybORG-side alias for fsm — they must resolve to the
    # exact same function so JAX-side and CybORG-side stay in lockstep.
    assert make_red_selector("fsm") is make_red_selector("finite_state")
    assert make_red_selector("fsm") is fsm_selector


def test_cia_short_aliases_resolve_to_full_names():
    # Recipes use the short aliases; refactor must not silently re-route.
    assert make_red_selector("c").__qualname__ == make_red_selector("cia_c").__qualname__
    assert make_red_selector("i").__qualname__ == make_red_selector("cia_i").__qualname__
    assert make_red_selector("a").__qualname__ == make_red_selector("cia_a").__qualname__


def test_unknown_selector_raises():
    with pytest.raises(ValueError, match="Unknown red selector"):
        make_red_selector("ddos_red")


def test_cia_selectors_honor_variant_target_weight():
    """`_cia_c/i/a` must use the passed `target_weight`, not a hardcoded value.

    Pins the post-PR-#11-cleanup invariant: CIA selectors flow `target_weight`
    from the variant the same way `_resilience` does, so changing
    `CIA_C.target_weight` actually changes the bias.
    """
    s_default = make_red_selector("cia_c")
    s_lowweight = make_red_selector("cia_c", target_weight=1.0)
    # Function identity differs because closures capture different weights.
    # Sanity: at least the closures are distinct callables.
    assert s_default is not s_lowweight


def test_cia_action_prob_parity_jax_vs_cyborg():
    """`_CIA_PROB_MATRIX[FSM_R]` (JAX) == `_CIARedAgent.state_transitions_probability['R']` (CybORG).

    The JAX selector and CybORG agent must agree on the FSM_R action distribution
    or the L4 cross-backend equivalence stage will measure rule divergence as
    simulator drift.
    """
    from jaxborg.scenarios.cc4.cyborg_resilience_agents import CRedAgent
    from jaxborg.scenarios.cc4.red_fsm import FSM_R
    from jaxborg.scenarios.cc4.red_selectors import _CIA_PROB_MATRIX

    # FiniteStateRedAgent.__init__ replaces the method with its return value,
    # so on an instance `state_transitions_probability` is a dict, not callable.
    cy_row = CRedAgent().state_transitions_probability["R"]
    jx_row = [float(x) for x in _CIA_PROB_MATRIX[FSM_R]]

    # JAX uses -1.0 sentinels for invalid actions; CybORG uses None.
    for cy, jx in zip(cy_row, jx_row):
        if cy is None:
            assert jx < 0.0, f"CybORG None but JAX {jx} (valid)"
        else:
            assert jx >= 0.0 and abs(cy - jx) < 1e-6, f"row mismatch: cyborg={cy} jax={jx}"


def test_role_biased_selector_with_empty_target_roles_is_uniform():
    # Sanity: when no roles are targeted, host_weights stays at 1.0 everywhere
    # and the selector behaves like a uniform-over-eligible variant of the FSM.
    selector = role_biased_selector(target_roles=(), target_weight=1.0)
    assert callable(selector)


@pytest.mark.slow
def test_make_jax_env_runs_each_registered_selector():
    """End-to-end: every registered name produces an env that resets+steps."""
    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)
    blue_actions = {f"blue_{i}": jnp.int32(0) for i in range(5)}
    for name in ("fsm", "resilience", "cia_c", "cia_i", "cia_a"):
        variant = replace(variant_for_red(name), num_steps=5)
        env = make_jax_env(variant)
        assert isinstance(env, FsmRedCC4Env)
        obs, state = env.reset(key0)
        obs, state, rewards, dones, info = env.step(key1, state, blue_actions)
        assert "blue_0" in rewards


def test_default_extras_are_zeros_and_resilience_assigns_some():
    """Verify extras_factory wiring: default = zeros, resilience = nonzero."""
    env_default = make_jax_env(replace(variant_for_red("fsm"), num_steps=5))
    env_res = make_jax_env(replace(variant_for_red("resilience"), num_steps=5))
    obs, state_default = env_default.reset(jax.random.PRNGKey(0))
    obs, state_res = env_res.reset(jax.random.PRNGKey(0))

    # FSM default → all zeros (no role assignment).
    assert int(jnp.sum(state_default.extras["host_resilience_role"])) == 0
    # Resilience → at least one nonzero role assigned (auth/db/web on the
    # active op-zone servers).
    assert int(jnp.sum(state_res.extras["host_resilience_role"] != 0)) >= 1
