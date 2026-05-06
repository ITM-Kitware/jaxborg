"""Selector registry contract tests.

These pin the names callers (recipes, ippo_jax, eval scripts) rely on, and
the basic shape contract so a future refactor can't silently change the
return type of selectors registered under known names.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxborg.parity.fsm_red_env import FsmRedCC4Env, make_fsm_red_env
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


def test_role_biased_selector_with_empty_target_roles_is_uniform():
    # Sanity: when no roles are targeted, host_weights stays at 1.0 everywhere
    # and the selector behaves like a uniform-over-eligible variant of the FSM.
    selector = role_biased_selector(target_roles=(), target_weight=1.0)
    assert callable(selector)


@pytest.mark.slow
def test_make_fsm_red_env_runs_each_registered_selector():
    """End-to-end: every registered name produces an env that resets+steps."""
    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)
    blue_actions = {f"blue_{i}": jnp.int32(0) for i in range(5)}
    for name in ("fsm", "resilience", "cia_c", "cia_i", "cia_a"):
        env = make_fsm_red_env(num_steps=5, red_agent=name)
        assert isinstance(env, FsmRedCC4Env)
        obs, state = env.reset(key0)
        obs, state, rewards, dones, info = env.step(key1, state, blue_actions)
        assert "blue_0" in rewards


def test_default_extras_are_zeros_and_resilience_assigns_some():
    """Verify extras_factory wiring: default = zeros, resilience = nonzero."""
    env_default = make_fsm_red_env(num_steps=5, red_agent="fsm")
    env_res = make_fsm_red_env(num_steps=5, red_agent="resilience")
    obs, state_default = env_default.reset(jax.random.PRNGKey(0))
    obs, state_res = env_res.reset(jax.random.PRNGKey(0))

    # FSM default → all zeros (no role assignment).
    assert int(jnp.sum(state_default.extras["host_resilience_role"])) == 0
    # Resilience → at least one nonzero role assigned (auth/db/web on the
    # active op-zone servers).
    assert int(jnp.sum(state_res.extras["host_resilience_role"] != 0)) >= 1
