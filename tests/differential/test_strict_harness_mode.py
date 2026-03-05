import inspect

import jax.numpy as jnp

import tests.differential.harness as harness_mod
from jaxborg.actions.encoding import BLUE_SLEEP, RED_SLEEP, encode_red_action
from jaxborg.actions.pending_source import PENDING_SOURCE_KIND_NONE
from jaxborg.constants import NUM_BLUE_AGENTS
from tests.differential.fuzzer import run_differential_fuzz
from tests.differential.harness import CC4DifferentialHarness


def test_harness_constructor_is_strict_only():
    params = inspect.signature(CC4DifferentialHarness.__init__).parameters
    assert "strict_differential" not in params


def test_fuzzer_api_is_strict_only():
    params = inspect.signature(run_differential_fuzz).parameters
    assert "strict_differential" not in params


def test_fsm_scan_without_bound_anchor_keeps_pending_source_unbound():
    harness = CC4DifferentialHarness(seed=42, max_steps=20, sync_green_rng=True)
    harness.reset()

    red_agent_id = 0
    target_host = int(harness.jax_const.red_start_hosts[red_agent_id])
    queued_scan = encode_red_action("StealthServiceDiscovery", target_host, red_agent_id)

    harness.jax_state = harness.jax_state.replace(
        red_sessions=harness.jax_state.red_sessions.at[red_agent_id].set(False),
        red_session_count=harness.jax_state.red_session_count.at[red_agent_id].set(0),
        red_session_is_abstract=harness.jax_state.red_session_is_abstract.at[red_agent_id].set(False),
        red_scan_anchor_host=harness.jax_state.red_scan_anchor_host.at[red_agent_id].set(-1),
        red_pending_ticks=harness.jax_state.red_pending_ticks.at[red_agent_id].set(0),
        red_pending_source_kind=harness.jax_state.red_pending_source_kind.at[red_agent_id].set(
            PENDING_SOURCE_KIND_NONE
        ),
        red_pending_source_host=harness.jax_state.red_pending_source_host.at[red_agent_id].set(-1),
    )

    original_selector = harness_mod._jit_fsm_red_get_action_and_info
    original_resolve = harness._resolve_red_action

    def _forced_fsm_action(_state, _const, agent_id, _key):
        if int(agent_id) == red_agent_id:
            return (
                jnp.int32(queued_scan),
                jnp.int32(target_host),
                jnp.int32(0),
                jnp.bool_(True),
            )
        return (
            jnp.int32(RED_SLEEP),
            jnp.int32(0),
            jnp.int32(0),
            jnp.bool_(False),
        )

    try:
        harness_mod._jit_fsm_red_get_action_and_info = _forced_fsm_action
        harness._resolve_red_action = lambda _controller, _agent_idx, proposed_action: proposed_action
        harness.full_step(blue_actions={b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)})
    finally:
        harness_mod._jit_fsm_red_get_action_and_info = original_selector
        harness._resolve_red_action = original_resolve

    assert int(harness.jax_state.red_pending_ticks[red_agent_id]) > 0
    assert int(harness.jax_state.red_pending_source_kind[red_agent_id]) == int(PENDING_SOURCE_KIND_NONE)
