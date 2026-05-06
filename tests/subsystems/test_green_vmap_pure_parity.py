"""Regression: vmap green ≡ sequential green in pure mode.

The vmap green path computes per-host intents in parallel, then applies
phishing in a sequential fori_loop where the source agent is re-derived
against the live carry_state. This test pins the equivalence to the
sequential `apply_green_agents` baseline across multiple seeds for the pure
generative path.

If this test fails, the vmap path has a parity bug — the differential
test suite may not catch it when CybORG replay tables provide the green
destination/source choices directly through `IndexedRNGTape`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.green import apply_green_agents
from jaxborg.actions.green_vmap import apply_green_agents_vmapped
from jaxborg.constants import CC4_CONFIG
from jaxborg.env import _init_red_state
from jaxborg.scenarios.cc4.topology import build_topology
from jaxborg.state import create_initial_state


def _fresh_state():
    const = build_topology(jax.random.PRNGKey(42), num_steps=500)
    state = create_initial_state(CC4_CONFIG)
    state = state.replace(
        host_services=jnp.array(const.initial_services),
        host_max_pid=const.host_initial_max_pid,
    )
    state = _init_red_state(const, state)
    return state, const


def _state_diff(seq, vmap):
    """Return list of (field_name, locator) tuples that differ."""
    diffs = []
    for field in seq.__dataclass_fields__:
        a = np.asarray(getattr(seq, field))
        b = np.asarray(getattr(vmap, field))
        if a.shape != b.shape or not np.array_equal(a, b):
            diffs.append(field)
    return diffs


@pytest.mark.parametrize("seed", list(range(8)))
def test_vmap_matches_sequential_pure_mode(seed):
    """For each RNG seed, sequential and vmap green produce identical state.

    Pure mode — the regime where the vmap path derives the phishing source
    from live state via _find_phishing_red_agent rather than replaying a
    CybORG-recorded source through IndexedRNGTape.
    """
    state, const = _fresh_state()
    key = jax.random.PRNGKey(seed)

    state_seq = apply_green_agents(state, const, key)
    state_vmap = apply_green_agents_vmapped(state, const, key)

    diffs = _state_diff(state_seq, state_vmap)
    assert not diffs, f"seed={seed}: divergent fields = {diffs}"
