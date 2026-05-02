"""Regression: vmap green ≡ sequential green in pure mode.

The vmap green path computes per-host intents in parallel, then applies
phishing in a sequential fori_loop where the source agent is re-derived
against the live carry_state. This test pins the equivalence to the
sequential `apply_green_agents` baseline across multiple seeds and
forces the load-bearing edge case: two green phishings in the same
step where the second's source choice depends on the first's session.

If this test fails, the vmap path has a parity bug — the differential
test suite likely won't catch it because differential mode bypasses
`_find_phishing_red_agent` via recorded `green_randoms[t, h, 5]`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions.green import apply_green_agents
from jaxborg.actions.green_vmap import (
    GREEN_LOCAL_WORK,
    NUM_GREEN_ACTIONS,
    PHISHING_ERROR_RATE,
    apply_green_agents_vmapped,
)
from jaxborg.constants import (
    CC4_CONFIG,
    GLOBAL_MAX_HOSTS,
    MAX_STEPS,
    NUM_GREEN_RANDOM_FIELDS,
)
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

    Pure mode (use_green_randoms=False) — the regime where the vmap path
    derives phishing source via _find_phishing_red_agent rather than
    reading the recorded source from green_randoms[t, h, 5].
    """
    state, const = _fresh_state()
    key = jax.random.PRNGKey(seed)

    state_seq = apply_green_agents(state, const, key)
    state_vmap = apply_green_agents_vmapped(state, const, key)

    diffs = _state_diff(state_seq, state_vmap)
    assert not diffs, f"seed={seed}: divergent fields = {diffs}"


def test_vmap_matches_sequential_with_two_phishings_same_step():
    """Force two phishings in the same step where the second's source agent
    selection depends on the first's newly-created red session.

    Mechanism: pre-populate `green_randoms` so that two green agents on
    different hosts in the same subnet both roll GREEN_LOCAL_WORK with
    high success and a phishing trigger. The second's `_find_phishing_red_agent`
    must see the red_session created by the first.
    """
    state, const = _fresh_state()

    # Find two active green hosts in the same subnet.
    active_greens = [int(h) for h in range(GLOBAL_MAX_HOSTS) if bool(const.green_agent_active[h])]
    pair = None
    for i, h1 in enumerate(active_greens):
        s1 = int(const.host_subnet[h1])
        for h2 in active_greens[i + 1 :]:
            if int(const.host_subnet[h2]) == s1:
                pair = (h1, h2)
                break
        if pair:
            break
    assert pair is not None, "test fixture: need two active greens in same subnet"
    host_a, host_b = pair

    # Pre-populate green_randoms: t=0 for both hosts, GREEN_LOCAL_WORK,
    # service-token=0 (some service), reliability roll=0 (always succeed),
    # phish-roll below PHISHING_ERROR_RATE so phishing triggers,
    # source agent: 0 (1+0 in field 5 means red_agent_0 in CybORG-record convention,
    # but since we want pure-mode to exercise live derivation, leave field 5 = 0
    # meaning "no precomputed source").
    randoms = np.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
    for h in (host_a, host_b):
        randoms[0, h, 0] = (GREEN_LOCAL_WORK + 0.5) / NUM_GREEN_ACTIONS  # action
        randoms[0, h, 1] = 0  # service token = first
        randoms[0, h, 2] = 0  # reliability roll = always succeed
        randoms[0, h, 3] = 0.5  # FP roll (>0.01 => no FP)
        randoms[0, h, 4] = PHISHING_ERROR_RATE * 0.1  # phish-roll < threshold => trigger
        randoms[0, h, 5] = 0  # no precomputed source => exercise live derivation
        randoms[0, h, 6] = 0.5  # access FP roll
        randoms[0, h, 7] = 1  # pid delta

    const_pure = const.replace(
        green_randoms=jnp.array(randoms),
        use_green_randoms=jnp.array(False),  # pure mode
    )

    key = jax.random.PRNGKey(99)
    state_seq = apply_green_agents(state, const_pure, key)
    state_vmap = apply_green_agents_vmapped(state, const_pure, key)

    diffs = _state_diff(state_seq, state_vmap)
    assert not diffs, f"two-phishing same-step divergence: {diffs}"
