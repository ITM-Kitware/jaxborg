"""Random sampling primitives.

Default implementations route to ``jax.random.*``.  For parity tests two swap
points exist:

* **Low-level impl swap** (:func:`set_impls`, :func:`rng_impls`) replaces the
  underlying ``uniform`` / ``randint`` / ``permutation`` calls.  Useful for
  narrow unit tests where a single sequential value stream is enough — see
  ``tests.differential.parity_rng_replay.RNGTape``.

* **Per-purpose dispatch swap** (:func:`set_purpose_impls`,
  :func:`indexed_rng_impls`) replaces the high-level *purpose* (detection,
  green, red_pid_delta, …) so each call site can be served from its own
  indexed table — needed for full-episode harness parity where CybORG and
  JAX traverse hosts/agents in different orders.  See
  ``tests.differential.parity_rng_replay.IndexedRNGTape``.

The tape implementations themselves live under ``tests/differential/`` so
production code never imports replay machinery.  Production calls go through
the default impls (:func:`jax.random.uniform` / ``randint`` / ``permutation``)
which are jit-compatible.

The per-purpose dispatchers default to calling the low-level impls, so a
swap of ``_uniform_impl`` propagates to every purpose unless that purpose
has been individually overridden.
"""

import contextlib

import jax
import jax.numpy as jnp

from jaxborg.state import SimulatorConst, SimulatorState

# --- Low-level swap points (uniform / randint / permutation) ----------------

_uniform_impl = jax.random.uniform
_randint_impl = jax.random.randint
_permutation_impl = jax.random.permutation


def set_impls(uniform=None, randint=None, permutation=None):
    """Swap one or more low-level sampling impls.  Returns the previous tuple
    so callers can restore.  Prefer the :func:`rng_impls` context manager.
    """
    global _uniform_impl, _randint_impl, _permutation_impl
    prev = (_uniform_impl, _randint_impl, _permutation_impl)
    if uniform is not None:
        _uniform_impl = uniform
    if randint is not None:
        _randint_impl = randint
    if permutation is not None:
        _permutation_impl = permutation
    return prev


def reset_impls():
    """Restore default ``jax.random.*`` low-level implementations."""
    global _uniform_impl, _randint_impl, _permutation_impl
    _uniform_impl = jax.random.uniform
    _randint_impl = jax.random.randint
    _permutation_impl = jax.random.permutation


@contextlib.contextmanager
def rng_impls(uniform=None, randint=None, permutation=None):
    """Context manager swap.  Prior impls are restored on exit even on error."""
    prev = set_impls(uniform=uniform, randint=randint, permutation=permutation)
    try:
        yield
    finally:
        global _uniform_impl, _randint_impl, _permutation_impl
        _uniform_impl, _randint_impl, _permutation_impl = prev


# --- Per-purpose dispatchers ------------------------------------------------
#
# Each high-level "purpose" has its own dispatch slot so harness-parity tests
# can serve different call sites from different tables.  Default dispatchers
# call the low-level impls, so the existing low-level swap continues to work
# when no purpose-specific override is installed.


def _default_detection(key):
    return _uniform_impl(key)


def _default_green(key, time, host_idx, field_idx, int_range):
    del time, host_idx, field_idx
    if int_range is not None:
        return _randint_impl(key, (), 0, jnp.maximum(int_range, 1))
    return _uniform_impl(key)


def _default_red_pid_delta(key, agent_id):
    del agent_id
    return _randint_impl(key, (), minval=1, maxval=10, dtype=jnp.int32)


def _default_red_privesc(key, agent_id, num_sessions):
    del agent_id
    return _randint_impl(key, (), minval=0, maxval=jnp.maximum(num_sessions, 1), dtype=jnp.int32)


def _default_red_session_check(key, agent_id, num_sessions_on_host):
    del agent_id
    return _randint_impl(key, (), minval=0, maxval=jnp.maximum(num_sessions_on_host, 1), dtype=jnp.int32)


def _default_exploit_session(key, agent_id, visible_sessions):
    del agent_id
    return _randint_impl(key, (), minval=0, maxval=jnp.maximum(visible_sessions, 1), dtype=jnp.int32)


def _default_blue_decoy_type(key, agent_id, compatibility):
    del agent_id
    perm = _permutation_impl(key, 4, independent=True)
    scores = jnp.where(compatibility, perm, jnp.int32(100))
    return jnp.argmin(scores).astype(jnp.int32)


def _default_blue_decoy_pid_delta(key, agent_id, respawn_index):
    del agent_id, respawn_index
    return _randint_impl(key, (), minval=1, maxval=10, dtype=jnp.int32)


def _default_red_policy(key, agent_id, field_idx, probs):
    """Categorical sample over ``probs``.

    Tape impls override this to return the recorded JAX index directly.
    """
    del agent_id, field_idx
    n = probs.shape[0]
    return jax.random.choice(key, n, p=probs).astype(jnp.int32)


def _default_green_dest_host(key, time, host_idx, sorted_servers, num_reachable):
    del time, host_idx
    idx = _randint_impl(key, (), 0, jnp.maximum(num_reachable, 1))
    return sorted_servers[idx]


_PURPOSE_DEFAULTS = {
    "detection": _default_detection,
    "green": _default_green,
    "green_dest_host": _default_green_dest_host,
    "red_pid_delta": _default_red_pid_delta,
    "red_privesc": _default_red_privesc,
    "red_session_check": _default_red_session_check,
    "exploit_session": _default_exploit_session,
    "blue_decoy_type": _default_blue_decoy_type,
    "blue_decoy_pid_delta": _default_blue_decoy_pid_delta,
    "red_policy": _default_red_policy,
}

_purpose_impls: dict = dict(_PURPOSE_DEFAULTS)


def set_purpose_impls(**overrides):
    """Swap one or more per-purpose dispatch impls.  Returns a dict snapshot
    of the previous impls so callers can restore.  Prefer
    :func:`indexed_rng_impls`.
    """
    prev = dict(_purpose_impls)
    for name, fn in overrides.items():
        if name not in _PURPOSE_DEFAULTS:
            raise ValueError(f"unknown RNG purpose: {name!r}; valid: {sorted(_PURPOSE_DEFAULTS)}")
        _purpose_impls[name] = fn
    return prev


def reset_purpose_impls():
    """Restore all per-purpose dispatchers to their defaults."""
    _purpose_impls.clear()
    _purpose_impls.update(_PURPOSE_DEFAULTS)


@contextlib.contextmanager
def indexed_rng_impls(**overrides):
    """Context manager: swap per-purpose dispatchers, restore on exit."""
    prev = set_purpose_impls(**overrides)
    try:
        yield
    finally:
        _purpose_impls.clear()
        _purpose_impls.update(prev)


# --- Public sampling API (call sites in production code use these) ---------


def sample_detection_random(state: SimulatorState, const: SimulatorConst, key: jax.Array):
    """Return ``(random_float, state)``.  State passes through unchanged."""
    del const
    return _purpose_impls["detection"](key), state


def sample_green_random(const: SimulatorConst, time, host_idx, field_idx, key, *, int_range=None):
    """Return a random green-agent value.

    ``int_range`` not None ⇒ int32 in ``[0, int_range)``; otherwise float32 in
    ``[0, 1)``.
    """
    del const
    return _purpose_impls["green"](key, time, host_idx, field_idx, int_range)


def sample_red_policy_choice(const: SimulatorConst, time, agent_id, field_idx, key, probs):
    """Return a JAX index sampled from ``probs`` for the red FSM.

    Production: categorical sample via ``jax.random.choice``.  Tape impls
    return the recorded JAX index directly so CybORG-vs-JAX cumsum bucket
    differences (probs and iteration order) cannot drift the replay.
    """
    del const, time
    return _purpose_impls["red_policy"](key, agent_id, field_idx, probs)


def sample_red_pid_delta(const: SimulatorConst, time, agent_id, key):
    """Return ``Host.create_pid`` delta in ``[1, 9]`` for exploit session creation."""
    del const, time
    return _purpose_impls["red_pid_delta"](key, agent_id)


def sample_red_privesc_choice(const: SimulatorConst, time, agent_id, key, num_sessions):
    """Return privesc session choice index in ``[0, num_sessions)``."""
    del const, time
    return _purpose_impls["red_privesc"](key, agent_id, num_sessions)


def sample_red_session_check_choice(const: SimulatorConst, time, agent_id, key, num_sessions_on_host):
    """Return session-check within-host slot index in ``[0, num_sessions_on_host)``."""
    del const, time
    return _purpose_impls["red_session_check"](key, agent_id, num_sessions_on_host)


def sample_exploit_session_choice(const: SimulatorConst, time, agent_id, key, visible_sessions):
    """Return exploit session choice index in ``[0, visible_sessions)``."""
    del const, time
    return _purpose_impls["exploit_session"](key, agent_id, visible_sessions)


def sample_blue_decoy_type_choice(const: SimulatorConst, time, agent_id, compatibility, key):
    """Return a compatible decoy type index in ``[0, 4)``."""
    del const, time
    return _purpose_impls["blue_decoy_type"](key, agent_id, compatibility)


def sample_blue_decoy_pid_delta(const: SimulatorConst, time, agent_id, key, respawn_index=0):
    """Return ``Host.create_pid`` delta in ``[1, 9]`` for blue decoy creation."""
    del const, time
    return _purpose_impls["blue_decoy_pid_delta"](key, agent_id, respawn_index)


def sample_green_dest_host(const: SimulatorConst, time, host_idx, key, sorted_servers, num_reachable):
    """Return the JAX host_idx of the chosen GreenAccessService destination.

    ``sorted_servers`` is the per-host pre-sorted reachable-server list and
    ``num_reachable`` is the count of valid entries.  The default impl picks
    ``sorted_servers[randint(0, num_reachable)]``; the tape impl returns the
    recorded dest host directly so CybORG's destination choice is honoured.
    """
    del const
    return _purpose_impls["green_dest_host"](key, time, host_idx, sorted_servers, num_reachable)
