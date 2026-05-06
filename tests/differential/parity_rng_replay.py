"""Host-side tape RNG for parity tests.

This module is **test/debug-only** and lives under ``tests/differential`` so
production code never imports replay machinery.  The two tape primitives are:

* :class:`RNGTape` — single FIFO queue per JAX dtype, drop-in replacement for
  ``jax.random.uniform`` / ``randint`` / ``permutation``.  Used by narrow
  subsystem unit tests where one sequential value stream is enough.

* :class:`IndexedRNGTape` — per-purpose indexed tables keyed by the call-site
  context (agent_id / host_idx / field_idx).  Used by the differential
  harness because CybORG and JAX traverse hosts/agents in different orders,
  so a FIFO queue can't replay multi-step parity.  All impls use
  :func:`jax.experimental.io_callback` (the indexed lookups use
  ``ordered=False`` for relaxed ordering; the FIFO detection queue uses
  ``ordered=True``).  ``io_callback`` is required (not ``pure_callback``)
  because the host fns read mutable per-step buffers — pure_callback is
  documented to elide/cache calls based on input arguments, which would
  return stale values across steps.  This makes the tape jit-compatible
  so the harness can run ``apply_all_actions`` under ``@jax.jit`` and
  keep parity-seed wall-clock reasonable.

Strict-mode miss handling (the default) raises a :class:`RuntimeError` when
a dispatch hits an unpopulated table entry.  This surfaces "we forgot to
replay this draw" as a hard test failure instead of silently substituting a
sentinel.  Pass ``strict=False`` to opt into the legacy lenient behaviour
(record the miss, return a sentinel, let the test continue).

Typical usage::

    from jaxborg.actions.rng import rng_impls
    from tests.differential.parity_rng_replay import RNGTape

    tape = RNGTape()
    tape.push_uniform(0.5)
    tape.push_randint(2)
    with rng_impls(uniform=tape.uniform, randint=tape.randint):
        ...
    assert tape.consumed == 2
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

from jaxborg.constants import (
    NUM_BLUE_AGENTS,
    NUM_DECOY_TYPES,
    NUM_RED_AGENTS,
    NUM_RED_POLICY_RANDOM_FIELDS,
)


class RNGTape:
    """Host-side replay tape for parity tests.

    Use :meth:`push_uniform`, :meth:`push_randint`, :meth:`push_permutation`
    to enqueue values.  Bind the bound methods :meth:`uniform`, :meth:`randint`,
    :meth:`permutation` into :func:`jaxborg.actions.rng.rng_impls` to redirect
    sampling.

    Notes:
        * In eager mode the tape pops directly when each method is called.
        * In ``"callback"`` mode the actual pop happens host-side via
          :func:`io_callback`; tracing succeeds but source-order pops are only
          guaranteed with ``ordered=True`` (which is set automatically).
    """

    def __init__(self, mode: str = "eager"):
        if mode not in ("eager", "callback"):
            raise ValueError(f"mode must be 'eager' or 'callback'; got {mode!r}")
        self._mode = mode
        self._uniform_q: deque[float] = deque()
        self._randint_q: deque[int] = deque()
        self._perm_q: deque[np.ndarray] = deque()
        self._consumed = 0

    @property
    def consumed(self) -> int:
        """Total number of pops across all queues."""
        return self._consumed

    @property
    def remaining(self) -> int:
        return len(self._uniform_q) + len(self._randint_q) + len(self._perm_q)

    def push_uniform(self, value: float) -> None:
        self._uniform_q.append(float(value))

    def push_uniforms(self, values: Iterable[float]) -> None:
        for v in values:
            self.push_uniform(v)

    def push_randint(self, value: int) -> None:
        self._randint_q.append(int(value))

    def push_randints(self, values: Iterable[int]) -> None:
        for v in values:
            self.push_randint(v)

    def push_permutation(self, value: Iterable[int]) -> None:
        self._perm_q.append(np.asarray(list(value), dtype=np.int32))

    def _pop_uniform(self) -> float:
        if not self._uniform_q:
            raise RuntimeError("RNGTape: uniform queue exhausted")
        self._consumed += 1
        return self._uniform_q.popleft()

    def _pop_randint(self) -> int:
        if not self._randint_q:
            raise RuntimeError("RNGTape: randint queue exhausted")
        self._consumed += 1
        return self._randint_q.popleft()

    def _pop_permutation(self, n: int) -> np.ndarray:
        if not self._perm_q:
            raise RuntimeError("RNGTape: permutation queue exhausted")
        self._consumed += 1
        v = self._perm_q.popleft()
        if v.shape != (n,):
            raise RuntimeError(f"RNGTape: permutation shape mismatch: queued {v.shape}, expected ({n},)")
        return v

    def uniform(self, key, shape=(), dtype=jnp.float32, minval=0.0, maxval=1.0):
        del key
        if self._mode == "eager":
            v = self._pop_uniform()
            v = float(minval) + v * (float(maxval) - float(minval))
            return jnp.asarray(v, dtype=dtype).reshape(shape)
        spec = jax.ShapeDtypeStruct(shape, dtype)

        def host():
            v = self._pop_uniform()
            v = float(minval) + v * (float(maxval) - float(minval))
            return np.asarray(v, dtype=dtype).reshape(shape)

        return io_callback(host, spec, ordered=True)

    def randint(self, key, shape, minval=0, maxval=None, dtype=jnp.int32):
        del key
        if self._mode == "eager":
            v = self._pop_randint()
            return jnp.asarray(v, dtype=dtype).reshape(shape)
        spec = jax.ShapeDtypeStruct(shape, dtype)

        def host():
            v = self._pop_randint()
            return np.asarray(v, dtype=dtype).reshape(shape)

        return io_callback(host, spec, ordered=True)

    def permutation(self, key, x, axis=0, independent=False):
        del key, axis, independent
        n = int(x) if isinstance(x, int) else int(x.shape[0])
        if self._mode == "eager":
            v = self._pop_permutation(n)
            return jnp.asarray(v, dtype=jnp.int32)
        spec = jax.ShapeDtypeStruct((n,), jnp.int32)

        def host():
            return self._pop_permutation(n)

        return io_callback(host, spec, ordered=True)


# Sentinel float for missing entries in lenient mode.  Nominal "uniform" so
# downstream math stays in-distribution even when a draw is unreplayed.
_LENIENT_FLOAT = 0.5
_LENIENT_INT = 0


class IndexedRNGTape:
    """Per-purpose indexed-lookup tape for full-episode harness parity.

    Unlike :class:`RNGTape` (single FIFO queue per JAX dtype), this tape
    serves each :mod:`jaxborg.actions.rng` *purpose* (detection, green,
    red_pid_delta, …) from its own fixed-shape numpy table indexed by the
    call-site context (agent_id, host_idx, …).  This decouples JAX's
    source-order traversal from CybORG's iteration order, which is
    necessary for multi-step parity replay.

    All dispatch impls go through :func:`jax.experimental.io_callback` so
    the tape works under ``@jax.jit`` and under green ``vmap``.  Indexed
    lookups use ``ordered=False`` (relaxed ordering — the result depends
    only on the index, not on call order); the FIFO detection queue uses
    ``ordered=True``.  Per-step writes to the underlying numpy buffers are
    visible without retracing because the host callback closures reference
    ``self`` and read the buffers at execution time.

    .. note::
       ``jax.pure_callback`` is *not* safe here — JAX is allowed to elide
       or cache pure callbacks based on input arguments, which would freeze
       the tape's first-step values across all subsequent steps.

    Strict mode (the default) raises :class:`RuntimeError` from inside the
    callback when a dispatch hits an unpopulated entry — exactly the failure
    mode the harness wants when CybORG drew a value JAX didn't replay.  Pass
    ``strict=False`` to opt into the legacy behaviour: record the miss in
    :attr:`misses` and return a benign sentinel (``0.5`` for floats, ``0``
    or ``1`` for ints).

    Typical usage (inside the differential harness)::

        tape = IndexedRNGTape(strict=False)  # harness allows under-fill
        tape.set_detection_queue([0.5, 0.99])
        tape.set_red_pid_delta(agent_id=0, value=3)
        # … more table writes …
        with indexed_rng_impls(**tape.as_overrides()):
            new_state = apply_all_actions(state, const, ...)
        tape.clear()  # ready for next step
    """

    def __init__(self, *, strict: bool = True):
        self._strict = strict
        # FIFO queue (order matters; replays through io_callback ordered=True).
        self._detection_q: deque[float] = deque()
        # Indexed tables.  Numpy buffers of fixed shape; mutate per step.
        # Parallel _set_* boolean masks track which entries the harness has
        # populated, so misses can be detected even when 0 / 0.0 is a valid
        # tape value.
        self._red_pid_delta_arr = np.zeros((NUM_RED_AGENTS,), dtype=np.int32)
        self._red_pid_delta_set = np.zeros((NUM_RED_AGENTS,), dtype=np.bool_)
        self._red_privesc_arr = np.zeros((NUM_RED_AGENTS,), dtype=np.int32)
        self._red_privesc_set = np.zeros((NUM_RED_AGENTS,), dtype=np.bool_)
        self._red_session_check_arr = np.zeros((NUM_RED_AGENTS,), dtype=np.int32)
        self._red_session_check_set = np.zeros((NUM_RED_AGENTS,), dtype=np.bool_)
        self._exploit_session_arr = np.zeros((NUM_RED_AGENTS,), dtype=np.int32)
        self._exploit_session_set = np.zeros((NUM_RED_AGENTS,), dtype=np.bool_)
        self._blue_decoy_type_arr = np.zeros((NUM_BLUE_AGENTS,), dtype=np.int32)
        self._blue_decoy_type_set = np.zeros((NUM_BLUE_AGENTS,), dtype=np.bool_)
        self._blue_decoy_pid_delta_arr = np.zeros((NUM_BLUE_AGENTS, NUM_DECOY_TYPES), dtype=np.int32)
        self._blue_decoy_pid_delta_set = np.zeros((NUM_BLUE_AGENTS, NUM_DECOY_TYPES), dtype=np.bool_)
        self._red_policy_arr = np.zeros((NUM_RED_AGENTS, NUM_RED_POLICY_RANDOM_FIELDS), dtype=np.int32)
        self._red_policy_set = np.zeros((NUM_RED_AGENTS, NUM_RED_POLICY_RANDOM_FIELDS), dtype=np.bool_)
        # Green (consumed under vmap; tables stay numpy until lookup time).
        self._green_table: np.ndarray | None = None  # (GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS) float32
        self._green_int_range: np.ndarray | None = None  # same shape int32; -1 sentinel = no override
        self._misses: list[str] = []
        self._used = 0

    # ---- Population API ---------------------------------------------------

    def clear(self) -> None:
        self._detection_q.clear()
        self._red_pid_delta_arr.fill(0)
        self._red_pid_delta_set.fill(False)
        self._red_privesc_arr.fill(0)
        self._red_privesc_set.fill(False)
        self._red_session_check_arr.fill(0)
        self._red_session_check_set.fill(False)
        self._exploit_session_arr.fill(0)
        self._exploit_session_set.fill(False)
        self._blue_decoy_type_arr.fill(0)
        self._blue_decoy_type_set.fill(False)
        self._blue_decoy_pid_delta_arr.fill(0)
        self._blue_decoy_pid_delta_set.fill(False)
        self._red_policy_arr.fill(0)
        self._red_policy_set.fill(False)
        self._green_table = None
        self._green_int_range = None
        self._misses.clear()
        self._used = 0

    def set_detection_queue(self, values: Iterable[float]) -> None:
        self._detection_q.clear()
        self._detection_q.extend(float(v) for v in values)

    def set_red_pid_delta(self, agent_id: int, value: int) -> None:
        self._red_pid_delta_arr[int(agent_id)] = int(value)
        self._red_pid_delta_set[int(agent_id)] = True

    def set_red_privesc(self, agent_id: int, value: int) -> None:
        self._red_privesc_arr[int(agent_id)] = int(value)
        self._red_privesc_set[int(agent_id)] = True

    def set_red_session_check(self, agent_id: int, value: int) -> None:
        self._red_session_check_arr[int(agent_id)] = int(value)
        self._red_session_check_set[int(agent_id)] = True

    def set_exploit_session(self, agent_id: int, value: int) -> None:
        self._exploit_session_arr[int(agent_id)] = int(value)
        self._exploit_session_set[int(agent_id)] = True

    def set_blue_decoy_type(self, agent_id: int, value: int) -> None:
        self._blue_decoy_type_arr[int(agent_id)] = int(value)
        self._blue_decoy_type_set[int(agent_id)] = True

    def set_blue_decoy_pid_delta(self, agent_id: int, respawn_index: int, value: int) -> None:
        self._blue_decoy_pid_delta_arr[int(agent_id), int(respawn_index)] = int(value)
        self._blue_decoy_pid_delta_set[int(agent_id), int(respawn_index)] = True

    def set_red_policy(self, agent_id: int, field_idx: int, value: int) -> None:
        """Record the JAX index chosen by CybORG for ``(agent_id, field_idx)``.

        Field 0 = host_idx, field 1 = fsm_action_id, field 2 = subnet_idx.
        """
        self._red_policy_arr[int(agent_id), int(field_idx)] = int(value)
        self._red_policy_set[int(agent_id), int(field_idx)] = True

    def set_green_uniform(self, table: np.ndarray) -> None:
        """``table`` shape ``(num_hosts, num_fields)`` of float32 uniforms."""
        self._green_table = np.asarray(table, dtype=np.float32)

    def set_green_int_range(self, table: np.ndarray) -> None:
        """``table`` shape ``(num_hosts, num_fields)`` of int32 values to return
        when ``int_range`` is supplied."""
        self._green_int_range = np.asarray(table, dtype=np.int32)

    # ---- Stats ------------------------------------------------------------

    @property
    def used(self) -> int:
        """Number of dispatch calls that pulled from the tape (vs missed)."""
        return self._used

    @property
    def misses(self) -> tuple[str, ...]:
        """Tuple of distinct miss descriptors (purpose + context)."""
        return tuple(self._misses)

    @property
    def strict(self) -> bool:
        return self._strict

    def _record_miss(self, descriptor: str) -> None:
        if descriptor not in self._misses:
            self._misses.append(descriptor)

    def _on_miss_int(self, descriptor: str, sentinel: int) -> int:
        if self._strict:
            raise RuntimeError(f"IndexedRNGTape miss in strict mode: {descriptor}")
        self._record_miss(descriptor)
        return sentinel

    def _on_miss_float(self, descriptor: str, sentinel: float) -> float:
        if self._strict:
            raise RuntimeError(f"IndexedRNGTape miss in strict mode: {descriptor}")
        self._record_miss(descriptor)
        return sentinel

    # ---- Indexed-lookup helper -------------------------------------------

    def _scalar_int_callback(self, agent_id, *, descriptor_prefix: str, arr_attr: str, set_attr: str, sentinel: int):
        """Dispatch a per-(agent_id) int32 lookup via :func:`io_callback`.

        The host fn closes over ``self`` and the attribute *names*, so each
        invocation reads the current numpy buffer at execution time.
        ``ordered=False`` because the result depends only on ``agent_id``,
        not on the relative order of dispatch calls.
        """
        spec = jax.ShapeDtypeStruct((), jnp.int32)

        def host(a):
            idx = int(a)
            arr = getattr(self, arr_attr)
            mask = getattr(self, set_attr)
            if not bool(mask[idx]):
                self._used += 0  # no-op; clarifies that misses don't count
                return np.int32(self._on_miss_int(f"{descriptor_prefix}: agent={idx}", sentinel))
            self._used += 1
            return np.int32(arr[idx])

        return io_callback(host, spec, agent_id, ordered=False)

    # ---- Per-purpose dispatch impls --------------------------------------

    def _detection(self, key):
        """FIFO uniform pop.  Order-sensitive → io_callback ordered=True."""
        del key
        spec = jax.ShapeDtypeStruct((), jnp.float32)

        def host():
            if not self._detection_q:
                return np.float32(self._on_miss_float("detection: queue exhausted", _LENIENT_FLOAT))
            self._used += 1
            return np.float32(self._detection_q.popleft())

        return io_callback(host, spec, ordered=True)

    def _green_impl(self, key, time, host_idx, field_idx, int_range):
        """Per-(host_idx, field_idx) lookup; consumed under vmap.

        Reads :attr:`_green_table` (float32 uniforms) and
        :attr:`_green_int_range` (int32 overrides; ``-1`` sentinel = none).
        Routes through :func:`io_callback` (``ordered=False``) so jit +
        vmap both work and per-step buffer mutations are visible.

        ``int_range`` semantics:
          * ``None`` → return ``table[h, f]`` as float32.
          * ``not None`` → prefer ``int_range_table[h, f]`` if ≥ 0,
            else ``floor(table[h, f] * int_range)`` (field 1 special-cases
            to ``0`` because the recorder stored the raw chosen-token id, not
            a uniform — see harness :meth:`_build_green_int_range_table`).
        """
        del key, time

        if int_range is None:
            spec = jax.ShapeDtypeStruct((), jnp.float32)

            def host_float(h, f):
                if self._green_table is None:
                    return np.float32(
                        self._on_miss_float(f"green: host={int(h)} field={int(f)} (no table)", _LENIENT_FLOAT)
                    )
                self._used += 1
                return np.float32(self._green_table[int(h), int(f)])

            return io_callback(host_float, spec, host_idx, field_idx, ordered=False)

        spec = jax.ShapeDtypeStruct((), jnp.int32)
        int_range_static = int(int_range) if isinstance(int_range, int) else None

        def host_int(h, f, ir):
            hi = int(h)
            fi = int(f)
            ri = int(ir)
            if self._green_int_range is not None:
                ov = int(self._green_int_range[hi, fi])
                if ov >= 0:
                    self._used += 1
                    return np.int32(ov)
            if self._green_table is None:
                return np.int32(
                    self._on_miss_int(f"green: host={hi} field={fi} int_range={ri} (no table)", _LENIENT_INT)
                )
            self._used += 1
            if fi == 1:
                # Field 1 stores raw chosen-token id, not a uniform — float
                # decode would land on the sorted_tokens sentinel.  See
                # harness commentary.
                return np.int32(0)
            v = float(self._green_table[hi, fi])
            return np.int32(int(np.floor(v * ri)))

        ir_arg = int_range if int_range_static is None else jnp.int32(int_range_static)
        return io_callback(host_int, spec, host_idx, field_idx, ir_arg, ordered=False)

    def _red_pid_delta_impl(self, key, agent_id):
        del key
        return self._scalar_int_callback(
            agent_id,
            descriptor_prefix="red_pid_delta",
            arr_attr="_red_pid_delta_arr",
            set_attr="_red_pid_delta_set",
            sentinel=1,
        )

    def _red_privesc_impl(self, key, agent_id, num_sessions):
        del key, num_sessions
        return self._scalar_int_callback(
            agent_id,
            descriptor_prefix="red_privesc",
            arr_attr="_red_privesc_arr",
            set_attr="_red_privesc_set",
            sentinel=0,
        )

    def _red_session_check_impl(self, key, agent_id, num_sessions_on_host):
        del key, num_sessions_on_host
        return self._scalar_int_callback(
            agent_id,
            descriptor_prefix="red_session_check",
            arr_attr="_red_session_check_arr",
            set_attr="_red_session_check_set",
            sentinel=0,
        )

    def _exploit_session_impl(self, key, agent_id, visible_sessions):
        del key, visible_sessions
        return self._scalar_int_callback(
            agent_id,
            descriptor_prefix="exploit_session",
            arr_attr="_exploit_session_arr",
            set_attr="_exploit_session_set",
            sentinel=0,
        )

    def _blue_decoy_type_impl(self, key, agent_id, compatibility):
        """Return the recorded decoy type for ``agent_id``.

        Pure tape lookup — compatibility resolution lives at the call site in
        ``apply_blue_decoy``.  In lenient mode a missing slot returns sentinel
        ``-1`` so the caller's compat-fallback path picks the lowest True
        index; strict mode raises.
        """
        del key, compatibility
        return self._scalar_int_callback(
            agent_id,
            descriptor_prefix="blue_decoy_type",
            arr_attr="_blue_decoy_type_arr",
            set_attr="_blue_decoy_type_set",
            sentinel=-1,
        )

    def _blue_decoy_pid_delta_impl(self, key, agent_id, respawn_index):
        del key
        spec = jax.ShapeDtypeStruct((), jnp.int32)

        def host(a, r):
            ai = int(a)
            ri = int(r)
            if not bool(self._blue_decoy_pid_delta_set[ai, ri]):
                return np.int32(self._on_miss_int(f"blue_decoy_pid_delta: agent={ai} respawn={ri}", 1))
            self._used += 1
            return np.int32(self._blue_decoy_pid_delta_arr[ai, ri])

        return io_callback(host, spec, agent_id, respawn_index, ordered=False)

    def _green_dest_host_impl(self, key, time, host_idx, sorted_servers, num_reachable):
        """Return the recorded GreenAccessService dest host directly.

        Reads ``_green_table[host_idx, 5]``.  If the table is missing we
        fall back to the default ``sorted_servers[randint(0, num_reachable)]``
        path so unrecorded hosts behave like normal generative play — using
        the *real* ``num_reachable`` (not ``sorted_servers.shape[0]``, which
        includes padding sentinels).
        """
        if self._green_table is None:
            from jaxborg.actions.rng import _default_green_dest_host

            return _default_green_dest_host(key, time, host_idx, sorted_servers, num_reachable)

        spec = jax.ShapeDtypeStruct((), jnp.int32)

        def host(h):
            self._used += 1
            return np.int32(self._green_table[int(h), 5])

        return io_callback(host, spec, host_idx, ordered=False)

    def _red_policy_impl(self, key, agent_id, field_idx, probs):
        """Return the recorded JAX index for the red FSM choice.

        Production callers pass ``probs`` so the default impl can categorical-
        sample; the tape ignores ``probs`` because the recorder already mapped
        CybORG's choice into the JAX index space at recording time.  Using the
        recorded index sidesteps any cumsum-bucket alignment between CybORG's
        iteration order and JAX's GLOBAL_MAX_HOSTS / NUM_SUBNETS index order.
        """
        del key, probs
        spec = jax.ShapeDtypeStruct((), jnp.int32)

        def host(a, f):
            ai = int(a)
            fi = int(f)
            if not bool(self._red_policy_set[ai, fi]):
                return np.int32(self._on_miss_int(f"red_policy: agent={ai} field={fi}", _LENIENT_INT))
            self._used += 1
            return np.int32(self._red_policy_arr[ai, fi])

        return io_callback(host, spec, agent_id, field_idx, ordered=False)

    def as_overrides(self) -> dict:
        """Dict suitable for ``indexed_rng_impls(**tape.as_overrides())``."""
        return {
            "detection": self._detection,
            "green": self._green_impl,
            "green_dest_host": self._green_dest_host_impl,
            "red_pid_delta": self._red_pid_delta_impl,
            "red_privesc": self._red_privesc_impl,
            "red_session_check": self._red_session_check_impl,
            "exploit_session": self._exploit_session_impl,
            "blue_decoy_type": self._blue_decoy_type_impl,
            "blue_decoy_pid_delta": self._blue_decoy_pid_delta_impl,
            "red_policy": self._red_policy_impl,
        }
