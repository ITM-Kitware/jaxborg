"""Host-side tape RNG for parity tests.

A :class:`RNGTape` holds a deque of pre-recorded values.  Its ``uniform`` /
``randint`` / ``permutation`` methods are drop-in replacements for the
matching ``jax.random.*`` functions: they ignore the ``key`` argument and pop
the next value from the tape.

Two execution modes are supported:

* **Eager mode** (default — recommended for parity tests).  The methods
  return ``jnp`` arrays directly.  Run inside ``with jax.disable_jit():`` so
  the production code paths invoke the methods at every call.
* **JIT mode** (``ordered=True`` :func:`jax.experimental.io_callback`).  The
  tape is read via host callbacks so JIT-compiled code stays jittable.  Set
  ``mode="callback"`` on the tape; callbacks preserve source-order under JIT.

Typical usage::

    from jaxborg.actions.rng import rng_impls
    from jaxborg.actions.rng_tape import RNGTape

    tape = RNGTape()
    tape.push_uniform(0.5)            # next jax.random.uniform → 0.5
    tape.push_randint(2)              # next jax.random.randint → 2
    with rng_impls(uniform=tape.uniform, randint=tape.randint), jax.disable_jit():
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


class IndexedRNGTape:
    """Per-purpose indexed-lookup tape for full-episode harness parity.

    Unlike :class:`RNGTape` (single FIFO queue per JAX dtype), this tape
    serves each :mod:`jaxborg.actions.rng` *purpose* (detection, green,
    red_pid_delta, …) from its own table indexed by the call-site context
    (agent_id, host_idx, …).  This decouples JAX's source-order traversal
    from CybORG's iteration order, which is necessary for multi-step
    parity replay.

    Use must be paired with ``jax.disable_jit()`` because the tables are
    plain Python state mutated per step — JIT would cache the first step's
    values into the compiled artifact.

    Typical usage (inside the differential harness)::

        tape = IndexedRNGTape()
        tape.set_detection_queue([0.5, 0.99])
        tape.set_red_pid_delta(agent_id=0, value=3)
        # … more table writes …
        with indexed_rng_impls(**tape.as_overrides()), jax.disable_jit():
            new_state = apply_all_actions(state, const, ...)
        tape.clear()  # ready for next step
    """

    def __init__(self):
        self._detection_q: deque[float] = deque()
        self._red_pid_delta: dict[int, int] = {}
        self._red_privesc: dict[int, int] = {}
        self._red_session_check: dict[int, int] = {}
        self._exploit_session: dict[int, int] = {}
        self._blue_decoy_type: dict[int, int] = {}
        self._blue_decoy_pid_delta: dict[tuple[int, int], int] = {}
        self._red_policy: dict[tuple[int, int], float] = {}
        self._green_table: np.ndarray | None = None  # (num_hosts, num_fields) of float32
        self._green_int_range: np.ndarray | None = None  # same shape as _green; used when int_range given
        self._misses: list[str] = []
        self._used = 0

    # ---- Population API ---------------------------------------------------

    def clear(self) -> None:
        self._detection_q.clear()
        self._red_pid_delta.clear()
        self._red_privesc.clear()
        self._red_session_check.clear()
        self._exploit_session.clear()
        self._blue_decoy_type.clear()
        self._blue_decoy_pid_delta.clear()
        self._red_policy.clear()
        self._green_table = None
        self._green_int_range = None
        self._misses.clear()
        self._used = 0

    def set_detection_queue(self, values: Iterable[float]) -> None:
        self._detection_q.clear()
        self._detection_q.extend(float(v) for v in values)

    def set_red_pid_delta(self, agent_id: int, value: int) -> None:
        self._red_pid_delta[int(agent_id)] = int(value)

    def set_red_privesc(self, agent_id: int, value: int) -> None:
        self._red_privesc[int(agent_id)] = int(value)

    def set_red_session_check(self, agent_id: int, value: int) -> None:
        self._red_session_check[int(agent_id)] = int(value)

    def set_exploit_session(self, agent_id: int, value: int) -> None:
        self._exploit_session[int(agent_id)] = int(value)

    def set_blue_decoy_type(self, agent_id: int, value: int) -> None:
        self._blue_decoy_type[int(agent_id)] = int(value)

    def set_blue_decoy_pid_delta(self, agent_id: int, respawn_index: int, value: int) -> None:
        self._blue_decoy_pid_delta[(int(agent_id), int(respawn_index))] = int(value)

    def set_red_policy(self, agent_id: int, field_idx: int, value: float) -> None:
        self._red_policy[(int(agent_id), int(field_idx))] = float(value)

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

    def _record_miss(self, descriptor: str) -> None:
        if descriptor not in self._misses:
            self._misses.append(descriptor)

    # ---- Per-purpose dispatch impls --------------------------------------

    def _detection(self, key):
        del key
        if not self._detection_q:
            self._record_miss("detection: queue exhausted")
            return jnp.float32(0.5)
        self._used += 1
        return jnp.float32(self._detection_q.popleft())

    def _green_impl(self, key, time, host_idx, field_idx, int_range):
        """Green is consumed under ``vmap`` so ``host_idx`` may be a tracer.

        We use traced gather into the table, which works for both eager and
        vmap paths.  When the table is missing, fall back to the low-level
        default to keep parity tests viable on a slow path.

        Legacy semantics for ``int_range``: when set, return
        ``floor(table[h, f] * int_range)`` from the float table.  This matches
        the historical ``const.green_randoms`` encoding produced by
        :mod:`jaxborg.parity.cyborg_green_recorder`.  An optional integer
        override table (``_green_int_range``) takes precedence — the harness
        uses this for fields where the legacy float→int decoding does not
        recover the desired integer (notably field 1, where the recorder
        stores the raw chosen-token id and JAX needs the index into its
        per-host ``sorted_tokens`` array).
        """
        del time
        if int_range is not None:
            # Prefer the explicit override table if present at this slot.
            if self._green_int_range is not None:
                int_table = jnp.asarray(self._green_int_range)
                int_val = int_table[host_idx, field_idx].astype(jnp.int32)
                # Sentinel < 0 ⇒ no override; fall through to float decoding.
                # We use jax.lax.cond so this works under vmap/tracers as well.
                if self._green_table is None:
                    return jnp.where(int_val >= 0, int_val, jnp.int32(0))
                table = jnp.asarray(self._green_table)
                int_range_arr = jnp.asarray(int_range, dtype=jnp.float32)
                # Field 1 (service-token selection) stores the raw chosen
                # token id, NOT a uniform — the float-decode formula
                # ``floor(value * int_range)`` would yield garbage indices
                # (often pointing at the sorted_tokens sentinel, which the
                # green vmap then misinterprets as a non-existent decoy and
                # spuriously raises green_lwf).  Return 0 as a safe fallback
                # so JAX picks the first available service/decoy instead.
                fallback = jnp.where(
                    field_idx == 1,
                    jnp.int32(0),
                    jnp.floor(table[host_idx, field_idx] * int_range_arr).astype(jnp.int32),
                )
                return jnp.where(int_val >= 0, int_val, fallback)
            if self._green_table is None:
                from jaxborg.actions.rng import _default_green

                return _default_green(key, 0, host_idx, field_idx, int_range)
            table = jnp.asarray(self._green_table)
            int_range_arr = jnp.asarray(int_range, dtype=jnp.float32)
            return jnp.floor(table[host_idx, field_idx] * int_range_arr).astype(jnp.int32)
        if self._green_table is None:
            from jaxborg.actions.rng import _default_green

            return _default_green(key, 0, host_idx, field_idx, None)
        table = jnp.asarray(self._green_table)
        return table[host_idx, field_idx].astype(jnp.float32)

    def _red_pid_delta_impl(self, key, agent_id):
        del key
        a = int(agent_id)
        if a not in self._red_pid_delta:
            self._record_miss(f"red_pid_delta: agent={a}")
            return jnp.int32(1)
        self._used += 1
        return jnp.int32(self._red_pid_delta[a])

    def _red_privesc_impl(self, key, agent_id, num_sessions):
        del key, num_sessions
        a = int(agent_id)
        if a not in self._red_privesc:
            self._record_miss(f"red_privesc: agent={a}")
            return jnp.int32(0)
        self._used += 1
        return jnp.int32(self._red_privesc[a])

    def _red_session_check_impl(self, key, agent_id, num_sessions_on_host):
        del key, num_sessions_on_host
        a = int(agent_id)
        if a not in self._red_session_check:
            self._record_miss(f"red_session_check: agent={a}")
            return jnp.int32(0)
        self._used += 1
        return jnp.int32(self._red_session_check[a])

    def _exploit_session_impl(self, key, agent_id, visible_sessions):
        del key, visible_sessions
        a = int(agent_id)
        if a not in self._exploit_session:
            self._record_miss(f"exploit_session: agent={a}")
            return jnp.int32(0)
        self._used += 1
        return jnp.int32(self._exploit_session[a])

    def _blue_decoy_type_impl(self, key, agent_id, compatibility):
        del key
        a = int(agent_id)
        if a not in self._blue_decoy_type:
            self._record_miss(f"blue_decoy_type: agent={a}")
            return jnp.int32(0)
        self._used += 1
        chosen = int(self._blue_decoy_type[a])
        # Honor compatibility: if the chosen type isn't compatible, fall back
        # to the lowest-index compatible type (matches the default's semantics
        # when an incompatible permutation slot would lose to a 100 score).
        compat = np.asarray(compatibility)
        if chosen < compat.shape[0] and bool(compat[chosen]):
            return jnp.int32(chosen)
        for i in range(compat.shape[0]):
            if bool(compat[i]):
                return jnp.int32(i)
        return jnp.int32(0)

    def _blue_decoy_pid_delta_impl(self, key, agent_id, respawn_index):
        del key
        k = (int(agent_id), int(respawn_index))
        if k not in self._blue_decoy_pid_delta:
            self._record_miss(f"blue_decoy_pid_delta: agent={k[0]} respawn={k[1]}")
            return jnp.int32(1)
        self._used += 1
        return jnp.int32(self._blue_decoy_pid_delta[k])

    def _green_dest_host_impl(self, key, time, host_idx, sorted_servers, num_reachable):
        """Return the recorded GreenAccessService dest host directly.

        Reads ``_green_table[host_idx, 5]`` which the recorder fills with the
        JAX host_idx of CybORG's chosen destination.  When the table is
        missing or the slot is zero (no recorded access this step), we fall
        back to the default sorted_servers[randint] path so behaviour is
        unchanged for unrecorded hosts.
        """
        if self._green_table is None:
            from jaxborg.actions.rng import _default_green_dest_host

            return _default_green_dest_host(key, time, host_idx, sorted_servers, num_reachable)
        table = jnp.asarray(self._green_table)
        recorded = table[host_idx, 5].astype(jnp.int32)
        # ``0`` is also a valid host_idx, but the recorder leaves the slot
        # untouched (initial zero) when the agent didn't run access service
        # this step — and in that case the host with idx 0 is unlikely to be
        # the actual choice.  We fall back to the default when num_reachable
        # is 0 (gates the host_activity write); otherwise honour the recorded
        # value.  The harness always runs eager, so this is fine.
        return recorded

    def _red_policy_impl(self, key, agent_id, field_idx):
        del key
        k = (int(agent_id), int(field_idx))
        if k not in self._red_policy:
            self._record_miss(f"red_policy: agent={k[0]} field={k[1]}")
            return jnp.float32(0.5)
        self._used += 1
        return jnp.float32(self._red_policy[k])

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
