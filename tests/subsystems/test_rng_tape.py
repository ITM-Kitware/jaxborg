"""Smoke tests for the host-side RNG tape primitive."""

import jax
import jax.numpy as jnp
import pytest

from jaxborg.actions.rng import (
    indexed_rng_impls,
    rng_impls,
    sample_blue_decoy_pid_delta,
    sample_blue_decoy_type_choice,
    sample_detection_random,
    sample_exploit_session_choice,
    sample_green_random,
    sample_red_pid_delta,
    sample_red_privesc_choice,
    sample_red_session_check_choice,
)
from tests.differential.parity_rng_replay import IndexedRNGTape, RNGTape


def test_eager_uniform_replay():
    tape = RNGTape()
    tape.push_uniforms([0.25, 0.75])
    with rng_impls(uniform=tape.uniform):
        a, _ = sample_detection_random(state=None, const=None, key=jax.random.PRNGKey(0))
        b, _ = sample_detection_random(state=None, const=None, key=jax.random.PRNGKey(1))
    assert float(a) == pytest.approx(0.25)
    assert float(b) == pytest.approx(0.75)
    assert tape.consumed == 2
    assert tape.remaining == 0


def test_eager_randint_replay():
    tape = RNGTape()
    tape.push_randints([3, 7])
    with rng_impls(randint=tape.randint):
        a = sample_red_pid_delta(const=None, time=0, agent_id=0, key=jax.random.PRNGKey(0))
        b = sample_blue_decoy_pid_delta(const=None, time=0, agent_id=0, key=jax.random.PRNGKey(1))
    assert int(a) == 3
    assert int(b) == 7
    assert tape.consumed == 2


def test_eager_permutation_replay():
    tape = RNGTape()
    tape.push_permutation([2, 0, 1, 3])
    compatibility = jnp.array([True, True, False, True])
    with rng_impls(permutation=tape.permutation):
        choice = sample_blue_decoy_type_choice(
            const=None,
            time=0,
            agent_id=0,
            compatibility=compatibility,
            key=jax.random.PRNGKey(0),
        )
    # perm = [2, 0, 1, 3]; compatible mask masks out idx=2 (score 100); argmin -> idx 1 (score 0)
    assert int(choice) == 1


def test_green_int_range_uses_randint():
    tape = RNGTape()
    tape.push_randint(5)
    tape.push_uniform(0.42)
    with rng_impls(uniform=tape.uniform, randint=tape.randint):
        a = sample_green_random(const=None, time=0, host_idx=0, field_idx=0, key=jax.random.PRNGKey(0), int_range=10)
        b = sample_green_random(const=None, time=0, host_idx=0, field_idx=0, key=jax.random.PRNGKey(1))
    assert int(a) == 5
    assert float(b) == pytest.approx(0.42)


def test_exhaustion_raises():
    tape = RNGTape()
    with rng_impls(uniform=tape.uniform):
        with pytest.raises(RuntimeError, match="exhausted"):
            sample_detection_random(state=None, const=None, key=jax.random.PRNGKey(0))


def test_impls_restore_on_exit():
    import jaxborg.actions.rng as rng

    default_uniform = rng._uniform_impl
    tape = RNGTape()
    tape.push_uniform(0.1)
    assert rng._uniform_impl is default_uniform
    with rng_impls(uniform=tape.uniform):
        assert rng._uniform_impl is not default_uniform
    assert rng._uniform_impl is default_uniform


def test_exploit_session_uses_randint():
    tape = RNGTape()
    tape.push_randint(0)
    with rng_impls(randint=tape.randint):
        choice = sample_exploit_session_choice(
            const=None, time=0, agent_id=0, key=jax.random.PRNGKey(0), visible_sessions=4
        )
    assert int(choice) == 0


def test_callback_mode_under_jit():
    """Verify io_callback path works under JIT.

    We jit-compile a simple call and check the popped value is observed.
    """
    tape = RNGTape(mode="callback")
    tape.push_uniform(0.625)

    @jax.jit
    def _go(key):
        return tape.uniform(key)

    out = _go(jax.random.PRNGKey(0))
    assert float(out) == pytest.approx(0.625)
    assert tape.consumed == 1


# ---------- IndexedRNGTape ----------


def test_indexed_tape_per_agent_lookup():
    tape = IndexedRNGTape()
    tape.set_red_pid_delta(agent_id=0, value=3)
    tape.set_red_pid_delta(agent_id=1, value=7)
    tape.set_red_privesc(agent_id=0, value=2)
    tape.set_red_session_check(agent_id=2, value=4)

    with indexed_rng_impls(**tape.as_overrides()):
        d0 = sample_red_pid_delta(const=None, time=0, agent_id=0, key=jax.random.PRNGKey(0))
        d1 = sample_red_pid_delta(const=None, time=0, agent_id=1, key=jax.random.PRNGKey(1))
        p0 = sample_red_privesc_choice(const=None, time=0, agent_id=0, key=jax.random.PRNGKey(2), num_sessions=5)
        sc2 = sample_red_session_check_choice(
            const=None, time=0, agent_id=2, key=jax.random.PRNGKey(3), num_sessions_on_host=5
        )
    assert int(d0) == 3
    assert int(d1) == 7
    assert int(p0) == 2
    assert int(sc2) == 4
    assert tape.used == 4


def test_indexed_tape_detection_queue():
    tape = IndexedRNGTape()
    tape.set_detection_queue([0.1, 0.9])
    with indexed_rng_impls(**tape.as_overrides()):
        a, _ = sample_detection_random(state=None, const=None, key=jax.random.PRNGKey(0))
        b, _ = sample_detection_random(state=None, const=None, key=jax.random.PRNGKey(1))
    assert float(a) == pytest.approx(0.1)
    assert float(b) == pytest.approx(0.9)


def test_indexed_tape_records_misses():
    tape = IndexedRNGTape(strict=False)
    # No tables populated — every call should miss.
    with indexed_rng_impls(**tape.as_overrides()):
        sample_red_pid_delta(const=None, time=0, agent_id=0, key=jax.random.PRNGKey(0))
        sample_exploit_session_choice(const=None, time=0, agent_id=1, key=jax.random.PRNGKey(0), visible_sessions=2)
    assert tape.used == 0
    assert any("red_pid_delta: agent=0" in m for m in tape.misses)
    assert any("exploit_session: agent=1" in m for m in tape.misses)


def test_indexed_tape_strict_miss_raises():
    """Default strict=True: a missing entry surfaces as RuntimeError."""
    tape = IndexedRNGTape()  # strict=True by default
    with indexed_rng_impls(**tape.as_overrides()):
        with pytest.raises(RuntimeError, match="strict mode"):
            sample_red_pid_delta(const=None, time=0, agent_id=0, key=jax.random.PRNGKey(0))


def test_indexed_tape_blue_decoy_type_returns_recorded_value():
    """The tape is a pure lookup primitive — it returns the recorded type
    verbatim regardless of compatibility.  Compatibility resolution lives in
    ``apply_blue_decoy`` so the tape stays a thin replay layer."""
    tape = IndexedRNGTape()
    tape.set_blue_decoy_type(agent_id=0, value=2)
    compat = jnp.array([False, True, False, True])
    with indexed_rng_impls(**tape.as_overrides()):
        choice = sample_blue_decoy_type_choice(
            const=None, time=0, agent_id=0, compatibility=compat, key=jax.random.PRNGKey(0)
        )
    assert int(choice) == 2


def test_indexed_tape_blue_decoy_pid_delta_per_respawn():
    tape = IndexedRNGTape()
    tape.set_blue_decoy_pid_delta(agent_id=0, respawn_index=0, value=4)
    tape.set_blue_decoy_pid_delta(agent_id=0, respawn_index=1, value=8)
    with indexed_rng_impls(**tape.as_overrides()):
        a = sample_blue_decoy_pid_delta(const=None, time=0, agent_id=0, key=jax.random.PRNGKey(0), respawn_index=0)
        b = sample_blue_decoy_pid_delta(const=None, time=0, agent_id=0, key=jax.random.PRNGKey(1), respawn_index=1)
    assert int(a) == 4
    assert int(b) == 8


def test_indexed_tape_partial_override():
    """Purposes not overridden fall through to defaults (which call jax.random.*)."""
    tape = IndexedRNGTape()
    tape.set_red_pid_delta(agent_id=0, value=5)
    with indexed_rng_impls(red_pid_delta=tape.as_overrides()["red_pid_delta"]):
        # red_pid_delta from tape:
        delta = sample_red_pid_delta(const=None, time=0, agent_id=0, key=jax.random.PRNGKey(0))
        # detection NOT overridden → falls through to jax.random.uniform:
        rand, _ = sample_detection_random(state=None, const=None, key=jax.random.PRNGKey(0))
    assert int(delta) == 5
    assert 0.0 <= float(rand) <= 1.0


def test_indexed_tape_green_uniform_table():
    import numpy as np

    tape = IndexedRNGTape()
    tape.set_green_uniform(np.full((10, 8), 0.42, dtype=np.float32))
    with indexed_rng_impls(**tape.as_overrides()):
        v = sample_green_random(const=None, time=0, host_idx=3, field_idx=2, key=jax.random.PRNGKey(0))
    assert float(v) == pytest.approx(0.42)


def test_indexed_tape_jit_compatible():
    """IndexedRNGTape impls work under @jax.jit.

    Verifies (a) tracing succeeds without disable_jit, and (b) re-running
    after mutating the tape returns the new values without recompiling
    (callbacks read self.<attr> at execution time, not trace time).
    """
    tape = IndexedRNGTape(strict=False)
    tape.set_red_pid_delta(agent_id=0, value=3)
    tape.set_red_pid_delta(agent_id=1, value=7)

    @jax.jit
    def go(agent_id):
        with indexed_rng_impls(**tape.as_overrides()):
            return sample_red_pid_delta(const=None, time=0, agent_id=agent_id, key=jax.random.PRNGKey(0))

    a = go(jnp.int32(0))
    b = go(jnp.int32(1))
    assert int(a) == 3
    assert int(b) == 7

    # Mutate tape, re-call: callback reads new value (no retrace required).
    tape.set_red_pid_delta(agent_id=0, value=42)
    c = go(jnp.int32(0))
    assert int(c) == 42


def test_indexed_tape_jit_green_vmap():
    """Green table lookup works under jit + vmap (the harness pattern)."""
    import numpy as np

    tape = IndexedRNGTape(strict=False)
    tape.set_green_uniform(np.tile(np.arange(8, dtype=np.float32) / 10.0, (10, 1)))

    @jax.jit
    def gather(host_indices):
        with indexed_rng_impls(**tape.as_overrides()):
            return jax.vmap(
                lambda h: sample_green_random(const=None, time=0, host_idx=h, field_idx=2, key=jax.random.PRNGKey(0))
            )(host_indices)

    out = gather(jnp.arange(5, dtype=jnp.int32))
    # field_idx=2 → 0.2 for every host.
    assert all(float(v) == pytest.approx(0.2) for v in out)


def test_indexed_tape_clear():
    tape = IndexedRNGTape(strict=False)
    tape.set_red_pid_delta(agent_id=0, value=5)
    tape.set_detection_queue([0.5])
    tape.clear()
    with indexed_rng_impls(**tape.as_overrides()):
        sample_red_pid_delta(const=None, time=0, agent_id=0, key=jax.random.PRNGKey(0))
    # cleared → miss recorded
    assert tape.misses
    assert tape.used == 0
