import jax.numpy as jnp

from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.state import create_initial_state


class TestEffectiveSessionCounts:
    def test_uses_explicit_red_session_count_not_legacy_flags(self):
        state = create_initial_state()
        state = state.replace(
            red_sessions=state.red_sessions.at[0, 10].set(True),
            red_session_count=state.red_session_count.at[0, 10].set(0),
        )

        counts = effective_session_counts(state)
        assert int(counts[0, 10]) == 0

    def test_preserves_nonzero_explicit_count(self):
        state = create_initial_state()
        state = state.replace(
            red_sessions=state.red_sessions.at[2, 5].set(False),
            red_session_count=state.red_session_count.at[2, 5].set(3),
        )

        counts = effective_session_counts(state)
        assert int(counts[2, 5]) == 3
        assert counts.dtype == jnp.int32
