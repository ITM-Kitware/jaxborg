import jax.numpy as jnp


def effective_session_counts(state):
    """Return explicit session multiplicity state."""
    return state.red_session_count.astype(jnp.int32)
