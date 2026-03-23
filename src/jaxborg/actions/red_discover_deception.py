import chex
import jax
import jax.numpy as jnp

from jaxborg.actions.red_common import select_bound_source_host
from jaxborg.actions.rng import sample_detection_random
from jaxborg.state import CC4Const, CC4State

DECEPTION_TP_RATE = 0.5
DECEPTION_FP_RATE = 0.1


def apply_discover_deception(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
    key: jax.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    source_host = select_bound_source_host(state, const, agent_id)
    source_idx = jnp.clip(source_host, 0, state.red_sessions.shape[1] - 1)
    has_bound_source = (source_host >= 0) & state.red_sessions[agent_id, source_idx] & const.host_active[source_idx]
    # CybORG DiscoverDeception checks only that a route exists in the base link
    # graph; it does not consult state.blocks / firewall rules.
    can_reach = has_bound_source

    success = is_active & can_reach

    def _on_success(state_in: CC4State) -> CC4State:
        k1, k2 = jax.random.split(key)
        # Consume two detection randoms to stay in sync with CybORG's RNG.
        # CybORG's FiniteStateRedAgent handles FSM transitions via the
        # success/failure transition matrices in _host_state_transition,
        # NOT inside the DiscoverDeception action.  Decoy detection only
        # records PIDs in host_service_decoy_status for exploit avoidance.
        _r1, next_state = sample_detection_random(state_in, const, k1)
        _r2, next_state = sample_detection_random(next_state, const, k2)
        return next_state

    return jax.lax.cond(success, _on_success, lambda state_in: state_in, state)
