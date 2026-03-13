import chex
import jax
import jax.numpy as jnp

from jaxborg.actions.red_common import select_bound_source_host
from jaxborg.actions.rng import sample_detection_random
from jaxborg.agents.fsm_red import FSM_K, FSM_KD, FSM_R, FSM_RD, FSM_S, FSM_SD, FSM_U, FSM_UD
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
    has_decoys = jnp.any(state.host_decoys[target_host])

    def _on_success(state_in: CC4State) -> CC4State:
        k1, k2 = jax.random.split(key)
        r1, next_state = sample_detection_random(state_in, const, k1)
        r2, next_state = sample_detection_random(next_state, const, k2)
        detected = (has_decoys & (r1 < DECEPTION_TP_RATE)) | (~has_decoys & (r2 < DECEPTION_FP_RATE))
        return next_state.replace(
            fsm_host_states=jnp.where(
                detected,
                _apply_decoy_detection(next_state.fsm_host_states, agent_id, target_host),
                next_state.fsm_host_states,
            ),
        )

    return jax.lax.cond(success, _on_success, lambda state_in: state_in, state)


def _apply_decoy_detection(fsm_host_states, agent_id, target_host):
    cur = fsm_host_states[agent_id, target_host]
    new_state = cur
    new_state = jnp.where(cur == FSM_K, FSM_KD, new_state)
    new_state = jnp.where(cur == FSM_S, FSM_SD, new_state)
    new_state = jnp.where(cur == FSM_U, FSM_UD, new_state)
    new_state = jnp.where(cur == FSM_R, FSM_RD, new_state)
    return fsm_host_states.at[agent_id, target_host].set(new_state)
