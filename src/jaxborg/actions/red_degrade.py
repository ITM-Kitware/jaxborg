import chex
import jax.numpy as jnp

from jaxborg.constants import ACTIVITY_EXPLOIT, COMPROMISE_PRIVILEGED
from jaxborg.state import CC4Const, CC4State

DEGRADE_AMOUNT = 20


def apply_degrade(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    has_session = state.red_sessions[agent_id, target_host]
    is_privileged = state.red_privilege[agent_id, target_host] >= COMPROMISE_PRIVILEGED
    has_services = jnp.any(state.host_services[target_host]) | jnp.any(state.host_decoys[target_host])
    success = is_active & has_session & is_privileged & has_services

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )

    current_reliability = state.host_service_reliability[target_host]
    active_services = state.host_services[target_host]
    degraded = jnp.maximum(current_reliability - DEGRADE_AMOUNT, 0)
    degraded = jnp.where(active_services, degraded, current_reliability)
    reliability = jnp.where(
        success,
        state.host_service_reliability.at[target_host].set(degraded),
        state.host_service_reliability,
    )
    current_decoy_reliability = state.host_decoy_reliability[target_host]
    active_decoys = state.host_decoys[target_host]
    degraded_decoys = jnp.maximum(current_decoy_reliability - DEGRADE_AMOUNT, 0)
    degraded_decoys = jnp.where(active_decoys, degraded_decoys, current_decoy_reliability)
    decoy_reliability = jnp.where(
        success,
        state.host_decoy_reliability.at[target_host].set(degraded_decoys),
        state.host_decoy_reliability,
    )

    return state.replace(
        red_activity_this_step=activity,
        host_service_reliability=reliability,
        host_decoy_reliability=decoy_reliability,
    )
