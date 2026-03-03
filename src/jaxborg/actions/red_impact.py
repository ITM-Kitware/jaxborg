import chex
import jax.numpy as jnp

from jaxborg.actions.red_common import select_bound_source_host
from jaxborg.constants import ACTIVITY_EXPLOIT, COMPROMISE_PRIVILEGED, SERVICE_IDS
from jaxborg.state import CC4Const, CC4State

OTSERVICE_IDX = SERVICE_IDS["OTSERVICE"]


def apply_impact(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    has_session = state.red_sessions[agent_id, target_host]
    is_privileged = state.red_privilege[agent_id, target_host] >= COMPROMISE_PRIVILEGED
    has_ot = state.host_services[target_host, OTSERVICE_IDX]
    has_bound_source = select_bound_source_host(state, const, agent_id) >= 0
    success = is_active & has_session & is_privileged & has_ot & has_bound_source

    # CybORG BlueRewardMachine penalizes ALL Impact attempts, not just successes,
    # because bool(TernaryEnum.FALSE) is True. See CYBORG_DIFFERENCES.md.
    red_impact_attempted = jnp.where(
        is_active,
        state.red_impact_attempted.at[target_host].set(True),
        state.red_impact_attempted,
    )

    host_services = jnp.where(
        success,
        state.host_services.at[target_host, OTSERVICE_IDX].set(False),
        state.host_services,
    )

    ot_service_stopped = jnp.where(
        success,
        state.ot_service_stopped.at[target_host].set(True),
        state.ot_service_stopped,
    )

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )

    return state.replace(
        host_services=host_services,
        ot_service_stopped=ot_service_stopped,
        red_activity_this_step=activity,
        red_impact_attempted=red_impact_attempted,
    )
