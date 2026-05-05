import chex
import jax.numpy as jnp

from jaxborg.actions.red_common import select_bound_source_host
from jaxborg.constants import ACTIVITY_EXPLOIT, COMPROMISE_PRIVILEGED, SERVICE_IDS
from jaxborg.state import SimulatorConst, SimulatorState

OTSERVICE_IDX = SERVICE_IDS["OTSERVICE"]


def apply_impact(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    target_host: chex.Array,
) -> SimulatorState:
    is_active = const.host_active[target_host]
    agent_has_session = jnp.any(state.red_sessions[agent_id] & const.host_active)
    has_session = state.red_sessions[agent_id, target_host]
    is_privileged = state.red_privilege[agent_id, target_host] >= COMPROMISE_PRIVILEGED
    has_ot = state.host_services[target_host, OTSERVICE_IDX]
    has_bound_source = select_bound_source_host(state, const, agent_id) >= 0
    success = is_active & has_session & is_privileged & has_ot & has_bound_source

    # CybORG BlueRewardMachine penalizes failed Impact actions when the red
    # agent still has an active session, because bool(TernaryEnum.FALSE) is
    # True. Sessionless pending Impacts are not penalized.
    #
    # CybORG checks ``len(active sessions) > 0`` at *reward computation* time,
    # which runs after ``different_subnet_agent_reassignment`` and the
    # per-agent ``RedSessionCheck`` end-turn actions (see
    # CybORG/Simulator/SimulationController.py:278-309).  An agent that
    # attempted Impact during the red phase but lost all of its sessions to
    # cross-subnet reassignment before reward computation is therefore *not*
    # charged the RIA penalty.  Track the attempt per-agent so the post-step
    # gate in ``apply_all_actions`` can re-evaluate session presence using the
    # final (post-reassignment, post-session-check) session counts.
    red_impact_attempted = jnp.where(
        is_active & agent_has_session,
        state.red_impact_attempted.at[target_host].set(True),
        state.red_impact_attempted,
    )
    red_impact_attempted_by_agent = jnp.where(
        is_active,
        state.red_impact_attempted_by_agent.at[agent_id, target_host].set(True),
        state.red_impact_attempted_by_agent,
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
        red_impact_attempted_by_agent=red_impact_attempted_by_agent,
    )
