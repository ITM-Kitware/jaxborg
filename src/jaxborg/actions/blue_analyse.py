import jax.numpy as jnp

from jaxborg.state import SimulatorConst, SimulatorState


def apply_blue_analyse(state: SimulatorState, const: SimulatorConst, agent_id: int, target_host: int) -> SimulatorState:
    if not const.cage4_enhanced_obs:
        return state
    valid = (target_host >= 0) & (target_host < const.host_active.size)
    valid &= const.host_active[target_host] & const.blue_agent_hosts[agent_id, target_host]
    evidence = state.blue_file_evidence.at[target_host].set(state.host_file_artifact[target_host])
    return state.replace(blue_file_evidence=jnp.where(valid, evidence, state.blue_file_evidence))
