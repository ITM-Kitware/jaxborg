import jax.numpy as jnp

from jaxborg.actions.pids import allocate_host_pid_from_delta
from jaxborg.actions.rng import sample_blue_decoy_pid_delta
from jaxborg.state import CC4Const, CC4State


def apply_blue_decoy(state: CC4State, const: CC4Const, agent_id: int, target_host: int, decoy_type: int) -> CC4State:
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    pid_delta = sample_blue_decoy_pid_delta(state, state.time, agent_id)
    new_pid = allocate_host_pid_from_delta(state, const, target_host, pid_delta)
    host_decoys = jnp.where(
        covers_host,
        state.host_decoys.at[target_host, decoy_type].set(True),
        state.host_decoys,
    )
    host_decoy_process_pids = jnp.where(
        covers_host,
        state.host_decoy_process_pids.at[target_host, decoy_type].set(new_pid),
        state.host_decoy_process_pids,
    )
    return state.replace(
        host_decoys=host_decoys,
        host_decoy_process_pids=host_decoy_process_pids,
    )
