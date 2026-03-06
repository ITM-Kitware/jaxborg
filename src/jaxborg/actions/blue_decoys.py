import jax.numpy as jnp

from jaxborg.actions.pids import allocate_host_pid_from_delta
from jaxborg.actions.rng import sample_blue_decoy_pid_delta
from jaxborg.constants import DECOY_IDS, SERVICE_IDS
from jaxborg.state import CC4Const, CC4State


def apply_blue_decoy(state: CC4State, const: CC4Const, agent_id: int, target_host: int, decoy_type: int) -> CC4State:
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    has_port_25 = (
        state.host_services[target_host, SERVICE_IDS["SMTP"]] | state.host_decoys[target_host, DECOY_IDS["HarakaSMPT"]]
    )
    has_port_80 = (
        state.host_services[target_host, SERVICE_IDS["APACHE2"]]
        | state.host_decoys[target_host, DECOY_IDS["Apache"]]
        | state.host_decoys[target_host, DECOY_IDS["Vsftpd"]]
    )
    has_port_443 = state.host_decoys[target_host, DECOY_IDS["Tomcat"]]

    compatible = jnp.array(True)
    compatible = jnp.where(decoy_type == DECOY_IDS["HarakaSMPT"], ~has_port_25, compatible)
    compatible = jnp.where(decoy_type == DECOY_IDS["Apache"], ~has_port_80, compatible)
    compatible = jnp.where(decoy_type == DECOY_IDS["Tomcat"], ~has_port_443, compatible)

    can_deploy = covers_host & compatible
    pid_delta = sample_blue_decoy_pid_delta(state, state.time, agent_id)
    new_pid = allocate_host_pid_from_delta(state, const, target_host, pid_delta)
    host_decoys = jnp.where(
        can_deploy,
        state.host_decoys.at[target_host, decoy_type].set(True),
        state.host_decoys,
    )
    host_decoy_process_pids = jnp.where(
        can_deploy,
        state.host_decoy_process_pids.at[target_host, decoy_type].set(new_pid),
        state.host_decoy_process_pids,
    )
    return state.replace(
        host_decoys=host_decoys,
        host_decoy_process_pids=host_decoy_process_pids,
    )
