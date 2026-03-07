import jax.numpy as jnp

from jaxborg.actions.pids import allocate_host_pid_from_delta
from jaxborg.actions.rng import sample_blue_decoy_pid_delta
from jaxborg.constants import DECOY_IDS, SERVICE_IDS
from jaxborg.state import CC4Const, CC4State


def host_decoy_compatibility_mask(host_services: jnp.ndarray, host_decoys: jnp.ndarray) -> jnp.ndarray:
    """Return per-decoy compatibility for one host.

    Matches the concrete decoy compatibility modeled by `apply_blue_decoy()`.
    """
    has_port_25 = host_services[SERVICE_IDS["SMTP"]] | host_decoys[DECOY_IDS["HarakaSMPT"]]
    has_port_80 = (
        host_services[SERVICE_IDS["APACHE2"]] | host_decoys[DECOY_IDS["Apache"]] | host_decoys[DECOY_IDS["Vsftpd"]]
    )
    has_port_443 = host_decoys[DECOY_IDS["Tomcat"]]

    return jnp.array(
        [
            ~has_port_25,
            ~has_port_80,
            ~has_port_443,
            True,
        ],
        dtype=jnp.bool_,
    )


def apply_blue_decoy(state: CC4State, const: CC4Const, agent_id: int, target_host: int, decoy_type: int) -> CC4State:
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    compatibility = host_decoy_compatibility_mask(state.host_services[target_host], state.host_decoys[target_host])
    compatible = compatibility[decoy_type]

    can_deploy = covers_host & compatible
    pid_delta = sample_blue_decoy_pid_delta(state, state.time, agent_id)
    new_pid = allocate_host_pid_from_delta(state, const, target_host, pid_delta)
    host_decoys = jnp.where(
        can_deploy,
        state.host_decoys.at[target_host, decoy_type].set(True),
        state.host_decoys,
    )
    host_decoy_reliability = jnp.where(
        can_deploy,
        state.host_decoy_reliability.at[target_host, decoy_type].set(100),
        state.host_decoy_reliability,
    )
    host_decoy_process_pids = jnp.where(
        can_deploy,
        state.host_decoy_process_pids.at[target_host, decoy_type].set(new_pid),
        state.host_decoy_process_pids,
    )
    return state.replace(
        host_decoys=host_decoys,
        host_decoy_reliability=host_decoy_reliability,
        host_decoy_process_pids=host_decoy_process_pids,
    )
