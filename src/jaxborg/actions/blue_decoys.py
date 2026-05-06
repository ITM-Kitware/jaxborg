import jax
import jax.numpy as jnp

from jaxborg.actions.pids import allocate_host_pid_from_delta
from jaxborg.actions.rng import sample_blue_decoy_pid_delta, sample_blue_decoy_type_choice
from jaxborg.constants import DECOY_IDS, SERVICE_IDS
from jaxborg.state import SimulatorConst, SimulatorState


def host_decoy_compatibility_mask(host_services: jnp.ndarray, host_decoys: jnp.ndarray) -> jnp.ndarray:
    """Return per-decoy compatibility for one host.

    Mirrors CybORG's actual (slightly quirky) port-occupancy semantics:

      * ``HarakaDecoyFactory`` (port 25) is incompatible if SMTP service or a
        prior Haraka decoy holds port 25.
      * ``ApacheDecoyFactory`` (port 80) is incompatible if APACHE2 service,
        a prior Apache decoy, *or* a prior Vsftpd decoy holds port 80.  The
        Vsftpd inclusion is intentional and matches CybORG: although CybORG's
        ``DecoyVsftpd.is_host_compatible`` checks port 21, the
        ``VsftpdDecoyFactory`` constant ``PORT == 80`` means the decoy
        process actually opens port 80, so subsequent Apache deploys see
        port 80 as taken.
      * ``TomcatDecoyFactory`` (port 443) is incompatible if a prior Tomcat
        decoy holds port 443 (no real service uses 443).
      * ``VsftpdDecoyFactory`` is *always compatible* (subject to OS, which
        is a static host property handled elsewhere).  CybORG's check uses
        port 21, which the Vsftpd decoy never actually claims, so re-deploys
        succeed unconditionally.
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


def apply_blue_decoy(
    state: SimulatorState, const: SimulatorConst, agent_id: int, target_host: int, decoy_type: int, key=None
) -> SimulatorState:
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    covers_host = const.blue_agent_hosts[agent_id, target_host]
    compatibility = host_decoy_compatibility_mask(state.host_services[target_host], state.host_decoys[target_host])

    # When decoy_type == -1 (collapsed action space), randomly select a compatible type.
    # When decoy_type >= 0 (direct call from tests), use the explicit type.
    random_type = sample_blue_decoy_type_choice(const, state.time, agent_id, compatibility, k1)
    # Compat-fallback: production sampler already respects compatibility, so
    # this only kicks in when a replay tape returns a type that isn't valid in
    # the current host state.  Falling back to ``argmax(compatibility)`` (the
    # lowest True index) matches the default sampler's permutation-based pick.
    n_decoys = compatibility.shape[0]
    in_range = (random_type >= 0) & (random_type < n_decoys)
    compat_at_chosen = jnp.where(in_range, compatibility[jnp.clip(random_type, 0, n_decoys - 1)], False)
    fallback_type = jnp.argmax(compatibility.astype(jnp.int32))
    random_type = jnp.where(compat_at_chosen, random_type, fallback_type).astype(jnp.int32)
    decoy_type = jnp.where(decoy_type < 0, random_type, decoy_type)

    compatible = compatibility[decoy_type]
    can_deploy = covers_host & compatible
    pid_delta = sample_blue_decoy_pid_delta(const, state.time, agent_id, k2)
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
    # CybORG's DecoyAction creates a new process without removing the old one
    # when redeploying the same decoy type.  The old process becomes an orphan
    # in host.processes, contributing to create_pid()'s max() calculation.
    # Track the max orphaned PID so recompute_host_max_pid stays correct.
    old_pid = state.host_decoy_process_pids[target_host, decoy_type]
    has_old = old_pid >= 0
    host_orphaned_decoy_max_pid = jnp.where(
        can_deploy & has_old,
        state.host_orphaned_decoy_max_pid.at[target_host].set(
            jnp.maximum(state.host_orphaned_decoy_max_pid[target_host], old_pid)
        ),
        state.host_orphaned_decoy_max_pid,
    )
    host_decoy_process_pids = jnp.where(
        can_deploy,
        state.host_decoy_process_pids.at[target_host, decoy_type].set(new_pid),
        state.host_decoy_process_pids,
    )
    host_max_pid = jnp.where(
        can_deploy,
        state.host_max_pid.at[target_host].set(jnp.maximum(state.host_max_pid[target_host], new_pid)),
        state.host_max_pid,
    )
    return state.replace(
        host_decoys=host_decoys,
        host_decoy_reliability=host_decoy_reliability,
        host_decoy_process_pids=host_decoy_process_pids,
        host_orphaned_decoy_max_pid=host_orphaned_decoy_max_pid,
        host_max_pid=host_max_pid,
    )
