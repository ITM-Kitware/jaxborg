import jax.numpy as jnp

from jaxborg.constants import (
    BLUE_ACTION_HOST_SLOTS,
    BLUE_MAX_OBSERVED_SUBNETS,
    DECOY_IDS,
    GLOBAL_MAX_HOSTS,
    NUM_DECOY_TYPES,
    NUM_SUBNETS,
    OBS_VECTOR_HOSTS_PER_SUBNET,
    SERVICE_IDS,
)
from jaxborg.state import SimulatorConst, SimulatorState


def _decoy_compat_vectorized(flat_services: jnp.ndarray, flat_decoys: jnp.ndarray) -> jnp.ndarray:
    """Compute per-slot decoy compatibility: True if ANY decoy type is compatible.

    Args:
        flat_services: (BLUE_ACTION_HOST_SLOTS, NUM_SERVICES) bool
        flat_decoys: (BLUE_ACTION_HOST_SLOTS, NUM_DECOY_TYPES) bool

    Returns:
        (BLUE_ACTION_HOST_SLOTS,) bool — True if at least one decoy type can be deployed
    """
    has_port_25 = flat_services[:, SERVICE_IDS["SMTP"]] | flat_decoys[:, DECOY_IDS["HarakaSMPT"]]
    has_port_80 = (
        flat_services[:, SERVICE_IDS["APACHE2"]]
        | flat_decoys[:, DECOY_IDS["Apache"]]
        | flat_decoys[:, DECOY_IDS["Vsftpd"]]
    )
    per_type = jnp.stack([~has_port_25, ~has_port_80, jnp.ones(flat_services.shape[0], dtype=jnp.bool_), ~has_port_80])
    return per_type.any(axis=0)


def compute_blue_action_mask(const: SimulatorConst, agent_id: int, state: SimulatorState | None = None) -> jnp.ndarray:
    """Return bool mask of valid actions for a blue agent.

    Uses agent-relative encoding: host slots index into the agent's 3 observed
    subnets via const.blue_obs_subnets[agent_id], and traffic dst indexes into
    the same 3 subnets.

    Matches CybORG's BlueFixedActionWrapper masking: host/subnet validity and
    router exclusion.  The mask is static per topology — it does NOT change
    based on pending-action (busy) state.  This matches CybORG, where the
    wrapper returns the same mask every tick and silently discards any action
    submitted while a multi-tick action is in progress.
    """
    agent_obs_subnets = const.blue_obs_subnets[agent_id]  # (3,) int, -1 = unused

    # Build (BLUE_MAX_OBSERVED_SUBNETS, OBS_VECTOR_HOSTS_PER_SUBNET) validity array
    # Only non-router slots (0..15); router at slot 16 is structurally excluded.
    def _subnet_validity(rel_idx):
        sid = agent_obs_subnets[rel_idx]
        safe_sid = jnp.clip(sid, 0, NUM_SUBNETS - 1)
        obs_valid = const.obs_host_map[safe_sid, :OBS_VECTOR_HOSTS_PER_SUBNET] < GLOBAL_MAX_HOSTS
        subnet_active = sid >= 0
        return obs_valid & subnet_active

    slot_valid = jnp.stack([_subnet_validity(i) for i in range(BLUE_MAX_OBSERVED_SUBNETS)])

    slot_valid_flat = slot_valid.reshape(-1)  # (BLUE_ACTION_HOST_SLOTS,)

    # Build services/decoys for the agent's observed subnets
    def _subnet_host_data(rel_idx):
        sid = agent_obs_subnets[rel_idx]
        safe_sid = jnp.clip(sid, 0, NUM_SUBNETS - 1)
        hosts = const.obs_host_map[safe_sid, :OBS_VECTOR_HOSTS_PER_SUBNET]
        safe_host_idx = jnp.where(hosts < GLOBAL_MAX_HOSTS, hosts, 0)
        return safe_host_idx

    # (BLUE_MAX_OBSERVED_SUBNETS, OBS_VECTOR_HOSTS_PER_SUBNET)
    safe_hosts = jnp.stack([_subnet_host_data(i) for i in range(BLUE_MAX_OBSERVED_SUBNETS)])
    safe_hosts_flat = safe_hosts.reshape(-1)  # (BLUE_ACTION_HOST_SLOTS,)

    if state is None:
        host_services = const.initial_services[safe_hosts_flat]
        host_decoys = jnp.zeros(host_services.shape[:-1] + (NUM_DECOY_TYPES,), dtype=jnp.bool_)
    else:
        host_services = state.host_services[safe_hosts_flat]
        host_decoys = state.host_decoys[safe_hosts_flat]

    flat_services = host_services.reshape(BLUE_ACTION_HOST_SLOTS, -1)
    flat_decoys = host_decoys.reshape(BLUE_ACTION_HOST_SLOTS, -1)
    any_decoy_compat = _decoy_compat_vectorized(flat_services, flat_decoys)
    decoy_mask = any_decoy_compat & slot_valid_flat

    # Traffic: src_offset ranges 0..(NUM_SUBNETS-2), dst is relative 0-2.
    # Self-loops (src==dst) are excluded structurally by the compressed encoding.
    # Layout: src_offset * BLUE_MAX_OBSERVED_SUBNETS + relative_dst
    abs_dst = agent_obs_subnets[None, :]  # (1, 3)
    dst_active = abs_dst >= 0
    traffic_flat = jnp.broadcast_to(dst_active, (NUM_SUBNETS - 1, BLUE_MAX_OBSERVED_SUBNETS)).reshape(-1)

    mask = jnp.concatenate(
        [
            jnp.array([True, True]),  # sleep + monitor
            slot_valid_flat,  # analyse
            slot_valid_flat,  # remove
            slot_valid_flat,  # restore
            decoy_mask,  # decoys (1 per host slot)
            traffic_flat,  # block traffic
            traffic_flat,  # allow traffic
        ]
    )

    return mask
