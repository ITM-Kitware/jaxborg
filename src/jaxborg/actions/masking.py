import jax.numpy as jnp

from jaxborg.constants import (
    ACTION_HOST_SLOTS,
    DECOY_IDS,
    GLOBAL_MAX_HOSTS,
    NUM_DECOY_TYPES,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
    SERVICE_IDS,
)
from jaxborg.state import CC4Const, CC4State


def _decoy_compat_vectorized(flat_services: jnp.ndarray, flat_decoys: jnp.ndarray) -> jnp.ndarray:
    """Compute per-slot decoy compatibility: True if ANY decoy type is compatible.

    Args:
        flat_services: (ACTION_HOST_SLOTS, NUM_SERVICES) bool
        flat_decoys: (ACTION_HOST_SLOTS, NUM_DECOY_TYPES) bool

    Returns:
        (ACTION_HOST_SLOTS,) bool — True if at least one decoy type can be deployed
    """
    has_port_25 = flat_services[:, SERVICE_IDS["SMTP"]] | flat_decoys[:, DECOY_IDS["HarakaSMPT"]]
    has_port_80 = (
        flat_services[:, SERVICE_IDS["APACHE2"]]
        | flat_decoys[:, DECOY_IDS["Apache"]]
        | flat_decoys[:, DECOY_IDS["Vsftpd"]]
    )
    # has_port_443 = flat_decoys[:, DECOY_IDS["Tomcat"]]  # Tomcat is always compatible

    # Any-compatible: Tomcat is always True, so at least one type is always available.
    # But we still gate by slot validity below, so just return True for all slots.
    # More precisely: HarakaSMPT needs ~port25, Apache needs ~port80, Tomcat always, Vsftpd needs ~port80
    # Since Tomcat is always compatible, any_compatible is always True for valid slots.
    # However, we keep the per-type check for correctness if Tomcat compatibility changes.
    per_type = jnp.stack([~has_port_25, ~has_port_80, jnp.ones(flat_services.shape[0], dtype=jnp.bool_), ~has_port_80])
    return per_type.any(axis=0)


def compute_blue_action_mask(const: CC4Const, agent_id: int, state: CC4State | None = None) -> jnp.ndarray:
    """Return bool mask of valid actions for a blue agent.

    Uses canonical (subnet, slot) encoding via obs_host_map for topology invariance.
    When `state` is provided, decoy validity reflects live services/decoys.
    Otherwise it falls back to reset-time services from `const`.

    Matches CybORG's BlueFixedActionWrapper masking: host/subnet validity and
    router exclusion only.  Pending-action state does NOT affect the mask
    (CybORG silently continues pending actions regardless of agent choice).
    """
    obs_valid = const.obs_host_map < GLOBAL_MAX_HOSTS
    agent_subnets = const.blue_agent_subnets[agent_id]
    slot_valid_flat = (obs_valid & agent_subnets[:, None]).reshape(-1)

    # Exclude router slots — CybORG's BlueFlatWrapper excludes routers from the
    # action space.  The router is always at position OBS_HOSTS_PER_SUBNET - 1
    # within each subnet.
    router_slot_mask = jnp.arange(OBS_HOSTS_PER_SUBNET) != (OBS_HOSTS_PER_SUBNET - 1)
    router_slot_mask = jnp.tile(router_slot_mask, NUM_SUBNETS)  # (ACTION_HOST_SLOTS,)
    slot_valid_flat = slot_valid_flat & router_slot_mask

    safe_host_idx = jnp.where(obs_valid, const.obs_host_map, 0)
    if state is None:
        host_services = const.initial_services[safe_host_idx]
        host_decoys = jnp.zeros(host_services.shape[:-1] + (NUM_DECOY_TYPES,), dtype=jnp.bool_)
    else:
        host_services = state.host_services[safe_host_idx]
        host_decoys = state.host_decoys[safe_host_idx]

    flat_services = host_services.reshape(ACTION_HOST_SLOTS, -1)
    flat_decoys = host_decoys.reshape(ACTION_HOST_SLOTS, -1)
    # (ACTION_HOST_SLOTS,) — True if any decoy type is compatible on this slot
    any_decoy_compat = _decoy_compat_vectorized(flat_services, flat_decoys)
    # Gate by slot validity
    decoy_mask = any_decoy_compat & slot_valid_flat

    # Traffic: agent controls dst subnet, src != dst
    src_idx = jnp.arange(NUM_SUBNETS)
    traffic_flat = (agent_subnets[None, :] & (src_idx[:, None] != jnp.arange(NUM_SUBNETS)[None, :])).reshape(-1)

    # Build mask as single concatenation: [sleep, monitor, analyse, remove, restore, decoys, block, allow]
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
