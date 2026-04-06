import jax.numpy as jnp

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_DECOY_START, BLUE_SLEEP
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
    """Return (BLUE_ALLOW_TRAFFIC_END,) bool mask of valid actions for a blue agent.

    Uses canonical (subnet, slot) encoding via obs_host_map for topology invariance.
    When `state` is provided, decoy validity reflects live services/decoys.
    Otherwise it falls back to reset-time services from `const`.
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

    if state is not None:
        busy = state.blue_pending_ticks[agent_id] > 0
        pending_action = state.blue_pending_action[agent_id]

        # When pending action is BLUE_SLEEP, the agent is busy with an
        # unsupported action (e.g. Restore on router) — mask everything.
        is_unsupported_pending = pending_action == BLUE_SLEEP

        # For pending DeployDecoy, CybORG allows ALL compatible decoy types
        # on the target host slot — not just the stored decoy type.  Build a
        # mask that enables every compatible decoy for the same host slot.
        is_decoy = (pending_action >= BLUE_DECOY_START) & (
            pending_action < BLUE_DECOY_START + ACTION_HOST_SLOTS * NUM_DECOY_TYPES
        )
        decoy_offset = pending_action - BLUE_DECOY_START
        host_slot = decoy_offset % ACTION_HOST_SLOTS

        # For non-decoy pending actions: one-hot for the stored action
        safe_action = jnp.clip(pending_action, 0, BLUE_ALLOW_TRAFFIC_END - 1)
        non_decoy_mask = jnp.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=jnp.bool_).at[safe_action].set(True)

        # For decoy pending actions: enable ALL compatible decoy types on the
        # same host slot (matching CybORG's DeployDecoy behavior).
        all_decoy_mask = jnp.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=jnp.bool_)
        for dt in range(NUM_DECOY_TYPES):
            idx = BLUE_DECOY_START + dt * ACTION_HOST_SLOTS + host_slot
            compat = decoy_mask[dt * ACTION_HOST_SLOTS + host_slot]
            all_decoy_mask = all_decoy_mask.at[idx].set(compat)
        # Always include the stored pending action itself — CybORG allows it
        # even on hosts (e.g. routers) where decoy_mask is False.
        # Applied after the loop so compat=False can't overwrite it.
        all_decoy_mask = all_decoy_mask.at[safe_action].set(True)

        pending_mask = jnp.where(is_decoy, all_decoy_mask, non_decoy_mask)

        # Unsupported pending → only Sleep is allowed (CybORG allows Sleep
        # even when the agent is busy with an unsupported action like Restore).
        unsupported_mask = jnp.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=jnp.bool_).at[BLUE_SLEEP].set(True)
        pending_mask = jnp.where(is_unsupported_pending, unsupported_mask, pending_mask)

        return jnp.where(busy, pending_mask, mask)

    return mask
