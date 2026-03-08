import jax.numpy as jnp

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.constants import (
    ACTION_HOST_SLOTS,
    DECOY_IDS,
    GLOBAL_MAX_HOSTS,
    NUM_DECOY_TYPES,
    NUM_SUBNETS,
    SERVICE_IDS,
)
from jaxborg.state import CC4Const, CC4State


def _decoy_compat_vectorized(flat_services: jnp.ndarray, flat_decoys: jnp.ndarray) -> jnp.ndarray:
    """Compute per-decoy compatibility for all host slots at once.

    Args:
        flat_services: (ACTION_HOST_SLOTS, NUM_SERVICES) bool
        flat_decoys: (ACTION_HOST_SLOTS, NUM_DECOY_TYPES) bool

    Returns:
        (NUM_DECOY_TYPES, ACTION_HOST_SLOTS) bool — decoy_type-major for concat
    """
    has_port_25 = flat_services[:, SERVICE_IDS["SMTP"]] | flat_decoys[:, DECOY_IDS["HarakaSMPT"]]
    has_port_80 = (
        flat_services[:, SERVICE_IDS["APACHE2"]]
        | flat_decoys[:, DECOY_IDS["Apache"]]
        | flat_decoys[:, DECOY_IDS["Vsftpd"]]
    )
    has_port_443 = flat_decoys[:, DECOY_IDS["Tomcat"]]

    return jnp.stack([~has_port_25, ~has_port_80, ~has_port_443, jnp.ones(flat_services.shape[0], dtype=jnp.bool_)])


def compute_blue_action_mask(const: CC4Const, agent_id: int, state: CC4State | None = None) -> jnp.ndarray:
    """Return (BLUE_ALLOW_TRAFFIC_END,) bool mask of valid actions for a blue agent.

    Uses canonical (subnet, slot) encoding via obs_host_map for topology invariance.
    When `state` is provided, decoy validity reflects live services/decoys.
    Otherwise it falls back to reset-time services from `const`.
    """
    obs_valid = const.obs_host_map < GLOBAL_MAX_HOSTS
    agent_subnets = const.blue_agent_subnets[agent_id]
    slot_valid_flat = (obs_valid & agent_subnets[:, None]).reshape(-1)

    safe_host_idx = jnp.where(obs_valid, const.obs_host_map, 0)
    if state is None:
        host_services = const.initial_services[safe_host_idx]
        host_decoys = jnp.zeros(host_services.shape[:-1] + (NUM_DECOY_TYPES,), dtype=jnp.bool_)
    else:
        host_services = state.host_services[safe_host_idx]
        host_decoys = state.host_decoys[safe_host_idx]

    flat_services = host_services.reshape(ACTION_HOST_SLOTS, -1)
    flat_decoys = host_decoys.reshape(ACTION_HOST_SLOTS, -1)
    # (NUM_DECOY_TYPES, ACTION_HOST_SLOTS) — decoy-type major
    decoy_compat = _decoy_compat_vectorized(flat_services, flat_decoys)
    # Gate each decoy type by slot validity
    decoy_mask = (decoy_compat & slot_valid_flat[None, :]).reshape(-1)

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
            decoy_mask,  # decoys (4 types x 144 slots)
            traffic_flat,  # block traffic
            traffic_flat,  # allow traffic
        ]
    )

    if state is not None:
        busy = state.blue_pending_ticks[agent_id] > 0
        pending_action = jnp.clip(state.blue_pending_action[agent_id], 0, BLUE_ALLOW_TRAFFIC_END - 1)
        pending_mask = jnp.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=jnp.bool_).at[pending_action].set(True)
        return jnp.where(busy, pending_mask, mask)

    return mask
