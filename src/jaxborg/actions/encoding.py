import jax.numpy as jnp

from jaxborg.actions.action_defs import (  # noqa: F401
    _EXPLOIT_RANGES,
    ACTION_TYPE_AGGRESSIVE_SCAN,
    ACTION_TYPE_DEGRADE,
    ACTION_TYPE_DISCOVER,
    ACTION_TYPE_DISCOVER_DECEPTION,
    ACTION_TYPE_EXPLOIT_BLUEKEEP,
    ACTION_TYPE_EXPLOIT_ETERNALBLUE,
    ACTION_TYPE_EXPLOIT_FTP,
    ACTION_TYPE_EXPLOIT_HARAKA,
    ACTION_TYPE_EXPLOIT_HTTP,
    ACTION_TYPE_EXPLOIT_HTTPS,
    ACTION_TYPE_EXPLOIT_SQL,
    ACTION_TYPE_EXPLOIT_SSH,
    ACTION_TYPE_IMPACT,
    ACTION_TYPE_PRIVESC,
    ACTION_TYPE_SCAN,
    ACTION_TYPE_SLEEP,
    ACTION_TYPE_STEALTH_SCAN,
    ACTION_TYPE_WITHDRAW,
    BLUE_ACTION_TYPE_ALLOW_TRAFFIC,
    BLUE_ACTION_TYPE_ANALYSE,
    BLUE_ACTION_TYPE_BLOCK_TRAFFIC,
    BLUE_ACTION_TYPE_DECOY,
    BLUE_ACTION_TYPE_MONITOR,
    BLUE_ACTION_TYPE_REMOVE,
    BLUE_ACTION_TYPE_RESTORE,
    BLUE_ACTION_TYPE_SLEEP,
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_ANALYSE_END,
    BLUE_ANALYSE_START,
    BLUE_BLOCK_TRAFFIC_END,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_DECOY_END,
    BLUE_DECOY_START,
    BLUE_MONITOR,
    BLUE_REMOVE_END,
    BLUE_REMOVE_START,
    BLUE_RESTORE_END,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
    RED_AGGRESSIVE_SCAN_END,
    RED_AGGRESSIVE_SCAN_START,
    RED_DEGRADE_END,
    RED_DEGRADE_START,
    RED_DISCOVER_DECEPTION_END,
    RED_DISCOVER_DECEPTION_START,
    RED_DISCOVER_END,
    RED_DISCOVER_START,
    RED_EXPLOIT_BLUEKEEP_END,
    RED_EXPLOIT_BLUEKEEP_START,
    RED_EXPLOIT_ETERNALBLUE_END,
    RED_EXPLOIT_ETERNALBLUE_START,
    RED_EXPLOIT_FTP_END,
    RED_EXPLOIT_FTP_START,
    RED_EXPLOIT_HARAKA_END,
    RED_EXPLOIT_HARAKA_START,
    RED_EXPLOIT_HTTP_END,
    RED_EXPLOIT_HTTP_START,
    RED_EXPLOIT_HTTPS_END,
    RED_EXPLOIT_HTTPS_START,
    RED_EXPLOIT_SQL_END,
    RED_EXPLOIT_SQL_START,
    RED_EXPLOIT_SSH_END,
    RED_EXPLOIT_SSH_START,
    RED_IMPACT_END,
    RED_IMPACT_START,
    RED_PRIVESC_END,
    RED_PRIVESC_START,
    RED_SCAN_END,
    RED_SCAN_START,
    RED_SLEEP,
    RED_STEALTH_SCAN_END,
    RED_STEALTH_SCAN_START,
    RED_WITHDRAW_END,
    RED_WITHDRAW_START,
    encode_blue_action,
    encode_red_action,
)
from jaxborg.constants import BLUE_MAX_OBSERVED_SUBNETS, NUM_SUBNETS, OBS_VECTOR_HOSTS_PER_SUBNET
from jaxborg.state import SimulatorConst

RED_ACTION_DURATIONS = jnp.array(
    #  Sleep Discover Scan  SSH  FTP  HTTP HTTPS Haraka SQL  EBlue BKeep PEsc Impact AggSc StlSc DcDec Degrd Withd
    [1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 1, 3, 2, 2, 1],
    dtype=jnp.int32,
)

BLUE_ACTION_DURATIONS = jnp.array(
    #  Sleep Monitor Analyse Remove Restore Decoy Block Allow
    [1, 1, 2, 3, 5, 2, 1, 1],
    dtype=jnp.int32,
)


def get_red_action_duration(action_idx: int, const: SimulatorConst) -> jnp.int32:
    action_type, _, _ = decode_red_action(action_idx, 0, const)
    return RED_ACTION_DURATIONS[action_type]


def get_blue_action_duration(action_idx: int, const: SimulatorConst) -> jnp.int32:
    action_type, _, _, _, _ = decode_blue_action(action_idx, 0, const)
    return BLUE_ACTION_DURATIONS[action_type]


def decode_red_action(action_idx: int, agent_id: int, const: SimulatorConst):
    is_discover = (action_idx >= RED_DISCOVER_START) & (action_idx < RED_DISCOVER_END)
    is_scan = (action_idx >= RED_SCAN_START) & (action_idx < RED_SCAN_END)

    action_type = jnp.where(is_discover, ACTION_TYPE_DISCOVER, jnp.where(is_scan, ACTION_TYPE_SCAN, ACTION_TYPE_SLEEP))
    target_host = jnp.where(is_scan, action_idx - RED_SCAN_START, jnp.int32(-1))

    for start, end, atype in _EXPLOIT_RANGES:
        in_range = (action_idx >= start) & (action_idx < end)
        action_type = jnp.where(in_range, atype, action_type)
        target_host = jnp.where(in_range, action_idx - start, target_host)

    is_privesc = (action_idx >= RED_PRIVESC_START) & (action_idx < RED_PRIVESC_END)
    action_type = jnp.where(is_privesc, ACTION_TYPE_PRIVESC, action_type)
    target_host = jnp.where(is_privesc, action_idx - RED_PRIVESC_START, target_host)

    is_impact = (action_idx >= RED_IMPACT_START) & (action_idx < RED_IMPACT_END)
    action_type = jnp.where(is_impact, ACTION_TYPE_IMPACT, action_type)
    target_host = jnp.where(is_impact, action_idx - RED_IMPACT_START, target_host)

    is_aggressive_scan = (action_idx >= RED_AGGRESSIVE_SCAN_START) & (action_idx < RED_AGGRESSIVE_SCAN_END)
    action_type = jnp.where(is_aggressive_scan, ACTION_TYPE_AGGRESSIVE_SCAN, action_type)
    target_host = jnp.where(is_aggressive_scan, action_idx - RED_AGGRESSIVE_SCAN_START, target_host)

    is_stealth_scan = (action_idx >= RED_STEALTH_SCAN_START) & (action_idx < RED_STEALTH_SCAN_END)
    action_type = jnp.where(is_stealth_scan, ACTION_TYPE_STEALTH_SCAN, action_type)
    target_host = jnp.where(is_stealth_scan, action_idx - RED_STEALTH_SCAN_START, target_host)

    is_discover_deception = (action_idx >= RED_DISCOVER_DECEPTION_START) & (action_idx < RED_DISCOVER_DECEPTION_END)
    action_type = jnp.where(is_discover_deception, ACTION_TYPE_DISCOVER_DECEPTION, action_type)
    target_host = jnp.where(is_discover_deception, action_idx - RED_DISCOVER_DECEPTION_START, target_host)

    is_degrade = (action_idx >= RED_DEGRADE_START) & (action_idx < RED_DEGRADE_END)
    action_type = jnp.where(is_degrade, ACTION_TYPE_DEGRADE, action_type)
    target_host = jnp.where(is_degrade, action_idx - RED_DEGRADE_START, target_host)

    is_withdraw = (action_idx >= RED_WITHDRAW_START) & (action_idx < RED_WITHDRAW_END)
    action_type = jnp.where(is_withdraw, ACTION_TYPE_WITHDRAW, action_type)
    target_host = jnp.where(is_withdraw, action_idx - RED_WITHDRAW_START, target_host)

    target_subnet = jnp.where(is_discover, action_idx - RED_DISCOVER_START, jnp.int32(-1))
    return action_type, target_subnet, target_host


def _slot_to_global_host(const: SimulatorConst, relative_slot, agent_id):
    """Resolve an agent-relative slot to a global host index via obs_host_map.

    relative_slot = relative_subnet_idx * OBS_VECTOR_HOSTS_PER_SUBNET + slot_within
    where relative_subnet_idx indexes into const.blue_obs_subnets[agent_id].
    """
    relative_subnet = relative_slot // OBS_VECTOR_HOSTS_PER_SUBNET
    slot_within = relative_slot % OBS_VECTOR_HOSTS_PER_SUBNET
    subnet_id = const.blue_obs_subnets[agent_id, relative_subnet]
    # Clamp to valid range for safe indexing; invalid subnets (-1) will
    # resolve to GLOBAL_MAX_HOSTS (sentinel) via obs_host_map padding.
    safe_subnet = jnp.clip(subnet_id, 0, NUM_SUBNETS - 1)
    host = const.obs_host_map[safe_subnet, slot_within]
    # If the subnet was invalid (-1), force result to -1
    return jnp.where(subnet_id >= 0, host, jnp.int32(-1))


def decode_blue_action(action_idx: int, agent_id: int, const: SimulatorConst):
    is_analyse = (action_idx >= BLUE_ANALYSE_START) & (action_idx < BLUE_ANALYSE_END)
    is_remove = (action_idx >= BLUE_REMOVE_START) & (action_idx < BLUE_REMOVE_END)
    is_restore = (action_idx >= BLUE_RESTORE_START) & (action_idx < BLUE_RESTORE_END)
    is_decoy = (action_idx >= BLUE_DECOY_START) & (action_idx < BLUE_DECOY_END)
    is_block = (action_idx >= BLUE_BLOCK_TRAFFIC_START) & (action_idx < BLUE_BLOCK_TRAFFIC_END)
    is_allow = (action_idx >= BLUE_ALLOW_TRAFFIC_START) & (action_idx < BLUE_ALLOW_TRAFFIC_END)

    action_type = jnp.where(action_idx == BLUE_MONITOR, BLUE_ACTION_TYPE_MONITOR, BLUE_ACTION_TYPE_SLEEP)
    action_type = jnp.where(is_analyse, BLUE_ACTION_TYPE_ANALYSE, action_type)
    action_type = jnp.where(is_remove, BLUE_ACTION_TYPE_REMOVE, action_type)
    action_type = jnp.where(is_restore, BLUE_ACTION_TYPE_RESTORE, action_type)
    action_type = jnp.where(is_decoy, BLUE_ACTION_TYPE_DECOY, action_type)
    action_type = jnp.where(is_block, BLUE_ACTION_TYPE_BLOCK_TRAFFIC, action_type)
    action_type = jnp.where(is_allow, BLUE_ACTION_TYPE_ALLOW_TRAFFIC, action_type)

    # Resolve agent-relative slot → global host via blue_obs_subnets + obs_host_map
    flat_slot = jnp.int32(0)
    flat_slot = jnp.where(is_analyse, action_idx - BLUE_ANALYSE_START, flat_slot)
    flat_slot = jnp.where(is_remove, action_idx - BLUE_REMOVE_START, flat_slot)
    flat_slot = jnp.where(is_restore, action_idx - BLUE_RESTORE_START, flat_slot)

    decoy_type = jnp.int32(-1)  # type selected at execution time
    flat_slot = jnp.where(is_decoy, action_idx - BLUE_DECOY_START, flat_slot)

    is_host_action = is_analyse | is_remove | is_restore | is_decoy
    target_host = jnp.where(is_host_action, _slot_to_global_host(const, flat_slot, agent_id), jnp.int32(-1))

    # Traffic: offset = src_offset * BLUE_MAX_OBSERVED_SUBNETS + relative_dst
    # src_offset is compressed (0..NUM_SUBNETS-2), skipping the self-loop (src==dst).
    traffic_offset_block = action_idx - BLUE_BLOCK_TRAFFIC_START
    traffic_offset_allow = action_idx - BLUE_ALLOW_TRAFFIC_START
    src_subnet = jnp.int32(-1)
    dst_subnet = jnp.int32(-1)
    # Block traffic
    block_src_offset = traffic_offset_block // BLUE_MAX_OBSERVED_SUBNETS
    block_rel_dst = traffic_offset_block % BLUE_MAX_OBSERVED_SUBNETS
    block_dst = const.blue_obs_subnets[agent_id, block_rel_dst]
    block_src = jnp.where(block_src_offset >= block_dst, block_src_offset + 1, block_src_offset)
    src_subnet = jnp.where(is_block, block_src, src_subnet)
    dst_subnet = jnp.where(is_block, block_dst, dst_subnet)
    # Allow traffic
    allow_src_offset = traffic_offset_allow // BLUE_MAX_OBSERVED_SUBNETS
    allow_rel_dst = traffic_offset_allow % BLUE_MAX_OBSERVED_SUBNETS
    allow_dst = const.blue_obs_subnets[agent_id, allow_rel_dst]
    allow_src = jnp.where(allow_src_offset >= allow_dst, allow_src_offset + 1, allow_src_offset)
    src_subnet = jnp.where(is_allow, allow_src, src_subnet)
    dst_subnet = jnp.where(is_allow, allow_dst, dst_subnet)

    return action_type, target_host, decoy_type, src_subnet, dst_subnet
