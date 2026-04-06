import jax.numpy as jnp

from jaxborg.constants import (
    BLUE_ACTION_HOST_SLOTS,
    BLUE_MAX_OBSERVED_SUBNETS,
    BLUE_TRAFFIC_SLOTS,
    GLOBAL_MAX_HOSTS,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
)
from jaxborg.state import CC4Const

RED_SLEEP = 0
RED_DISCOVER_START = 1
RED_DISCOVER_END = RED_DISCOVER_START + NUM_SUBNETS
RED_SCAN_START = RED_DISCOVER_END
RED_SCAN_END = RED_SCAN_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_SSH_START = RED_SCAN_END
RED_EXPLOIT_SSH_END = RED_EXPLOIT_SSH_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_FTP_START = RED_EXPLOIT_SSH_END
RED_EXPLOIT_FTP_END = RED_EXPLOIT_FTP_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_HTTP_START = RED_EXPLOIT_FTP_END
RED_EXPLOIT_HTTP_END = RED_EXPLOIT_HTTP_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_HTTPS_START = RED_EXPLOIT_HTTP_END
RED_EXPLOIT_HTTPS_END = RED_EXPLOIT_HTTPS_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_HARAKA_START = RED_EXPLOIT_HTTPS_END
RED_EXPLOIT_HARAKA_END = RED_EXPLOIT_HARAKA_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_SQL_START = RED_EXPLOIT_HARAKA_END
RED_EXPLOIT_SQL_END = RED_EXPLOIT_SQL_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_ETERNALBLUE_START = RED_EXPLOIT_SQL_END
RED_EXPLOIT_ETERNALBLUE_END = RED_EXPLOIT_ETERNALBLUE_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_BLUEKEEP_START = RED_EXPLOIT_ETERNALBLUE_END
RED_EXPLOIT_BLUEKEEP_END = RED_EXPLOIT_BLUEKEEP_START + GLOBAL_MAX_HOSTS
RED_PRIVESC_START = RED_EXPLOIT_BLUEKEEP_END
RED_PRIVESC_END = RED_PRIVESC_START + GLOBAL_MAX_HOSTS
RED_IMPACT_START = RED_PRIVESC_END
RED_IMPACT_END = RED_IMPACT_START + GLOBAL_MAX_HOSTS
RED_AGGRESSIVE_SCAN_START = RED_IMPACT_END
RED_AGGRESSIVE_SCAN_END = RED_AGGRESSIVE_SCAN_START + GLOBAL_MAX_HOSTS
RED_STEALTH_SCAN_START = RED_AGGRESSIVE_SCAN_END
RED_STEALTH_SCAN_END = RED_STEALTH_SCAN_START + GLOBAL_MAX_HOSTS
RED_DISCOVER_DECEPTION_START = RED_STEALTH_SCAN_END
RED_DISCOVER_DECEPTION_END = RED_DISCOVER_DECEPTION_START + GLOBAL_MAX_HOSTS
RED_DEGRADE_START = RED_DISCOVER_DECEPTION_END
RED_DEGRADE_END = RED_DEGRADE_START + GLOBAL_MAX_HOSTS
RED_WITHDRAW_START = RED_DEGRADE_END
RED_WITHDRAW_END = RED_WITHDRAW_START + GLOBAL_MAX_HOSTS

ACTION_TYPE_SLEEP = 0
ACTION_TYPE_DISCOVER = 1
ACTION_TYPE_SCAN = 2
ACTION_TYPE_EXPLOIT_SSH = 3
ACTION_TYPE_EXPLOIT_FTP = 4
ACTION_TYPE_EXPLOIT_HTTP = 5
ACTION_TYPE_EXPLOIT_HTTPS = 6
ACTION_TYPE_EXPLOIT_HARAKA = 7
ACTION_TYPE_EXPLOIT_SQL = 8
ACTION_TYPE_EXPLOIT_ETERNALBLUE = 9
ACTION_TYPE_EXPLOIT_BLUEKEEP = 10
ACTION_TYPE_PRIVESC = 11
ACTION_TYPE_IMPACT = 12
ACTION_TYPE_AGGRESSIVE_SCAN = 13
ACTION_TYPE_STEALTH_SCAN = 14
ACTION_TYPE_DISCOVER_DECEPTION = 15
ACTION_TYPE_DEGRADE = 16
ACTION_TYPE_WITHDRAW = 17

_EXPLOIT_RANGES = (
    (RED_EXPLOIT_SSH_START, RED_EXPLOIT_SSH_END, ACTION_TYPE_EXPLOIT_SSH),
    (RED_EXPLOIT_FTP_START, RED_EXPLOIT_FTP_END, ACTION_TYPE_EXPLOIT_FTP),
    (RED_EXPLOIT_HTTP_START, RED_EXPLOIT_HTTP_END, ACTION_TYPE_EXPLOIT_HTTP),
    (RED_EXPLOIT_HTTPS_START, RED_EXPLOIT_HTTPS_END, ACTION_TYPE_EXPLOIT_HTTPS),
    (RED_EXPLOIT_HARAKA_START, RED_EXPLOIT_HARAKA_END, ACTION_TYPE_EXPLOIT_HARAKA),
    (RED_EXPLOIT_SQL_START, RED_EXPLOIT_SQL_END, ACTION_TYPE_EXPLOIT_SQL),
    (RED_EXPLOIT_ETERNALBLUE_START, RED_EXPLOIT_ETERNALBLUE_END, ACTION_TYPE_EXPLOIT_ETERNALBLUE),
    (RED_EXPLOIT_BLUEKEEP_START, RED_EXPLOIT_BLUEKEEP_END, ACTION_TYPE_EXPLOIT_BLUEKEEP),
)

_ENCODE_MAP = {
    "ExploitRemoteService_cc4SSHBruteForce": RED_EXPLOIT_SSH_START,
    "ExploitRemoteService_cc4FTPDirectoryTraversal": RED_EXPLOIT_FTP_START,
    "ExploitRemoteService_cc4HTTPRFI": RED_EXPLOIT_HTTP_START,
    "ExploitRemoteService_cc4HTTPSRFI": RED_EXPLOIT_HTTPS_START,
    "ExploitRemoteService_cc4HarakaRCE": RED_EXPLOIT_HARAKA_START,
    "ExploitRemoteService_cc4SQLInjection": RED_EXPLOIT_SQL_START,
    "ExploitRemoteService_cc4EternalBlue": RED_EXPLOIT_ETERNALBLUE_START,
    "ExploitRemoteService_cc4BlueKeep": RED_EXPLOIT_BLUEKEEP_START,
    "PrivilegeEscalate": RED_PRIVESC_START,
    "Impact": RED_IMPACT_START,
    "AggressiveServiceDiscovery": RED_AGGRESSIVE_SCAN_START,
    "StealthServiceDiscovery": RED_STEALTH_SCAN_START,
    "DiscoverDeception": RED_DISCOVER_DECEPTION_START,
    "DegradeServices": RED_DEGRADE_START,
    "Withdraw": RED_WITHDRAW_START,
}

BLUE_SLEEP = 0
BLUE_MONITOR = 1
BLUE_ANALYSE_START = 2
BLUE_ANALYSE_END = BLUE_ANALYSE_START + BLUE_ACTION_HOST_SLOTS
BLUE_REMOVE_START = BLUE_ANALYSE_END
BLUE_REMOVE_END = BLUE_REMOVE_START + BLUE_ACTION_HOST_SLOTS
BLUE_RESTORE_START = BLUE_REMOVE_END
BLUE_RESTORE_END = BLUE_RESTORE_START + BLUE_ACTION_HOST_SLOTS
BLUE_DECOY_START = BLUE_RESTORE_END
BLUE_DECOY_END = BLUE_DECOY_START + BLUE_ACTION_HOST_SLOTS
BLUE_BLOCK_TRAFFIC_START = BLUE_DECOY_END
BLUE_BLOCK_TRAFFIC_END = BLUE_BLOCK_TRAFFIC_START + BLUE_TRAFFIC_SLOTS
BLUE_ALLOW_TRAFFIC_START = BLUE_BLOCK_TRAFFIC_END
BLUE_ALLOW_TRAFFIC_END = BLUE_ALLOW_TRAFFIC_START + BLUE_TRAFFIC_SLOTS

BLUE_ACTION_TYPE_SLEEP = 0
BLUE_ACTION_TYPE_MONITOR = 1
BLUE_ACTION_TYPE_ANALYSE = 2
BLUE_ACTION_TYPE_REMOVE = 3
BLUE_ACTION_TYPE_RESTORE = 4
BLUE_ACTION_TYPE_DECOY = 5
BLUE_ACTION_TYPE_BLOCK_TRAFFIC = 6
BLUE_ACTION_TYPE_ALLOW_TRAFFIC = 7


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


def get_red_action_duration(action_idx: int, const: CC4Const) -> jnp.int32:
    action_type, _, _ = decode_red_action(action_idx, 0, const)
    return RED_ACTION_DURATIONS[action_type]


def get_blue_action_duration(action_idx: int, const: CC4Const) -> jnp.int32:
    action_type, _, _, _, _ = decode_blue_action(action_idx, 0, const)
    return BLUE_ACTION_DURATIONS[action_type]


def encode_red_action(action_name: str, target: int, agent_id: int) -> int:
    if action_name == "Sleep":
        return RED_SLEEP
    if action_name == "DiscoverRemoteSystems":
        return RED_DISCOVER_START + target
    if action_name == "DiscoverNetworkServices":
        return RED_SCAN_START + target
    base = _ENCODE_MAP.get(action_name)
    if base is not None:
        return base + target
    raise NotImplementedError(f"Unknown red action {action_name}")


def decode_red_action(action_idx: int, agent_id: int, const: CC4Const):
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


_BLUE_ENCODE_MAP = {
    "Analyse": BLUE_ANALYSE_START,
    "Remove": BLUE_REMOVE_START,
    "Restore": BLUE_RESTORE_START,
}

_BLUE_DECOY_ENCODE_NAMES = {
    "DeployDecoy",
    "DeployDecoy_HarakaSMPT",
    "DeployDecoy_Apache",
    "DeployDecoy_Tomcat",
    "DeployDecoy_Vsftpd",
}


def _global_host_to_relative_slot(const: CC4Const, global_host: int, agent_id: int) -> int:
    """Convert a global host index to an agent-relative slot.

    Returns relative_subnet_idx * OBS_HOSTS_PER_SUBNET + slot_within,
    where relative_subnet_idx is 0, 1, or 2 indexing into
    const.blue_obs_subnets[agent_id].

    Returns -1 if the host's subnet is not in the agent's observed subnets
    or the host is not in obs_host_map.
    """
    sid = int(const.host_subnet[global_host])
    # Find which relative index (0, 1, 2) this subnet maps to for this agent
    rel_idx = -1
    for i in range(BLUE_MAX_OBSERVED_SUBNETS):
        if int(const.blue_obs_subnets[agent_id, i]) == sid:
            rel_idx = i
            break
    if rel_idx < 0:
        return -1
    for slot in range(OBS_HOSTS_PER_SUBNET):
        if int(const.obs_host_map[sid, slot]) == global_host:
            return rel_idx * OBS_HOSTS_PER_SUBNET + slot
    return -1


def _abs_subnet_to_relative(const: CC4Const, subnet_id: int, agent_id: int) -> int:
    """Convert an absolute subnet ID to an agent-relative index (0, 1, or 2).

    Returns -1 if the subnet is not in the agent's observed subnets.
    """
    for i in range(BLUE_MAX_OBSERVED_SUBNETS):
        if int(const.blue_obs_subnets[agent_id, i]) == subnet_id:
            return i
    return -1


def encode_blue_action(
    action_name: str,
    target_host: int,
    agent_id: int,
    *,
    const: CC4Const = None,
    src_subnet: int = -1,
    dst_subnet: int = -1,
) -> int:
    if action_name == "Sleep":
        return BLUE_SLEEP
    if action_name == "Monitor":
        return BLUE_MONITOR
    base = _BLUE_ENCODE_MAP.get(action_name)
    if base is not None:
        slot = _global_host_to_relative_slot(const, target_host, agent_id)
        if slot < 0:
            return BLUE_SLEEP
        return base + slot
    if action_name in _BLUE_DECOY_ENCODE_NAMES:
        slot = _global_host_to_relative_slot(const, target_host, agent_id)
        if slot < 0:
            return BLUE_SLEEP
        return BLUE_DECOY_START + slot
    if action_name == "BlockTrafficZone":
        rel_dst = _abs_subnet_to_relative(const, dst_subnet, agent_id)
        if rel_dst < 0:
            return BLUE_SLEEP
        return BLUE_BLOCK_TRAFFIC_START + src_subnet * BLUE_MAX_OBSERVED_SUBNETS + rel_dst
    if action_name == "AllowTrafficZone":
        rel_dst = _abs_subnet_to_relative(const, dst_subnet, agent_id)
        if rel_dst < 0:
            return BLUE_SLEEP
        return BLUE_ALLOW_TRAFFIC_START + src_subnet * BLUE_MAX_OBSERVED_SUBNETS + rel_dst
    raise NotImplementedError(f"Unknown blue action {action_name}")


def _slot_to_global_host(const: CC4Const, relative_slot, agent_id):
    """Resolve an agent-relative slot to a global host index via obs_host_map.

    relative_slot = relative_subnet_idx * OBS_HOSTS_PER_SUBNET + slot_within
    where relative_subnet_idx indexes into const.blue_obs_subnets[agent_id].
    """
    relative_subnet = relative_slot // OBS_HOSTS_PER_SUBNET
    slot_within = relative_slot % OBS_HOSTS_PER_SUBNET
    subnet_id = const.blue_obs_subnets[agent_id, relative_subnet]
    # Clamp to valid range for safe indexing; invalid subnets (-1) will
    # resolve to GLOBAL_MAX_HOSTS (sentinel) via obs_host_map padding.
    safe_subnet = jnp.clip(subnet_id, 0, NUM_SUBNETS - 1)
    host = const.obs_host_map[safe_subnet, slot_within]
    # If the subnet was invalid (-1), force result to -1
    return jnp.where(subnet_id >= 0, host, jnp.int32(-1))


def decode_blue_action(action_idx: int, agent_id: int, const: CC4Const):
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

    # Traffic: offset = src_subnet * BLUE_MAX_OBSERVED_SUBNETS + relative_dst
    traffic_offset_block = action_idx - BLUE_BLOCK_TRAFFIC_START
    traffic_offset_allow = action_idx - BLUE_ALLOW_TRAFFIC_START
    src_subnet = jnp.int32(-1)
    dst_subnet = jnp.int32(-1)
    # Block traffic
    block_src = traffic_offset_block // BLUE_MAX_OBSERVED_SUBNETS
    block_rel_dst = traffic_offset_block % BLUE_MAX_OBSERVED_SUBNETS
    block_dst = const.blue_obs_subnets[agent_id, block_rel_dst]
    src_subnet = jnp.where(is_block, block_src, src_subnet)
    dst_subnet = jnp.where(is_block, block_dst, dst_subnet)
    # Allow traffic
    allow_src = traffic_offset_allow // BLUE_MAX_OBSERVED_SUBNETS
    allow_rel_dst = traffic_offset_allow % BLUE_MAX_OBSERVED_SUBNETS
    allow_dst = const.blue_obs_subnets[agent_id, allow_rel_dst]
    src_subnet = jnp.where(is_allow, allow_src, src_subnet)
    dst_subnet = jnp.where(is_allow, allow_dst, dst_subnet)

    return action_type, target_host, decoy_type, src_subnet, dst_subnet
