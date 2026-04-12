"""Pure-Python action encoding constants and encode functions.

This module is intentionally free of JAX imports so that it can be used
in multiprocessing workers that only need CybORG + NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxborg.constants import (
    BLUE_ACTION_HOST_SLOTS,
    BLUE_MAX_OBSERVED_SUBNETS,
    BLUE_TRAFFIC_SLOTS,
    GLOBAL_MAX_HOSTS,
    NUM_SUBNETS,
    OBS_VECTOR_HOSTS_PER_SUBNET,
)

if TYPE_CHECKING:
    from jaxborg.state import CC4Const

# ---------------------------------------------------------------------------
# Red action ranges
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Red action type IDs
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Blue action ranges
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Blue action type IDs
# ---------------------------------------------------------------------------

BLUE_ACTION_TYPE_SLEEP = 0
BLUE_ACTION_TYPE_MONITOR = 1
BLUE_ACTION_TYPE_ANALYSE = 2
BLUE_ACTION_TYPE_REMOVE = 3
BLUE_ACTION_TYPE_RESTORE = 4
BLUE_ACTION_TYPE_DECOY = 5
BLUE_ACTION_TYPE_BLOCK_TRAFFIC = 6
BLUE_ACTION_TYPE_ALLOW_TRAFFIC = 7

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


# ---------------------------------------------------------------------------
# Encode functions (pure Python — no JAX)
# ---------------------------------------------------------------------------


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


def _global_host_to_relative_slot(const: CC4Const, global_host: int, agent_id: int) -> int:
    """Convert a global host index to an agent-relative slot.

    Returns relative_subnet_idx * OBS_VECTOR_HOSTS_PER_SUBNET + slot_within,
    where relative_subnet_idx is 0, 1, or 2 indexing into
    const.blue_obs_subnets[agent_id].

    Only searches slots 0..OBS_VECTOR_HOSTS_PER_SUBNET-1 (excludes router at
    slot 16), so router hosts return -1.

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
    for slot in range(OBS_VECTOR_HOSTS_PER_SUBNET):
        if int(const.obs_host_map[sid, slot]) == global_host:
            return rel_idx * OBS_VECTOR_HOSTS_PER_SUBNET + slot
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
        if rel_dst < 0 or src_subnet == dst_subnet:
            return BLUE_SLEEP
        src_offset = src_subnet if src_subnet < dst_subnet else src_subnet - 1
        return BLUE_BLOCK_TRAFFIC_START + src_offset * BLUE_MAX_OBSERVED_SUBNETS + rel_dst
    if action_name == "AllowTrafficZone":
        rel_dst = _abs_subnet_to_relative(const, dst_subnet, agent_id)
        if rel_dst < 0 or src_subnet == dst_subnet:
            return BLUE_SLEEP
        src_offset = src_subnet if src_subnet < dst_subnet else src_subnet - 1
        return BLUE_ALLOW_TRAFFIC_START + src_offset * BLUE_MAX_OBSERVED_SUBNETS + rel_dst
    raise NotImplementedError(f"Unknown blue action {action_name}")
