from dataclasses import dataclass

from jaxborg.actions.encoding import (
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
from jaxborg.constants import BLUE_MAX_OBSERVED_SUBNETS, GLOBAL_MAX_HOSTS, OBS_VECTOR_HOSTS_PER_SUBNET
from jaxborg.topology import CYBORG_SUBNET_SUFFIX, CYBORG_SUFFIX_TO_ID


@dataclass
class CC4Mappings:
    hostname_to_idx: dict
    idx_to_hostname: dict
    ip_to_hostname: dict
    hostname_to_ip: dict
    subnet_cidrs: dict
    subnet_names: dict
    cidr_to_subnet_idx: dict
    num_hosts: int


def build_mappings_from_cyborg(cyborg_env) -> CC4Mappings:
    state = cyborg_env.environment_controller.state

    sorted_hostnames = sorted(state.hosts.keys())
    hostname_to_idx = {h: i for i, h in enumerate(sorted_hostnames)}
    idx_to_hostname = {i: h for h, i in hostname_to_idx.items()}

    ip_to_hostname = dict(state.ip_addresses)
    hostname_to_ip = {v: k for k, v in state.ip_addresses.items()}

    subnet_cidrs = {}
    subnet_names = {}
    cidr_to_subnet_idx = {}
    for jax_name, cyborg_suffix in CYBORG_SUBNET_SUFFIX.items():
        sid = CYBORG_SUFFIX_TO_ID[cyborg_suffix]
        subnet_names[sid] = cyborg_suffix
        if cyborg_suffix in state.subnet_name_to_cidr:
            cidr = state.subnet_name_to_cidr[cyborg_suffix]
            subnet_cidrs[sid] = cidr
            cidr_to_subnet_idx[cidr] = sid

    return CC4Mappings(
        hostname_to_idx=hostname_to_idx,
        idx_to_hostname=idx_to_hostname,
        ip_to_hostname=ip_to_hostname,
        hostname_to_ip=hostname_to_ip,
        subnet_cidrs=subnet_cidrs,
        subnet_names=subnet_names,
        cidr_to_subnet_idx=cidr_to_subnet_idx,
        num_hosts=len(sorted_hostnames),
    )


def _slot_to_global(flat_slot: int, const, agent_id: int) -> int:
    """Resolve an agent-relative flat slot to a global host index.

    flat_slot = relative_subnet_idx * OBS_VECTOR_HOSTS_PER_SUBNET + slot_within,
    where relative_subnet_idx indexes into const.blue_obs_subnets[agent_id].
    """
    relative_subnet = flat_slot // OBS_VECTOR_HOSTS_PER_SUBNET
    slot_within = flat_slot % OBS_VECTOR_HOSTS_PER_SUBNET
    sid = int(const.blue_obs_subnets[agent_id, relative_subnet])
    return int(const.obs_host_map[sid, slot_within])


_RED_HOST_RANGES = [
    (RED_SCAN_START, RED_SCAN_END, "DiscoverNetworkServices"),
    (RED_EXPLOIT_SSH_START, RED_EXPLOIT_SSH_END, "ExploitRemoteService_cc4SSHBruteForce"),
    (RED_EXPLOIT_FTP_START, RED_EXPLOIT_FTP_END, "ExploitRemoteService_cc4FTPDirectoryTraversal"),
    (RED_EXPLOIT_HTTP_START, RED_EXPLOIT_HTTP_END, "ExploitRemoteService_cc4HTTPRFI"),
    (RED_EXPLOIT_HTTPS_START, RED_EXPLOIT_HTTPS_END, "ExploitRemoteService_cc4HTTPSRFI"),
    (RED_EXPLOIT_HARAKA_START, RED_EXPLOIT_HARAKA_END, "ExploitRemoteService_cc4HarakaRCE"),
    (RED_EXPLOIT_SQL_START, RED_EXPLOIT_SQL_END, "ExploitRemoteService_cc4SQLInjection"),
    (RED_EXPLOIT_ETERNALBLUE_START, RED_EXPLOIT_ETERNALBLUE_END, "ExploitRemoteService_cc4EternalBlue"),
    (RED_EXPLOIT_BLUEKEEP_START, RED_EXPLOIT_BLUEKEEP_END, "ExploitRemoteService_cc4BlueKeep"),
    (RED_PRIVESC_START, RED_PRIVESC_END, "PrivilegeEscalate"),
    (RED_IMPACT_START, RED_IMPACT_END, "Impact"),
    (RED_AGGRESSIVE_SCAN_START, RED_AGGRESSIVE_SCAN_END, "AggressiveServiceDiscovery"),
    (RED_STEALTH_SCAN_START, RED_STEALTH_SCAN_END, "StealthServiceDiscovery"),
    (RED_DISCOVER_DECEPTION_START, RED_DISCOVER_DECEPTION_END, "DiscoverDeception"),
    (RED_DEGRADE_START, RED_DEGRADE_END, "DegradeServices"),
    (RED_WITHDRAW_START, RED_WITHDRAW_END, "Withdraw"),
]

_CYBORG_EXPLOIT_TO_ENCODE = {
    "SSHBruteForce": "ExploitRemoteService_cc4SSHBruteForce",
    "FTPDirectoryTraversal": "ExploitRemoteService_cc4FTPDirectoryTraversal",
    "HTTPRFI": "ExploitRemoteService_cc4HTTPRFI",
    "HTTPSRFI": "ExploitRemoteService_cc4HTTPSRFI",
    "HarakaRCE": "ExploitRemoteService_cc4HarakaRCE",
    "SQLInjection": "ExploitRemoteService_cc4SQLInjection",
    "EternalBlue": "ExploitRemoteService_cc4EternalBlue",
    "BlueKeep": "ExploitRemoteService_cc4BlueKeep",
}

_ENCODE_TO_CYBORG_EXPLOIT = {v: k for k, v in _CYBORG_EXPLOIT_TO_ENCODE.items()}

_CYBORG_BLUE_DECOY_NAMES = {
    "DecoyHarakaSMPT": "DeployDecoy_HarakaSMPT",
    "DecoyApache": "DeployDecoy_Apache",
    "DecoyTomcat": "DeployDecoy_Tomcat",
    "DecoyVsftpd": "DeployDecoy_Vsftpd",
}


class _FixedExploitSelector:
    """Forces ExploitRemoteService to use a specific concrete exploit class."""

    def __init__(self, exploit_class):
        self._cls = exploit_class

    def get_exploit_action(self, *, state, session, agent, ip_address, priority=None):
        return self._cls(session=session, agent=agent, ip_address=ip_address)


def _build_exploit_range_map():
    from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.BlueKeep import BlueKeep
    from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.EternalBlue import EternalBlue
    from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.FTPDirectoryTraversal import FTPDirectoryTraversal
    from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.HarakaRCE import HarakaRCE
    from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.HTTPRFI import HTTPRFI
    from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.HTTPSRFI import HTTPSRFI
    from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.SQLInjection import SQLInjection
    from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.SSHBruteForce import SSHBruteForce

    return [
        (RED_EXPLOIT_SSH_START, SSHBruteForce),
        (RED_EXPLOIT_FTP_START, FTPDirectoryTraversal),
        (RED_EXPLOIT_HTTP_START, HTTPRFI),
        (RED_EXPLOIT_HTTPS_START, HTTPSRFI),
        (RED_EXPLOIT_HARAKA_START, HarakaRCE),
        (RED_EXPLOIT_SQL_START, SQLInjection),
        (RED_EXPLOIT_ETERNALBLUE_START, EternalBlue),
        (RED_EXPLOIT_BLUEKEEP_START, BlueKeep),
    ]


_EXPLOIT_RANGE_TO_CLASS = None


def _get_exploit_range_map():
    global _EXPLOIT_RANGE_TO_CLASS
    if _EXPLOIT_RANGE_TO_CLASS is None:
        _EXPLOIT_RANGE_TO_CLASS = _build_exploit_range_map()
    return _EXPLOIT_RANGE_TO_CLASS


def _agent_idx(agent_name: str) -> int:
    return int(agent_name.split("_")[-1])


def _host_idx_from_ip(ip, mappings: CC4Mappings) -> int:
    hostname = mappings.ip_to_hostname[ip]
    return mappings.hostname_to_idx[hostname]


def _host_idx_from_hostname(hostname: str, mappings: CC4Mappings) -> int:
    return mappings.hostname_to_idx[hostname]


def cyborg_red_to_jax(action, agent_name: str, mappings: CC4Mappings) -> int:
    cls_name = type(action).__name__
    aid = _agent_idx(agent_name)

    if cls_name == "Sleep":
        return RED_SLEEP

    if cls_name == "DiscoverRemoteSystems":
        sid = mappings.cidr_to_subnet_idx[action.subnet]
        return encode_red_action("DiscoverRemoteSystems", sid, aid)

    if cls_name == "DiscoverNetworkServices":
        return encode_red_action("DiscoverNetworkServices", _host_idx_from_ip(action.ip_address, mappings), aid)

    if cls_name in _CYBORG_EXPLOIT_TO_ENCODE:
        return encode_red_action(
            _CYBORG_EXPLOIT_TO_ENCODE[cls_name],
            _host_idx_from_ip(action.ip_address, mappings),
            aid,
        )

    if cls_name == "ExploitRemoteService":
        return encode_red_action(
            "ExploitRemoteService_cc4SSHBruteForce",
            _host_idx_from_ip(action.ip_address, mappings),
            aid,
        )

    ip_actions = {"AggressiveServiceDiscovery", "StealthServiceDiscovery", "DiscoverDeception"}
    if cls_name in ip_actions:
        return encode_red_action(cls_name, _host_idx_from_ip(action.ip_address, mappings), aid)

    hostname_actions = {"PrivilegeEscalate", "Impact", "DegradeServices"}
    if cls_name in hostname_actions:
        return encode_red_action(cls_name, _host_idx_from_hostname(action.hostname, mappings), aid)

    if cls_name == "Withdraw":
        return encode_red_action("Withdraw", _host_idx_from_hostname(action.hostname, mappings), aid)

    raise ValueError(f"Unknown CybORG red action class: {cls_name}")


def jax_red_to_cyborg(action_idx: int, agent_id: int, mappings: CC4Mappings, session_override=None):
    from CybORG.Simulator.Actions import (
        AggressiveServiceDiscovery,
        DegradeServices,
        DiscoverDeception,
        DiscoverNetworkServices,
        DiscoverRemoteSystems,
        Impact,
        PrivilegeEscalate,
        Sleep,
        StealthServiceDiscovery,
        Withdraw,
    )

    agent_name = f"red_agent_{agent_id}"
    session = session_override if session_override is not None else 0

    if action_idx == RED_SLEEP:
        return Sleep()

    if RED_DISCOVER_START <= action_idx < RED_DISCOVER_END:
        subnet_idx = action_idx - RED_DISCOVER_START
        return DiscoverRemoteSystems(subnet=mappings.subnet_cidrs[subnet_idx], session=session, agent=agent_name)

    _ip_class_map = {
        "DiscoverNetworkServices": (RED_SCAN_START, DiscoverNetworkServices),
        "AggressiveServiceDiscovery": (RED_AGGRESSIVE_SCAN_START, AggressiveServiceDiscovery),
        "StealthServiceDiscovery": (RED_STEALTH_SCAN_START, StealthServiceDiscovery),
        "DiscoverDeception": (RED_DISCOVER_DECEPTION_START, DiscoverDeception),
    }
    for _, (start, cls) in _ip_class_map.items():
        end = start + GLOBAL_MAX_HOSTS
        if start <= action_idx < end:
            host_idx = action_idx - start
            hostname = mappings.idx_to_hostname[host_idx]
            ip = mappings.hostname_to_ip[hostname]
            return cls(session=session, agent=agent_name, ip_address=ip)

    for start, exploit_cls in _get_exploit_range_map():
        end = start + GLOBAL_MAX_HOSTS
        if start <= action_idx < end:
            from CybORG.Simulator.Actions import ExploitRemoteService

            host_idx = action_idx - start
            hostname = mappings.idx_to_hostname[host_idx]
            ip = mappings.hostname_to_ip[hostname]
            action = ExploitRemoteService(session=session, agent=agent_name, ip_address=ip)
            action.exploit_action_selector = _FixedExploitSelector(exploit_cls)
            return action

    _hostname_actions = [
        (RED_PRIVESC_START, PrivilegeEscalate),
        (RED_IMPACT_START, Impact),
        (RED_DEGRADE_START, DegradeServices),
    ]
    for start, cls in _hostname_actions:
        end = start + GLOBAL_MAX_HOSTS
        if start <= action_idx < end:
            host_idx = action_idx - start
            hostname = mappings.idx_to_hostname[host_idx]
            return cls(hostname=hostname, session=session, agent=agent_name)

    if RED_WITHDRAW_START <= action_idx < RED_WITHDRAW_END:
        host_idx = action_idx - RED_WITHDRAW_START
        hostname = mappings.idx_to_hostname[host_idx]
        ip = mappings.hostname_to_ip[hostname]
        return Withdraw(session=session, agent=agent_name, ip_address=ip, hostname=hostname)

    raise ValueError(f"Unknown JAX red action index: {action_idx}")


def cyborg_blue_to_jax(action, agent_name: str, mappings: CC4Mappings, const=None) -> int:
    cls_name = type(action).__name__
    aid = _agent_idx(agent_name)

    if cls_name == "Sleep":
        return BLUE_SLEEP

    if cls_name == "Monitor":
        return BLUE_MONITOR

    hostname_actions = {"Analyse": "Analyse", "Remove": "Remove", "Restore": "Restore"}
    if cls_name in hostname_actions:
        host_idx = _host_idx_from_hostname(action.hostname, mappings)
        return encode_blue_action(hostname_actions[cls_name], host_idx, aid, const=const)

    if cls_name in _CYBORG_BLUE_DECOY_NAMES or cls_name == "DeployDecoy":
        host_idx = _host_idx_from_hostname(action.hostname, mappings)
        return encode_blue_action("DeployDecoy", host_idx, aid, const=const)

    if cls_name == "BlockTrafficZone":
        from_sid = CYBORG_SUFFIX_TO_ID[action.from_subnet]
        to_sid = CYBORG_SUFFIX_TO_ID[action.to_subnet]
        return encode_blue_action("BlockTrafficZone", -1, aid, const=const, src_subnet=from_sid, dst_subnet=to_sid)

    if cls_name == "AllowTrafficZone":
        from_sid = CYBORG_SUFFIX_TO_ID[action.from_subnet]
        to_sid = CYBORG_SUFFIX_TO_ID[action.to_subnet]
        return encode_blue_action("AllowTrafficZone", -1, aid, const=const, src_subnet=from_sid, dst_subnet=to_sid)

    raise ValueError(f"Unknown CybORG blue action class: {cls_name}")


def jax_blue_to_cyborg(action_idx: int, agent_id: int, mappings: CC4Mappings, const=None):
    from CybORG.Simulator.Actions import (
        Analyse,
        Monitor,
        Remove,
        Restore,
        Sleep,
    )
    from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
        AllowTrafficZone,
        BlockTrafficZone,
    )
    from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DeployDecoy import DeployDecoy

    agent_name = f"blue_agent_{agent_id}"
    session = 0

    if action_idx == BLUE_SLEEP:
        return Sleep()

    if action_idx == BLUE_MONITOR:
        return Monitor(session=session, agent=agent_name)

    _hostname_actions = [
        (BLUE_ANALYSE_START, BLUE_ANALYSE_END, Analyse),
        (BLUE_REMOVE_START, BLUE_REMOVE_END, Remove),
        (BLUE_RESTORE_START, BLUE_RESTORE_END, Restore),
    ]
    for start, end, cls in _hostname_actions:
        if start <= action_idx < end:
            flat_slot = action_idx - start
            host_idx = _slot_to_global(flat_slot, const, agent_id)
            hostname = mappings.idx_to_hostname[host_idx]
            return cls(session=session, agent=agent_name, hostname=hostname)

    if BLUE_DECOY_START <= action_idx < BLUE_DECOY_END:
        flat_slot = action_idx - BLUE_DECOY_START
        host_idx = _slot_to_global(flat_slot, const, agent_id)
        hostname = mappings.idx_to_hostname[host_idx]
        return DeployDecoy(session=session, agent=agent_name, hostname=hostname)

    if BLUE_BLOCK_TRAFFIC_START <= action_idx < BLUE_BLOCK_TRAFFIC_END:
        offset = action_idx - BLUE_BLOCK_TRAFFIC_START
        src_offset = offset // BLUE_MAX_OBSERVED_SUBNETS
        rel_dst = offset % BLUE_MAX_OBSERVED_SUBNETS
        dst = int(const.blue_obs_subnets[agent_id, rel_dst])
        src = src_offset if src_offset < dst else src_offset + 1
        from_subnet = mappings.subnet_names[src]
        to_subnet = mappings.subnet_names[dst]
        return BlockTrafficZone(session=session, agent=agent_name, from_subnet=from_subnet, to_subnet=to_subnet)

    if BLUE_ALLOW_TRAFFIC_START <= action_idx < BLUE_ALLOW_TRAFFIC_END:
        offset = action_idx - BLUE_ALLOW_TRAFFIC_START
        src_offset = offset // BLUE_MAX_OBSERVED_SUBNETS
        rel_dst = offset % BLUE_MAX_OBSERVED_SUBNETS
        dst = int(const.blue_obs_subnets[agent_id, rel_dst])
        src = src_offset if src_offset < dst else src_offset + 1
        from_subnet = mappings.subnet_names[src]
        to_subnet = mappings.subnet_names[dst]
        return AllowTrafficZone(session=session, agent=agent_name, from_subnet=from_subnet, to_subnet=to_subnet)

    raise ValueError(f"Unknown JAX blue action index: {action_idx}")


def jax_blue_to_cyborg_wrapper_action(action_idx: int, agent_id: int, mappings: CC4Mappings, const=None):
    """Translate a JAX blue action into the BlueFlatWrapper-compatible action space.

    Now identical to jax_blue_to_cyborg since the action space uses generic
    DeployDecoy (one per host slot, type selected at execution time).
    """
    return jax_blue_to_cyborg(action_idx, agent_id, mappings, const=const)


def describe_red_action(action_idx: int, mappings: CC4Mappings) -> str:
    if action_idx == RED_SLEEP:
        return "Sleep"
    if RED_DISCOVER_START <= action_idx < RED_DISCOVER_END:
        subnet_idx = action_idx - RED_DISCOVER_START
        name = mappings.subnet_names.get(subnet_idx, f"subnet_{subnet_idx}")
        return f"DiscoverRemoteSystems({name})"
    for start, end, action_name in _RED_HOST_RANGES:
        if start <= action_idx < end:
            host_idx = action_idx - start
            hostname = mappings.idx_to_hostname.get(host_idx, f"host_{host_idx}")
            return f"{action_name}({hostname})"
    return f"Unknown({action_idx})"


def describe_blue_action(action_idx: int, mappings: CC4Mappings, const=None, agent_id: int = 0) -> str:
    if action_idx == BLUE_SLEEP:
        return "Sleep"
    if action_idx == BLUE_MONITOR:
        return "Monitor"

    _blue_host_ranges = [
        (BLUE_ANALYSE_START, BLUE_ANALYSE_END, "Analyse"),
        (BLUE_REMOVE_START, BLUE_REMOVE_END, "Remove"),
        (BLUE_RESTORE_START, BLUE_RESTORE_END, "Restore"),
    ]
    for start, end, name in _blue_host_ranges:
        if start <= action_idx < end:
            flat_slot = action_idx - start
            if const is not None:
                host_idx = _slot_to_global(flat_slot, const, agent_id)
            else:
                host_idx = flat_slot
            hostname = mappings.idx_to_hostname.get(host_idx, f"slot_{flat_slot}")
            return f"{name}({hostname})"

    if BLUE_DECOY_START <= action_idx < BLUE_DECOY_END:
        flat_slot = action_idx - BLUE_DECOY_START
        if const is not None:
            host_idx = _slot_to_global(flat_slot, const, agent_id)
        else:
            host_idx = flat_slot
        hostname = mappings.idx_to_hostname.get(host_idx, f"slot_{flat_slot}")
        return f"DeployDecoy({hostname})"

    if BLUE_BLOCK_TRAFFIC_START <= action_idx < BLUE_BLOCK_TRAFFIC_END:
        offset = action_idx - BLUE_BLOCK_TRAFFIC_START
        src_offset = offset // BLUE_MAX_OBSERVED_SUBNETS
        rel_dst = offset % BLUE_MAX_OBSERVED_SUBNETS
        if const is not None:
            dst = int(const.blue_obs_subnets[agent_id, rel_dst])
            src = src_offset if src_offset < dst else src_offset + 1
        else:
            dst = rel_dst
            src = src_offset
        return f"BlockTrafficZone({mappings.subnet_names.get(src, src)}->{mappings.subnet_names.get(dst, dst)})"

    if BLUE_ALLOW_TRAFFIC_START <= action_idx < BLUE_ALLOW_TRAFFIC_END:
        offset = action_idx - BLUE_ALLOW_TRAFFIC_START
        src_offset = offset // BLUE_MAX_OBSERVED_SUBNETS
        rel_dst = offset % BLUE_MAX_OBSERVED_SUBNETS
        if const is not None:
            dst = int(const.blue_obs_subnets[agent_id, rel_dst])
            src = src_offset if src_offset < dst else src_offset + 1
        else:
            dst = rel_dst
            src = src_offset
        return f"AllowTrafficZone({mappings.subnet_names.get(src, src)}->{mappings.subnet_names.get(dst, dst)})"

    return f"Unknown({action_idx})"
