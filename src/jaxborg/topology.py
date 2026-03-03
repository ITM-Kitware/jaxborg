import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    MAX_SERVER_HOSTS,
    MAX_USER_HOSTS,
    MISSION_PHASES,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
    SERVICE_IDS,
    SERVICE_NAMES,
    SUBNET_IDS,
    SUBNET_NAMES,
)
from jaxborg.state import CC4Const

CYBORG_SUBNET_SUFFIX = {
    "RESTRICTED_ZONE_A": "restricted_zone_a_subnet",
    "RESTRICTED_ZONE_B": "restricted_zone_b_subnet",
    "OPERATIONAL_ZONE_A": "operational_zone_a_subnet",
    "OPERATIONAL_ZONE_B": "operational_zone_b_subnet",
    "CONTRACTOR_NETWORK": "contractor_network_subnet",
    "ADMIN_NETWORK": "admin_network_subnet",
    "OFFICE_NETWORK": "office_network_subnet",
    "PUBLIC_ACCESS_ZONE": "public_access_zone_subnet",
    "INTERNET": "internet_subnet",
}

CYBORG_SUFFIX_TO_ID = {v: SUBNET_IDS[k] for k, v in CYBORG_SUBNET_SUFFIX.items()}


def _subnet_nacl_adjacency() -> np.ndarray:
    """Build the default NACL-based subnet adjacency matrix.

    Returns (NUM_SUBNETS, NUM_SUBNETS) bool numpy array where [i,j]=True means
    traffic can flow from subnet i to subnet j.
    """
    S = SUBNET_IDS
    adj = np.zeros((NUM_SUBNETS, NUM_SUBNETS), dtype=bool)

    adj[S["RESTRICTED_ZONE_A"], S["OPERATIONAL_ZONE_A"]] = True
    adj[S["RESTRICTED_ZONE_A"], S["CONTRACTOR_NETWORK"]] = True
    adj[S["RESTRICTED_ZONE_A"], S["PUBLIC_ACCESS_ZONE"]] = True

    adj[S["OPERATIONAL_ZONE_A"], S["RESTRICTED_ZONE_A"]] = True

    adj[S["RESTRICTED_ZONE_B"], S["OPERATIONAL_ZONE_B"]] = True
    adj[S["RESTRICTED_ZONE_B"], S["CONTRACTOR_NETWORK"]] = True
    adj[S["RESTRICTED_ZONE_B"], S["PUBLIC_ACCESS_ZONE"]] = True

    adj[S["OPERATIONAL_ZONE_B"], S["RESTRICTED_ZONE_B"]] = True

    adj[S["CONTRACTOR_NETWORK"], S["RESTRICTED_ZONE_A"]] = True
    adj[S["CONTRACTOR_NETWORK"], S["RESTRICTED_ZONE_B"]] = True
    adj[S["CONTRACTOR_NETWORK"], S["PUBLIC_ACCESS_ZONE"]] = True

    adj[S["PUBLIC_ACCESS_ZONE"], S["RESTRICTED_ZONE_A"]] = True
    adj[S["PUBLIC_ACCESS_ZONE"], S["RESTRICTED_ZONE_B"]] = True
    adj[S["PUBLIC_ACCESS_ZONE"], S["CONTRACTOR_NETWORK"]] = True
    adj[S["PUBLIC_ACCESS_ZONE"], S["ADMIN_NETWORK"]] = True
    adj[S["PUBLIC_ACCESS_ZONE"], S["OFFICE_NETWORK"]] = True

    adj[S["ADMIN_NETWORK"], S["PUBLIC_ACCESS_ZONE"]] = True
    adj[S["ADMIN_NETWORK"], S["OFFICE_NETWORK"]] = True

    adj[S["OFFICE_NETWORK"], S["PUBLIC_ACCESS_ZONE"]] = True
    adj[S["OFFICE_NETWORK"], S["ADMIN_NETWORK"]] = True

    adj[S["INTERNET"], S["RESTRICTED_ZONE_A"]] = True
    adj[S["INTERNET"], S["OPERATIONAL_ZONE_A"]] = True
    adj[S["INTERNET"], S["RESTRICTED_ZONE_B"]] = True
    adj[S["INTERNET"], S["OPERATIONAL_ZONE_B"]] = True
    adj[S["INTERNET"], S["CONTRACTOR_NETWORK"]] = True
    adj[S["INTERNET"], S["PUBLIC_ACCESS_ZONE"]] = True
    adj[S["INTERNET"], S["ADMIN_NETWORK"]] = True
    adj[S["INTERNET"], S["OFFICE_NETWORK"]] = True

    return adj


_ROUTER_LINKS = {
    "INTERNET": [
        "RESTRICTED_ZONE_A",
        "RESTRICTED_ZONE_B",
        "CONTRACTOR_NETWORK",
        "PUBLIC_ACCESS_ZONE",
    ],
    "RESTRICTED_ZONE_A": ["INTERNET", "OPERATIONAL_ZONE_A"],
    "RESTRICTED_ZONE_B": ["INTERNET", "OPERATIONAL_ZONE_B"],
    "CONTRACTOR_NETWORK": ["INTERNET"],
    "PUBLIC_ACCESS_ZONE": ["INTERNET", "ADMIN_NETWORK", "OFFICE_NETWORK"],
    "OPERATIONAL_ZONE_A": ["RESTRICTED_ZONE_A"],
    "OPERATIONAL_ZONE_B": ["RESTRICTED_ZONE_B"],
    "ADMIN_NETWORK": ["PUBLIC_ACCESS_ZONE"],
    "OFFICE_NETWORK": ["PUBLIC_ACCESS_ZONE"],
}

BLUE_AGENT_SUBNETS = [
    ["RESTRICTED_ZONE_A"],
    ["OPERATIONAL_ZONE_A"],
    ["RESTRICTED_ZONE_B"],
    ["OPERATIONAL_ZONE_B"],
    ["PUBLIC_ACCESS_ZONE", "ADMIN_NETWORK", "OFFICE_NETWORK"],
]

RED_AGENT_SUBNETS = [
    ["CONTRACTOR_NETWORK"],
    ["RESTRICTED_ZONE_A"],
    ["OPERATIONAL_ZONE_A"],
    ["RESTRICTED_ZONE_B"],
    ["OPERATIONAL_ZONE_B"],
    ["PUBLIC_ACCESS_ZONE", "ADMIN_NETWORK", "OFFICE_NETWORK"],
]


def _build_data_links(
    host_subnet: np.ndarray,
    host_is_router: np.ndarray,
    num_hosts: int,
    subnet_router_idx: np.ndarray,
) -> np.ndarray:
    """Build host-level data_links adjacency from CybORG router topology rules."""
    links = np.zeros((GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS), dtype=bool)

    for h in range(num_hosts):
        s = int(host_subnet[h])
        sname = SUBNET_NAMES[s]

        if sname == "INTERNET":
            for neighbor_name in _ROUTER_LINKS["INTERNET"]:
                neighbor_sid = SUBNET_IDS[neighbor_name]
                r = int(subnet_router_idx[neighbor_sid])
                if r >= 0:
                    links[h, r] = True
                    links[r, h] = True
        elif host_is_router[h]:
            for neighbor_name in _ROUTER_LINKS.get(sname, []):
                neighbor_sid = SUBNET_IDS[neighbor_name]
                if neighbor_name == "INTERNET":
                    internet_host = int(subnet_router_idx[SUBNET_IDS["INTERNET"]])
                    if internet_host >= 0:
                        links[h, internet_host] = True
                        links[internet_host, h] = True
                else:
                    r = int(subnet_router_idx[neighbor_sid])
                    if r >= 0:
                        links[h, r] = True
                        links[r, h] = True
        else:
            r = int(subnet_router_idx[s])
            if r >= 0:
                links[h, r] = True
                links[r, h] = True

    return links


def build_const_from_cyborg(cyborg_env) -> CC4Const:
    """Extract static topology from a live CybORG environment."""
    state = cyborg_env.environment_controller.state
    scenario = state.scenario

    hostname_to_idx = {}
    host_active = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_subnet = np.zeros(GLOBAL_MAX_HOSTS, dtype=np.int32)
    host_is_router = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_is_server = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_is_user = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_respond_to_ping = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_has_bruteforceable_user = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_has_rfi = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_initial_max_pid = np.zeros(GLOBAL_MAX_HOSTS, dtype=np.int32)
    initial_services = np.zeros((GLOBAL_MAX_HOSTS, len(SERVICE_NAMES)), dtype=bool)
    subnet_router_idx = np.full(NUM_SUBNETS, -1, dtype=np.int32)

    sorted_hostnames = sorted(state.hosts.keys())
    for idx, hostname in enumerate(sorted_hostnames):
        hostname_to_idx[hostname] = idx

    num_hosts = len(sorted_hostnames)
    assert num_hosts <= GLOBAL_MAX_HOSTS

    for hostname, idx in hostname_to_idx.items():
        host = state.hosts[hostname]
        subnet_name_cyborg = state.hostname_subnet_map[hostname]
        sid = CYBORG_SUFFIX_TO_ID[subnet_name_cyborg]

        host_active[idx] = True
        host_subnet[idx] = sid

        if hostname == "root_internet_host_0":
            subnet_router_idx[SUBNET_IDS["INTERNET"]] = idx
        elif "_router" in hostname:
            host_is_router[idx] = True
            subnet_router_idx[sid] = idx
        elif "_server_host_" in hostname:
            host_is_server[idx] = True
        elif "_user_host_" in hostname:
            host_is_user[idx] = True

        host_respond_to_ping[idx] = host.respond_to_ping
        if host.processes:
            process_pids = [int(proc.pid) for proc in host.processes if proc.pid is not None]
            if process_pids:
                host_initial_max_pid[idx] = np.int32(max(process_pids))

        for user in host.users:
            if getattr(user, "bruteforceable", False):
                host_has_bruteforceable_user[idx] = True
                break

        if host.processes:
            for proc in host.processes:
                if hasattr(proc, "properties") and proc.properties and "rfi" in proc.properties:
                    host_has_rfi[idx] = True

        if host.services:
            for svc_name in host.services:
                svc_str = str(svc_name).split(".")[-1] if "." in str(svc_name) else str(svc_name)
                if svc_str in SERVICE_IDS:
                    initial_services[idx, SERVICE_IDS[svc_str]] = True

    data_links = _build_data_links(host_subnet, host_is_router, num_hosts, subnet_router_idx)

    _fill_data_links_from_cyborg(data_links, state, hostname_to_idx)

    subnet_adjacency = _subnet_nacl_adjacency()

    blue_agent_subnets = np.zeros((NUM_BLUE_AGENTS, NUM_SUBNETS), dtype=bool)
    blue_agent_hosts = np.zeros((NUM_BLUE_AGENTS, GLOBAL_MAX_HOSTS), dtype=bool)
    for i, snames in enumerate(BLUE_AGENT_SUBNETS):
        for sname in snames:
            sid = SUBNET_IDS[sname]
            blue_agent_subnets[i, sid] = True
            for h in range(num_hosts):
                if host_active[h] and host_subnet[h] == sid:
                    blue_agent_hosts[i, h] = True

    red_start_hosts = np.zeros(NUM_RED_AGENTS, dtype=np.int32)
    red_agent_active = np.zeros(NUM_RED_AGENTS, dtype=bool)
    red_agent_subnets = np.zeros((NUM_RED_AGENTS, NUM_SUBNETS), dtype=bool)
    for agent_name, agent_info in scenario.agents.items():
        if not agent_name.startswith("red_agent_"):
            continue
        red_idx = int(agent_name.split("_")[-1])
        if red_idx >= NUM_RED_AGENTS:
            continue
        if agent_info.starting_sessions:
            sess = agent_info.starting_sessions[0]
            if sess.hostname in hostname_to_idx:
                red_start_hosts[red_idx] = hostname_to_idx[sess.hostname]
        red_agent_active[red_idx] = agent_info.active
        if agent_info.allowed_subnets:
            for sub_enum in agent_info.allowed_subnets:
                cyborg_suffix = str(sub_enum)
                if cyborg_suffix in CYBORG_SUFFIX_TO_ID:
                    red_agent_subnets[red_idx, CYBORG_SUFFIX_TO_ID[cyborg_suffix]] = True
    red_initial_discovered_hosts = np.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=bool)
    red_initial_scanned_hosts = np.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=bool)
    for red_idx in range(NUM_RED_AGENTS):
        if red_agent_active[red_idx]:
            red_initial_discovered_hosts[red_idx, red_start_hosts[red_idx]] = True
    host_info_links = np.zeros((GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS), dtype=bool)
    for src_hostname, host in state.hosts.items():
        if src_hostname not in hostname_to_idx:
            continue
        src_idx = hostname_to_idx[src_hostname]
        for dst_hostname in getattr(host, "info", {}).keys():
            if dst_hostname in hostname_to_idx:
                host_info_links[src_idx, hostname_to_idx[dst_hostname]] = True

    green_agent_host = np.full(GLOBAL_MAX_HOSTS, -1, dtype=np.int32)
    green_agent_active = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    green_count = 0
    for agent_name, agent_info in sorted(scenario.agents.items()):
        if not agent_name.startswith("green_agent_"):
            continue
        if agent_info.starting_sessions:
            sess = agent_info.starting_sessions[0]
            if sess.hostname in hostname_to_idx:
                hidx = hostname_to_idx[sess.hostname]
                green_agent_host[hidx] = green_count
                green_agent_active[hidx] = True
        green_count += 1

    phase_boundaries = _compute_phase_boundaries(scenario.mission_phases)
    allowed_subnet_pairs = _compute_allowed_subnet_pairs(scenario.allowed_subnets_per_mphase)

    obs_host_map = _build_obs_host_map(host_subnet, host_is_server, host_is_user, host_active, num_hosts)

    return CC4Const(
        host_active=jnp.array(host_active),
        host_subnet=jnp.array(host_subnet),
        host_is_router=jnp.array(host_is_router),
        host_is_server=jnp.array(host_is_server),
        host_is_user=jnp.array(host_is_user),
        subnet_adjacency=jnp.array(subnet_adjacency),
        data_links=jnp.array(data_links),
        initial_services=jnp.array(initial_services),
        host_has_bruteforceable_user=jnp.array(host_has_bruteforceable_user),
        host_has_rfi=jnp.array(host_has_rfi),
        host_respond_to_ping=jnp.array(host_respond_to_ping),
        host_initial_max_pid=jnp.array(host_initial_max_pid),
        blue_agent_subnets=jnp.array(blue_agent_subnets),
        blue_agent_hosts=jnp.array(blue_agent_hosts),
        red_start_hosts=jnp.array(red_start_hosts),
        red_agent_active=jnp.array(red_agent_active),
        red_agent_subnets=jnp.array(red_agent_subnets),
        red_initial_discovered_hosts=jnp.array(red_initial_discovered_hosts),
        red_initial_scanned_hosts=jnp.array(red_initial_scanned_hosts),
        host_info_links=jnp.array(host_info_links),
        green_agent_host=jnp.array(green_agent_host),
        green_agent_active=jnp.array(green_agent_active),
        num_green_agents=green_count,
        phase_rewards=jnp.array(_build_phase_rewards_from_cyborg(cyborg_env)),
        phase_boundaries=jnp.array(phase_boundaries),
        allowed_subnet_pairs=jnp.array(allowed_subnet_pairs),
        obs_host_map=jnp.array(obs_host_map),
        blue_obs_subnets=jnp.array(_build_blue_obs_subnets()),
        comms_policy=jnp.array(_build_comms_policy()),
        max_steps=500,
        num_hosts=num_hosts,
    )


def _fill_data_links_from_cyborg(links: np.ndarray, state, hostname_to_idx: dict) -> None:
    """Overwrite data_links from CybORG's actual interface data_links."""
    links[:] = False
    for hostname, host in state.hosts.items():
        h = hostname_to_idx[hostname]
        for iface in host.interfaces:
            if iface.interface_type == "wired":
                for dl_name in iface.data_links:
                    if dl_name in hostname_to_idx:
                        j = hostname_to_idx[dl_name]
                        links[h, j] = True
                        links[j, h] = True


def _compute_phase_boundaries(mission_phases) -> np.ndarray:
    boundaries = np.zeros(MISSION_PHASES, dtype=np.int32)
    cumulative = 0
    for i, phase_len in enumerate(mission_phases):
        boundaries[i] = cumulative
        cumulative += phase_len
    return boundaries


def _compute_allowed_subnet_pairs(allowed_per_mphase) -> np.ndarray:
    pairs = np.zeros((MISSION_PHASES, NUM_SUBNETS, NUM_SUBNETS), dtype=bool)
    for phase_idx, phase_pairs in enumerate(allowed_per_mphase):
        for src_enum, dst_enum in phase_pairs:
            src_name = str(src_enum).split(".")[-1] if "." in str(src_enum) else str(src_enum)
            dst_name = str(dst_enum).split(".")[-1] if "." in str(dst_enum) else str(dst_enum)
            src_cyborg = src_name.lower() + "_subnet"
            dst_cyborg = dst_name.lower() + "_subnet"
            if src_cyborg in CYBORG_SUFFIX_TO_ID and dst_cyborg in CYBORG_SUFFIX_TO_ID:
                si = CYBORG_SUFFIX_TO_ID[src_cyborg]
                di = CYBORG_SUFFIX_TO_ID[dst_cyborg]
                pairs[phase_idx, si, di] = True
                pairs[phase_idx, di, si] = True
    return pairs


def build_topology(key: jax.Array, num_steps: int = 500) -> CC4Const:
    """Build CC4 topology purely in JAX/numpy (no CybORG dependency).

    Mimics EnterpriseScenarioGenerator: for each non-internet subnet, generates
    1 router + random user hosts (3-10) + random server hosts (1-6).
    Internet subnet gets 1 host (root_internet_host_0).
    """
    rng = np.random.default_rng(int(key[0]) if hasattr(key, "__getitem__") else int(key))

    host_active = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_subnet_arr = np.zeros(GLOBAL_MAX_HOSTS, dtype=np.int32)
    host_is_router = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_is_server = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_is_user = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_respond_to_ping = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_has_bruteforceable_user = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_has_rfi = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_initial_max_pid = np.zeros(GLOBAL_MAX_HOSTS, dtype=np.int32)
    initial_services = np.zeros((GLOBAL_MAX_HOSTS, len(SERVICE_NAMES)), dtype=bool)
    subnet_router_idx = np.full(NUM_SUBNETS, -1, dtype=np.int32)
    subnet_host_counts = np.zeros(NUM_SUBNETS, dtype=np.int32)

    host_idx = 0
    host_names = []

    for sname in SUBNET_NAMES:
        sid = SUBNET_IDS[sname]

        if sname == "INTERNET":
            host_active[host_idx] = True
            host_subnet_arr[host_idx] = sid
            subnet_router_idx[sid] = host_idx
            subnet_host_counts[sid] = 1
            host_names.append("root_internet_host_0")
            host_idx += 1
            continue

        cyborg_suffix = CYBORG_SUBNET_SUFFIX[sname]
        router_name = f"{cyborg_suffix}_router"
        host_active[host_idx] = True
        host_subnet_arr[host_idx] = sid
        host_is_router[host_idx] = True
        subnet_router_idx[sid] = host_idx
        host_names.append(router_name)
        host_idx += 1

        num_users = rng.integers(3, 10, endpoint=True)
        for u in range(num_users):
            uname = f"{cyborg_suffix}_user_host_{u}"
            host_active[host_idx] = True
            host_subnet_arr[host_idx] = sid
            host_is_user[host_idx] = True
            host_respond_to_ping[host_idx] = True
            host_has_bruteforceable_user[host_idx] = True
            host_initial_max_pid[host_idx] = 5000

            initial_services[host_idx, SERVICE_IDS["SSHD"]] = True
            if "operational" in cyborg_suffix:
                initial_services[host_idx, SERVICE_IDS["OTSERVICE"]] = True
            _assign_random_addon_services(rng, initial_services, host_idx)

            host_names.append(uname)
            host_idx += 1

        num_servers = rng.integers(1, 6, endpoint=True)
        for sv in range(num_servers):
            svname = f"{cyborg_suffix}_server_host_{sv}"
            host_active[host_idx] = True
            host_subnet_arr[host_idx] = sid
            host_is_server[host_idx] = True
            host_respond_to_ping[host_idx] = True
            host_has_bruteforceable_user[host_idx] = True
            host_initial_max_pid[host_idx] = 5000

            initial_services[host_idx, SERVICE_IDS["SSHD"]] = True
            if "operational" in cyborg_suffix:
                initial_services[host_idx, SERVICE_IDS["OTSERVICE"]] = True
            _assign_random_addon_services(rng, initial_services, host_idx)

            host_names.append(svname)
            host_idx += 1

        subnet_host_counts[sid] = 1 + num_users + num_servers

    num_hosts = host_idx

    data_links = _build_data_links(host_subnet_arr, host_is_router, num_hosts, subnet_router_idx)

    subnet_adjacency = _subnet_nacl_adjacency()

    blue_agent_subnets = np.zeros((NUM_BLUE_AGENTS, NUM_SUBNETS), dtype=bool)
    blue_agent_hosts = np.zeros((NUM_BLUE_AGENTS, GLOBAL_MAX_HOSTS), dtype=bool)
    for i, snames in enumerate(BLUE_AGENT_SUBNETS):
        for sn in snames:
            sid = SUBNET_IDS[sn]
            blue_agent_subnets[i, sid] = True
            for h in range(num_hosts):
                if host_active[h] and host_subnet_arr[h] == sid:
                    blue_agent_hosts[i, h] = True

    red_start_hosts = np.zeros(NUM_RED_AGENTS, dtype=np.int32)
    red_agent_active = np.zeros(NUM_RED_AGENTS, dtype=bool)
    red_agent_subnets = np.zeros((NUM_RED_AGENTS, NUM_SUBNETS), dtype=bool)
    for i, snames in enumerate(RED_AGENT_SUBNETS):
        for sn in snames:
            red_agent_subnets[i, SUBNET_IDS[sn]] = True
        non_router_hosts = [
            h
            for h in range(num_hosts)
            if host_active[h]
            and SUBNET_NAMES[int(host_subnet_arr[h])] in snames
            and not host_is_router[h]
            and host_names[h] != "root_internet_host_0"
        ]
        if non_router_hosts:
            red_start_hosts[i] = rng.choice(non_router_hosts)
            red_agent_active[i] = True
    red_initial_discovered_hosts = np.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=bool)
    red_initial_scanned_hosts = np.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=bool)
    for red_idx in range(NUM_RED_AGENTS):
        if red_agent_active[red_idx]:
            red_initial_discovered_hosts[red_idx, red_start_hosts[red_idx]] = True
    host_info_links = np.zeros((GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS), dtype=bool)
    host_name_to_idx = {name: idx for idx, name in enumerate(host_names)}
    server0_idx_by_subnet = np.full(NUM_SUBNETS, -1, dtype=np.int32)
    for sname in SUBNET_NAMES:
        if sname == "INTERNET":
            continue
        sid = SUBNET_IDS[sname]
        server0_name = f"{CYBORG_SUBNET_SUFFIX[sname]}_server_host_0"
        if server0_name in host_name_to_idx:
            server0_idx_by_subnet[sid] = host_name_to_idx[server0_name]
    for src_name, neighbor_names in _ROUTER_LINKS.items():
        if src_name == "INTERNET":
            continue
        src_sid = SUBNET_IDS[src_name]
        src_idx = int(server0_idx_by_subnet[src_sid])
        if src_idx < 0:
            continue
        for dst_name in neighbor_names:
            if dst_name == "INTERNET":
                continue
            dst_sid = SUBNET_IDS[dst_name]
            dst_idx = int(server0_idx_by_subnet[dst_sid])
            if dst_idx >= 0:
                host_info_links[src_idx, dst_idx] = True

    green_agent_host = np.full(GLOBAL_MAX_HOSTS, -1, dtype=np.int32)
    green_agent_active = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    green_count = 0
    for h in range(num_hosts):
        if host_is_user[h]:
            green_agent_host[h] = green_count
            green_agent_active[h] = True
            green_count += 1

    phase_boundaries = _compute_phase_boundaries(_compute_mission_phases(num_steps))
    allowed_subnet_pairs = _build_allowed_subnet_pairs_pure()

    obs_host_map = _build_obs_host_map(host_subnet_arr, host_is_server, host_is_user, host_active, num_hosts)

    return CC4Const(
        host_active=jnp.array(host_active),
        host_subnet=jnp.array(host_subnet_arr),
        host_is_router=jnp.array(host_is_router),
        host_is_server=jnp.array(host_is_server),
        host_is_user=jnp.array(host_is_user),
        subnet_adjacency=jnp.array(subnet_adjacency),
        data_links=jnp.array(data_links),
        initial_services=jnp.array(initial_services),
        host_has_bruteforceable_user=jnp.array(host_has_bruteforceable_user),
        host_has_rfi=jnp.array(host_has_rfi),
        host_respond_to_ping=jnp.array(host_respond_to_ping),
        host_initial_max_pid=jnp.array(host_initial_max_pid),
        blue_agent_subnets=jnp.array(blue_agent_subnets),
        blue_agent_hosts=jnp.array(blue_agent_hosts),
        red_start_hosts=jnp.array(red_start_hosts),
        red_agent_active=jnp.array(red_agent_active),
        red_agent_subnets=jnp.array(red_agent_subnets),
        red_initial_discovered_hosts=jnp.array(red_initial_discovered_hosts),
        red_initial_scanned_hosts=jnp.array(red_initial_scanned_hosts),
        host_info_links=jnp.array(host_info_links),
        green_agent_host=jnp.array(green_agent_host),
        green_agent_active=jnp.array(green_agent_active),
        num_green_agents=green_count,
        phase_rewards=jnp.array(_build_phase_rewards()),
        phase_boundaries=jnp.array(phase_boundaries),
        allowed_subnet_pairs=jnp.array(allowed_subnet_pairs),
        obs_host_map=jnp.array(obs_host_map),
        blue_obs_subnets=jnp.array(_build_blue_obs_subnets()),
        comms_policy=jnp.array(_build_comms_policy()),
        max_steps=num_steps,
        num_hosts=num_hosts,
    )


def _assign_random_addon_services(
    rng: np.random.Generator,
    services: np.ndarray,
    host_idx: int,
) -> None:
    addon_options = ["APACHE2", "MYSQLD", "SMTP"]
    num_addons = rng.integers(0, len(addon_options), endpoint=True)
    chosen = rng.choice(addon_options, size=num_addons, replace=False)
    for svc in chosen:
        services[host_idx, SERVICE_IDS[svc]] = True


def _compute_mission_phases(steps: int) -> tuple:
    quotient, remainder = divmod(steps, 3)
    if remainder == 2:
        return (quotient + 1, quotient + 1, quotient)
    if remainder == 1:
        return (quotient + 1, quotient, quotient)
    return (quotient, quotient, quotient)


def _build_phase_rewards() -> np.ndarray:
    S = SUBNET_IDS
    # (MISSION_PHASES, NUM_SUBNETS, 3) where columns are [LWF, ASF, RIA]
    pr = np.zeros((MISSION_PHASES, NUM_SUBNETS, 3), dtype=np.float32)

    # Phase 0 (Preplanning)
    pr[0, S["RESTRICTED_ZONE_A"]] = [-1, -3, -1]
    pr[0, S["OPERATIONAL_ZONE_A"]] = [-1, -1, -1]
    pr[0, S["RESTRICTED_ZONE_B"]] = [-1, -3, -1]
    pr[0, S["OPERATIONAL_ZONE_B"]] = [-1, -1, -1]
    pr[0, S["CONTRACTOR_NETWORK"]] = [0, -5, -5]
    pr[0, S["ADMIN_NETWORK"]] = [-1, -1, -3]
    pr[0, S["OFFICE_NETWORK"]] = [-1, -1, -3]
    pr[0, S["PUBLIC_ACCESS_ZONE"]] = [-1, -1, -3]
    pr[0, S["INTERNET"]] = [0, 0, -1]

    # Phase 1 (MissionA)
    pr[1, S["RESTRICTED_ZONE_A"]] = [-2, -1, -3]
    pr[1, S["OPERATIONAL_ZONE_A"]] = [-10, 0, -10]
    pr[1, S["RESTRICTED_ZONE_B"]] = [-1, -1, -1]
    pr[1, S["OPERATIONAL_ZONE_B"]] = [-1, -1, -1]
    pr[1, S["CONTRACTOR_NETWORK"]] = [0, 0, 0]
    pr[1, S["ADMIN_NETWORK"]] = [-1, -1, -3]
    pr[1, S["OFFICE_NETWORK"]] = [-1, -1, -3]
    pr[1, S["PUBLIC_ACCESS_ZONE"]] = [-1, -1, -3]
    pr[1, S["INTERNET"]] = [0, 0, 0]

    # Phase 2 (MissionB)
    pr[2, S["RESTRICTED_ZONE_A"]] = [-1, -3, -3]
    pr[2, S["OPERATIONAL_ZONE_A"]] = [-1, -1, -1]
    pr[2, S["RESTRICTED_ZONE_B"]] = [-2, -1, -3]
    pr[2, S["OPERATIONAL_ZONE_B"]] = [-10, 0, -10]
    pr[2, S["CONTRACTOR_NETWORK"]] = [0, 0, 0]
    pr[2, S["ADMIN_NETWORK"]] = [-1, -1, -3]
    pr[2, S["OFFICE_NETWORK"]] = [-1, -1, -3]
    pr[2, S["PUBLIC_ACCESS_ZONE"]] = [-1, -1, -3]
    pr[2, S["INTERNET"]] = [0, 0, 0]

    return pr


def _build_phase_rewards_from_cyborg(cyborg_env) -> np.ndarray:
    from CybORG.Shared.BlueRewardMachine import BlueRewardMachine

    brm = BlueRewardMachine("")
    pr = np.zeros((MISSION_PHASES, NUM_SUBNETS, 3), dtype=np.float32)
    for phase in range(MISSION_PHASES):
        table = brm.get_phase_rewards(phase)
        for cyborg_name, rewards in table.items():
            sid = CYBORG_SUFFIX_TO_ID[cyborg_name]
            pr[phase, sid, 0] = rewards["LWF"]
            pr[phase, sid, 1] = rewards["ASF"]
            pr[phase, sid, 2] = rewards["RIA"]
    return pr


JAX_TO_CYBORG_ORDER = np.array([5, 4, 8, 6, 2, 3, 7, 0, 1], dtype=np.int32)


def _build_obs_host_map(
    host_subnet: np.ndarray,
    host_is_server: np.ndarray,
    host_is_user: np.ndarray,
    host_active: np.ndarray,
    num_hosts: int,
) -> np.ndarray:
    obs_map = np.full((NUM_SUBNETS, OBS_HOSTS_PER_SUBNET), GLOBAL_MAX_HOSTS, dtype=np.int32)
    for sid in range(NUM_SUBNETS):
        servers = []
        users = []
        for h in range(num_hosts):
            if not host_active[h] or host_subnet[h] != sid:
                continue
            if host_is_server[h]:
                servers.append(h)
            elif host_is_user[h]:
                users.append(h)
        for i, h in enumerate(servers[:MAX_SERVER_HOSTS]):
            obs_map[sid, i] = h
        for i, h in enumerate(users[:MAX_USER_HOSTS]):
            obs_map[sid, MAX_SERVER_HOSTS + i] = h
    return obs_map


def _build_blue_obs_subnets() -> np.ndarray:
    result = np.full((NUM_BLUE_AGENTS, 3), -1, dtype=np.int32)
    for agent_idx, snames in enumerate(BLUE_AGENT_SUBNETS):
        cyborg_sorted = sorted(CYBORG_SUBNET_SUFFIX[s] for s in snames)
        for slot, cyborg_name in enumerate(cyborg_sorted):
            result[agent_idx, slot] = CYBORG_SUFFIX_TO_ID[cyborg_name]
    return result


def _build_comms_policy() -> np.ndarray:
    S = SUBNET_IDS
    base_hosts = [
        "INTERNET",
        "ADMIN_NETWORK",
        "OFFICE_NETWORK",
        "PUBLIC_ACCESS_ZONE",
        "CONTRACTOR_NETWORK",
        "RESTRICTED_ZONE_A",
        "RESTRICTED_ZONE_B",
    ]
    base_ids = [S[n] for n in base_hosts]

    adj = np.zeros((MISSION_PHASES, NUM_SUBNETS, NUM_SUBNETS), dtype=bool)
    for phase in range(MISSION_PHASES):
        for i_idx in range(len(base_ids)):
            for j_idx in range(i_idx + 1, len(base_ids)):
                adj[phase, base_ids[i_idx], base_ids[j_idx]] = True
                adj[phase, base_ids[j_idx], base_ids[i_idx]] = True
        adj[phase, S["RESTRICTED_ZONE_A"], S["OPERATIONAL_ZONE_A"]] = True
        adj[phase, S["OPERATIONAL_ZONE_A"], S["RESTRICTED_ZONE_A"]] = True
        adj[phase, S["RESTRICTED_ZONE_B"], S["OPERATIONAL_ZONE_B"]] = True
        adj[phase, S["OPERATIONAL_ZONE_B"], S["RESTRICTED_ZONE_B"]] = True

    remove_phase1 = [
        (S["RESTRICTED_ZONE_A"], S["OPERATIONAL_ZONE_A"]),
        (S["RESTRICTED_ZONE_A"], S["CONTRACTOR_NETWORK"]),
        (S["RESTRICTED_ZONE_A"], S["RESTRICTED_ZONE_B"]),
        (S["RESTRICTED_ZONE_A"], S["INTERNET"]),
    ]
    for a, b in remove_phase1:
        adj[1, a, b] = False
        adj[1, b, a] = False

    remove_phase2 = [
        (S["RESTRICTED_ZONE_B"], S["OPERATIONAL_ZONE_B"]),
        (S["RESTRICTED_ZONE_B"], S["CONTRACTOR_NETWORK"]),
        (S["RESTRICTED_ZONE_B"], S["RESTRICTED_ZONE_A"]),
        (S["RESTRICTED_ZONE_B"], S["INTERNET"]),
    ]
    for a, b in remove_phase2:
        adj[2, a, b] = False
        adj[2, b, a] = False

    return ~adj


def _build_allowed_subnet_pairs_pure() -> np.ndarray:
    """Build allowed_subnet_pairs matching CybORG's _set_allowed_subnets_per_mission_phase."""
    S = SUBNET_IDS

    policy_1 = [
        (S["PUBLIC_ACCESS_ZONE"], S["CONTRACTOR_NETWORK"]),
        (S["ADMIN_NETWORK"], S["CONTRACTOR_NETWORK"]),
        (S["OFFICE_NETWORK"], S["CONTRACTOR_NETWORK"]),
        (S["PUBLIC_ACCESS_ZONE"], S["RESTRICTED_ZONE_A"]),
        (S["ADMIN_NETWORK"], S["RESTRICTED_ZONE_A"]),
        (S["OFFICE_NETWORK"], S["RESTRICTED_ZONE_A"]),
        (S["PUBLIC_ACCESS_ZONE"], S["RESTRICTED_ZONE_B"]),
        (S["ADMIN_NETWORK"], S["RESTRICTED_ZONE_B"]),
        (S["OFFICE_NETWORK"], S["RESTRICTED_ZONE_B"]),
        (S["RESTRICTED_ZONE_A"], S["CONTRACTOR_NETWORK"]),
        (S["OPERATIONAL_ZONE_A"], S["RESTRICTED_ZONE_A"]),
        (S["RESTRICTED_ZONE_B"], S["CONTRACTOR_NETWORK"]),
        (S["RESTRICTED_ZONE_B"], S["RESTRICTED_ZONE_A"]),
        (S["OPERATIONAL_ZONE_B"], S["RESTRICTED_ZONE_B"]),
    ]

    policy_2 = [
        (S["PUBLIC_ACCESS_ZONE"], S["CONTRACTOR_NETWORK"]),
        (S["ADMIN_NETWORK"], S["CONTRACTOR_NETWORK"]),
        (S["OFFICE_NETWORK"], S["CONTRACTOR_NETWORK"]),
        (S["PUBLIC_ACCESS_ZONE"], S["RESTRICTED_ZONE_A"]),
        (S["ADMIN_NETWORK"], S["RESTRICTED_ZONE_A"]),
        (S["OFFICE_NETWORK"], S["RESTRICTED_ZONE_A"]),
        (S["PUBLIC_ACCESS_ZONE"], S["RESTRICTED_ZONE_B"]),
        (S["ADMIN_NETWORK"], S["RESTRICTED_ZONE_B"]),
        (S["OFFICE_NETWORK"], S["RESTRICTED_ZONE_B"]),
        (S["RESTRICTED_ZONE_B"], S["CONTRACTOR_NETWORK"]),
        (S["OPERATIONAL_ZONE_B"], S["RESTRICTED_ZONE_B"]),
    ]

    policy_3 = [
        (S["PUBLIC_ACCESS_ZONE"], S["CONTRACTOR_NETWORK"]),
        (S["ADMIN_NETWORK"], S["CONTRACTOR_NETWORK"]),
        (S["OFFICE_NETWORK"], S["CONTRACTOR_NETWORK"]),
        (S["PUBLIC_ACCESS_ZONE"], S["RESTRICTED_ZONE_A"]),
        (S["ADMIN_NETWORK"], S["RESTRICTED_ZONE_A"]),
        (S["OFFICE_NETWORK"], S["RESTRICTED_ZONE_A"]),
        (S["PUBLIC_ACCESS_ZONE"], S["RESTRICTED_ZONE_B"]),
        (S["ADMIN_NETWORK"], S["RESTRICTED_ZONE_B"]),
        (S["OFFICE_NETWORK"], S["RESTRICTED_ZONE_B"]),
        (S["RESTRICTED_ZONE_A"], S["CONTRACTOR_NETWORK"]),
        (S["OPERATIONAL_ZONE_A"], S["RESTRICTED_ZONE_A"]),
    ]

    pairs = np.zeros((MISSION_PHASES, NUM_SUBNETS, NUM_SUBNETS), dtype=bool)
    for phase_idx, policy in enumerate([policy_1, policy_2, policy_3]):
        for si, di in policy:
            pairs[phase_idx, si, di] = True
            pairs[phase_idx, di, si] = True
    return pairs
