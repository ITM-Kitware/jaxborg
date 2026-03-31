import hashlib
import pickle
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    MAX_DETECTION_RANDOMS,
    MAX_SERVER_HOSTS,
    MAX_STEPS,
    MAX_USER_HOSTS,
    MISSION_PHASES,
    NUM_BLUE_AGENTS,
    NUM_GREEN_RANDOM_FIELDS,
    NUM_RED_AGENTS,
    NUM_RED_POLICY_RANDOM_FIELDS,
    NUM_SERVICES,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
    SERVICE_IDS,
    SERVICE_NAMES,
    SUBNET_IDS,
    SUBNET_NAMES,
    TOTAL_ACTION_ACTOR_SLOTS,
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


def cyborg_bank_index_from_key(key: jax.Array, bank_size: int) -> jax.Array:
    """Map a JAX reset key onto a cached CybORG bank entry."""
    bank_size = jnp.int32(bank_size)
    return jnp.bitwise_xor(key[0], key[1]) % bank_size


def cyborg_bank_seed_from_seed(seed: int, bank_size: int) -> int:
    """Return the cached CybORG topology seed corresponding to a JAX episode seed."""
    key = jax.random.PRNGKey(seed)
    return int(cyborg_bank_index_from_key(key, bank_size))


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
    red_agent_subnets = np.zeros((NUM_RED_AGENTS, NUM_SUBNETS), dtype=bool)
    _red_agent_initially_active = np.zeros(NUM_RED_AGENTS, dtype=bool)
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
        _red_agent_initially_active[red_idx] = agent_info.active
        if agent_info.allowed_subnets:
            for sub_enum in agent_info.allowed_subnets:
                cyborg_suffix = str(sub_enum)
                if cyborg_suffix in CYBORG_SUFFIX_TO_ID:
                    red_agent_subnets[red_idx, CYBORG_SUFFIX_TO_ID[cyborg_suffix]] = True
    red_initial_discovered_hosts = np.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=bool)
    red_initial_scanned_hosts = np.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=bool)
    controller = cyborg_env.environment_controller
    known_hosts_by_red = [set() for _ in range(NUM_RED_AGENTS)]
    scanned_hosts_by_red = [set() for _ in range(NUM_RED_AGENTS)]
    for red_idx in range(NUM_RED_AGENTS):
        iface = controller.agent_interfaces.get(f"red_agent_{red_idx}")
        if iface is None:
            continue
        action_space = getattr(iface, "action_space", None)
        if action_space is not None:
            for ip, known in getattr(action_space, "ip_address", {}).items():
                if not known:
                    continue
                hostname = state.ip_addresses.get(ip)
                if hostname in hostname_to_idx:
                    known_hosts_by_red[red_idx].add(hostname_to_idx[hostname])
        for sess in state.sessions.get(f"red_agent_{red_idx}", {}).values():
            for ip in getattr(sess, "ports", {}).keys():
                hostname = state.ip_addresses.get(ip)
                if hostname in hostname_to_idx:
                    scanned_hosts_by_red[red_idx].add(hostname_to_idx[hostname])
    for red_idx in range(NUM_RED_AGENTS):
        if known_hosts_by_red[red_idx]:
            red_start_hosts[red_idx] = min(known_hosts_by_red[red_idx])
        if _red_agent_initially_active[red_idx]:
            red_initial_discovered_hosts[red_idx, red_start_hosts[red_idx]] = True
            for hidx in known_hosts_by_red[red_idx]:
                red_initial_discovered_hosts[red_idx, hidx] = True
        # Inactive agents: DON'T pre-seed discovery from aspace.ip_address.
        # CybORG's FSM starts with empty host_states; the pre-populated IP
        # doesn't enter host_states until the agent processes an observation.
        for hidx in scanned_hosts_by_red[red_idx]:
            red_initial_scanned_hosts[red_idx, hidx] = True
    host_info_links = np.zeros((GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS), dtype=bool)
    for src_hostname, host in state.hosts.items():
        if src_hostname not in hostname_to_idx:
            continue
        src_idx = hostname_to_idx[src_hostname]
        for dst_hostname in getattr(host, "info", {}).keys():
            if dst_hostname in hostname_to_idx:
                host_info_links[src_idx, hostname_to_idx[dst_hostname]] = True

    green_agent_host, green_agent_active, green_count = _build_green_agent_map_numpy(
        host_active=host_active,
        host_subnet=host_subnet,
        host_is_user=host_is_user,
        num_hosts=num_hosts,
    )

    phase_boundaries = _compute_phase_boundaries(scenario.mission_phases)
    allowed_subnet_pairs = _compute_allowed_subnet_pairs(scenario.allowed_subnets_per_mphase)

    obs_host_map = _build_obs_host_map(
        host_subnet, host_is_server, host_is_user, host_is_router, host_active, num_hosts
    )

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
        red_agent_subnets=jnp.array(red_agent_subnets),
        red_initial_discovered_hosts=jnp.array(red_initial_discovered_hosts),
        red_initial_scanned_hosts=jnp.array(red_initial_scanned_hosts),
        host_info_links=jnp.array(host_info_links),
        green_agent_host=jnp.array(green_agent_host),
        green_agent_active=jnp.array(green_agent_active),
        num_green_agents=jnp.int32(green_count),
        phase_rewards=jnp.array(_build_phase_rewards_from_cyborg(cyborg_env)),
        phase_boundaries=jnp.array(phase_boundaries),
        allowed_subnet_pairs=jnp.array(allowed_subnet_pairs),
        obs_host_map=jnp.array(obs_host_map),
        blue_obs_subnets=jnp.array(_build_blue_obs_subnets()),
        comms_policy=jnp.array(_build_comms_policy()),
        max_steps=500,
        num_hosts=jnp.int32(num_hosts),
        green_agents_active=jnp.array(True),
        # Precomputed RNG arrays (defaults: disabled / zeros)
        green_randoms=jnp.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=jnp.float32),
        use_green_randoms=jnp.array(False),
        red_policy_randoms=jnp.full((MAX_STEPS, NUM_RED_AGENTS, NUM_RED_POLICY_RANDOM_FIELDS), 0.5, dtype=jnp.float32),
        use_red_policy_randoms=jnp.array(False),
        detection_randoms=jnp.zeros(MAX_DETECTION_RANDOMS, dtype=jnp.float32),
        use_detection_randoms=jnp.array(False),
        red_pid_deltas=jnp.zeros((MAX_STEPS, NUM_RED_AGENTS), dtype=jnp.int32),
        use_red_pid_deltas=jnp.array(False),
        blue_decoy_pid_deltas=jnp.zeros((MAX_STEPS, NUM_BLUE_AGENTS), dtype=jnp.int32),
        use_blue_decoy_pid_deltas=jnp.array(False),
        red_privesc_choices=jnp.zeros((MAX_STEPS, NUM_RED_AGENTS), dtype=jnp.int32),
        use_red_privesc_choices=jnp.array(False),
        red_session_check_choices=jnp.zeros((MAX_STEPS, NUM_RED_AGENTS), dtype=jnp.int32),
        red_session_check_hosts=jnp.full((MAX_STEPS, NUM_RED_AGENTS), -1, dtype=jnp.int32),
        use_red_session_check_choices=jnp.array(False),
        blue_decoy_type_choices=jnp.zeros((MAX_STEPS, NUM_BLUE_AGENTS), dtype=jnp.int32),
        use_blue_decoy_type_choices=jnp.array(False),
        green_host_order=jnp.zeros((MAX_STEPS, TOTAL_ACTION_ACTOR_SLOTS), dtype=jnp.int32),
        use_green_host_order=jnp.array(False),
        cyborg_random_exploit_source=jnp.array(False),
    )


_BANK_CACHE_DIR = Path(__file__).resolve().parents[2] / ".bank_cache"


def _hash_paths(*relative_paths: str) -> str:
    digest = hashlib.md5()
    root = Path(__file__).resolve().parent
    for rel in relative_paths:
        digest.update((root / rel).read_bytes())
    return digest.hexdigest()[:12]


def _topology_cache_key(num_steps: int, bank_size: int) -> str:
    return f"steps{num_steps}_bank{bank_size}_{_hash_paths('topology.py')}"


def _green_cache_key(num_steps: int, bank_size: int) -> str:
    return f"steps{num_steps}_bank{bank_size}_{_hash_paths('topology.py', 'cyborg_green_recorder.py')}"


def _red_policy_cache_key(num_steps: int, bank_size: int) -> str:
    return (
        f"steps{num_steps}_bank{bank_size}_"
        f"{_hash_paths('topology.py', 'cyborg_red_policy_recorder.py', 'agents/fsm_red.py')}"
    )


def _build_topology_bank(num_steps: int, bank_size: int) -> CC4Const:
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    consts = []
    for seed in range(bank_size):
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=num_steps,
        )
        cyborg = CybORG(scenario_generator=scenario, seed=seed)
        cyborg.reset()
        consts.append(build_const_from_cyborg(cyborg))

    return jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0), *consts)


def _build_green_random_bank(num_steps: int, bank_size: int) -> jax.Array:
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Actions import Sleep
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.cyborg_green_recorder import GreenRecorder
    from jaxborg.translate import build_mappings_from_cyborg

    green_randoms = []
    for seed in range(bank_size):
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=num_steps,
        )
        cyborg = CybORG(scenario_generator=scenario, seed=seed)
        wrapper = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        wrapper.reset()

        mappings = build_mappings_from_cyborg(cyborg)
        recorder = GreenRecorder()
        recorder.install(cyborg, mappings)

        sleep_actions = {agent: Sleep() for agent in wrapper.agents}
        for step_idx in range(num_steps):
            wrapper.step(actions=sleep_actions)
            recorder.extract_step(step_idx)

        green_randoms.append(recorder.to_jax_array())

    return jnp.stack(green_randoms, axis=0)


def _build_red_policy_random_bank(num_steps: int, bank_size: int) -> jax.Array:
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Actions import Sleep
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.cyborg_red_policy_recorder import RedPolicyRecorder
    from jaxborg.translate import build_mappings_from_cyborg

    tapes = []
    for seed in range(bank_size):
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=num_steps,
        )
        cyborg = CybORG(scenario_generator=scenario, seed=seed)
        wrapper = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        wrapper.reset()

        recorder = RedPolicyRecorder()
        recorder.install(cyborg, build_mappings_from_cyborg(cyborg))

        sleep_actions = {agent: Sleep() for agent in wrapper.agents}
        for _step in range(num_steps):
            wrapper.step(actions=sleep_actions)

        tapes.append(recorder.to_jax_array())

    return jnp.stack(tapes, axis=0)


@lru_cache(maxsize=None)
def get_cyborg_topology_bank(num_steps: int, bank_size: int) -> CC4Const:
    """Build (or load cached) bank of CybORG topologies for JAX resets."""
    if bank_size <= 0:
        raise ValueError(f"bank_size must be > 0, got {bank_size}")

    cache_dir = _BANK_CACHE_DIR
    key = _topology_cache_key(num_steps, bank_size)
    cache_path = cache_dir / f"topo_{key}.pkl"

    if cache_path.exists():
        print(f"Loading cached topology bank from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            np_tree = pickle.load(f)
        return jax.tree.map(jnp.asarray, np_tree)

    print(f"Building topology bank ({bank_size} seeds)...", flush=True)
    bank = _build_topology_bank(num_steps, bank_size)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np_tree = jax.tree.map(np.asarray, bank)
    with open(cache_path, "wb") as f:
        pickle.dump(np_tree, f)
    print(f"Cached topology bank to {cache_path}", flush=True)
    return bank


@lru_cache(maxsize=None)
def get_cyborg_green_random_bank(num_steps: int, bank_size: int) -> jax.Array:
    """Build (or load cached) bank of CybORG green random tapes for JAX resets."""
    if bank_size <= 0:
        raise ValueError(f"bank_size must be > 0, got {bank_size}")

    cache_dir = _BANK_CACHE_DIR
    key = _green_cache_key(num_steps, bank_size)
    cache_path = cache_dir / f"green_{key}.pkl"

    if cache_path.exists():
        print(f"Loading cached green random bank from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            arr = pickle.load(f)
        return jnp.asarray(arr)

    print(f"Building green random bank ({bank_size} seeds)...", flush=True)
    bank = _build_green_random_bank(num_steps, bank_size)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(np.asarray(bank), f)
    print(f"Cached green random bank to {cache_path}", flush=True)
    return bank


@lru_cache(maxsize=None)
def get_cyborg_red_policy_random_bank(num_steps: int, bank_size: int) -> jax.Array:
    """Build (or load cached) bank of CybORG native red-policy choice tapes."""
    if bank_size <= 0:
        raise ValueError(f"bank_size must be > 0, got {bank_size}")

    cache_dir = _BANK_CACHE_DIR
    key = _red_policy_cache_key(num_steps, bank_size)
    cache_path = cache_dir / f"red_policy_{key}.pkl"

    if cache_path.exists():
        print(f"Loading cached red policy random bank from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            arr = pickle.load(f)
        return jnp.asarray(arr)

    print(f"Building red policy random bank ({bank_size} seeds)...", flush=True)
    bank = _build_red_policy_random_bank(num_steps, bank_size)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(np.asarray(bank), f)
    print(f"Cached red policy random bank to {cache_path}", flush=True)
    return bank


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


def build_topology(key: jax.Array, num_steps: int = 500, *, training_mode: bool = False) -> CC4Const:
    """Build CC4 topology in pure JAX — JIT-compatible.

    Mimics EnterpriseScenarioGenerator: for each non-internet subnet, generates
    1 router + random server hosts (1-6) + random user hosts (3-10).
    Internet subnet gets 1 host (root_internet_host_0).

    Host indices follow alphabetical hostname ordering (same as build_const_from_cyborg):
    subnets ordered by CYBORG_SUBNET_SUFFIX, within each subnet: router < servers < users.
    """
    k_counts, k_services, k_red = jax.random.split(key, 3)
    k_users, k_servers = jax.random.split(k_counts)

    n_users = jax.random.randint(k_users, (8,), 3, 11)
    n_servers = jax.random.randint(k_servers, (8,), 1, 7)

    hosts_per_alpha = jnp.concatenate([1 + n_servers + n_users, jnp.array([1])])
    cumsum = jnp.cumsum(hosts_per_alpha)
    starts = jnp.concatenate([jnp.array([0]), cumsum[:-1]])
    num_hosts = cumsum[-1]

    j = jnp.arange(GLOBAL_MAX_HOSTS)
    alpha_idx = jnp.clip(jnp.searchsorted(cumsum, j + 1), 0, 8)
    offset = j - starts[alpha_idx]
    host_active = j < num_hosts

    host_subnet = _ALPHA_SUBNET_ORDER[alpha_idx]
    n_servers_pad = jnp.concatenate([n_servers, jnp.array([0])])

    host_is_internet = (alpha_idx == 8) & host_active
    host_is_router = (offset == 0) & (alpha_idx < 8) & host_active
    host_is_server = (offset >= 1) & (offset <= n_servers_pad[alpha_idx]) & (alpha_idx < 8) & host_active
    host_is_user = (offset > n_servers_pad[alpha_idx]) & (alpha_idx < 8) & host_active

    host_respond_to_ping = host_is_server | host_is_user
    host_has_bruteforceable_user = host_is_server | host_is_user
    host_has_rfi = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
    host_initial_max_pid = jnp.where(host_is_server | host_is_user, 5000, 0).astype(jnp.int32)

    svc_host = host_is_server | host_is_user
    is_operational = (host_subnet == SUBNET_IDS["OPERATIONAL_ZONE_A"]) | (
        host_subnet == SUBNET_IDS["OPERATIONAL_ZONE_B"]
    )
    initial_services = jnp.zeros((GLOBAL_MAX_HOSTS, NUM_SERVICES), dtype=jnp.bool_)
    initial_services = initial_services.at[:, SERVICE_IDS["SSHD"]].set(svc_host)
    initial_services = initial_services.at[:, SERVICE_IDS["OTSERVICE"]].set(svc_host & is_operational)

    k_addon_n, k_addon_sel = jax.random.split(k_services)
    num_addons = jax.random.randint(k_addon_n, (GLOBAL_MAX_HOSTS,), 0, 4)
    gumbel = jax.random.gumbel(k_addon_sel, (GLOBAL_MAX_HOSTS, 3))
    ranks = jnp.argsort(jnp.argsort(-gumbel, axis=1), axis=1)
    addon_selected = (ranks < num_addons[:, None]) & svc_host[:, None]
    initial_services = initial_services.at[:, SERVICE_IDS["APACHE2"]].set(
        initial_services[:, SERVICE_IDS["APACHE2"]] | addon_selected[:, 0]
    )
    initial_services = initial_services.at[:, SERVICE_IDS["MYSQLD"]].set(
        initial_services[:, SERVICE_IDS["MYSQLD"]] | addon_selected[:, 1]
    )
    initial_services = initial_services.at[:, SERVICE_IDS["SMTP"]].set(
        initial_services[:, SERVICE_IDS["SMTP"]] | addon_selected[:, 2]
    )

    subnet_router_idx = jnp.full(NUM_SUBNETS, -1, dtype=jnp.int32)
    for alpha_i in range(9):
        sid = int(_ALPHA_SUBNET_ORDER_NP[alpha_i])
        subnet_router_idx = subnet_router_idx.at[sid].set(starts[alpha_i])

    data_links = jnp.zeros((GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_)
    router_of = subnet_router_idx[host_subnet]
    is_regular = host_active & ~host_is_router & ~host_is_internet
    data_links = data_links.at[j, router_of].set(is_regular)
    data_links = data_links.at[router_of, j].set(is_regular)
    for src_name, neighbor_names in _ROUTER_LINKS.items():
        src_r = subnet_router_idx[SUBNET_IDS[src_name]]
        for dst_name in neighbor_names:
            dst_r = subnet_router_idx[SUBNET_IDS[dst_name]]
            data_links = data_links.at[src_r, dst_r].set(True)
            data_links = data_links.at[dst_r, src_r].set(True)

    blue_agent_hosts = jnp.zeros((NUM_BLUE_AGENTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_)
    for i, snames in enumerate(BLUE_AGENT_SUBNETS):
        mask = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
        for sn in snames:
            mask = mask | (host_active & (host_subnet == SUBNET_IDS[sn]))
        blue_agent_hosts = blue_agent_hosts.at[i].set(mask)

    red_start_hosts = jnp.zeros(NUM_RED_AGENTS, dtype=jnp.int32)
    for i, snames in enumerate(RED_AGENT_SUBNETS):
        k_i = jax.random.fold_in(k_red, i)
        subnet_mask = jnp.zeros(NUM_SUBNETS, dtype=jnp.bool_)
        for sn in snames:
            subnet_mask = subnet_mask.at[SUBNET_IDS[sn]].set(True)
        valid = host_active & ~host_is_router & ~host_is_internet & subnet_mask[host_subnet]
        gumbel_noise = jax.random.gumbel(k_i, (GLOBAL_MAX_HOSTS,))
        masked_gumbel = jnp.where(valid, gumbel_noise, jnp.float32(-1e9))
        red_start_hosts = red_start_hosts.at[i].set(jnp.argmax(masked_gumbel))

    # Only red_agent_0 is initially active; others activate via session reassignment
    red_initial_discovered = jnp.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_)
    red_initial_scanned = jnp.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_)
    red_initial_discovered = red_initial_discovered.at[0, red_start_hosts[0]].set(True)

    host_info_links = jnp.zeros((GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_)
    server0_idx = jnp.full(NUM_SUBNETS, -1, dtype=jnp.int32)
    for alpha_i in range(8):
        sid = int(_ALPHA_SUBNET_ORDER_NP[alpha_i])
        server0_idx = server0_idx.at[sid].set(starts[alpha_i] + 1)
    for src_name, neighbor_names in _ROUTER_LINKS.items():
        if src_name == "INTERNET":
            continue
        src_s0 = server0_idx[SUBNET_IDS[src_name]]
        for dst_name in neighbor_names:
            if dst_name == "INTERNET":
                continue
            dst_s0 = server0_idx[SUBNET_IDS[dst_name]]
            host_info_links = host_info_links.at[src_s0, dst_s0].set(True)

    green_agent_host, green_agent_active, num_green_agents = _build_green_agent_map_jax(
        host_active=host_active,
        host_subnet=host_subnet.astype(jnp.int32),
        host_is_user=host_is_user,
    )

    obs_host_map = jnp.full((NUM_SUBNETS, OBS_HOSTS_PER_SUBNET), GLOBAL_MAX_HOSTS, dtype=jnp.int32)
    router_slot = MAX_SERVER_HOSTS + MAX_USER_HOSTS
    host_indices = jnp.arange(GLOBAL_MAX_HOSTS)
    for sid in range(NUM_SUBNETS):
        is_srv = host_active & host_is_server & (host_subnet == sid)
        srv_idx = jnp.where(is_srv, j, GLOBAL_MAX_HOSTS)
        sorted_srv = jnp.sort(srv_idx)
        for slot in range(MAX_SERVER_HOSTS):
            obs_host_map = obs_host_map.at[sid, slot].set(sorted_srv[slot])
        is_usr = host_active & host_is_user & (host_subnet == sid)
        usr_idx = jnp.where(is_usr, j, GLOBAL_MAX_HOSTS)
        sorted_usr = jnp.sort(usr_idx)
        for slot in range(MAX_USER_HOSTS):
            obs_host_map = obs_host_map.at[sid, MAX_SERVER_HOSTS + slot].set(sorted_usr[slot])
        is_rtr = host_active & host_is_router & (host_subnet == sid)
        rtr_idx = jnp.where(is_rtr, host_indices, GLOBAL_MAX_HOSTS)
        obs_host_map = obs_host_map.at[sid, router_slot].set(jnp.min(rtr_idx))

    phase_boundaries = jnp.array(_compute_phase_boundaries(_compute_mission_phases(num_steps)))

    return CC4Const(
        host_active=host_active,
        host_subnet=host_subnet.astype(jnp.int32),
        host_is_router=host_is_router,
        host_is_server=host_is_server,
        host_is_user=host_is_user,
        subnet_adjacency=_SUBNET_ADJACENCY,
        data_links=data_links,
        initial_services=initial_services,
        host_has_bruteforceable_user=host_has_bruteforceable_user,
        host_has_rfi=host_has_rfi,
        host_respond_to_ping=host_respond_to_ping,
        host_initial_max_pid=host_initial_max_pid,
        blue_agent_subnets=_BLUE_AGENT_SUBNETS_BOOL,
        blue_agent_hosts=blue_agent_hosts,
        red_start_hosts=red_start_hosts,
        red_agent_subnets=_RED_AGENT_SUBNETS_BOOL,
        red_initial_discovered_hosts=red_initial_discovered,
        red_initial_scanned_hosts=red_initial_scanned,
        host_info_links=host_info_links,
        green_agent_host=green_agent_host,
        green_agent_active=green_agent_active,
        num_green_agents=num_green_agents,
        phase_rewards=_PHASE_REWARDS,
        phase_boundaries=phase_boundaries,
        allowed_subnet_pairs=_ALLOWED_SUBNET_PAIRS,
        obs_host_map=obs_host_map,
        blue_obs_subnets=_BLUE_OBS_SUBNETS,
        comms_policy=_COMMS_POLICY,
        max_steps=num_steps,
        num_hosts=num_hosts,
        green_agents_active=jnp.array(True),
        # Precomputed RNG arrays (defaults: disabled / zeros).
        # In training_mode, use minimal (1,...) shapes to reduce PyTree size and
        # XLA trace complexity.  The use_* flags are always False here, so the
        # full-size arrays are never read at runtime — but XLA still traces both
        # branches of each jax.lax.cond and includes the array shapes in the
        # HLO program.  Smaller shapes → smaller HLO → faster compile + possibly
        # smaller GPU kernels.
        green_randoms=jnp.zeros(
            (1, 1, 1) if training_mode else (MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS),
            dtype=jnp.float32,
        ),
        use_green_randoms=jnp.array(False),
        red_policy_randoms=jnp.full(
            (1, 1, 1) if training_mode else (MAX_STEPS, NUM_RED_AGENTS, NUM_RED_POLICY_RANDOM_FIELDS),
            0.5,
            dtype=jnp.float32,
        ),
        use_red_policy_randoms=jnp.array(False),
        detection_randoms=jnp.zeros(
            (1,) if training_mode else (MAX_DETECTION_RANDOMS,),
            dtype=jnp.float32,
        ),
        use_detection_randoms=jnp.array(False),
        red_pid_deltas=jnp.zeros(
            (1, 1) if training_mode else (MAX_STEPS, NUM_RED_AGENTS),
            dtype=jnp.int32,
        ),
        use_red_pid_deltas=jnp.array(False),
        blue_decoy_pid_deltas=jnp.zeros(
            (1, 1) if training_mode else (MAX_STEPS, NUM_BLUE_AGENTS),
            dtype=jnp.int32,
        ),
        use_blue_decoy_pid_deltas=jnp.array(False),
        red_privesc_choices=jnp.zeros(
            (1, 1) if training_mode else (MAX_STEPS, NUM_RED_AGENTS),
            dtype=jnp.int32,
        ),
        use_red_privesc_choices=jnp.array(False),
        red_session_check_choices=jnp.zeros(
            (1, 1) if training_mode else (MAX_STEPS, NUM_RED_AGENTS),
            dtype=jnp.int32,
        ),
        red_session_check_hosts=jnp.full(
            (1, 1) if training_mode else (MAX_STEPS, NUM_RED_AGENTS),
            -1,
            dtype=jnp.int32,
        ),
        use_red_session_check_choices=jnp.array(False),
        blue_decoy_type_choices=jnp.zeros(
            (1, 1) if training_mode else (MAX_STEPS, NUM_BLUE_AGENTS),
            dtype=jnp.int32,
        ),
        use_blue_decoy_type_choices=jnp.array(False),
        green_host_order=jnp.zeros(
            (1, 1) if training_mode else (MAX_STEPS, TOTAL_ACTION_ACTOR_SLOTS),
            dtype=jnp.int32,
        ),
        use_green_host_order=jnp.array(False),
        cyborg_random_exploit_source=jnp.array(False),
    )


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

_ALPHA_SUBNET_ORDER_NP = np.array([5, 4, 6, 2, 3, 7, 0, 1, 8])
_ALPHA_SUBNET_ORDER = jnp.array(_ALPHA_SUBNET_ORDER_NP)
_CYBORG_GENERATION_SUBNET_ORDER_NP = np.array(
    [
        SUBNET_IDS["RESTRICTED_ZONE_A"],
        SUBNET_IDS["OPERATIONAL_ZONE_A"],
        SUBNET_IDS["RESTRICTED_ZONE_B"],
        SUBNET_IDS["OPERATIONAL_ZONE_B"],
        SUBNET_IDS["CONTRACTOR_NETWORK"],
        SUBNET_IDS["PUBLIC_ACCESS_ZONE"],
        SUBNET_IDS["ADMIN_NETWORK"],
        SUBNET_IDS["OFFICE_NETWORK"],
        SUBNET_IDS["INTERNET"],
    ],
    dtype=np.int32,
)
_CYBORG_GENERATION_SUBNET_ORDER = jnp.array(_CYBORG_GENERATION_SUBNET_ORDER_NP)


def _build_green_agent_map_numpy(
    host_active: np.ndarray,
    host_subnet: np.ndarray,
    host_is_user: np.ndarray,
    num_hosts: int,
) -> tuple[np.ndarray, np.ndarray, np.int32]:
    green_agent_host = np.full(GLOBAL_MAX_HOSTS, -1, dtype=np.int32)
    green_agent_active = host_active & host_is_user
    green_count = 0
    for sid in _CYBORG_GENERATION_SUBNET_ORDER_NP:
        for host_idx in range(num_hosts):
            if not host_active[host_idx]:
                continue
            if host_subnet[host_idx] != sid:
                continue
            if not host_is_user[host_idx]:
                continue
            green_agent_host[host_idx] = green_count
            green_count += 1
    return green_agent_host, green_agent_active, np.int32(green_count)


def _build_green_agent_map_jax(
    host_active: jax.Array,
    host_subnet: jax.Array,
    host_is_user: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green_agent_host = jnp.full(GLOBAL_MAX_HOSTS, -1, dtype=jnp.int32)
    green_agent_active = host_active & host_is_user
    next_idx = jnp.int32(0)
    host_indices = jnp.arange(GLOBAL_MAX_HOSTS, dtype=jnp.int32)

    for sid in _CYBORG_GENERATION_SUBNET_ORDER_NP:
        user_mask = host_active & host_is_user & (host_subnet == sid)
        sorted_users = jnp.sort(jnp.where(user_mask, host_indices, GLOBAL_MAX_HOSTS))
        for slot in range(MAX_USER_HOSTS):
            host_idx = sorted_users[slot]
            valid = host_idx < GLOBAL_MAX_HOSTS
            green_agent_host = jax.lax.cond(
                valid,
                lambda arr: arr.at[host_idx].set(next_idx),
                lambda arr: arr,
                green_agent_host,
            )
            next_idx = next_idx + valid.astype(jnp.int32)

    return green_agent_host, green_agent_active, next_idx


def _build_obs_host_map(
    host_subnet: np.ndarray,
    host_is_server: np.ndarray,
    host_is_user: np.ndarray,
    host_is_router: np.ndarray,
    host_active: np.ndarray,
    num_hosts: int,
) -> np.ndarray:
    obs_map = np.full((NUM_SUBNETS, OBS_HOSTS_PER_SUBNET), GLOBAL_MAX_HOSTS, dtype=np.int32)
    router_slot = MAX_SERVER_HOSTS + MAX_USER_HOSTS
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
        router_hosts = sorted(
            [h for h in range(num_hosts) if host_active[h] and host_subnet[h] == sid and host_is_router[h]]
        )
        if router_hosts:
            obs_map[sid, router_slot] = router_hosts[0]
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


def _build_blue_agent_subnets_bool():
    arr = np.zeros((NUM_BLUE_AGENTS, NUM_SUBNETS), dtype=bool)
    for i, snames in enumerate(BLUE_AGENT_SUBNETS):
        for sn in snames:
            arr[i, SUBNET_IDS[sn]] = True
    return jnp.array(arr)


def _build_red_agent_subnets_bool():
    arr = np.zeros((NUM_RED_AGENTS, NUM_SUBNETS), dtype=bool)
    for i, snames in enumerate(RED_AGENT_SUBNETS):
        for sn in snames:
            arr[i, SUBNET_IDS[sn]] = True
    return jnp.array(arr)


_SUBNET_ADJACENCY = jnp.array(_subnet_nacl_adjacency())
_PHASE_REWARDS = jnp.array(_build_phase_rewards())
_ALLOWED_SUBNET_PAIRS = jnp.array(_build_allowed_subnet_pairs_pure())
_COMMS_POLICY = jnp.array(_build_comms_policy())
_BLUE_OBS_SUBNETS = jnp.array(_build_blue_obs_subnets())
_BLUE_AGENT_SUBNETS_BOOL = _build_blue_agent_subnets_bool()
_RED_AGENT_SUBNETS_BOOL = _build_red_agent_subnets_bool()
