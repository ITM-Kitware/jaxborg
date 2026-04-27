import hashlib
import multiprocessing
import os
import pickle
import signal
from functools import lru_cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import (
    CYBORG_SUBNET_SUFFIX,  # noqa: F401 — re-exported
    CYBORG_SUFFIX_TO_ID,  # noqa: F401 — re-exported
    GLOBAL_MAX_HOSTS,
    MAX_DETECTION_RANDOMS,
    MAX_SERVER_HOSTS,
    MAX_STEPS,
    MAX_USER_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_DECOY_TYPES,
    NUM_GREEN_RANDOM_FIELDS,
    NUM_RED_AGENTS,
    NUM_RED_POLICY_RANDOM_FIELDS,
    NUM_SERVICES,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
    SERVICE_IDS,
    SUBNET_IDS,
    TOTAL_ACTION_ACTOR_SLOTS,
)
from jaxborg.state import CC4Const
from jaxborg.topology_numpy import (
    BLUE_AGENT_SUBNETS,
    RED_AGENT_SUBNETS,
    _build_allowed_subnet_pairs_pure,
    _build_blue_obs_subnets,
    _build_comms_policy,
    _build_phase_rewards,
    _compute_allowed_subnet_pairs,  # noqa: F401 — re-exported
    _compute_mission_phases,
    _compute_phase_boundaries,
    _subnet_nacl_adjacency,
    build_const_arrays_from_cyborg,
    get_phase_rewards_bank,
    get_router_link_bank,
)

# Eagerly materialize the JAX-side router-link bank as a module constant.
# Building inside build_topology (which is traced) and caching to a global
# would leak a tracer; computing once at import time gives JAX a concrete
# constant that gets inlined during JIT compilation.
_ROUTER_LINK_BANK_JNP: jax.Array = jnp.asarray(get_router_link_bank())

# Axis C: phase-reward bank materialized eagerly for the same reason.
_PHASE_REWARDS_BANK_JNP: jax.Array = jnp.asarray(get_phase_rewards_bank())


def _router_link_bank_jnp() -> jax.Array:
    return _ROUTER_LINK_BANK_JNP


def _phase_rewards_bank_jnp() -> jax.Array:
    return _PHASE_REWARDS_BANK_JNP


def cyborg_bank_index_from_key(key: jax.Array, bank_size: int) -> jax.Array:
    """Map a JAX reset key onto a cached CybORG bank entry."""
    bank_size = jnp.int32(bank_size)
    return jnp.bitwise_xor(key[0], key[1]) % bank_size


def cyborg_bank_seed_from_seed(seed: int, bank_size: int) -> int:
    """Return the cached CybORG topology seed corresponding to a JAX episode seed."""
    key = jax.random.PRNGKey(seed)
    return int(cyborg_bank_index_from_key(key, bank_size))


def build_const_from_cyborg(cyborg_env) -> CC4Const:
    """Extract static topology from a live CybORG environment."""
    arrays = build_const_arrays_from_cyborg(cyborg_env)
    return CC4Const(**{k: jnp.asarray(v) for k, v in arrays.items()})


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


_PARALLEL_THRESHOLD = 8  # use multiprocessing for bank_size >= this


def _pool_init():
    """Ignore SIGINT in workers so Ctrl-C is handled by the main process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _pool_workers(bank_size: int) -> int:
    """Choose worker count: min(bank_size, cpus - 8), clamped to [1, 56]."""
    cpus = os.cpu_count() or 1
    return max(1, min(bank_size, cpus - 8, 56))


def _build_topology_bank(num_steps: int, bank_size: int) -> CC4Const:
    if bank_size >= _PARALLEL_THRESHOLD:
        from jaxborg.topology_workers import _build_one_topology

        workers = _pool_workers(bank_size)
        print(f"  Building topology bank ({bank_size} seeds, {workers} workers)...", flush=True)
        worker_fn = partial(_build_one_topology, num_steps=num_steps)
        with multiprocessing.get_context("spawn").Pool(workers, initializer=_pool_init) as pool:
            dicts = list(pool.imap(worker_fn, range(bank_size)))
        stacked = {k: jnp.stack([jnp.asarray(d[k]) for d in dicts]) for k in dicts[0]}
        return CC4Const(**stacked)
    else:
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        dicts = []
        for seed in range(bank_size):
            scenario = EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=EnterpriseGreenAgent,
                red_agent_class=FiniteStateRedAgent,
                steps=num_steps,
            )
            cyborg = CybORG(scenario_generator=scenario, seed=seed)
            cyborg.reset()
            dicts.append(build_const_arrays_from_cyborg(cyborg))

    stacked = {k: jnp.stack([jnp.asarray(d[k]) for d in dicts]) for k in dicts[0]}
    return CC4Const(**stacked)


def _build_green_random_bank(num_steps: int, bank_size: int) -> jax.Array:
    if bank_size >= _PARALLEL_THRESHOLD:
        from jaxborg.topology_workers import _build_one_green

        workers = _pool_workers(bank_size)
        print(f"  Building green random bank ({bank_size} seeds, {workers} workers)...", flush=True)
        worker_fn = partial(_build_one_green, num_steps=num_steps)
        with multiprocessing.get_context("spawn").Pool(workers, initializer=_pool_init) as pool:
            arrays = list(pool.imap(worker_fn, range(bank_size)))
    else:
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.cyborg_green_recorder import GreenRecorder
        from jaxborg.translate import build_mappings_from_cyborg

        arrays = []
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

            arrays.append(np.asarray(recorder.to_jax_array()))

    return jnp.stack(arrays, axis=0)


def _build_red_policy_random_bank(num_steps: int, bank_size: int) -> jax.Array:
    if bank_size >= _PARALLEL_THRESHOLD:
        from jaxborg.topology_workers import _build_one_red_policy

        workers = _pool_workers(bank_size)
        print(f"  Building red policy bank ({bank_size} seeds, {workers} workers)...", flush=True)
        worker_fn = partial(_build_one_red_policy, num_steps=num_steps)
        with multiprocessing.get_context("spawn").Pool(workers, initializer=_pool_init) as pool:
            arrays = list(pool.imap(worker_fn, range(bank_size)))
    else:
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        from jaxborg.cyborg_red_policy_recorder import RedPolicyRecorder
        from jaxborg.translate import build_mappings_from_cyborg

        arrays = []
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
            for _ in range(num_steps):
                wrapper.step(actions=sleep_actions)

            arrays.append(np.asarray(recorder.to_jax_array()))

    return jnp.stack(arrays, axis=0)


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


def build_topology(
    key: jax.Array,
    num_steps: int = 500,
    *,
    training_mode: bool = False,
    vary_router_links: bool = False,
    vary_phase_rewards: bool = False,
) -> CC4Const:
    """Build CC4 topology in pure JAX — JIT-compatible.

    Mimics EnterpriseScenarioGenerator: for each non-internet subnet, generates
    1 router + random server hosts (1-6) + random user hosts (3-10).
    Internet subnet gets 1 host (root_internet_host_0).

    Host indices follow alphabetical hostname ordering (same as build_const_from_cyborg):
    subnets ordered by CYBORG_SUBNET_SUFFIX, within each subnet: router < servers < users.

    Axis B: when ``vary_router_links=True``, the inter-router (subnet-to-subnet)
    backbone topology is sampled per reset key from the precomputed bank in
    :func:`jaxborg.topology_numpy.get_router_link_bank` (default tree + extra
    candidate edges). When False, bank entry 0 (default tree) is used and the
    behavior matches the legacy static ``_ROUTER_LINKS`` dict.

    Axis C: when ``vary_phase_rewards=True``, the per-(phase, subnet, component)
    reward landscape is sampled per reset key from the precomputed bank in
    :func:`jaxborg.topology_numpy.get_phase_rewards_bank` (default matrix +
    variants reassigning which subnet is the high-value target in phases 1, 2).
    When False, bank entry 0 (the default) is used.
    """
    k_counts, k_services, k_red, k_pids, k_router, k_phase = jax.random.split(key, 6)
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

    # Randomize initial max PID per host to match CybORG's _generate_pid().
    # CybORG samples each service process PID from randint(1000, 10000);
    # host_initial_max_pid = max over those draws.  Routers/internet have
    # no service processes so remain 0.
    pid_samples = jax.random.randint(k_pids, (GLOBAL_MAX_HOSTS, NUM_SERVICES), 1000, 10000)
    masked_pids = jnp.where(initial_services, pid_samples, jnp.int32(0))
    host_initial_max_pid = jnp.max(masked_pids, axis=1).astype(jnp.int32)

    subnet_router_idx = jnp.full(NUM_SUBNETS, -1, dtype=jnp.int32)
    for alpha_i in range(9):
        sid = int(_ALPHA_SUBNET_ORDER_NP[alpha_i])
        subnet_router_idx = subnet_router_idx.at[sid].set(starts[alpha_i])

    # Axis B: select router-to-router adjacency matrix.
    # Bank[0] is the default tree (matches legacy _ROUTER_LINKS exactly).
    router_bank = _router_link_bank_jnp()
    bank_size = router_bank.shape[0]
    bank_idx = jax.lax.cond(
        vary_router_links,
        lambda: jax.random.randint(k_router, (), 0, bank_size, dtype=jnp.int32),
        lambda: jnp.int32(0),
    )
    router_adj = router_bank[bank_idx]  # (NUM_SUBNETS, NUM_SUBNETS) bool

    data_links = jnp.zeros((GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_)
    router_of = subnet_router_idx[host_subnet]
    is_regular = host_active & ~host_is_router & ~host_is_internet
    data_links = data_links.at[j, router_of].set(is_regular)
    data_links = data_links.at[router_of, j].set(is_regular)

    # Apply inter-router edges from the selected router_adj matrix. Vectorized
    # scatter over the (NUM_SUBNETS, NUM_SUBNETS) grid: for each (i, j), if
    # router_adj[i, j] is True, set data_links at (router_idx[i], router_idx[j]).
    sub_i, sub_j = jnp.meshgrid(jnp.arange(NUM_SUBNETS), jnp.arange(NUM_SUBNETS), indexing="ij")
    src_h = subnet_router_idx[sub_i.flatten()]
    dst_h = subnet_router_idx[sub_j.flatten()]
    adj_flat = router_adj.flatten()
    existing_dl = data_links[src_h, dst_h]
    data_links = data_links.at[src_h, dst_h].set(existing_dl | adj_flat)

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

    # Axis B: vectorized server0-to-server0 link installation.  Skip pairs
    # where either endpoint is INTERNET (no server0 there).  router_adj is the
    # same matrix selected above for data_links.
    s0_valid = server0_idx >= 0  # (NUM_SUBNETS,)
    valid_pair = s0_valid[:, None] & s0_valid[None, :]
    info_adj = router_adj & valid_pair  # (NUM_SUBNETS, NUM_SUBNETS) bool
    s0_safe = jnp.maximum(server0_idx, 0)  # clamp -1 → 0; masked False below
    src_s0 = s0_safe[sub_i.flatten()]
    dst_s0 = s0_safe[sub_j.flatten()]
    info_flat = info_adj.flatten()
    existing_info = host_info_links[src_s0, dst_s0]
    host_info_links = host_info_links.at[src_s0, dst_s0].set(existing_info | info_flat)

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

    # Axis C: select phase_rewards from the precomputed bank.  Bank[0] is the
    # default matrix (matches legacy _build_phase_rewards exactly).
    phase_rewards_bank = _phase_rewards_bank_jnp()
    pr_bank_size = phase_rewards_bank.shape[0]
    pr_bank_idx = jax.lax.cond(
        vary_phase_rewards,
        lambda: jax.random.randint(k_phase, (), 0, pr_bank_size, dtype=jnp.int32),
        lambda: jnp.int32(0),
    )
    phase_rewards = phase_rewards_bank[pr_bank_idx]

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
        phase_rewards=phase_rewards,
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
            (1, 1, 1) if training_mode else (MAX_STEPS, NUM_BLUE_AGENTS, NUM_DECOY_TYPES),
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
        red_exploit_session_choices=jnp.zeros(
            (1, 1) if training_mode else (MAX_STEPS, NUM_RED_AGENTS),
            dtype=jnp.int32,
        ),
        use_red_exploit_session_choices=jnp.array(False),
    )


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
