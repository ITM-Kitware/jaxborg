import json
import os
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import (
    CC4_CONFIG,
    CYBORG_SUBNET_SUFFIX,  # noqa: F401 — re-exported
    CYBORG_SUFFIX_TO_ID,  # noqa: F401 — re-exported
    GLOBAL_MAX_HOSTS,
    MAX_SERVER_HOSTS,
    MAX_USER_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
    NUM_SERVICES,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
    SERVICE_IDS,
    SUBNET_IDS,
)
from jaxborg.scenarios.cc4.topology_numpy import (
    _ROUTER_LINKS,
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
)
from jaxborg.scenarios.config import ScenarioConfig
from jaxborg.state import SimulatorConst, create_initial_const

TOPOLOGY_SNAPSHOT_METADATA_KEY = "__metadata_json__"
TOPOLOGY_SNAPSHOT_FORMAT = "jaxborg.cc4.topology"
TOPOLOGY_SNAPSHOT_VERSION = 1

TOPOLOGY_SNAPSHOT_FIELDS = (
    "host_active",
    "host_subnet",
    "host_is_router",
    "host_is_server",
    "host_is_user",
    "subnet_adjacency",
    "data_links",
    "initial_services",
    "host_has_bruteforceable_user",
    "host_has_rfi",
    "host_respond_to_ping",
    "host_initial_max_pid",
    "blue_agent_subnets",
    "blue_agent_hosts",
    "red_start_hosts",
    "red_agent_subnets",
    "red_initial_discovered_hosts",
    "red_initial_scanned_hosts",
    "host_info_links",
    "green_agent_host",
    "green_agent_active",
    "num_green_agents",
    "green_agents_active",
    "phase_rewards",
    "phase_boundaries",
    "allowed_subnet_pairs",
    "obs_host_map",
    "blue_obs_subnets",
    "comms_policy",
    "max_steps",
    "num_hosts",
)


def build_const_from_cyborg(cyborg_env) -> SimulatorConst:
    """Extract static topology from a live CybORG environment."""
    arrays = build_const_arrays_from_cyborg(cyborg_env)
    return SimulatorConst(**{k: jnp.asarray(v) for k, v in arrays.items()})


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[4],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


def _cyborg_version() -> str:
    try:
        return importlib_metadata.version("cyborg")
    except importlib_metadata.PackageNotFoundError:
        return ""


_SCENARIO_CONFIG_DIGEST_FIELDS = (
    "num_hosts",
    "num_subnets",
    "num_blue_agents",
    "num_red_agents",
    "num_services",
    "num_decoy_types",
    "mission_phases",
    "max_steps",
    "message_length",
    "blue_max_observed_subnets",
    "max_tracked_session_pids",
    "max_tracked_suspicious_pids",
    "obs_hosts_per_subnet",
)


def _scenario_config_digest(cfg: ScenarioConfig) -> dict[str, int]:
    return {name: int(getattr(cfg, name)) for name in _SCENARIO_CONFIG_DIGEST_FIELDS}


def _snapshot_metadata(metadata: Mapping[str, Any] | None, cfg: ScenarioConfig) -> dict[str, Any]:
    out = {
        "format": TOPOLOGY_SNAPSHOT_FORMAT,
        "format_version": TOPOLOGY_SNAPSHOT_VERSION,
        "cyborg_version": _cyborg_version(),
        "jaxborg_git_sha": _git_sha(),
        "scenario_config": _scenario_config_digest(cfg),
    }
    if metadata:
        out.update(dict(metadata))
    return out


def save_topology(
    const: SimulatorConst,
    path: str | os.PathLike,
    metadata: Mapping[str, Any] | None = None,
    *,
    scenario_config: ScenarioConfig = CC4_CONFIG,
) -> None:
    """Save a static CC4 topology snapshot as an npz file.

    The active ``ScenarioConfig`` is embedded in metadata so ``load_topology``
    can detect snapshots saved against an incompatible config (different
    ``max_steps``, host/subnet counts, etc.) and refuse to load.

    Replay/random tape fields are not part of the runtime const ABI.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {name: np.asarray(getattr(const, name)) for name in TOPOLOGY_SNAPSHOT_FIELDS}
    arrays[TOPOLOGY_SNAPSHOT_METADATA_KEY] = np.asarray(
        json.dumps(_snapshot_metadata(metadata, scenario_config), sort_keys=True)
    )
    np.savez_compressed(out_path, **arrays)


def load_topology_metadata(path: str | os.PathLike) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        if TOPOLOGY_SNAPSHOT_METADATA_KEY not in data:
            return {}
        return json.loads(str(np.asarray(data[TOPOLOGY_SNAPSHOT_METADATA_KEY]).item()))


def load_topology(
    path: str | os.PathLike,
    *,
    training_mode: bool = False,
    scenario_config: ScenarioConfig = CC4_CONFIG,
) -> SimulatorConst:
    """Load a static CC4 topology snapshot.

    Validates that the snapshot was saved against a compatible ``ScenarioConfig``
    (matching host/subnet counts, ``max_steps``, mission phases, etc.).  Mismatch
    raises ``ValueError`` rather than silently producing a const with
    inconsistent shape arrays.

    ``training_mode`` is accepted for API symmetry with ``build_topology``.
    """
    del training_mode
    snapshot_path = Path(path)
    with np.load(snapshot_path, allow_pickle=False) as data:
        missing = [name for name in TOPOLOGY_SNAPSHOT_FIELDS if name not in data]
        if missing:
            raise ValueError(f"{snapshot_path} is missing topology snapshot fields: {', '.join(missing)}")

        if TOPOLOGY_SNAPSHOT_METADATA_KEY in data:
            metadata = json.loads(str(np.asarray(data[TOPOLOGY_SNAPSHOT_METADATA_KEY]).item()))
            saved_digest = metadata.get("scenario_config")
            current_digest = _scenario_config_digest(scenario_config)
            if saved_digest is not None and saved_digest != current_digest:
                diffs = {
                    k: (saved_digest.get(k), current_digest.get(k))
                    for k in current_digest
                    if saved_digest.get(k) != current_digest.get(k)
                }
                raise ValueError(
                    f"{snapshot_path} was saved against an incompatible ScenarioConfig; "
                    f"mismatched fields (saved → current): {diffs}"
                )

        replacements = {}
        for name in TOPOLOGY_SNAPSHOT_FIELDS:
            value = data[name]
            replacements[name] = int(np.asarray(value)) if name == "max_steps" else jnp.asarray(value)

    return create_initial_const(scenario_config).replace(**replacements)


def build_topology(
    key: jax.Array,
    num_steps: int = 500,
    *,
    training_mode: bool = False,
    op_zone_min_servers: int | tuple[int, int] | None = None,
) -> SimulatorConst:
    """Build CC4 topology in pure JAX — JIT-compatible.

    Mimics EnterpriseScenarioGenerator: for each non-internet subnet, generates
    1 router + random server hosts (1-6) + random user hosts (3-10).
    Internet subnet gets 1 host (root_internet_host_0).

    ``op_zone_min_servers`` controls the operational-zone server floor.
    Pass an int to force both OPS-A and OPS-B to the same value (legacy
    behavior). Pass a 2-tuple ``(a_floor, b_floor)`` to set them
    independently — used by the topology bank builder to produce totals
    that aren't multiples of 2 (e.g. 3 = 1+2 for balanced AUTH/DB/WEB
    role assignment across three resilience candidates).

    Host indices follow alphabetical hostname ordering (same as build_const_from_cyborg):
    subnets ordered by CYBORG_SUBNET_SUFFIX, within each subnet: router < servers < users.
    """
    k_counts, k_services, k_red, k_pids = jax.random.split(key, 4)
    k_users, k_servers = jax.random.split(k_counts)

    n_users = jax.random.randint(k_users, (8,), 3, 11)
    random_n = jax.random.randint(k_servers, (8,), 1, 7)
    if op_zone_min_servers is not None:
        if isinstance(op_zone_min_servers, tuple):
            a_floor, b_floor = op_zone_min_servers
        else:
            a_floor = b_floor = int(op_zone_min_servers)
        # alpha-order positions 3 and 4 are OPERATIONAL_ZONE_A / OPERATIONAL_ZONE_B.
        floor_per_alpha = jnp.array(
            [-1, -1, -1, int(a_floor), int(b_floor), -1, -1, -1], dtype=jnp.int32
        )
        n_servers = jnp.where(floor_per_alpha >= 0, floor_per_alpha, random_n)
    else:
        n_servers = random_n

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

    return SimulatorConst(
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
