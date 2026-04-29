"""Numpy-only topology extraction functions. No JAX imports — safe for multiprocessing workers."""

import numpy as np

from jaxborg.constants import (
    CYBORG_SUBNET_SUFFIX,
    CYBORG_SUFFIX_TO_ID,
    GLOBAL_MAX_HOSTS,
    MAX_DETECTION_RANDOMS,
    MAX_SERVER_HOSTS,
    MAX_STEPS,
    MAX_USER_HOSTS,
    MISSION_PHASES,
    NUM_BLUE_AGENTS,
    NUM_DECOY_TYPES,
    NUM_GREEN_RANDOM_FIELDS,
    NUM_RED_AGENTS,
    NUM_RED_POLICY_RANDOM_FIELDS,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
    SERVICE_IDS,
    SERVICE_NAMES,
    SUBNET_IDS,
    SUBNET_NAMES,
    TOTAL_ACTION_ACTOR_SLOTS,
)

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


# Axis B (env-diversity for CEC): per-key router topology variation.
# We keep the default tree above as the always-present base, then per reset key
# sample a subset of these candidate edges to add. Each candidate is a
# plausible "extra" inter-router edge that could exist via misconfiguration or
# design choice in a real enterprise network. See plans/jax/cc4/prompts/phase1-plan.md.
ROUTER_LINK_CANDIDATE_EDGE_NAMES: tuple[tuple[str, str], ...] = (
    ("RESTRICTED_ZONE_A", "RESTRICTED_ZONE_B"),
    ("ADMIN_NETWORK", "RESTRICTED_ZONE_A"),
    ("ADMIN_NETWORK", "RESTRICTED_ZONE_B"),
    ("OFFICE_NETWORK", "RESTRICTED_ZONE_A"),
    ("OFFICE_NETWORK", "RESTRICTED_ZONE_B"),
    ("CONTRACTOR_NETWORK", "PUBLIC_ACCESS_ZONE"),
    ("CONTRACTOR_NETWORK", "ADMIN_NETWORK"),
    ("CONTRACTOR_NETWORK", "OFFICE_NETWORK"),
    ("OPERATIONAL_ZONE_A", "OPERATIONAL_ZONE_B"),
    ("OPERATIONAL_ZONE_A", "RESTRICTED_ZONE_B"),
    ("OPERATIONAL_ZONE_B", "RESTRICTED_ZONE_A"),
    ("INTERNET", "ADMIN_NETWORK"),
)
NUM_ROUTER_LINK_CANDIDATES = len(ROUTER_LINK_CANDIDATE_EDGE_NAMES)


def _default_router_adj() -> np.ndarray:
    """Symmetric (NUM_SUBNETS, NUM_SUBNETS) bool adjacency for the default tree."""
    adj = np.zeros((NUM_SUBNETS, NUM_SUBNETS), dtype=bool)
    for src_name, neighbors in _ROUTER_LINKS.items():
        si = SUBNET_IDS[src_name]
        for dst_name in neighbors:
            di = SUBNET_IDS[dst_name]
            adj[si, di] = True
            adj[di, si] = True
    return adj


def _bfs_reachable(adj: np.ndarray, start: int) -> np.ndarray:
    """Boolean (NUM_SUBNETS,) of which subnets are reachable from `start`."""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    frontier = [start]
    while frontier:
        nxt = []
        for u in frontier:
            for v in range(n):
                if adj[u, v] and not visited[v]:
                    visited[v] = True
                    nxt.append(v)
        frontier = nxt
    return visited


def _validate_router_adj(adj: np.ndarray) -> bool:
    """All required CC4 connectivity invariants for axis B router topologies.

    1. Every subnet reachable from INTERNET (red entry connectivity).
    2. OPERATIONAL_ZONE_A reachable from CONTRACTOR_NETWORK (red_agent_0 must be
       able to reach phase-1 high-value target).
    3. OPERATIONAL_ZONE_B reachable from CONTRACTOR_NETWORK (phase-2 target).
    """
    S = SUBNET_IDS
    from_internet = _bfs_reachable(adj, S["INTERNET"])
    if not from_internet.all():
        return False
    from_contractor = _bfs_reachable(adj, S["CONTRACTOR_NETWORK"])
    if not from_contractor[S["OPERATIONAL_ZONE_A"]]:
        return False
    if not from_contractor[S["OPERATIONAL_ZONE_B"]]:
        return False
    return True


def _build_router_link_bank() -> np.ndarray:
    """Enumerate all 2^k candidate-edge subsets, return a stack of valid adjacencies.

    Bank entry 0 is always the default tree (no extra edges) so the
    vary_router_links=False / topology_fixed_key=0 path matches legacy behavior.
    """
    base = _default_router_adj()
    cand = ROUTER_LINK_CANDIDATE_EDGE_NAMES
    n_cand = len(cand)
    cand_pairs = [(SUBNET_IDS[a], SUBNET_IDS[b]) for a, b in cand]
    valid: list[np.ndarray] = [base.copy()]
    for mask in range(1, 1 << n_cand):
        adj = base.copy()
        for k in range(n_cand):
            if mask & (1 << k):
                a, b = cand_pairs[k]
                adj[a, b] = True
                adj[b, a] = True
        if _validate_router_adj(adj):
            valid.append(adj)
    return np.stack(valid, axis=0)


_ROUTER_LINK_BANK_CACHE: np.ndarray | None = None


def get_router_link_bank() -> np.ndarray:
    """Cached (N_valid, NUM_SUBNETS, NUM_SUBNETS) bool router-adjacency bank.

    Bank[0] is the default tree. All entries pass the connectivity validator.
    """
    global _ROUTER_LINK_BANK_CACHE
    if _ROUTER_LINK_BANK_CACHE is None:
        _ROUTER_LINK_BANK_CACHE = _build_router_link_bank()
    return _ROUTER_LINK_BANK_CACHE


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


def _compute_mission_phases(steps: int) -> tuple:
    quotient, remainder = divmod(steps, 3)
    if remainder == 2:
        return (quotient + 1, quotient + 1, quotient)
    if remainder == 1:
        return (quotient + 1, quotient, quotient)
    return (quotient, quotient, quotient)


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


# Axis C (env-diversity for CEC): per-key phase-reward variation.
# We pick a "primary target" subnet for each of mission phases 1 and 2 (the
# zone that gets the high-value [-10, 0, -10] reward profile).  Defaults
# correspond to OPERATIONAL_ZONE_A (phase 1) and OPERATIONAL_ZONE_B (phase 2);
# variants choose ordered pairs from the candidate set below.
PHASE_REWARDS_PRIMARY_TARGET_NAMES: tuple[str, ...] = (
    "ADMIN_NETWORK",
    "OFFICE_NETWORK",
    "OPERATIONAL_ZONE_A",
    "OPERATIONAL_ZONE_B",
    "RESTRICTED_ZONE_A",
    "RESTRICTED_ZONE_B",
)
NUM_PHASE_REWARDS_PRIMARY_TARGETS = len(PHASE_REWARDS_PRIMARY_TARGET_NAMES)


def _build_phase_rewards_variant(phase1_target_sid: int, phase2_target_sid: int) -> np.ndarray:
    """Return a (MISSION_PHASES, NUM_SUBNETS, 3) variant of the default phase-rewards.

    Starts from :func:`_build_phase_rewards` and re-assigns the high-value
    ``[-10, 0, -10]`` profile in phase 1 (resp. 2) to ``phase1_target_sid``
    (resp. ``phase2_target_sid``). The original op-zone (OPERATIONAL_ZONE_A in
    phase 1, OPERATIONAL_ZONE_B in phase 2) is reset to ``[-1, -1, -1]`` (the
    non-target value the other op-zone already has by default).

    Passing the original op-zones reproduces the default matrix exactly.
    """
    S = SUBNET_IDS
    pr = _build_phase_rewards().copy()
    high = np.array([-10.0, 0.0, -10.0], dtype=np.float32)
    inactive = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

    # Phase 1: original primary is OPERATIONAL_ZONE_A.
    pr[1, S["OPERATIONAL_ZONE_A"]] = inactive
    pr[1, phase1_target_sid] = high

    # Phase 2: original primary is OPERATIONAL_ZONE_B.
    pr[2, S["OPERATIONAL_ZONE_B"]] = inactive
    pr[2, phase2_target_sid] = high

    return pr


def _build_phase_rewards_bank() -> np.ndarray:
    """Build the full phase-rewards bank.

    Bank entry 0 is exactly :func:`_build_phase_rewards` so that the
    ``vary_phase_rewards=False`` path matches legacy behavior. Remaining
    entries enumerate ordered pairs ``(phase1_target, phase2_target)`` drawn
    from :data:`PHASE_REWARDS_PRIMARY_TARGET_NAMES` with the two targets
    distinct (6 × 5 = 30 ordered pairs → 31-entry bank).
    """
    entries: list[np.ndarray] = [_build_phase_rewards()]
    candidate_sids = [SUBNET_IDS[name] for name in PHASE_REWARDS_PRIMARY_TARGET_NAMES]
    for p1 in candidate_sids:
        for p2 in candidate_sids:
            if p1 == p2:
                continue
            entries.append(_build_phase_rewards_variant(p1, p2))
    return np.stack(entries, axis=0).astype(np.float32)


# Phase 3 mission-objective family: per-key sampling of the CIA-component
# multiplier triple (LWF, ASF, RIA).  bank[0] is the default (1, 1, 1) so the
# vary_mission_profile=False path matches legacy behavior.  The remaining three
# entries each privilege one CIA dimension at 10x with off-axis components held
# at 1.0 (amplify-only, no damping — see Phase 3 spike 1: per-profile reward
# swing is dominated by amplification, damping contributes <1%).
MISSION_PROFILE_MULTIPLIERS: tuple[tuple[float, float, float], ...] = (
    # (LWF, ASF, RIA)
    (1.0, 1.0, 1.0),  # default — balanced
    (1.0, 10.0, 1.0),  # availability-heavy: amplify ASF
    (10.0, 1.0, 1.0),  # productivity-heavy: amplify LWF
    (1.0, 1.0, 10.0),  # CI-heavy: amplify RIA
)
NUM_MISSION_PROFILES = len(MISSION_PROFILE_MULTIPLIERS)


def get_mission_profile_multipliers() -> np.ndarray:
    """(NUM_MISSION_PROFILES, 3) float32 multipliers in (LWF, ASF, RIA) order."""
    return np.asarray(MISSION_PROFILE_MULTIPLIERS, dtype=np.float32)


_PHASE_REWARDS_BANK_CACHE: np.ndarray | None = None


def get_phase_rewards_bank() -> np.ndarray:
    """Cached (N, MISSION_PHASES, NUM_SUBNETS, 3) float32 phase-rewards bank.

    ``bank[0]`` equals :func:`_build_phase_rewards`. Remaining entries are
    valid variants per :func:`_build_phase_rewards_variant`.
    """
    global _PHASE_REWARDS_BANK_CACHE
    if _PHASE_REWARDS_BANK_CACHE is None:
        _PHASE_REWARDS_BANK_CACHE = _build_phase_rewards_bank()
    return _PHASE_REWARDS_BANK_CACHE


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


# Axis D (Phase 2 OOD eval): per-key allowed_subnet_pairs variation.  Bank[0]
# is the default policy matrix above so vary_subnet_pairs=False reproduces
# legacy behavior.  Remaining entries permute the per-phase policies and
# rotate which mission phase carries which policy, then validate that the
# default green-allowable communication links survive (the ones referenced
# by the comms_policy bypass list — RZ_A↔OZ_A, RZ_B↔OZ_B, plus a baseline
# of CN-anchored links).
def _required_pairs_per_phase() -> list[set[tuple[int, int]]]:
    """Pairs that MUST be allowed in every phase to keep green plays solvable.

    Common base set — the comms-bypass uplinks that the always-active green
    flows depend on.  Per-phase op-zone uplinks are deliberately *not*
    required, so axis-D permutations can rotate which mission phase carries
    which op-zone link.
    """
    S = SUBNET_IDS
    base = {
        (S["PUBLIC_ACCESS_ZONE"], S["CONTRACTOR_NETWORK"]),
        (S["ADMIN_NETWORK"], S["CONTRACTOR_NETWORK"]),
        (S["OFFICE_NETWORK"], S["CONTRACTOR_NETWORK"]),
        (S["PUBLIC_ACCESS_ZONE"], S["RESTRICTED_ZONE_A"]),
        (S["PUBLIC_ACCESS_ZONE"], S["RESTRICTED_ZONE_B"]),
    }
    return [base, base, base]


def _phase_pairs_set(pairs: np.ndarray, phase_idx: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for i in range(NUM_SUBNETS):
        for j in range(NUM_SUBNETS):
            if pairs[phase_idx, i, j]:
                out.add((i, j))
                out.add((j, i))
    return out


def _validate_subnet_pairs(pairs: np.ndarray) -> bool:
    required = _required_pairs_per_phase()
    for phase_idx, req in enumerate(required):
        present = _phase_pairs_set(pairs, phase_idx)
        for si, di in req:
            if (si, di) not in present:
                return False
    return True


def _build_subnet_pairs_bank() -> np.ndarray:
    """Bank of validated allowed_subnet_pairs matrices.

    Bank[0] is the default :func:`_build_allowed_subnet_pairs_pure`.  Remaining
    entries rotate the three per-phase policies and add a swapped variant
    where phase 1 and phase 2 use each other's primary op-zone link.  Each
    candidate is validated against :func:`_required_pairs_per_phase`.
    """
    S = SUBNET_IDS
    default = _build_allowed_subnet_pairs_pure()

    # Decompose default into per-phase policies (sets of (si,di) ordered pairs
    # — matrix is symmetric by construction so we keep both directions).
    phase_sets = [
        {(i, j) for i in range(NUM_SUBNETS) for j in range(NUM_SUBNETS) if default[k, i, j]}
        for k in range(MISSION_PHASES)
    ]

    def assemble(per_phase_sets: list[set[tuple[int, int]]]) -> np.ndarray:
        arr = np.zeros((MISSION_PHASES, NUM_SUBNETS, NUM_SUBNETS), dtype=bool)
        for k, s in enumerate(per_phase_sets):
            for si, di in s:
                arr[k, si, di] = True
                arr[k, di, si] = True
        return arr

    candidates: list[np.ndarray] = [default]

    # Variant 1: rotate (phase 0 → 1 → 2 → 0).
    candidates.append(assemble([phase_sets[2], phase_sets[0], phase_sets[1]]))
    # Variant 2: reverse rotate.
    candidates.append(assemble([phase_sets[1], phase_sets[2], phase_sets[0]]))

    # Variant 3: swap phase 1 ↔ phase 2 op-zone connectivity.
    swapped = [set(s) for s in phase_sets]
    pair_a = (S["OPERATIONAL_ZONE_A"], S["RESTRICTED_ZONE_A"])
    pair_b = (S["OPERATIONAL_ZONE_B"], S["RESTRICTED_ZONE_B"])
    swapped[1].discard(pair_a)
    swapped[1].discard(pair_a[::-1])
    swapped[1].add(pair_b)
    swapped[1].add(pair_b[::-1])
    swapped[2].discard(pair_b)
    swapped[2].discard(pair_b[::-1])
    swapped[2].add(pair_a)
    swapped[2].add(pair_a[::-1])
    candidates.append(assemble(swapped))

    # Variant 4: union — phase 0 always sees every link from any phase.
    union_p0 = phase_sets[0] | phase_sets[1] | phase_sets[2]
    candidates.append(assemble([union_p0, phase_sets[1], phase_sets[2]]))

    valid = [c for c in candidates if _validate_subnet_pairs(c)]
    return np.stack(valid, axis=0)


_SUBNET_PAIRS_BANK_CACHE: np.ndarray | None = None


def get_subnet_pairs_bank() -> np.ndarray:
    """Cached (N, MISSION_PHASES, NUM_SUBNETS, NUM_SUBNETS) bool bank.

    ``bank[0]`` reproduces :func:`_build_allowed_subnet_pairs_pure`.  Remaining
    entries pass :func:`_validate_subnet_pairs`.
    """
    global _SUBNET_PAIRS_BANK_CACHE
    if _SUBNET_PAIRS_BANK_CACHE is None:
        _SUBNET_PAIRS_BANK_CACHE = _build_subnet_pairs_bank()
    return _SUBNET_PAIRS_BANK_CACHE


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


def build_const_arrays_from_cyborg(cyborg_env) -> dict:
    """Extract static topology from a live CybORG environment.

    Returns a plain dict of numpy arrays with keys matching CC4Const field names.
    """
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

    return {
        "host_active": np.array(host_active),
        "host_subnet": np.array(host_subnet),
        "host_is_router": np.array(host_is_router),
        "host_is_server": np.array(host_is_server),
        "host_is_user": np.array(host_is_user),
        "subnet_adjacency": np.array(subnet_adjacency),
        "data_links": np.array(data_links),
        "initial_services": np.array(initial_services),
        "host_has_bruteforceable_user": np.array(host_has_bruteforceable_user),
        "host_has_rfi": np.array(host_has_rfi),
        "host_respond_to_ping": np.array(host_respond_to_ping),
        "host_initial_max_pid": np.array(host_initial_max_pid),
        "blue_agent_subnets": np.array(blue_agent_subnets),
        "blue_agent_hosts": np.array(blue_agent_hosts),
        "red_start_hosts": np.array(red_start_hosts),
        "red_agent_subnets": np.array(red_agent_subnets),
        "red_initial_discovered_hosts": np.array(red_initial_discovered_hosts),
        "red_initial_scanned_hosts": np.array(red_initial_scanned_hosts),
        "host_info_links": np.array(host_info_links),
        "green_agent_host": np.array(green_agent_host),
        "green_agent_active": np.array(green_agent_active),
        "num_green_agents": np.int32(green_count),
        "phase_rewards": np.array(_build_phase_rewards_from_cyborg(cyborg_env)),
        "phase_boundaries": np.array(phase_boundaries),
        "allowed_subnet_pairs": np.array(allowed_subnet_pairs),
        "obs_host_map": np.array(obs_host_map),
        "blue_obs_subnets": np.array(_build_blue_obs_subnets()),
        "comms_policy": np.array(_build_comms_policy()),
        "max_steps": np.int32(500),
        "num_hosts": np.int32(num_hosts),
        "green_agents_active": np.array(True),
        # Precomputed RNG arrays (defaults: disabled / zeros)
        "green_randoms": np.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32),
        "use_green_randoms": np.array(False),
        "red_policy_randoms": np.full((MAX_STEPS, NUM_RED_AGENTS, NUM_RED_POLICY_RANDOM_FIELDS), 0.5, dtype=np.float32),
        "use_red_policy_randoms": np.array(False),
        "detection_randoms": np.zeros(MAX_DETECTION_RANDOMS, dtype=np.float32),
        "use_detection_randoms": np.array(False),
        "red_pid_deltas": np.zeros((MAX_STEPS, NUM_RED_AGENTS), dtype=np.int32),
        "use_red_pid_deltas": np.array(False),
        "blue_decoy_pid_deltas": np.zeros((MAX_STEPS, NUM_BLUE_AGENTS, NUM_DECOY_TYPES), dtype=np.int32),
        "use_blue_decoy_pid_deltas": np.array(False),
        "red_privesc_choices": np.zeros((MAX_STEPS, NUM_RED_AGENTS), dtype=np.int32),
        "use_red_privesc_choices": np.array(False),
        "red_session_check_choices": np.zeros((MAX_STEPS, NUM_RED_AGENTS), dtype=np.int32),
        "red_session_check_hosts": np.full((MAX_STEPS, NUM_RED_AGENTS), -1, dtype=np.int32),
        "use_red_session_check_choices": np.array(False),
        "blue_decoy_type_choices": np.zeros((MAX_STEPS, NUM_BLUE_AGENTS), dtype=np.int32),
        "use_blue_decoy_type_choices": np.array(False),
        "green_host_order": np.zeros((MAX_STEPS, TOTAL_ACTION_ACTOR_SLOTS), dtype=np.int32),
        "use_green_host_order": np.array(False),
        "red_exploit_session_choices": np.zeros((MAX_STEPS, NUM_RED_AGENTS), dtype=np.int32),
        "use_red_exploit_session_choices": np.array(False),
        "mission_profile_index": np.int32(0),
        "mission_multipliers": np.ones(3, dtype=np.float32),
        "obs_mission_goal": np.array(False),
        "subnet_pairs_bank_index": np.int32(0),
    }
