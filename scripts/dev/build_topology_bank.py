"""Build a bank of CC4 topology snapshots covering structural variation.

Phase 6 stream S1 — produces the "kitchen layouts" analog: 16 (or N) valid
topology snapshots that vary along three axes:

1. Router adjacency — perturbations of `_ROUTER_LINKS` post-applied to
   `data_links` (e.g. add an op-zone-A ↔ op-zone-B router cross-link, drop
   the office ↔ admin link).
2. Subnet sizing — `op_zone_min_servers ∈ {2, 3, 4}` plus seed-driven user
   host counts.
3. Cross-segment allow-list — perturbations of `allowed_subnet_pairs`
   (zero-out one phase pair on certain shapes).

Every emitted snapshot is loaded via `load_topology` and validated under
`_validate_resilience_topology(CIA_RESILIENCE, [p])`; bad shapes abort
the run.

Usage::

    python scripts/dev/build_topology_bank.py \
        --out-dir scripts/dev/topology_bank/ --count 16 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import GLOBAL_MAX_HOSTS, SUBNET_IDS
from jaxborg.evaluation.jax_env_factory import _validate_resilience_topology
from jaxborg.scenarios.cc4.game_variants import CIA_RESILIENCE
from jaxborg.scenarios.cc4.topology import build_topology, load_topology, save_topology
from jaxborg.state import SimulatorConst

# 4 perturbation patterns × 4 base seeds = 16 shapes by default.
# Patterns vary along multiple axes simultaneously so that even a small
# bank covers router-adjacency, sizing, and allow-list variation.
_PATTERNS = (
    {
        "name": "P0_baseline_minservers3",
        "op_zone_min_servers": 3,
        "router_perturbation": "none",
        "allowlist_perturbation": "none",
    },
    {
        "name": "P1_minservers2_dropOA",
        "op_zone_min_servers": 2,
        "router_perturbation": "drop_office_admin",
        "allowlist_perturbation": "none",
    },
    {
        "name": "P2_minservers4_OPxlink",
        "op_zone_min_servers": 4,
        "router_perturbation": "add_opzone_xlink",
        "allowlist_perturbation": "none",
    },
    {
        "name": "P3_minservers3_restrictpairs",
        "op_zone_min_servers": 3,
        "router_perturbation": "none",
        "allowlist_perturbation": "drop_phase1_contractor_admin",
    },
)


def _router_host_idx_per_subnet(const: SimulatorConst) -> dict[int, int]:
    """Return {subnet_id: host_idx_of_router} for active subnets."""
    host_subnet = np.asarray(const.host_subnet)
    host_is_router = np.asarray(const.host_is_router)
    host_active = np.asarray(const.host_active)
    out: dict[int, int] = {}
    for h in range(GLOBAL_MAX_HOSTS):
        if host_active[h] and host_is_router[h]:
            sid = int(host_subnet[h])
            # Only one router per subnet — first wins.
            out.setdefault(sid, h)
    return out


def _apply_router_perturbation(const: SimulatorConst, kind: str) -> SimulatorConst:
    if kind == "none":
        return const

    routers = _router_host_idx_per_subnet(const)
    data_links = np.asarray(const.data_links).copy()

    if kind == "add_opzone_xlink":
        # Wire the two operational-zone routers directly to each other.
        a = routers.get(SUBNET_IDS["OPERATIONAL_ZONE_A"])
        b = routers.get(SUBNET_IDS["OPERATIONAL_ZONE_B"])
        if a is not None and b is not None:
            data_links[a, b] = True
            data_links[b, a] = True
    elif kind == "drop_office_admin":
        # Cut the office ↔ admin secondary path; both still reach PAZ.
        # In stock _ROUTER_LINKS, office and admin only connect via
        # PUBLIC_ACCESS_ZONE — there's no direct link to drop. To still
        # produce a meaningful router-adjacency perturbation, drop the
        # OFFICE_NETWORK ↔ PUBLIC_ACCESS_ZONE router link instead, which
        # isolates OFFICE_NETWORK from the rest of the routing fabric.
        office = routers.get(SUBNET_IDS["OFFICE_NETWORK"])
        paz = routers.get(SUBNET_IDS["PUBLIC_ACCESS_ZONE"])
        if office is not None and paz is not None:
            data_links[office, paz] = False
            data_links[paz, office] = False
    else:
        raise ValueError(f"unknown router_perturbation: {kind!r}")

    return const.replace(data_links=jnp.asarray(data_links))


def _apply_allowlist_perturbation(const: SimulatorConst, kind: str) -> SimulatorConst:
    if kind == "none":
        return const

    pairs = np.asarray(const.allowed_subnet_pairs).copy()

    if kind == "drop_phase1_contractor_admin":
        # Zero a single phase-1 cross-segment pair so the allow-list bank
        # actually varies. CONTRACTOR_NETWORK ↔ ADMIN_NETWORK is present
        # in the stock phase-1 allow-list, so dropping it produces a
        # measurable reduction in the bank summary table.
        s_con = SUBNET_IDS["CONTRACTOR_NETWORK"]
        s_adm = SUBNET_IDS["ADMIN_NETWORK"]
        pairs[1, s_con, s_adm] = False
        pairs[1, s_adm, s_con] = False
    else:
        raise ValueError(f"unknown allowlist_perturbation: {kind!r}")

    return const.replace(allowed_subnet_pairs=jnp.asarray(pairs))


def _data_link_router_hash(const: SimulatorConst) -> str:
    """Hash of the inter-subnet router-router data_links (adjacency fingerprint)."""
    routers = _router_host_idx_per_subnet(const)
    sids = sorted(routers.keys())
    data_links = np.asarray(const.data_links)
    bits = []
    for i, si in enumerate(sids):
        for sj in sids[i + 1 :]:
            ri = routers[si]
            rj = routers[sj]
            bits.append("1" if data_links[ri, rj] else "0")
    return hashlib.sha1("".join(bits).encode()).hexdigest()[:8]


def _allowed_pair_count(const: SimulatorConst) -> int:
    return int(np.asarray(const.allowed_subnet_pairs).sum())


def _summarize(const: SimulatorConst) -> dict:
    host_subnet = np.asarray(const.host_subnet)
    host_active = np.asarray(const.host_active)
    host_is_user = np.asarray(const.host_is_user)
    host_is_server = np.asarray(const.host_is_server)

    user_subnets = sorted({int(host_subnet[h]) for h in range(GLOBAL_MAX_HOSTS) if host_active[h] and host_is_user[h]})
    server_subnets = sorted(
        {int(host_subnet[h]) for h in range(GLOBAL_MAX_HOSTS) if host_active[h] and host_is_server[h]}
    )

    op_zone_a = int(SUBNET_IDS["OPERATIONAL_ZONE_A"])
    op_zone_b = int(SUBNET_IDS["OPERATIONAL_ZONE_B"])
    op_zone_servers = sum(
        1
        for h in range(GLOBAL_MAX_HOSTS)
        if host_active[h] and host_is_server[h] and host_subnet[h] in (op_zone_a, op_zone_b)
    )

    return {
        "router_hash": _data_link_router_hash(const),
        "num_user_subnets": len(user_subnets),
        "num_server_subnets": len(server_subnets),
        "op_zone_servers": op_zone_servers,
        "allowed_pair_count": _allowed_pair_count(const),
        "num_hosts": int(np.asarray(const.num_hosts)),
    }


def build_one(seed: int, pattern: dict) -> SimulatorConst:
    const = build_topology(
        jax.random.PRNGKey(seed),
        op_zone_min_servers=pattern["op_zone_min_servers"],
    )
    const = _apply_router_perturbation(const, pattern["router_perturbation"])
    const = _apply_allowlist_perturbation(const, pattern["allowlist_perturbation"])
    return const


def build_bank(out_dir: Path, count: int, base_seed: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    n_patterns = len(_PATTERNS)
    rows: list[tuple[str, dict, int]] = []

    for i in range(count):
        pattern = _PATTERNS[i % n_patterns]
        seed = base_seed + (i // n_patterns)
        const = build_one(seed, pattern)
        # ``np.savez_compressed`` appends ``.npz`` if not present; use the
        # extension explicitly so the on-disk filename matches what we
        # return and what callers (recipe + tests) reference.
        path = out_dir / f"shape_{i:02d}.snapshot.npz"
        save_topology(
            const,
            path,
            metadata={
                "source": "phase6_topology_bank",
                "shape_index": i,
                "pattern_name": pattern["name"],
                "source_seed": seed,
                "op_zone_min_servers": pattern["op_zone_min_servers"],
                "router_perturbation": pattern["router_perturbation"],
                "allowlist_perturbation": pattern["allowlist_perturbation"],
            },
        )
        paths.append(path)

    # Validate that every snapshot loads cleanly and passes the
    # resilience-topology check for the strictest variant we care about.
    _validate_resilience_topology(CIA_RESILIENCE, paths)
    for p in paths:
        # Round-trip load to confirm the on-disk snapshot is parseable.
        const = load_topology(p)
        rows.append((p.name, _summarize(const), int(p.stat().st_size)))

    _print_summary_table(rows)
    return paths


def _print_summary_table(rows: list[tuple[str, dict, int]]) -> None:
    header = (
        f"{'shape':<22}{'router_hash':<13}{'#user_sub':<11}{'#srv_sub':<10}"
        f"{'#opzone_srv':<13}{'#allowpair':<12}{'#hosts':<8}{'bytes':<10}"
    )
    print(header)
    print("-" * len(header))
    for name, s, nbytes in rows:
        print(
            f"{name:<22}{s['router_hash']:<13}{s['num_user_subnets']:<11}"
            f"{s['num_server_subnets']:<10}{s['op_zone_servers']:<13}"
            f"{s['allowed_pair_count']:<12}{s['num_hosts']:<8}{nbytes:<10}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bank of CC4 topology snapshots for Phase 6 Axis A.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scripts/dev/topology_bank"),
        help="Directory to emit shape_NN.snapshot files into.",
    )
    parser.add_argument("--count", type=int, default=16, help="Number of snapshots to emit.")
    parser.add_argument("--seed", type=int, default=0, help="Base PRNG seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = build_bank(args.out_dir, count=args.count, base_seed=args.seed)
    print(f"\nEmitted {len(paths)} snapshots to {args.out_dir}")


if __name__ == "__main__":
    main()
