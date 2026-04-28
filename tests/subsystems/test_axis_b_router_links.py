"""Axis B (CEC env-diversity) router-topology variation tests.

Covers:
1. Default-tree parity: vary_router_links=False reproduces legacy data_links
   and host_info_links exactly (already covered implicitly elsewhere; here
   we cross-check against the literal _ROUTER_LINKS dict).
2. Bank well-formedness: every entry symmetric, validator-passing,
   bank[0] equals the default tree.
3. vary_router_links=True produces multiple distinct topologies across keys.
4. topology_fixed_key collapses every reset to a single topology.
5. Default-tree edges are ALWAYS present (we never remove edges).
"""

import jax
import numpy as np

from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_SUBNETS, SUBNET_IDS
from jaxborg.topology import build_topology
from jaxborg.topology_numpy import (
    _ROUTER_LINKS,
    _default_router_adj,
    _validate_router_adj,
    get_router_link_bank,
)


def test_bank_shape_and_validity():
    bank = get_router_link_bank()
    assert bank.ndim == 3
    assert bank.shape[1] == NUM_SUBNETS
    assert bank.shape[2] == NUM_SUBNETS
    # 12 candidate edges, default tree always valid → all 2^12 subsets pass.
    assert bank.shape[0] == 4096


def test_bank_entries_symmetric_and_valid():
    bank = get_router_link_bank()
    # Sample a chunk for symmetry + validator (full sweep is 4096 BFS calls).
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(bank.shape[0], size=200, replace=False)
    for i in sample_idx:
        adj = bank[i]
        assert np.array_equal(adj, adj.T), f"bank[{i}] not symmetric"
        assert _validate_router_adj(adj), f"bank[{i}] fails validator"


def test_bank_zero_is_default_tree():
    bank = get_router_link_bank()
    assert np.array_equal(bank[0], _default_router_adj())


def test_default_tree_matches_legacy_router_links_dict():
    base = _default_router_adj()
    for src, neighbors in _ROUTER_LINKS.items():
        si = SUBNET_IDS[src]
        for dst in neighbors:
            di = SUBNET_IDS[dst]
            assert base[si, di], f"default missing {src}->{dst}"
            assert base[di, si], f"default missing {dst}->{src}"


def test_default_path_data_links_unchanged():
    """Inter-router edges from the default tree must be present in data_links."""
    for seed in [0, 1, 7, 42]:
        c = build_topology(jax.random.PRNGKey(seed), vary_router_links=False)
        sri = np.full(NUM_SUBNETS, -1, dtype=np.int32)
        host_subnet = np.array(c.host_subnet)
        host_active = np.array(c.host_active)
        host_is_router = np.array(c.host_is_router)
        for h in range(GLOBAL_MAX_HOSTS):
            if not host_active[h]:
                continue
            sid = int(host_subnet[h])
            if host_is_router[h]:
                sri[sid] = h
        # Internet host is the only active host in INTERNET subnet (no router flag).
        for h in range(GLOBAL_MAX_HOSTS):
            if host_active[h] and host_subnet[h] == SUBNET_IDS["INTERNET"]:
                sri[SUBNET_IDS["INTERNET"]] = h
                break
        dl = np.array(c.data_links)
        for src_name, neighbors in _ROUTER_LINKS.items():
            sr = int(sri[SUBNET_IDS[src_name]])
            for dst_name in neighbors:
                dr = int(sri[SUBNET_IDS[dst_name]])
                assert dl[sr, dr], f"seed={seed} missing data_link {src_name}->{dst_name}"
                assert dl[dr, sr], f"seed={seed} missing data_link {dst_name}->{src_name}"


def test_vary_router_links_produces_distinct_topologies():
    sums = set()
    for seed in range(32):
        c = build_topology(jax.random.PRNGKey(seed), vary_router_links=True)
        sums.add(int(c.data_links.sum()))
    # With 32 keys over a 4096-bank we expect many distinct topologies.
    assert len(sums) >= 5, f"only {len(sums)} distinct data_links sums in 32 keys"


def test_vary_router_links_default_edges_always_present():
    """Even when extra edges are added, default tree edges must remain."""
    base = _default_router_adj()
    for seed in range(24):
        c = build_topology(jax.random.PRNGKey(seed), vary_router_links=True)
        # Reconstruct router_idx
        sri = np.full(NUM_SUBNETS, -1, dtype=np.int32)
        host_subnet = np.array(c.host_subnet)
        host_active = np.array(c.host_active)
        host_is_router = np.array(c.host_is_router)
        for h in range(GLOBAL_MAX_HOSTS):
            if host_active[h] and host_is_router[h]:
                sri[int(host_subnet[h])] = h
        for h in range(GLOBAL_MAX_HOSTS):
            if host_active[h] and host_subnet[h] == SUBNET_IDS["INTERNET"]:
                sri[SUBNET_IDS["INTERNET"]] = h
                break
        dl = np.array(c.data_links)
        for i in range(NUM_SUBNETS):
            for j in range(NUM_SUBNETS):
                if base[i, j]:
                    si, sj = int(sri[i]), int(sri[j])
                    assert dl[si, sj], f"seed={seed} default edge {i}->{j} missing"


def test_topology_fixed_key_constant_across_envs():
    """topology_fixed_key collapses topology across reset keys."""
    from jaxborg.fsm_red_env import FsmRedCC4Env

    env_fixed = FsmRedCC4Env(num_steps=10, topology_mode="generative", topology_fixed_key=42)
    obs0, st0 = env_fixed.reset(jax.random.PRNGKey(0))
    obs1, st1 = env_fixed.reset(jax.random.PRNGKey(1))
    obs2, st2 = env_fixed.reset(jax.random.PRNGKey(99999))
    # Same topology each reset.
    assert (st0.const.data_links == st1.const.data_links).all()
    assert (st0.const.data_links == st2.const.data_links).all()
    assert (st0.const.host_subnet == st1.const.host_subnet).all()
